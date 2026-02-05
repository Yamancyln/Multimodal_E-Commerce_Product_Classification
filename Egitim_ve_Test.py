import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel

# Ayarlar
# dosya yolunu bulma
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'data.csv')
IMG_DIR = os.path.join(script_dir, 'images')

CONFIG = {
    'BATCH_SIZE': 8,         
    'EPOCHS': 3,            
    'LR': 2e-5,
    'MAX_LEN': 32,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"İşlem {CONFIG['DEVICE']} cihazında başlatılıyor...")
print(f"Veri Yolu: {DATA_PATH}")

# DATASET SINIFI 
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # resim yükleme
        img_name = str(row['image_id']) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Resim bulunamazsa siyah ekran verir (Hata önleyici)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        # metin işleme
        text = str(row['text_description'])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=CONFIG['MAX_LEN'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

# Görüntü Transformasyonları 
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resmi modelin beklediği standart boyuta (224x224 piksel) getir.
    transforms.ToTensor(), # Resmi (0-255 arası renkler) Pytorch'un anlayacağı sayı dizisine (Tensor) çevirir, sayıları 0 ile 1 arasına sıkıştırır.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# Sayıları normalize et (İstatistiksel dengeleme). 
    # Bu sayılar ImageNet veri setinin standart ortalamasıdır. 
    # Modelin daha hızlı ve kararlı öğrenmesini sağlar.
])

# FUSION MİMARİLERİ

# Özellik Çıkarıcılar
# Ortak Parçalar (Encoder'lar)
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        base = models.resnet18(pretrained=True) # Dünyadaki nesneleri zaten bilen, eğitilmiş ResNet18 modelini indir.
        self.base = nn.Sequential(*list(base.children())[:-1])# resmin özelliklerinin sayısal değerlerini alır.
        
    def forward(self, x):
        x = self.base(x)
        return x.view(x.size(0), -1) # çıkan sonucu düzleştirir [Batch, 512, 1, 1] olan boyutu [Batch, 512] yapar.

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') #ingilizceyi bilen eğitilmiş BERT modelini indirir.
        
    def forward(self, input_ids, mask):
        out = self.bert(input_ids=input_ids, attention_mask=mask) # kelimeleri BERT e verir.
        return out.last_hidden_state[:, 0, :] # Cümlenin tamamını temsil eden ilk tokeni (CLS token) alır bu 768 tane sayıdan oluşan bir özettir

# --- FUSION MODELLERİ ---

# MODEL 1: EARLY FUSION
class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.img_enc = ImageEncoder()
        self.text_enc = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256), # Giriş: 512 (Resimden) + 768 (Metinden) = 1280 sayı giriyor.
            nn.ReLU(), # Negatif sayıları sıfırlar
            nn.Dropout(0.2), # Ezberlemeyi önlemek için bilgilerin %20'sini rastgele unutur
            nn.Linear(256, num_classes) # son karar 3 sınıftan biridir
        )
    def forward(self, img, ids, mask):
        i_feat = self.img_enc(img) # Resimden özellikleri çıkarır (512 sayı)
        t_feat = self.text_enc(ids, mask) # Metinden özellikleri çıkarır (768 sayı)
        combined = torch.cat((i_feat, t_feat), dim=1) # İki listeyi yan yana yapıştırır. Örn: Resimler + Metinler = Hepsi
        return self.classifier(combined)

# MODEL 2: INTERMEDIATE FUSION (CROSS-ATTENTION)
class IntermediateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(IntermediateFusionModel, self).__init__()
        self.img_enc = ImageEncoder()
        self.text_enc = TextEncoder()
        
        # boyut eşitleme
        # İkisin dilini eşitlemek için boyutları 256'ya düşürür.
        self.img_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(768, 256)
        # Dikkat Mekanizması: Modelin odaklanmasını sağlar.
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc = nn.Linear(256, num_classes) # Son karar katmanı
        
    def forward(self, img, ids, mask):
        i_feat = self.img_proj(self.img_enc(img)).unsqueeze(1) # [B, 1, 256]
        t_feat = self.text_proj(self.text_enc(ids, mask)).unsqueeze(1) # [B, 1, 256]
        #Cross-Attention
        # Query (Soran): Resim
        # Key/Value (Cevap Anahtarı): Metin
        # "Bendeki bu şekil, sendeki hangi kelimeye benziyor mu"
        attn_out, _ = self.attn(query=i_feat, key=t_feat, value=t_feat)
        return self.fc(attn_out.squeeze(1))

# eğitim 
def train_model(model, train_dl, val_dl, name="Model"):
    print(f"\n>>> {name} EĞİTİMİ BAŞLIYOR...")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR']) #Modelin hatasını düzelten algoritma
    criterion = nn.CrossEntropyLoss() #Hata hesaplayıcı (Cevap anahtarı kontrolü)
    model.to(CONFIG['DEVICE'])
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for batch in train_dl:
            img = batch['image'].to(CONFIG['DEVICE'])
            ids = batch['input_ids'].to(CONFIG['DEVICE'])
            mask = batch['attention_mask'].to(CONFIG['DEVICE'])
            lbl = batch['label'].to(CONFIG['DEVICE'])
            
            optimizer.zero_grad() #öncekin temizleme
            out = model(img, ids, mask) # 1. Tahmin Yap (Forward Pass)
            loss = criterion(out, lbl) # 2. Hatayı Hesapla (Loss) - "Ne kadar yanlış bildim?"
            loss.backward() # 3. Hatayı Geriye Yay (Backward Pass) - "Hatanın sebebi hangi nöron?"
            optimizer.step() # 4. Ağırlıkları Güncelle - "Bir dahaki sefere daha doğru yap."
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Loss: {total_loss/len(train_dl):.4f}")

    # Test
    model.eval() # Öğrenmeyi kapatır sadece bildiğini okur
    preds, true_lbls = [], []
    with torch.no_grad():
        for batch in val_dl:
            img = batch['image'].to(CONFIG['DEVICE'])
            ids = batch['input_ids'].to(CONFIG['DEVICE'])
            mask = batch['attention_mask'].to(CONFIG['DEVICE'])
            lbl = batch['label'].to(CONFIG['DEVICE'])
            out = model(img, ids, mask) #verileri alır
            preds.extend(torch.argmax(out, 1).cpu().numpy()) # En yüksek puanı alan sınıfı seçer
            true_lbls.extend(lbl.cpu().numpy())
    # Sonuçları hesaplar        
    acc = accuracy_score(true_lbls, preds)
    f1 = f1_score(true_lbls, preds, average='macro')
    print(f"{name} SONUÇ -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    return acc, f1

#çalıştırma
if __name__ == "__main__":
    # dosya kontrolü ve veri okuma
    if not os.path.exists(DATA_PATH):
        print(f"HATA: {DATA_PATH} bulunamadı!")
        print("Lütfen veri_hazirla.py dosyasını çalıştırdığından emin ol.")
        exit()
        
    df = pd.read_csv(DATA_PATH)
    
    try:
    # data.csv dosyasındaki veriyi 300'e düşürerek her sınıftan eşit sayıda (100'er) örnek alma
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=100, random_state=42))
        print(f"Veri seti dengeli şekilde 300'e düşürüldü (Her sınıftan 100 adet).")
    except ValueError:
        print("Uyarı: Bazı sınıflarda 100'den az veri vardı, hepsi alındı.") # Eğer bir sınıfta 100'den az veri varsa olduğu kadarını alır (Hata önleyici)

    # Veriyi karıştırma
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sınıf sayısını tekrar hesaplar
    num_classes = len(df['label'].unique())
    print(f"Veri Yüklendi. Toplam: {len(df)}, Sınıf Sayısı: {num_classes}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Train/Val Split (Eğitim ve Test verisini ayırır)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Dataset ve Dataloader oluşturur     
    # IMG_DIR burada parametre olarak geçiliyor
    train_ds = MultimodalDataset(train_df, IMG_DIR, tokenizer, transform)
    val_ds = MultimodalDataset(val_df, IMG_DIR, tokenizer, transform)
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'])
    
    # EARLY FUSION eğitimi    
    model_early = EarlyFusionModel(num_classes)
    train_model(model_early, train_dl, val_dl, name="EARLY FUSION")
    # INTERMEDIATE FUSION eğitimi
    model_inter = IntermediateFusionModel(num_classes)
    train_model(model_inter, train_dl, val_dl, name="INTERMEDIATE FUSION")
    
    print("\n İşlem Tamamlandı!")