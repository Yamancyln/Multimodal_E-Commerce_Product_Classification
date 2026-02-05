import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'data.csv')
IMG_DIR = os.path.join(script_dir, 'images')
DEVICE = 'cpu' 

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
        img_name = str(row['image_id']) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        text = str(row['text_description'])
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=32,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'text_raw': text, # ham metni döndürür
            'img_name': img_name
        }

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resmi modelin beklediği standart boyuta (224x224 piksel) getir.
    transforms.ToTensor(), # Resmi (0-255 arası renkler) Pytorch'un anlayacağı sayı dizisine (Tensor) çevirir, sayıları 0 ile 1 arasına sıkıştırır.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Sayıları normalize et (İstatistiksel dengeleme). 
    # Bu sayılar ImageNet veri setinin standart ortalamasıdır. 
    # Modelin daha hızlı ve kararlı öğrenmesini sağlar.
])

#modeller
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        base = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
    def forward(self, x):
        return self.base(x).view(x.size(0), -1)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, input_ids, mask):
        return self.bert(input_ids=input_ids, attention_mask=mask).last_hidden_state[:, 0, :]

class IntermediateFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(IntermediateFusionModel, self).__init__()
        self.img_enc = ImageEncoder()
        self.text_enc = TextEncoder()
        self.img_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(768, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, img, ids, mask):
        i_feat = self.img_proj(self.img_enc(img)).unsqueeze(1)
        t_feat = self.text_proj(self.text_enc(ids, mask)).unsqueeze(1)
        attn_out, _ = self.attn(query=i_feat, key=t_feat, value=t_feat)
        return self.fc(attn_out.squeeze(1))

#gösrselleştirme
def visualize_results(model, dataloader, label_map):
    model.eval()
    
    # batch verisi çeker
    data = next(iter(dataloader))
    
    images = data['image']
    input_ids = data['input_ids']
    masks = data['attention_mask']
    labels = data['label']
    raw_texts = data['text_raw']
    
    #Tahmin
    with torch.no_grad():
        outputs = model(images, input_ids, masks)
        _, preds = torch.max(outputs, 1)
    
    # Görüntüyü eski haline döndür (Normalize işlemini tersine çevir)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    plt.figure(figsize=(15, 8))
    for i in range(min(5, len(images))):
        img_tensor = images[i] * std + mean 
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        true_label = label_map[labels[i].item()]
        pred_label = label_map[preds[i].item()]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img_np)
        plt.axis('off')
        
        title = f"Gerçek: {true_label}\nTahmin: {pred_label}"
        plt.title(title, color=color, fontweight='bold')
        
        # Konsola da yazdıralım
        print(f"--- Örnek {i+1} ---")
        print(f"Metin: {raw_texts[i]}")
        print(f"Gerçek: {true_label} | Model Tahmini: {pred_label}")
        print("-" * 30)

    # Resmi kaydetme
    save_path = os.path.join(script_dir, 'sonuc_tablosu.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n Görsel oluşturuldu: {save_path}")

#çalıştırma
if __name__ == "__main__":
    print("hazırlanıyor...")
    
    # veriyi yükle
    df = pd.read_csv(DATA_PATH)
    
    # etiket isimlerini data.csv içinde label_name sütunundan alır
    unique_labels = df[['label', 'label_name']].drop_duplicates().sort_values('label')
    label_map = dict(zip(unique_labels['label'], unique_labels['label_name']))
    
    print(f"Sınıflar: {label_map}")

    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=100, random_state=42))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = MultimodalDataset(df, IMG_DIR, tokenizer, transform)
    dl = DataLoader(ds, batch_size=5, shuffle=True)

    # modeli eğitme
    model = IntermediateFusionModel(num_classes=len(label_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    print("Model eğitiliyor...")
    model.train()
    for batch in dl:
        optimizer.zero_grad()
        out = model(batch['image'], batch['input_ids'], batch['attention_mask'])
        loss = criterion(out, batch['label'])
        loss.backward()
        optimizer.step()
        
    print("Eğitim bitti! Sonuçlar gösteriliyor...")
    visualize_results(model, dl, label_map)