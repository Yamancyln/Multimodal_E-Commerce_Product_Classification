import pandas as pd
import os

# --- GÜNCELLEME: OTOMATİK YOL BULUCU ---
# Kodun çalıştığı klasörü otomatik bulur
current_dir = os.path.dirname(os.path.abspath(__file__))
# styles.csv dosyasının tam yolunu oluşturur
csv_path = os.path.join(current_dir, 'styles.csv')

print(f"Hedef Dosya Yolu: {csv_path}")

# 1. Dosyayı Oku
if not os.path.exists(csv_path):
    print("HATA: styles.csv dosyası bulunamadı!")
    print(f"Lütfen 'styles.csv' dosyasının şu klasörde olduğundan emin ol: {current_dir}")
    exit()

print("Dosya bulundu, okunuyor...")

# Pandas yeni sürüm uyumlu okuma (on_bad_lines='skip')
try:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
except TypeError:
    # Çok eski pandas sürümü varsa yedek plan
    df = pd.read_csv(csv_path, error_bad_lines=False)

# 2. Sütun isimlerini düzelt
df.columns = df.columns.str.strip()

# 3. Bizim için önemli sütunları al
# id: Resim ismi, articleType: Sınıf (Tshirt vs), productDisplayName: Metin
# Olası sütun ismi hatalarına karşı kontrol
required_cols = ['id', 'articleType', 'productDisplayName']
if not all(col in df.columns for col in required_cols):
    print(f"HATA: CSV dosyasında beklenen sütunlar yok. Mevcut sütunlar: {list(df.columns)}")
    print("Lütfen doğru styles.csv dosyasını indirdiğinden emin ol.")
    exit()

df = df[['id', 'articleType', 'productDisplayName']]
df.columns = ['image_id', 'label', 'text_description']

# 4. En çok verisi olan 3 sınıfı bul
top_3_classes = df['label'].value_counts().head(3).index.tolist()
print(f"Seçilen Sınıflar: {top_3_classes}")

# 5. Her sınıftan tam 500 tane örnek al
final_df = pd.DataFrame()

for category in top_3_classes:
    subset = df[df['label'] == category]
    if len(subset) >= 500:
        sampled = subset.sample(n=500, random_state=42)
    else:
        sampled = subset
    final_df = pd.concat([final_df, sampled])

# 6. Etiketleri Sayıya Çevir
label_map = {label: idx for idx, label in enumerate(top_3_classes)}
final_df['label_code'] = final_df['label'].map(label_map)
final_df = final_df.rename(columns={'label': 'label_name', 'label_code': 'label'})

# 7. Kaydet (Yine script'in olduğu yere kaydet)
save_path = os.path.join(current_dir, "data.csv")
final_df.to_csv(save_path, index=False)

print(f"GÖREV TAMAMLANDI! Dosya şuraya kaydedildi: {save_path}")
print(f"Toplam Veri Sayısı: {len(final_df)}")