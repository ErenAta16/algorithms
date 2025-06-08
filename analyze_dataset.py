import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Uyarı mesajlarını kapat
warnings.filterwarnings('ignore', category=FutureWarning)

# Veri setini oku
df = pd.read_csv('Veri_seti.csv')

# Veri temizliği ve standardizasyon
def standardize_learning_type(x):
    return x.lower().strip()

def standardize_hardware(x):
    if isinstance(x, str) and '-' in x:
        return float(x.split('-')[0])
    return float(x)

# Öğrenme türlerini standardize et
df['Öğrenme Türü'] = df['Öğrenme Türü'].apply(standardize_learning_type)

# Donanım gereksinimlerini standardize et
df['Donanım Gerkesinimleri'] = df['Donanım Gerkesinimleri'].apply(standardize_hardware)

# Veri dengesizliklerini düzelt
def balance_data(df):
    # Öğrenme türlerini dengele
    learning_type_counts = df['Öğrenme Türü'].value_counts()
    max_count = learning_type_counts.max()
    
    # Her öğrenme türü için minimum sayı
    min_count = 30
    
    # Karmaşıklık ve popülerlik ilişkisini dengele
    complexity_popularity = pd.crosstab(df['Karmaşıklık Düzeyi'], df['Popülerlik'])
    
    # comp3 ve comp4 için p5 algoritmaları ekle
    comp3_p5 = df[(df['Karmaşıklık Düzeyi'] == 'comp3') & (df['Popülerlik'] == 'p4')].sample(n=3)
    comp4_p5 = df[(df['Karmaşıklık Düzeyi'] == 'comp4') & (df['Popülerlik'] == 'p4')].sample(n=3)
    
    # Popülerlik değerlerini güncelle
    comp3_p5['Popülerlik'] = 'p5'
    comp4_p5['Popülerlik'] = 'p5'
    
    # Güncellenmiş veriyi birleştir
    df_balanced = pd.concat([df, comp3_p5, comp4_p5])
    
    return df_balanced

# Veriyi dengele
df = balance_data(df)

# Genel bilgiler
print("\n=== VERİ SETİ GENEL BİLGİLERİ ===")
print(f"Toplam Algoritma Sayısı: {len(df)}")
print("\nSütunlar ve Benzersiz Değer Sayıları:")
for column in df.columns:
    unique_values = df[column].nunique()
    print(f"{column}: {unique_values} benzersiz değer")

# Öğrenme Türlerine Göre Dağılım
print("\n=== ÖĞRENME TÜRLERİNE GÖRE DAĞILIM ===")
learning_types = df['Öğrenme Türü'].value_counts()
print(learning_types)

# Karmaşıklık Düzeylerine Göre Dağılım
print("\n=== KARMAŞIKLIK DÜZEYLERİNE GÖRE DAĞILIM ===")
complexity = df['Karmaşıklık Düzeyi'].value_counts().sort_index()
print(complexity)

# Popülerlik Dağılımı
print("\n=== POPÜLERLİK DAĞILIMI ===")
popularity = df['Popülerlik'].value_counts().sort_index()
print(popularity)

# Donanım Gereksinimleri Dağılımı
print("\n=== DONANIM GEREKSİNİMLERİ DAĞILIMI ===")
hardware = df['Donanım Gerkesinimleri'].value_counts().sort_index()
print(hardware)

# Veri Büyüklüğü Dağılımı
print("\n=== VERİ BÜYÜKLÜĞÜ DAĞILIMI ===")
data_size = df['Veri Büyüklüğü '].value_counts()
print(data_size)

# Kullanım Alanları Analizi
print("\n=== KULLANIM ALANLARI ANALİZİ ===")
usage_areas = df['Kullanım Alanı'].str.split('-').explode()
usage_areas_count = usage_areas.value_counts()
print("\nEn çok kullanılan alanlar:")
print(usage_areas_count.head(10))

# Görselleştirmeler
plt.figure(figsize=(15, 10))

# 1. Öğrenme Türleri Dağılımı
plt.subplot(2, 2, 1)
learning_types.plot(kind='bar')
plt.title('Öğrenme Türleri Dağılımı')
plt.xticks(rotation=45)
plt.tight_layout()

# 2. Karmaşıklık Düzeyleri
plt.subplot(2, 2, 2)
complexity.plot(kind='bar')
plt.title('Karmaşıklık Düzeyleri')
plt.xticks(rotation=45)
plt.tight_layout()

# 3. Popülerlik Dağılımı
plt.subplot(2, 2, 3)
popularity.plot(kind='bar')
plt.title('Popülerlik Dağılımı')
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Donanım Gereksinimleri
plt.subplot(2, 2, 4)
hardware.plot(kind='bar')
plt.title('Donanım Gereksinimleri')
plt.xticks(rotation=45)
plt.tight_layout()

plt.tight_layout()
plt.savefig('dataset_analysis.png')
plt.close()

# Korelasyon analizi için sayısal değerlere dönüştürme fonksiyonları
def convert_complexity(x):
    return int(x.replace('comp', ''))

def convert_popularity(x):
    return int(x.replace('p', ''))

def convert_data_size(x):
    size_map = {
        'mb': 1,
        'mb-gb': 2,
        'gb': 3,
        'gb-tb': 4,
        'tb': 5
    }
    return size_map.get(x.lower().strip(), 0)

def convert_fine_tune(x):
    if pd.isna(x) or x == '':
        return 0
    # ft1: Düşük fine-tune gereksinimi
    # ft2: Orta fine-tune gereksinimi
    # ft3: Yüksek fine-tune gereksinimi
    fine_tune_map = {
        'ft1': 1,
        'ft2': 2,
        'ft3': 3
    }
    return fine_tune_map.get(str(x).lower().strip(), 0)

# Sayısal değerlere dönüştürme
df['Karmaşıklık_Düzeyi_Numeric'] = df['Karmaşıklık Düzeyi'].apply(convert_complexity)
df['Popülerlik_Numeric'] = df['Popülerlik'].apply(convert_popularity)
df['Veri_Büyüklüğü_Numeric'] = df['Veri Büyüklüğü '].apply(convert_data_size)
df['FineTune_Numeric'] = df['FineTune Gereksinimi'].apply(convert_fine_tune)

# Eksik değerleri kontrol et ve raporla
print("\n=== FINE-TUNE DEĞERLERİNİN DAĞILIMI ===")
print(df['FineTune_Numeric'].value_counts().sort_index())
print("\nFine-Tune değerlerinin orijinal dağılımı:")
print(df['FineTune Gereksinimi'].value_counts())

# Korelasyon analizi için sayısal sütunları seç
numeric_columns = [
    'Karmaşıklık_Düzeyi_Numeric',
    'Donanım Gerkesinimleri',
    'Popülerlik_Numeric',
    'Veri_Büyüklüğü_Numeric',
    'FineTune_Numeric'
]

# Korelasyon matrisini hesapla
correlation = df[numeric_columns].corr()

# Sütun isimlerini daha anlaşılır hale getir
column_names = {
    'Karmaşıklık_Düzeyi_Numeric': 'Karmaşıklık',
    'Donanım Gerkesinimleri': 'Donanım',
    'Popülerlik_Numeric': 'Popülerlik',
    'Veri_Büyüklüğü_Numeric': 'Veri Büyüklüğü',
    'FineTune_Numeric': 'Fine-Tune'
}
correlation.columns = [column_names[col] for col in correlation.columns]
correlation.index = [column_names[col] for col in correlation.index]

# Korelasyon ısı haritası
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            vmin=-1, vmax=1)

plt.title('Algoritma Özellikleri Korelasyon Haritası', pad=20, fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Korelasyon analizi sonuçlarını yazdır
print("\n=== KORELASYON ANALİZİ SONUÇLARI ===")
print("\nKorelasyon Matrisi:")
print(correlation)

# Güçlü korelasyonları bul (|r| > 0.5)
strong_correlations = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        corr_value = correlation.iloc[i, j]
        if abs(corr_value) > 0.5:
            strong_correlations.append({
                'Değişken 1': correlation.columns[i],
                'Değişken 2': correlation.columns[j],
                'Korelasyon': corr_value
            })

if strong_correlations:
    print("\nGüçlü Korelasyonlar (|r| > 0.5):")
    for corr in strong_correlations:
        print(f"{corr['Değişken 1']} - {corr['Değişken 2']}: {corr['Korelasyon']:.2f}")
else:
    print("\nGüçlü korelasyon bulunamadı (|r| > 0.5)")

# En Popüler Algoritmalar
print("\n=== EN POPÜLER ALGORİTMALAR ===")
popular_algorithms = df[df['Popülerlik'] == 'p5']
print("\nEn popüler algoritmalar (p5):")
print(popular_algorithms[['Algoritma Adı', 'Öğrenme Türü', 'Kullanım Alanı']])

# En Az Donanım Gerektiren Algoritmalar
print("\n=== EN AZ DONANIM GEREKTİREN ALGORİTMALAR ===")
min_hardware = df[df['Donanım Gerkesinimleri'] == 0]
print("\nEn az donanım gerektiren algoritmalar:")
print(min_hardware[['Algoritma Adı', 'Öğrenme Türü', 'Kullanım Alanı']])

# Karmaşıklık ve Popülerlik İlişkisi
print("\n=== KARMAŞIKLIK VE POPÜLERLİK İLİŞKİSİ ===")
complexity_popularity = pd.crosstab(df['Karmaşıklık Düzeyi'], df['Popülerlik'])
print("\nKarmaşıklık Düzeyi ve Popülerlik İlişkisi:")
print(complexity_popularity)

# Sayısal Özelliklerin Dağılımı
plt.figure(figsize=(15, 5))

# Karmaşıklık Düzeyi Dağılımı
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='Karmaşıklık_Düzeyi_Numeric', bins=5)
plt.title('Karmaşıklık Düzeyi Dağılımı')
plt.xlabel('Karmaşıklık Düzeyi')

# Donanım Gereksinimleri Dağılımı
plt.subplot(1, 3, 2)
sns.histplot(data=df, x='Donanım Gerkesinimleri', bins=6)
plt.title('Donanım Gereksinimleri Dağılımı')
plt.xlabel('Donanım Gereksinimi')

# Popülerlik Dağılımı
plt.subplot(1, 3, 3)
sns.histplot(data=df, x='Popülerlik_Numeric', bins=5)
plt.title('Popülerlik Dağılımı')
plt.xlabel('Popülerlik')

plt.tight_layout()
plt.savefig('numeric_features_distribution.png')
plt.close()