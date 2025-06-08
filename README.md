# 🧮 Algoritma Kümeleme Analizi

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

Bu proje, çeşitli makine öğrenmesi algoritmalarının özelliklerini ve kullanım alanlarını içeren bir veri seti üzerinde kümeleme analizi yapmayı amaçlamaktadır. K-means ve Ward algoritmaları kullanılarak algoritmaların benzerliklerine göre gruplandırılması sağlanmıştır.

## 📊 Veri Seti İçeriği

Veri seti aşağıdaki özellikleri içermektedir:

| Özellik | Açıklama |
|---------|-----------|
| Algoritma Adı | Algoritmanın ismi |
| Öğrenme Türü | Denetimli/Denetimsiz öğrenme |
| Kullanım Alanı | Algoritmanın kullanıldığı alanlar |
| Karmaşıklık Düzeyi | Algoritmanın karmaşıklık seviyesi |
| Model Yapısı | Algoritmanın yapısal özellikleri |
| Aşırı Öğrenme Eğilimi | Overfitting eğilimi |
| Katman Tipi | Kullanılan katman türleri |
| Veri Tipi | Desteklenen veri tipleri |
| Donanım Gereksinimleri | Gerekli donanım özellikleri |
| Veri Büyüklüğü | İşlenebilecek veri boyutu |
| FineTune Gereksinimi | İnce ayar ihtiyacı |
| Popülerlik | Algoritmanın kullanım yaygınlığı |

## 📁 Proje Yapısı

```
.
├── clustering_analysis.py     # Ana analiz scripti
├── kumeleme_analizi_raporu.txt # Detaylı analiz raporu
├── Veri_seti.csv             # CSV formatında veri seti
├── Veri_seti.xlsx            # Excel formatında veri seti
└── requirements.txt          # Gerekli Python paketleri
```

## 🚀 Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/ErenAta16/algorithms.git
cd algorithms
```

2. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

1. Veri setini hazırlayın:
   - `Veri_seti.csv` veya `Veri_seti.xlsx` dosyasını proje dizinine kopyalayın

2. Analizi çalıştırın:
```bash
python clustering_analysis.py
```

3. Sonuçları inceleyin:
   - Analiz sonuçları konsola yazdırılacaktır
   - Detaylı rapor `kumeleme_analizi_raporu.txt` dosyasında bulunabilir
   - Eğitilmiş modeller aşağıdaki dosyalarda saklanacaktır:
     - `kmeans_model.joblib`
     - `ward_model.joblib`
     - `scaler.joblib`
     - `pca.joblib`

## 🔍 Algoritma Karşılaştırması

### Performans Metrikleri Karşılaştırması

| Metrik | K-means | Ward |
|--------|---------|------|
| Silhouette Skoru | 0.5051 | 0.4856 |
| Calinski-Harabasz Skoru | 304.8974 | 272.8796 |
| Davies-Bouldin İndeksi | 0.6711 | 0.8802 |
| SSE | - | 85.0346 |
| Homogeneity | 1.0000 | 1.0000 |
| Completeness | 1.0000 | 1.0000 |
| V-measure | 1.0000 | 1.0000 |

### Sonuç Analizi

K-means ve Ward algoritmalarının performans metriklerini karşılaştırdığımızda, her iki algoritmanın da güçlü yönleri olduğunu görüyoruz:

- **K-means Algoritması**:
  - Daha iyi küme ayrışması (Silhouette: 0.5051)
  - Daha yüksek küme ayrımı (Calinski-Harabasz: 304.8974)
  - Daha düşük Davies-Bouldin indeksi (0.6711)

- **Ward Algoritması**:
  - Daha kompakt kümeler (SSE: 85.0346)
  - Daha yoğun küme yapısı
  - Benzer iç tutarlılık (Homogeneity: 1.0000)

## 📈 Analiz Sonuçları

### K-means Algoritması
- Optimal küme sayısı: 4
- Silhouette skoru: 0.5051
- Calinski-Harabasz skoru: 304.8974
- Davies-Bouldin indeksi: 0.6711

### Ward Algoritması
- Optimal küme sayısı: 5
- Silhouette skoru: 0.4856
- Calinski-Harabasz skoru: 272.8796
- Davies-Bouldin indeksi: 0.8802

## 🤝 Katkıda Bulunma

1. Bu repoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun
