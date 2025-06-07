import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import json
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import ParameterGrid, GridSearchCV, cross_val_score
from tqdm import tqdm
import warnings
import joblib
warnings.filterwarnings('ignore')

def silhouette_scorer(estimator, X):
    """
    GridSearchCV için özel silhouette scoring fonksiyonu.
    """
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

def preprocess_data(df):
    """
    Veri ön işleme adımlarını uygular
    """
    print("\nVeri ön işleme başlatılıyor...")
    
    # Kategorik değişkenleri işle
    def process_categorical_features(df):
        print("Kategorik özellikler işleniyor...")
        categorical_columns = ['Öğrenme Türü', 'Model Yapısı', 'Katman Tipi', 'Veri Tipi']
        
        # Label encoding uygula
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    # Sayısal değişkenleri işle
    def process_numeric_features(df):
        print("Sayısal özellikler işleniyor...")
        numeric_columns = ['Karmaşıklık Düzeyi', 'Popülerlik', 'Donanım Gerkesinimleri']
        
        # Karmaşıklık düzeyini sayısal değere dönüştür
        df['Karmaşıklık Düzeyi'] = df['Karmaşıklık Düzeyi'].str.replace('comp', '').astype(int)
        
        # Popülerliği sayısal değere dönüştür
        df['Popülerlik'] = df['Popülerlik'].str.replace('p', '').astype(int)
        
        # Donanım gereksinimlerini sayısal değere dönüştür
        def convert_hardware(x):
            if '-' in str(x):
                start, end = map(float, x.split('-'))
                return (start + end) / 2
            return float(x)
        
        df['Donanım Gerkesinimleri'] = df['Donanım Gerkesinimleri'].apply(convert_hardware)
        
        # Robust Scaler uygula (aykırı değerlere karşı daha dayanıklı)
        scaler = RobustScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df
    
    # Kullanım alanlarını işle
    def process_usage_areas(df):
        print("Kullanım alanları işleniyor...")
        # Kullanım alanlarını binary encoding ile dönüştür
        usage_areas = df['Kullanım Alanı'].str.get_dummies(sep='-')
        # Sadece en sık kullanılan 3 alanı seç (daha az gürültü)
        top_areas = usage_areas.sum().nlargest(3).index
        usage_areas = usage_areas[top_areas]
        df = pd.concat([df, usage_areas], axis=1)
        df = df.drop('Kullanım Alanı', axis=1)
        return df
    
    # Özellik seçimi
    def select_features(df):
        print("Özellik seçimi yapılıyor...")
        # Sadece sayısal sütunları seç
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Korelasyon matrisini hesapla
        corr_matrix = numeric_df.corr().abs()
        
        # Yüksek korelasyonlu özellikleri kaldır (daha sıkı eşik)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        
        # Korelasyonlu özellikleri kaldır
        numeric_df = numeric_df.drop(to_drop, axis=1)
        
        # PCA uygula (varyansın %90'ını koru)
        pca = PCA(n_components=0.90)
        pca_result = pca.fit_transform(numeric_df)
        
        # PCA sonuçlarını DataFrame'e dönüştür
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        
        return pca_df
    
    # Tüm ön işleme adımlarını uygula
    df = process_categorical_features(df)
    df = process_numeric_features(df)
    df = process_usage_areas(df)
    df = select_features(df)
    
    print("Veri ön işleme tamamlandı!")
    return df

def find_optimal_clusters(X, max_clusters=10):
    """
    Birden fazla metrik kullanarak optimal küme sayısını belirler
    """
    print("\nOptimal küme sayısı hesaplanıyor...")
    metrics = {
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    K = range(2, max_clusters + 1)
    
    for k in tqdm(K, desc="Küme sayısı optimizasyonu"):
        # K-means++ ile başlat ve daha fazla iterasyon kullan
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=200,  # Daha fazla başlatma
            init='k-means++',
            max_iter=2000,  # Daha fazla iterasyon
            tol=1e-6  # Daha hassas tolerans
        )
        labels = kmeans.fit_predict(X)
        
        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
    
    # Optimal küme sayısını belirle
    optimal_k = {
        'silhouette': K[np.argmax(metrics['silhouette'])],
        'calinski_harabasz': K[np.argmax(metrics['calinski_harabasz'])],
        'davies_bouldin': K[np.argmin(metrics['davies_bouldin'])]
    }
    
    # En sık önerilen küme sayısını bul
    recommended_k = max(set(optimal_k.values()), key=list(optimal_k.values()).count)
    print(f"\nÖnerilen küme sayısı: {recommended_k}")
    
    # Metrikleri görselleştir
    visualize_metrics(metrics, optimal_k)
    
    return recommended_k, metrics

def optimize_algorithm_parameters(X, algorithm='kmeans'):
    """
    Algoritma parametrelerini optimize eder
    """
    print(f"\n{algorithm} algoritması için parametreler optimize ediliyor...")
    
    if algorithm == 'kmeans':
        param_grid = {
            'n_clusters': [3, 4, 5],
            'init': ['k-means++'],
            'n_init': [200],  # Daha fazla başlatma
            'max_iter': [2000],  # Daha fazla iterasyon
            'algorithm': ['lloyd'],
            'tol': [1e-6]  # Daha hassas tolerans
        }
        model = KMeans(random_state=42)
    elif algorithm == 'ward':
        param_grid = {
            'n_clusters': [3, 4, 5],
            'linkage': ['ward', 'complete'],  # Farklı bağlantı yöntemleri
            'compute_full_tree': ['auto']
        }
        model = AgglomerativeClustering()
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=silhouette_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X)
    print(f"En iyi parametreler: {grid_search.best_params_}")
    return grid_search.best_params_

def evaluate_and_report_metrics(X, labels, algorithm_name):
    """
    Detaylı metrik değerlendirmesi ve raporlama
    """
    print(f"\n{algorithm_name} algoritması için metrikler hesaplanıyor...")
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'homogeneity': homogeneity_score(labels, labels),
        'completeness': completeness_score(labels, labels),
        'v_measure': v_measure_score(labels, labels)
    }
    
    interpretations = {
        'silhouette': 'İyi' if metrics['silhouette'] > 0.5 else 'Orta' if metrics['silhouette'] > 0.2 else 'Kötü',
        'calinski_harabasz': 'İyi' if metrics['calinski_harabasz'] > 100 else 'Orta' if metrics['calinski_harabasz'] > 50 else 'Kötü',
        'davies_bouldin': 'İyi' if metrics['davies_bouldin'] < 1 else 'Orta' if metrics['davies_bouldin'] < 2 else 'Kötü'
    }
    
    recommendations = generate_recommendations(metrics, interpretations)
    
    report = {
        'algorithm': algorithm_name,
        'metrics': metrics,
        'interpretations': interpretations,
        'recommendations': recommendations
    }
    
    print_metrics_report(report)
    return report

def generate_recommendations(metrics, interpretations):
    """
    Metrik sonuçlarına göre öneriler oluşturur
    """
    recommendations = []
    
    if metrics['silhouette'] < 0.2:
        recommendations.append("Silhouette skoru düşük: Küme sayısını değiştirmeyi veya farklı bir algoritma denemeyi düşünün")
    
    if metrics['calinski_harabasz'] < 50:
        recommendations.append("Calinski-Harabasz skoru düşük: Veri ön işleme adımlarını gözden geçirin")
    
    if metrics['davies_bouldin'] > 2:
        recommendations.append("Davies-Bouldin skoru yüksek: Özellik seçimini ve normalizasyonu kontrol edin")
    
    return recommendations

def print_metrics_report(report):
    """
    Metrik raporunu yazdırır
    """
    print("\n" + "="*50)
    print(f"{report['algorithm']} Algoritması Metrik Raporu")
    print("="*50)
    
    print("\nMetrikler:")
    for metric, value in report['metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nYorumlar:")
    for metric, interpretation in report['interpretations'].items():
        print(f"{metric}: {interpretation}")
    
    print("\nÖneriler:")
    for recommendation in report['recommendations']:
        print(f"- {recommendation}")
    print("="*50)

def visualize_metrics(metrics_history, optimal_k):
    """
    Metrik geçmişini görselleştirir
    """
    plt.figure(figsize=(15, 5))
    
    # Silhouette skoru
    plt.subplot(1, 3, 1)
    plt.plot(range(2, len(metrics_history['silhouette']) + 2), metrics_history['silhouette'], 'bo-')
    plt.axvline(x=optimal_k['silhouette'], color='r', linestyle='--')
    plt.title('Silhouette Skoru')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    
    # Calinski-Harabasz skoru
    plt.subplot(1, 3, 2)
    plt.plot(range(2, len(metrics_history['calinski_harabasz']) + 2), metrics_history['calinski_harabasz'], 'go-')
    plt.axvline(x=optimal_k['calinski_harabasz'], color='r', linestyle='--')
    plt.title('Calinski-Harabasz Skoru')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    
    # Davies-Bouldin skoru
    plt.subplot(1, 3, 3)
    plt.plot(range(2, len(metrics_history['davies_bouldin']) + 2), metrics_history['davies_bouldin'], 'ro-')
    plt.axvline(x=optimal_k['davies_bouldin'], color='r', linestyle='--')
    plt.title('Davies-Bouldin Skoru')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    
    plt.tight_layout()
    plt.savefig('metrik_optimizasyonu.png')
    plt.close()

def visualize_results(X, kmeans_labels, ward_labels, optimal_k):
    """
    Kümeleme sonuçlarını görselleştirir
    """
    # PCA ile 2 boyuta indir
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # K-means sonuçları
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-means Kümeleme Sonuçları')
    plt.xlabel('Birinci Bileşen')
    plt.ylabel('İkinci Bileşen')
    plt.colorbar(scatter, label='Küme')
    
    # Ward sonuçları
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ward_labels, cmap='viridis')
    plt.title('Ward Kümeleme Sonuçları')
    plt.xlabel('Birinci Bileşen')
    plt.ylabel('İkinci Bileşen')
    plt.colorbar(scatter, label='Küme')
    
    plt.tight_layout()
    plt.savefig('kumeleme_sonuclari.png')
    plt.close()
    
    # Küme boyutları
    plt.figure(figsize=(10, 5))
    
    # K-means küme boyutları
    plt.subplot(1, 2, 1)
    kmeans_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
    plt.bar(kmeans_sizes.index, kmeans_sizes.values)
    plt.title('K-means Küme Boyutları')
    plt.xlabel('Küme')
    plt.ylabel('Veri Noktası Sayısı')
    
    # Ward küme boyutları
    plt.subplot(1, 2, 2)
    ward_sizes = pd.Series(ward_labels).value_counts().sort_index()
    plt.bar(ward_sizes.index, ward_sizes.values)
    plt.title('Ward Küme Boyutları')
    plt.xlabel('Küme')
    plt.ylabel('Veri Noktası Sayısı')
    
    plt.tight_layout()
    plt.savefig('kume_boyutlari.png')
    plt.close()

def calculate_sse(X, labels):
    """
    Sum of Squared Errors (SSE) hesaplar
    """
    sse = 0
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def gelismis_metrik_analizi(X, labels, true_labels=None):
    """
    Gelişmiş metrik analizi yapar
    """
    print("\n=== GELİŞMİŞ METRİK ANALİZİ ===")
    
    metrics = {}
    
    # 1. SSE/WCSS Hesaplama
    metrics['SSE'] = {
        'kmeans': calculate_sse(X, labels['kmeans']),
        'ward': calculate_sse(X, labels['ward'])
    }
    
    # 2. Davies-Bouldin İndeksi
    metrics['Davies_Bouldin'] = {
        'kmeans': davies_bouldin_score(X, labels['kmeans']),
        'ward': davies_bouldin_score(X, labels['ward'])
    }
    
    # 3. Etiketli Veri Metrikleri (eğer true_labels varsa)
    if true_labels is not None:
        metrics['ARI'] = {
            'kmeans': adjusted_rand_score(true_labels, labels['kmeans']),
            'ward': adjusted_rand_score(true_labels, labels['ward'])
        }
        
        metrics['NMI'] = {
            'kmeans': normalized_mutual_info_score(true_labels, labels['kmeans']),
            'ward': normalized_mutual_info_score(true_labels, labels['ward'])
        }
        
        metrics['FMI'] = {
            'kmeans': fowlkes_mallows_score(true_labels, labels['kmeans']),
            'ward': fowlkes_mallows_score(true_labels, labels['ward'])
        }
    
    # Metrikleri görselleştir
    plt.figure(figsize=(15, 10))
    
    # SSE karşılaştırması
    plt.subplot(2, 2, 1)
    plt.bar(['K-means', 'Ward'], [metrics['SSE']['kmeans'], metrics['SSE']['ward']])
    plt.title('SSE Karşılaştırması')
    plt.ylabel('SSE Değeri')
    
    # Davies-Bouldin karşılaştırması
    plt.subplot(2, 2, 2)
    plt.bar(['K-means', 'Ward'], [metrics['Davies_Bouldin']['kmeans'], metrics['Davies_Bouldin']['ward']])
    plt.title('Davies-Bouldin İndeksi Karşılaştırması')
    plt.ylabel('Davies-Bouldin Değeri')
    
    if true_labels is not None:
        # ARI karşılaştırması
        plt.subplot(2, 2, 3)
        plt.bar(['K-means', 'Ward'], [metrics['ARI']['kmeans'], metrics['ARI']['ward']])
        plt.title('ARI Karşılaştırması')
        plt.ylabel('ARI Değeri')
        
        # NMI ve FMI karşılaştırması
        plt.subplot(2, 2, 4)
        x = np.arange(2)
        width = 0.35
        plt.bar(x - width/2, [metrics['NMI']['kmeans'], metrics['NMI']['ward']], width, label='NMI')
        plt.bar(x + width/2, [metrics['FMI']['kmeans'], metrics['FMI']['ward']], width, label='FMI')
        plt.title('NMI ve FMI Karşılaştırması')
        plt.xticks(x, ['K-means', 'Ward'])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('gelismis_metrik_analizi.png')
    plt.close()
    
    # Metrikleri raporla
    print("\nMetrik Sonuçları:")
    for metric_name, values in metrics.items():
        print(f"\n{metric_name}:")
        print(f"K-means: {values['kmeans']:.4f}")
        print(f"Ward: {values['ward']:.4f}")
    
    return metrics

def performans_raporu_olustur(X, labels, kmeans_model, ward_model, true_labels=None):
    """
    Kapsamlı performans raporu oluşturur
    """
    # 1. Temel metrikler
    basic_metrics = evaluate_and_report_metrics(X, labels['kmeans'], 'K-means')
    ward_metrics = evaluate_and_report_metrics(X, labels['ward'], 'Ward')
    
    # 2. Gelişmiş metrikler
    advanced_metrics = gelismis_metrik_analizi(X, labels, true_labels)
    
    # 3. Küme boyutları analizi
    kmeans_sizes = np.bincount(labels['kmeans'])
    ward_sizes = np.bincount(labels['ward'])
    
    # 4. Rapor oluşturma
    rapor = {
        'temel_metrikler': {
            'kmeans': {
                'metrics': {k: float(v) for k, v in basic_metrics['metrics'].items()},
                'interpretations': basic_metrics['interpretations'],
                'recommendations': basic_metrics['recommendations']
            },
            'ward': {
                'metrics': {k: float(v) for k, v in ward_metrics['metrics'].items()},
                'interpretations': ward_metrics['interpretations'],
                'recommendations': ward_metrics['recommendations']
            }
        },
        'gelismis_metrikler': {
            k: {sk: float(sv) for sk, sv in v.items()} 
            for k, v in advanced_metrics.items()
        },
        'kume_boyutlari': {
            'kmeans': kmeans_sizes.tolist(),
            'ward': ward_sizes.tolist()
        },
        'sonuc': {
            'kmeans_basari': bool(basic_metrics['metrics']['silhouette'] > 0.3),
            'ward_basari': bool(ward_metrics['metrics']['silhouette'] > 0.3),
            'oneri_edilen_model': 'Ward' if ward_metrics['metrics']['silhouette'] > basic_metrics['metrics']['silhouette'] else 'K-means'
        }
    }
    
    # Raporu JSON dosyasına kaydet
    with open('performans_raporu.json', 'w', encoding='utf-8') as f:
        json.dump(rapor, f, ensure_ascii=False, indent=4)
    
    return rapor

def save_models(kmeans_model, ward_model, scaler, pca):
    """
    Eğitilmiş modelleri ve ön işleme bileşenlerini kaydeder
    """
    print("\nModeller kaydediliyor...")
    
    # Modelleri kaydet
    joblib.dump(kmeans_model, 'kmeans_model.joblib')
    joblib.dump(ward_model, 'ward_model.joblib')
    
    # Ön işleme bileşenlerini kaydet
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(pca, 'pca.joblib')
    
    print("Modeller ve ön işleme bileşenleri kaydedildi!")

def main():
    # Veri setini oku
    print("Veri seti okunuyor...")
    df = pd.read_csv('Veri_seti.csv')
    
    # Veri ön işleme
    X = preprocess_data(df)
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA uygula
    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X_scaled)
    
    # Optimal küme sayısını bul
    optimal_k, metrics_history = find_optimal_clusters(X_pca)
    
    # Algoritma parametrelerini optimize et
    kmeans_params = optimize_algorithm_parameters(X_pca, 'kmeans')
    ward_params = optimize_algorithm_parameters(X_pca, 'ward')
    
    # Modelleri eğit
    kmeans = KMeans(**kmeans_params, random_state=42)
    ward = AgglomerativeClustering(**ward_params)
    
    kmeans_labels = kmeans.fit_predict(X_pca)
    ward_labels = ward.fit_predict(X_pca)
    
    labels = {
        'kmeans': kmeans_labels,
        'ward': ward_labels
    }
    
    # Performans raporu oluştur
    true_labels = None
    performans_raporu = performans_raporu_olustur(X_pca, labels, kmeans, ward, true_labels)
    
    # Sonuçları görselleştir
    visualize_results(X_pca, kmeans_labels, ward_labels, optimal_k)
    
    # Modelleri kaydet
    save_models(kmeans, ward, scaler, pca)
    
    print("\nAnaliz tamamlandı! Sonuçlar kaydedildi.")

if __name__ == "__main__":
    main() 