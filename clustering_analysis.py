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
import os
warnings.filterwarnings('ignore')

def explain_metrics(metrics, algorithm_name):
    """
    Kümeleme metriklerini detaylı olarak açıklar
    """
    print(f"\n=== {algorithm_name} ALGORİTMASI METRİK AÇIKLAMALARI ===")
    
    # Silhouette Skoru Açıklaması
    print("\n1. Silhouette Skoru:")
    print(f"Değer: {metrics['silhouette']:.4f}")
    print("Açıklama:")
    print("- Küme içi ve küme arası mesafeleri karşılaştırır")
    print("- SSW (Küme içi kareler toplamı) ve SSB (Küme arası kareler toplamı) kullanır")
    print("- -1 ile 1 arasında değer alır")
    print("- Yüksek değerler iyi kümeleme gösterir")
    print(f"Yorum: {metrics['silhouette']:.4f} değeri, kümelerin {'iyi' if metrics['silhouette'] > 0.5 else 'orta' if metrics['silhouette'] > 0.2 else 'zayıf'} ayrıştığını gösterir")
    
    # Calinski-Harabasz Skoru Açıklaması
    print("\n2. Calinski-Harabasz Skoru:")
    print(f"Değer: {metrics['calinski_harabasz']:.4f}")
    print("Açıklama:")
    print("- SSB/SSW oranını kullanır")
    print("- Küme arası varyasyonun küme içi varyasyona oranını gösterir")
    print("- Yüksek değerler iyi kümeleme gösterir")
    print(f"Yorum: {metrics['calinski_harabasz']:.4f} değeri, kümelerin {'iyi' if metrics['calinski_harabasz'] > 100 else 'orta' if metrics['calinski_harabasz'] > 50 else 'zayıf'} ayrıştığını gösterir")
    
    # Davies-Bouldin İndeksi Açıklaması
    print("\n3. Davies-Bouldin İndeksi:")
    print(f"Değer: {metrics['davies_bouldin']:.4f}")
    print("Açıklama:")
    print("- Küme içi ve küme arası mesafelerin oranını kullanır")
    print("- Düşük değerler iyi kümeleme gösterir")
    print("- 0'a yakın değerler ideal kümeleme gösterir")
    print(f"Yorum: {metrics['davies_bouldin']:.4f} değeri, kümelerin {'iyi' if metrics['davies_bouldin'] < 1 else 'orta' if metrics['davies_bouldin'] < 2 else 'zayıf'} ayrıştığını gösterir")
    
    # SSE (Sum of Squared Errors) Açıklaması
    if 'SSE' in metrics:
        print("\n4. SSE (Sum of Squared Errors):")
        print(f"Değer: {metrics['SSE']:.4f}")
        print("Açıklama:")
        print("- Küme içi kareler toplamı (SSW)")
        print("- Küçük değerler kompakt kümeleri gösterir")
        print("- Kümeleme kalitesinin bir göstergesidir")
        print(f"Yorum: {metrics['SSE']:.4f} değeri, kümelerin {'kompakt' if metrics['SSE'] < 100 else 'orta' if metrics['SSE'] < 200 else 'dağınık'} olduğunu gösterir")

def generate_detailed_metrics_report(metrics, algorithm_name):
    """
    Detaylı metrik raporu oluşturur
    """
    report = {
        'algorithm': algorithm_name,
        'metrics': metrics,
        'interpretations': {
            'silhouette': {
                'value': metrics['silhouette'],
                'interpretation': 'İyi' if metrics['silhouette'] > 0.5 else 'Orta' if metrics['silhouette'] > 0.2 else 'Zayıf',
                'explanation': 'Küme içi ve küme arası mesafelerin karşılaştırması'
            },
            'calinski_harabasz': {
                'value': metrics['calinski_harabasz'],
                'interpretation': 'İyi' if metrics['calinski_harabasz'] > 100 else 'Orta' if metrics['calinski_harabasz'] > 50 else 'Zayıf',
                'explanation': 'SSB/SSW oranı - Küme arası varyasyonun küme içi varyasyona oranı'
            },
            'davies_bouldin': {
                'value': metrics['davies_bouldin'],
                'interpretation': 'İyi' if metrics['davies_bouldin'] < 1 else 'Orta' if metrics['davies_bouldin'] < 2 else 'Zayıf',
                'explanation': 'Küme içi ve küme arası mesafelerin oranı'
            }
        },
        'recommendations': []
    }
    
    # Öneriler oluştur
    if metrics['silhouette'] < 0.3:
        report['recommendations'].append("Silhouette skoru düşük: Küme sayısını değiştirmeyi veya farklı bir kümeleme algoritması denemeyi düşünün")
    if metrics['calinski_harabasz'] < 50:
        report['recommendations'].append("Calinski-Harabasz skoru düşük: Kümeler arası ayrımı artırmak için özellik seçimini gözden geçirin")
    if metrics['davies_bouldin'] > 2:
        report['recommendations'].append("Davies-Bouldin indeksi yüksek: Kümelerin kompaktlığını artırmak için veri ön işlemeyi gözden geçirin")
    
    return report

def create_directories():
    """
    Gerekli dizinleri oluşturur
    """
    directories = [
        'visualizations/clustering',
        'visualizations/metrics',
        'visualizations/analysis',
        'models',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def silhouette_scorer(estimator, X):
    """
    GridSearchCV için özel silhouette scoring fonksiyonu.
    """
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

def preprocess_categorical_features(df):
    """
    Kategorik özellikleri ön işleme ve ağırlıklandırma
    """
    # Kategorik sütunları seç
    categorical_cols = [
        'Öğrenme Türü', 'Model Yapısı', 'Aşırı Öğrenme Eğilimi',
        'Katman Tipi', 'Veri Tipi', 'FineTune Gereksinimi'
    ]
    
    # Her kategorik sütun için one-hot encoding uygula
    categorical_features = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
    
    # Kategorik özellikler için ağırlıklar - En iyi sonuç veren ağırlıklar
    categorical_weights = {
        'Öğrenme Türü': 3.5,           # En iyi sonuç veren değer
        'Model Yapısı': 3.0,           # En iyi sonuç veren değer
        'Aşırı Öğrenme Eğilimi': 2.5,  # En iyi sonuç veren değer
        'Katman Tipi': 2.0,            # En iyi sonuç veren değer
        'Veri Tipi': 3.2,              # En iyi sonuç veren değer
        'FineTune Gereksinimi': 2.0    # En iyi sonuç veren değer
    }
    
    # Ağırlıklandırma uygula
    for col in categorical_cols:
        # Sütun adına göre ilgili one-hot encoding sütunlarını bul
        col_prefix = f"{col}_"
        matching_cols = [c for c in categorical_features.columns if c.startswith(col_prefix)]
        
        # Bu sütunlara ağırlık uygula
        for matching_col in matching_cols:
            categorical_features[matching_col] = categorical_features[matching_col] * categorical_weights[col]
    
    return categorical_features

def preprocess_numerical_features(df):
    """
    Sayısal özellikleri ön işleme ve ağırlıklandırma
    """
    # Yeni bir DataFrame oluştur
    numerical_df = df.copy()
    
    # Karmaşıklık düzeyini sayısal değere dönüştür
    numerical_df['Karmaşıklık Düzeyi'] = numerical_df['Karmaşıklık Düzeyi'].str.replace('comp', '').astype(int)
    
    # Popülerliği sayısal değere dönüştür
    numerical_df['Popülerlik'] = numerical_df['Popülerlik'].str.replace('p', '').astype(int)
    
    # Donanım gereksinimlerini sayısal değere dönüştür
    def convert_hardware(x):
        if isinstance(x, str):
            if '-' in x:
                start, end = map(float, x.split('-'))
                return (start + end) / 2
            return float(x)
        return x
    
    numerical_df['Donanım Gerkesinimleri'] = numerical_df['Donanım Gerkesinimleri'].apply(convert_hardware)
    
    # Veri büyüklüğünü sayısal değere dönüştür
    def convert_data_size(x):
        try:
            if pd.isna(x) or x == '' or x == ' ':
                return 0.0
            if isinstance(x, str):
                # Birimleri kaldır ve sayısal değere dönüştür
                x = x.strip()
                if x == 'MB':
                    return 1.0
                elif x == 'GB':
                    return 1024.0  # 1 GB = 1024 MB
                elif x == 'TB':
                    return 1024.0 * 1024.0  # 1 TB = 1024 GB = 1024 * 1024 MB
                elif '-' in x:
                    # Aralık değerleri için ortalama al
                    start, end = x.split('-')
                    start = start.strip()
                    end = end.strip()
                    
                    # Başlangıç değerini dönüştür
                    if start == 'MB':
                        start_val = 1.0
                    elif start == 'GB':
                        start_val = 1024.0
                    elif start == 'TB':
                        start_val = 1024.0 * 1024.0
                    else:
                        start_val = float(start)
                    
                    # Bitiş değerini dönüştür
                    if end == 'MB':
                        end_val = 1.0
                    elif end == 'GB':
                        end_val = 1024.0
                    elif end == 'TB':
                        end_val = 1024.0 * 1024.0
                    else:
                        end_val = float(end)
                    
                    return (start_val + end_val) / 2
                else:
                    return float(x)
            return float(x)
        except Exception as e:
            print(f"Hata: {x} değeri dönüştürülemedi. Hata: {str(e)}")
            return 0.0
    
    numerical_df['Veri Büyüklüğü '] = numerical_df['Veri Büyüklüğü '].apply(convert_data_size)
    
    # Sayısal özellikler için ağırlıklar - En iyi sonuç veren ağırlıklar
    numerical_weights = {
        'Karmaşıklık Düzeyi': 4.0,      # En iyi sonuç veren değer
        'Popülerlik': 3.2,              # En iyi sonuç veren değer
        'Donanım Gerkesinimleri': 2.5,   # En iyi sonuç veren değer
        'Veri Büyüklüğü ': 3.0           # En iyi sonuç veren değer
    }
    
    # Ağırlıklandırma uygula
    for col, weight in numerical_weights.items():
        numerical_df[col] = numerical_df[col] * weight
    
    numerical_cols = ['Karmaşıklık Düzeyi', 'Popülerlik', 'Donanım Gerkesinimleri', 'Veri Büyüklüğü ']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numerical_df[numerical_cols])
    return pd.DataFrame(scaled_features, columns=numerical_cols)

def preprocess_usage_areas(df):
    """
    Kullanım alanlarını ön işleme
    """
    usage_areas = df['Kullanım Alanı'].str.get_dummies('-')
    return usage_areas

def select_features(df):
    """
    Özellik seçimi ve birleştirme
    """
    categorical_features = preprocess_categorical_features(df)
    numerical_features = preprocess_numerical_features(df)
    usage_features = preprocess_usage_areas(df)
    
    # Tüm özellikleri birleştir
    features = pd.concat([categorical_features, numerical_features, usage_features], axis=1)
    return features

def find_optimal_clusters(X, max_clusters=27):
    """
    En optimal küme sayısını belirler
    """
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    sse_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # K-means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Ward kümeleme
        ward = AgglomerativeClustering(n_clusters=n_clusters)
        ward_labels = ward.fit_predict(X)
        
        # Metrikleri hesapla
        silhouette_scores.append({
            'kmeans': silhouette_score(X, kmeans_labels),
            'ward': silhouette_score(X, ward_labels)
        })
        
        calinski_scores.append({
            'kmeans': calinski_harabasz_score(X, kmeans_labels),
            'ward': calinski_harabasz_score(X, ward_labels)
        })
        
        davies_scores.append({
            'kmeans': davies_bouldin_score(X, kmeans_labels),
            'ward': davies_bouldin_score(X, ward_labels)
        })
        
        sse_scores.append({
            'kmeans': kmeans.inertia_,
            'ward': np.sum((X - ward.fit(X).labels_.reshape(-1, 1)) ** 2)
        })
    
    return {
        'silhouette': silhouette_scores,
        'calinski': calinski_scores,
        'davies': davies_scores,
        'sse': sse_scores
    }

def optimize_algorithm_parameters(X, algorithm='kmeans', optimal_k=4):
    """
    Algoritma parametrelerini optimize eder
    """
    print(f"\n{algorithm} algoritması için parametreler optimize ediliyor...")
    
    if algorithm == 'kmeans':
        param_grid = {
            'n_clusters': [optimal_k],  # Optimal küme sayısını kullan
            'init': ['k-means++'],
            'n_init': [200],
            'max_iter': [2000],
            'algorithm': ['lloyd'],
            'tol': [1e-6]
        }
        model = KMeans(random_state=42)
    elif algorithm == 'ward':
        param_grid = {
            'n_clusters': [optimal_k],  # Optimal küme sayısını kullan
            'linkage': ['ward', 'complete'],
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
    Metrikleri görselleştirir ve açıklamalar ekler
    """
    plt.figure(figsize=(15, 10))
    
    # Silhouette skoru
    plt.subplot(2, 2, 1)
    plt.plot(range(2, len(metrics_history['silhouette']) + 2), metrics_history['silhouette'], 'bo-')
    plt.axvline(x=optimal_k['silhouette'], color='r', linestyle='--')
    plt.title('Silhouette Skoru\n(Küme İçi ve Arası Mesafe Oranı)')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    plt.grid(True)
    
    # Calinski-Harabasz skoru
    plt.subplot(2, 2, 2)
    plt.plot(range(2, len(metrics_history['calinski_harabasz']) + 2), metrics_history['calinski_harabasz'], 'go-')
    plt.axvline(x=optimal_k['calinski_harabasz'], color='r', linestyle='--')
    plt.title('Calinski-Harabasz Skoru\n(SSB/SSW Oranı)')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    plt.grid(True)
    
    # Davies-Bouldin skoru
    plt.subplot(2, 2, 3)
    plt.plot(range(2, len(metrics_history['davies_bouldin']) + 2), metrics_history['davies_bouldin'], 'ro-')
    plt.axvline(x=optimal_k['davies_bouldin'], color='r', linestyle='--')
    plt.title('Davies-Bouldin İndeksi\n(Küme İçi/Arası Mesafe Oranı)')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Skor')
    plt.grid(True)
    
    # Metrik karşılaştırması
    plt.subplot(2, 2, 4)
    k = range(2, len(metrics_history['silhouette']) + 2)
    plt.plot(k, metrics_history['silhouette'], 'b-', label='Silhouette')
    plt.plot(k, np.array(metrics_history['calinski_harabasz'])/100, 'g-', label='Calinski-Harabasz/100')
    plt.plot(k, metrics_history['davies_bouldin'], 'r-', label='Davies-Bouldin')
    plt.title('Metrik Karşılaştırması\n(Normalize Edilmiş)')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Normalize Skor')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics/detayli_metrik_analizi.png')
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
    plt.savefig('visualizations/clustering/kumeleme_sonuclari.png')
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
    plt.savefig('visualizations/clustering/kume_boyutlari.png')
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
    plt.savefig('visualizations/metrics/gelismis_metrik_analizi.png')
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
    with open('reports/performans_raporu.json', 'w', encoding='utf-8') as f:
        json.dump(rapor, f, ensure_ascii=False, indent=4)
    
    return rapor

def save_models(kmeans_model, ward_model, scaler, pca, save_dir='models'):
    """
    Eğitilmiş modelleri ve dönüştürücüleri kaydeder
    """
    # Dizin yoksa oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    # Modelleri kaydet
    joblib.dump(kmeans_model, os.path.join(save_dir, 'kmeans_model.joblib'))
    joblib.dump(ward_model, os.path.join(save_dir, 'ward_model.joblib'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(save_dir, 'pca.joblib'))
    
    print("\nModeller başarıyla kaydedildi:")
    print(f"- K-means modeli: {os.path.join(save_dir, 'kmeans_model.joblib')}")
    print(f"- Ward modeli: {os.path.join(save_dir, 'ward_model.joblib')}")
    print(f"- Scaler: {os.path.join(save_dir, 'scaler.joblib')}")
    print(f"- PCA: {os.path.join(save_dir, 'pca.joblib')}")

def visualize_elbow_method(X, max_clusters=10):
    """
    Dirsek yöntemi grafiğini oluşturur
    """
    print("\nDirsek yöntemi grafiği oluşturuluyor...")
    
    # SSE değerlerini hesapla
    sse = []
    K = range(1, max_clusters + 1)
    
    for k in tqdm(K, desc="Dirsek yöntemi hesaplanıyor"):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=200,
            init='k-means++',
            max_iter=2000,
            tol=1e-6
        )
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # Grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.plot(K, sse, 'bo-')
    plt.xlabel('Küme Sayısı (k)')
    plt.ylabel('SSE')
    plt.title('Dirsek Yöntemi')
    plt.grid(True)
    
    # Grafiği kaydet
    plt.savefig('visualizations/metrics/dirsek_yontemi.png')
    plt.close()

def analyze_cluster_contents(df, labels, algorithm_name):
    """
    Her kümenin içeriğini detaylı olarak analiz eder ve raporlar
    """
    print(f"\n=== {algorithm_name} ALGORİTMASI KÜME İÇERİKLERİ ===")
    
    # Her küme için analiz yap
    for cluster_id in range(max(labels) + 1):
        # Kümedeki algoritmaları al
        cluster_algorithms = df[labels == cluster_id]['Algoritma Adı'].tolist()
        
        # Küme özelliklerini analiz et
        cluster_data = df[labels == cluster_id]
        
        # Ortalama değerleri hesapla
        avg_complexity = cluster_data['Karmaşıklık Düzeyi'].mean()
        avg_hardware = cluster_data['Donanım Gerkesinimleri'].mean()
        avg_popularity = cluster_data['Popülerlik'].mean()
        
        # En yaygın özellikleri bul
        learning_types = cluster_data['Öğrenme Türü'].value_counts().head(3)
        model_structures = cluster_data['Model Yapısı'].value_counts().head(3)
        layer_types = cluster_data['Katman Tipi'].value_counts().head(3)
        
        print(f"\nKüme {cluster_id + 1} Analizi:")
        print("-" * 50)
        print(f"Küme Büyüklüğü: {len(cluster_algorithms)} algoritma")
        print("\nKüme Özellikleri:")
        print(f"Ortalama Karmaşıklık Düzeyi: {avg_complexity:.2f}")
        print(f"Ortalama Donanım Gereksinimi: {avg_hardware:.2f}")
        print(f"Ortalama Popülerlik: {avg_popularity:.2f}")
        
        print("\nEn Yaygın Öğrenme Türleri:")
        for lt, count in learning_types.items():
            print(f"- {lt}: {count} algoritma")
            
        print("\nEn Yaygın Model Yapıları:")
        for ms, count in model_structures.items():
            print(f"- {ms}: {count} algoritma")
            
        print("\nEn Yaygın Katman Tipleri:")
        for lt, count in layer_types.items():
            print(f"- {lt}: {count} algoritma")
            
        print("\nKümedeki Algoritmalar:")
        for i, algo in enumerate(cluster_algorithms, 1):
            print(f"{i}. {algo}")
        print("-" * 50)

def visualize_dendrogram(X, max_clusters=27):
    """
    Hiyerarşik kümeleme dendrogramını görselleştirir
    """
    # Hiyerarşik kümeleme
    linkage_matrix = linkage(X, method='ward')
    
    # Dendrogram
    plt.figure(figsize=(20, 15))
    dendrogram(linkage_matrix,
               truncate_mode='lastp',
               p=max_clusters,
               leaf_rotation=90.,
               leaf_font_size=12.,
               show_contracted=True)
    plt.title('Hiyerarşik Kümeleme Dendrogramı (27 Küme)')
    plt.xlabel('Örnek İndeksi')
    plt.ylabel('Uzaklık')
    plt.grid(True)
    plt.savefig('visualizations/clustering/dendrogram.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 27 küme için renkli görselleştirme
    ward = AgglomerativeClustering(n_clusters=27)
    ward_labels = ward.fit_predict(X)
    
    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=ward_labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('27 Küme ile Ward Kümeleme Görselleştirmesi')
    plt.xlabel('Birinci Bileşen')
    plt.ylabel('İkinci Bileşen')
    plt.grid(True)
    plt.savefig('visualizations/clustering/ward_clustering_27.png')
    plt.close()

def visualize_cluster_metrics(metrics, max_clusters=27):
    """
    Kümeleme metriklerini görselleştirir
    """
    n_clusters = range(2, max_clusters + 1)
    
    # Silhouette skoru
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, [m['kmeans'] for m in metrics['silhouette']], 'b-', label='K-means')
    plt.plot(n_clusters, [m['ward'] for m in metrics['silhouette']], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/silhouette_scores.png')
    plt.close()
    
    # Calinski-Harabasz skoru
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, [m['kmeans'] for m in metrics['calinski']], 'b-', label='K-means')
    plt.plot(n_clusters, [m['ward'] for m in metrics['calinski']], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/calinski_scores.png')
    plt.close()
    
    # Davies-Bouldin indeksi
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, [m['kmeans'] for m in metrics['davies']], 'b-', label='K-means')
    plt.plot(n_clusters, [m['ward'] for m in metrics['davies']], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/davies_scores.png')
    plt.close()
    
    # SSE
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, [m['kmeans'] for m in metrics['sse']], 'b-', label='K-means')
    plt.plot(n_clusters, [m['ward'] for m in metrics['sse']], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/sse_scores.png')
    plt.close()

def analyze_all_cluster_numbers(X, max_clusters=27):
    """
    Tüm küme sayıları için performans analizi yapar
    """
    print("\nTüm küme sayıları için performans analizi yapılıyor...")
    results = []
    
    for k in tqdm(range(2, max_clusters + 1), desc="Küme sayıları analiz ediliyor"):
        # K-means kümeleme
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Ward kümeleme
        ward = AgglomerativeClustering(n_clusters=k)
        ward_labels = ward.fit_predict(X)
        
        # Metrikleri hesapla
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        ward_silhouette = silhouette_score(X, ward_labels)
        
        kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)
        ward_calinski = calinski_harabasz_score(X, ward_labels)
        
        kmeans_davies = davies_bouldin_score(X, kmeans_labels)
        ward_davies = davies_bouldin_score(X, ward_labels)
        
        kmeans_sse = kmeans.inertia_
        ward_sse = sum(np.sum((X[ward_labels == i] - np.mean(X[ward_labels == i], axis=0)) ** 2) 
                      for i in range(k))
        
        results.append({
            'k': k,
            'kmeans_silhouette': kmeans_silhouette,
            'ward_silhouette': ward_silhouette,
            'kmeans_calinski': kmeans_calinski,
            'ward_calinski': ward_calinski,
            'kmeans_davies': kmeans_davies,
            'ward_davies': ward_davies,
            'kmeans_sse': kmeans_sse,
            'ward_sse': ward_sse
        })
    
    # Sonuçları tablo olarak yazdır
    print("\nPerformans Analizi Sonuçları:")
    print("-" * 100)
    print(f"{'Küme Sayısı':<10} {'K-means Silhouette':<20} {'Ward Silhouette':<20} {'K-means Calinski':<20} {'Ward Calinski':<20}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['k']:<10} {r['kmeans_silhouette']:<20.4f} {r['ward_silhouette']:<20.4f} {r['kmeans_calinski']:<20.4f} {r['ward_calinski']:<20.4f}")
    
    print("\nDavies-Bouldin ve SSE Değerleri:")
    print("-" * 100)
    print(f"{'Küme Sayısı':<10} {'K-means Davies':<20} {'Ward Davies':<20} {'K-means SSE':<20} {'Ward SSE':<20}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['k']:<10} {r['kmeans_davies']:<20.4f} {r['ward_davies']:<20.4f} {r['kmeans_sse']:<20.4f} {r['ward_sse']:<20.4f}")
    
    # En iyi performans gösteren küme sayılarını bul
    best_kmeans_silhouette = max(results, key=lambda x: x['kmeans_silhouette'])
    best_ward_silhouette = max(results, key=lambda x: x['ward_silhouette'])
    
    print("\nEn İyi Performans Gösteren Küme Sayıları:")
    print(f"K-means (Silhouette): {best_kmeans_silhouette['k']} küme (Skor: {best_kmeans_silhouette['kmeans_silhouette']:.4f})")
    print(f"Ward (Silhouette): {best_ward_silhouette['k']} küme (Skor: {best_ward_silhouette['ward_silhouette']:.4f})")
    
    # Metrikleri görselleştir
    visualize_metrics(results)
    
    return results

def visualize_metrics(results):
    """
    Metrikleri görselleştirir
    """
    k_values = [r['k'] for r in results]
    
    # Silhouette skorları
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, [r['kmeans_silhouette'] for r in results], 'b-', label='K-means')
    plt.plot(k_values, [r['ward_silhouette'] for r in results], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/silhouette_scores.png')
    plt.close()
    
    # Calinski-Harabasz skorları
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, [r['kmeans_calinski'] for r in results], 'b-', label='K-means')
    plt.plot(k_values, [r['ward_calinski'] for r in results], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/calinski_scores.png')
    plt.close()
    
    # Davies-Bouldin indeksleri
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, [r['kmeans_davies'] for r in results], 'b-', label='K-means')
    plt.plot(k_values, [r['ward_davies'] for r in results], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/davies_scores.png')
    plt.close()
    
    # SSE değerleri
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, [r['kmeans_sse'] for r in results], 'b-', label='K-means')
    plt.plot(k_values, [r['ward_sse'] for r in results], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE vs Küme Sayısı')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/sse_scores.png')
    plt.close()

def visualize_all_metrics(X, max_clusters=27):
    """
    Tüm metrik görselleştirmelerini oluşturur
    """
    # 1. Dirsek Yöntemi (Elbow Method)
    distortions = []
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(15, 10))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Dirsek Yöntemi (Elbow Method)')
    plt.grid(True)
    
    # 2. Detaylı Metrik Analizi
    metrics = {
        'silhouette': [],
        'calinski': [],
        'davies': [],
        'sse': []
    }
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski'].append(calinski_harabasz_score(X, labels))
        metrics['davies'].append(davies_bouldin_score(X, labels))
        metrics['sse'].append(kmeans.inertia_)
    
    plt.figure(figsize=(20, 15))
    
    # Silhouette
    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_clusters + 1), metrics['silhouette'], 'bo-')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru Analizi')
    plt.grid(True)
    
    # Calinski-Harabasz
    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_clusters + 1), metrics['calinski'], 'ro-')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru Analizi')
    plt.grid(True)
    
    # Davies-Bouldin
    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_clusters + 1), metrics['davies'], 'go-')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi Analizi')
    plt.grid(True)
    
    # SSE
    plt.subplot(2, 2, 4)
    plt.plot(range(2, max_clusters + 1), metrics['sse'], 'mo-')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE Analizi')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics/detayli_metrik_analizi.png')
    plt.close()
    
    # 3. Gelişmiş Metrik Analizi
    plt.figure(figsize=(20, 15))
    
    # Normalize edilmiş metrikler
    normalized_metrics = {
        'silhouette': np.array(metrics['silhouette']) / max(metrics['silhouette']),
        'calinski': np.array(metrics['calinski']) / max(metrics['calinski']),
        'davies': 1 - (np.array(metrics['davies']) / max(metrics['davies'])),
        'sse': 1 - (np.array(metrics['sse']) / max(metrics['sse']))
    }
    
    # Tüm metriklerin ortalaması
    avg_score = np.mean([normalized_metrics[m] for m in normalized_metrics], axis=0)
    
    plt.plot(range(2, max_clusters + 1), avg_score, 'ko-', label='Ortalama Skor')
    plt.plot(range(2, max_clusters + 1), normalized_metrics['silhouette'], 'bo-', label='Silhouette')
    plt.plot(range(2, max_clusters + 1), normalized_metrics['calinski'], 'ro-', label='Calinski-Harabasz')
    plt.plot(range(2, max_clusters + 1), normalized_metrics['davies'], 'go-', label='Davies-Bouldin')
    plt.plot(range(2, max_clusters + 1), normalized_metrics['sse'], 'mo-', label='SSE')
    
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Normalize Edilmiş Skor')
    plt.title('Gelişmiş Metrik Analizi')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/metrics/gelismis_metrik_analizi.png')
    plt.close()
    
    # 4. Performans Metrikleri
    plt.figure(figsize=(20, 15))
    
    # K-means ve Ward karşılaştırması
    kmeans_metrics = {
        'silhouette': [],
        'calinski': [],
        'davies': [],
        'sse': []
    }
    
    ward_metrics = {
        'silhouette': [],
        'calinski': [],
        'davies': [],
        'sse': []
    }
    
    for k in range(2, max_clusters + 1):
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_metrics['silhouette'].append(silhouette_score(X, kmeans_labels))
        kmeans_metrics['calinski'].append(calinski_harabasz_score(X, kmeans_labels))
        kmeans_metrics['davies'].append(davies_bouldin_score(X, kmeans_labels))
        kmeans_metrics['sse'].append(kmeans.inertia_)
        
        # Ward
        ward = AgglomerativeClustering(n_clusters=k)
        ward_labels = ward.fit_predict(X)
        ward_metrics['silhouette'].append(silhouette_score(X, ward_labels))
        ward_metrics['calinski'].append(calinski_harabasz_score(X, ward_labels))
        ward_metrics['davies'].append(davies_bouldin_score(X, ward_labels))
        ward_metrics['sse'].append(np.sum((X - ward.fit(X).labels_.reshape(-1, 1)) ** 2))
    
    # Silhouette
    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_clusters + 1), kmeans_metrics['silhouette'], 'b-', label='K-means')
    plt.plot(range(2, max_clusters + 1), ward_metrics['silhouette'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # Calinski-Harabasz
    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_clusters + 1), kmeans_metrics['calinski'], 'b-', label='K-means')
    plt.plot(range(2, max_clusters + 1), ward_metrics['calinski'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # Davies-Bouldin
    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_clusters + 1), kmeans_metrics['davies'], 'b-', label='K-means')
    plt.plot(range(2, max_clusters + 1), ward_metrics['davies'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # SSE
    plt.subplot(2, 2, 4)
    plt.plot(range(2, max_clusters + 1), kmeans_metrics['sse'], 'b-', label='K-means')
    plt.plot(range(2, max_clusters + 1), ward_metrics['sse'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics/performans_metrikleri.png')
    plt.close()

def plot_metrics_comparison(metrics_kmeans, metrics_ward, n_clusters_range, save_path='visualizations/metrics/metrics_comparison.png'):
    """
    Metrikleri karşılaştırmalı olarak görselleştir
    """
    plt.figure(figsize=(15, 10))
    
    # Silhouette Skoru
    plt.subplot(2, 2, 1)
    plt.plot(n_clusters_range, metrics_kmeans['silhouette'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['silhouette'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # Calinski-Harabasz Skoru
    plt.subplot(2, 2, 2)
    plt.plot(n_clusters_range, metrics_kmeans['calinski'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['calinski'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # Davies-Bouldin İndeksi
    plt.subplot(2, 2, 3)
    plt.plot(n_clusters_range, metrics_kmeans['davies'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['davies'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # SSE
    plt.subplot(2, 2, 4)
    plt.plot(n_clusters_range, metrics_kmeans['sse'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['sse'], 'r-', label='Ward')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_distribution(labels_kmeans, labels_ward, save_path='visualizations/clustering/cluster_distribution.png'):
    """
    Küme dağılımlarını görselleştir
    """
    plt.figure(figsize=(15, 6))
    
    # K-means küme dağılımı
    plt.subplot(1, 2, 1)
    kmeans_counts = pd.Series(labels_kmeans).value_counts().sort_index()
    plt.bar(kmeans_counts.index, kmeans_counts.values, color='skyblue')
    plt.xlabel('Küme')
    plt.ylabel('Algoritma Sayısı')
    plt.title('K-means Küme Dağılımı')
    plt.grid(True, axis='y')
    
    # Ward küme dağılımı
    plt.subplot(1, 2, 2)
    ward_counts = pd.Series(labels_ward).value_counts().sort_index()
    plt.bar(ward_counts.index, ward_counts.values, color='lightgreen')
    plt.xlabel('Küme')
    plt.ylabel('Algoritma Sayısı')
    plt.title('Ward Küme Dağılımı')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_clusters(X_pca, labels_kmeans, labels_ward, save_path='visualizations/clustering/pca_clusters.png'):
    """
    PCA ile kümeleme sonuçlarını görselleştir
    """
    plt.figure(figsize=(15, 6))
    
    # K-means PCA görselleştirmesi
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Küme')
    plt.xlabel('Birinci Bileşen')
    plt.ylabel('İkinci Bileşen')
    plt.title('K-means Kümeleme (PCA)')
    plt.grid(True)
    
    # Ward PCA görselleştirmesi
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_ward, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Küme')
    plt.xlabel('Birinci Bileşen')
    plt.ylabel('İkinci Bileşen')
    plt.title('Ward Kümeleme (PCA)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimal_clusters(metrics_kmeans, metrics_ward, n_clusters_range, save_path='visualizations/metrics/optimal_clusters.png'):
    """
    Optimal küme sayısını belirlemek için metrikleri görselleştir
    """
    plt.figure(figsize=(15, 10))
    
    # Silhouette Skoru
    plt.subplot(2, 2, 1)
    plt.plot(n_clusters_range, metrics_kmeans['silhouette'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['silhouette'], 'r-', label='Ward')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru - Optimal Küme Sayısı')
    plt.legend()
    plt.grid(True)
    
    # Calinski-Harabasz Skoru
    plt.subplot(2, 2, 2)
    plt.plot(n_clusters_range, metrics_kmeans['calinski'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['calinski'], 'r-', label='Ward')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru - Optimal Küme Sayısı')
    plt.legend()
    plt.grid(True)
    
    # Davies-Bouldin İndeksi
    plt.subplot(2, 2, 3)
    plt.plot(n_clusters_range, metrics_kmeans['davies'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['davies'], 'r-', label='Ward')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi - Optimal Küme Sayısı')
    plt.legend()
    plt.grid(True)
    
    # SSE
    plt.subplot(2, 2, 4)
    plt.plot(n_clusters_range, metrics_kmeans['sse'], 'b-', label='K-means')
    plt.plot(n_clusters_range, metrics_ward['sse'], 'r-', label='Ward')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE - Optimal Küme Sayısı')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(df, save_path='visualizations/analysis/correlation_heatmap.png'):
    """
    Sayısal özellikler arasındaki korelasyonu görselleştir
    """
    # Sayısal özellikleri seç ve dönüştür
    numeric_df = df.copy()
    
    # Karmaşıklık düzeyini sayısal değere dönüştür
    numeric_df['Karmaşıklık Düzeyi'] = numeric_df['Karmaşıklık Düzeyi'].str.replace('comp', '').astype(int)
    
    # Popülerliği sayısal değere dönüştür
    numeric_df['Popülerlik'] = numeric_df['Popülerlik'].str.replace('p', '').astype(int)
    
    # Donanım gereksinimlerini sayısal değere dönüştür
    def convert_hardware(x):
        if isinstance(x, str):
            if '-' in x:
                start, end = map(float, x.split('-'))
                return (start + end) / 2
            return float(x)
        return x
    
    numeric_df['Donanım Gerkesinimleri'] = numeric_df['Donanım Gerkesinimleri'].apply(convert_hardware)
    
    # Sayısal sütunları seç
    numeric_cols = ['Karmaşıklık Düzeyi', 'Popülerlik', 'Donanım Gerkesinimleri']
    numeric_df = numeric_df[numeric_cols]
    
    # Korelasyon matrisini hesapla
    corr_matrix = numeric_df.corr()
    
    # Isı haritası oluştur
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Sayısal Özellikler Arasındaki Korelasyon')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_analysis(df, save_path='visualizations/analysis/dataset_analysis.png'):
    """
    Veri seti analizini görselleştir
    """
    plt.figure(figsize=(15, 10))
    
    # Kategorik özelliklerin dağılımı
    plt.subplot(2, 2, 1)
    df['Öğrenme Türü'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Öğrenme Türü Dağılımı')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 2)
    df['Model Yapısı'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Model Yapısı Dağılımı')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 3)
    df['Katman Tipi'].value_counts().plot(kind='bar', color='salmon')
    plt.title('Katman Tipi Dağılımı')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.subplot(2, 2, 4)
    df['Veri Tipi'].value_counts().plot(kind='bar', color='orange')
    plt.title('Veri Tipi Dağılımı')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_numeric_features_distribution(df, save_path='visualizations/analysis/numeric_features_distribution.png'):
    """
    Sayısal özelliklerin dağılımını görselleştir
    """
    # Sayısal özellikleri dönüştür
    numeric_df = df.copy()
    
    # Karmaşıklık düzeyini sayısal değere dönüştür
    numeric_df['Karmaşıklık Düzeyi'] = numeric_df['Karmaşıklık Düzeyi'].str.replace('comp', '').astype(int)
    
    # Popülerliği sayısal değere dönüştür
    numeric_df['Popülerlik'] = numeric_df['Popülerlik'].str.replace('p', '').astype(int)
    
    # Donanım gereksinimlerini sayısal değere dönüştür
    def convert_hardware(x):
        if isinstance(x, str):
            if '-' in x:
                start, end = map(float, x.split('-'))
                return (start + end) / 2
            return float(x)
        return x
    
    numeric_df['Donanım Gerkesinimleri'] = numeric_df['Donanım Gerkesinimleri'].apply(convert_hardware)
    
    plt.figure(figsize=(15, 5))
    
    # Karmaşıklık Düzeyi
    plt.subplot(1, 3, 1)
    sns.histplot(data=numeric_df, x='Karmaşıklık Düzeyi', bins=10, color='skyblue')
    plt.title('Karmaşıklık Düzeyi Dağılımı')
    plt.grid(True)
    
    # Popülerlik
    plt.subplot(1, 3, 2)
    sns.histplot(data=numeric_df, x='Popülerlik', bins=10, color='lightgreen')
    plt.title('Popülerlik Dağılımı')
    plt.grid(True)
    
    # Donanım Gereksinimleri
    plt.subplot(1, 3, 3)
    sns.histplot(data=numeric_df, x='Donanım Gerkesinimleri', bins=10, color='salmon')
    plt.title('Donanım Gereksinimleri Dağılımı')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_metrics(metrics_kmeans, metrics_ward, n_clusters_range, save_path='visualizations/metrics/model_performance.png'):
    """
    Model performans metriklerini görselleştir
    """
    plt.figure(figsize=(15, 10))
    
    # Silhouette Skoru
    plt.subplot(2, 2, 1)
    plt.plot(n_clusters_range, metrics_kmeans['silhouette'], 'b-', label='K-means', marker='o')
    plt.plot(n_clusters_range, metrics_ward['silhouette'], 'r-', label='Ward', marker='s')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Silhouette Skoru - Model Performansı')
    plt.legend()
    plt.grid(True)
    
    # Calinski-Harabasz Skoru
    plt.subplot(2, 2, 2)
    plt.plot(n_clusters_range, metrics_kmeans['calinski'], 'b-', label='K-means', marker='o')
    plt.plot(n_clusters_range, metrics_ward['calinski'], 'r-', label='Ward', marker='s')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Calinski-Harabasz Skoru')
    plt.title('Calinski-Harabasz Skoru - Model Performansı')
    plt.legend()
    plt.grid(True)
    
    # Davies-Bouldin İndeksi
    plt.subplot(2, 2, 3)
    plt.plot(n_clusters_range, metrics_kmeans['davies'], 'b-', label='K-means', marker='o')
    plt.plot(n_clusters_range, metrics_ward['davies'], 'r-', label='Ward', marker='s')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Davies-Bouldin İndeksi')
    plt.title('Davies-Bouldin İndeksi - Model Performansı')
    plt.legend()
    plt.grid(True)
    
    # SSE
    plt.subplot(2, 2, 4)
    plt.plot(n_clusters_range, metrics_kmeans['sse'], 'b-', label='K-means', marker='o')
    plt.plot(n_clusters_range, metrics_ward['sse'], 'r-', label='Ward', marker='s')
    plt.axvline(x=27, color='g', linestyle='--', label='Optimal Küme Sayısı')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('SSE')
    plt.title('SSE - Model Performansı')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 27 küme için detaylı metrik tablosu
    print("\n27 Küme için Model Performans Metrikleri:")
    print("\nK-means Metrikleri:")
    print(f"Silhouette Skoru: {metrics_kmeans['silhouette'][7]:.4f}")
    print(f"Calinski-Harabasz Skoru: {metrics_kmeans['calinski'][7]:.4f}")
    print(f"Davies-Bouldin İndeksi: {metrics_kmeans['davies'][7]:.4f}")
    print(f"SSE: {metrics_kmeans['sse'][7]:.4f}")
    
    print("\nWard Metrikleri:")
    print(f"Silhouette Skoru: {metrics_ward['silhouette'][7]:.4f}")
    print(f"Calinski-Harabasz Skoru: {metrics_ward['calinski'][7]:.4f}")
    print(f"Davies-Bouldin İndeksi: {metrics_ward['davies'][7]:.4f}")
    print(f"SSE: {metrics_ward['sse'][7]:.4f}")

def check_cluster_separation(features, labels):
    """
    Küme içi ve küme arası mesafeleri hesapla
    """
    intra_cluster_dist = []
    inter_cluster_dist = []
    unique_labels = np.unique(labels)
    
    for i in unique_labels:
        cluster_points = features[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        
        # Küme içi mesafe
        intra_dist = np.mean(np.linalg.norm(cluster_points - cluster_center, axis=1))
        intra_cluster_dist.append(intra_dist)
        
        # Diğer kümelerle arasındaki mesafe
        for j in unique_labels:
            if i != j:
                other_cluster_points = features[labels == j]
                other_center = np.mean(other_cluster_points, axis=0)
                inter_dist = np.linalg.norm(cluster_center - other_center)
                inter_cluster_dist.append(inter_dist)
    
    return np.mean(intra_cluster_dist), np.mean(inter_cluster_dist)

def analyze_cluster_quality(features, labels_kmeans, labels_ward):
    """
    Küme kalitesini analiz et ve raporla
    """
    print("\nKüme Kalitesi Analizi:")
    
    # K-means için analiz
    intra_kmeans, inter_kmeans = check_cluster_separation(features, labels_kmeans)
    print("\nK-means Algoritması:")
    print(f"Ortalama Küme İçi Mesafe: {intra_kmeans:.4f}")
    print(f"Ortalama Küme Arası Mesafe: {inter_kmeans:.4f}")
    print(f"Mesafe Oranı (İç/Arası): {intra_kmeans/inter_kmeans:.4f}")
    
    # Ward için analiz
    intra_ward, inter_ward = check_cluster_separation(features, labels_ward)
    print("\nWard Algoritması:")
    print(f"Ortalama Küme İçi Mesafe: {intra_ward:.4f}")
    print(f"Ortalama Küme Arası Mesafe: {inter_ward:.4f}")
    print(f"Mesafe Oranı (İç/Arası): {intra_ward/inter_ward:.4f}")
    
    # Kalite değerlendirmesi
    print("\nKalite Değerlendirmesi:")
    if intra_kmeans/inter_kmeans < 0.5:
        print("K-means: İyi küme ayrımı (Küme içi mesafe, küme arası mesafenin yarısından az)")
    else:
        print("K-means: Küme ayrımı iyileştirilmeli")
        
    if intra_ward/inter_ward < 0.5:
        print("Ward: İyi küme ayrımı (Küme içi mesafe, küme arası mesafenin yarısından az)")
    else:
        print("Ward: Küme ayrımı iyileştirilmeli")

def main():
    print("Veri seti okunuyor...")
    # Veri setini oku ve sütun isimlerini kontrol et
    df = pd.read_csv('algorithms/Veri_seti.csv', encoding='utf-8')
    print("\nSütun isimleri:", df.columns.tolist())
    
    # Veri analizi görselleştirmeleri
    print("\nVeri analizi görselleştirmeleri oluşturuluyor...")
    plot_correlation_heatmap(df)
    plot_dataset_analysis(df)
    plot_numeric_features_distribution(df)
    
    # Veri ön işleme
    print("\nVeri ön işleme başlatılıyor...")
    print("Kategorik özellikler işleniyor...")
    categorical_features = preprocess_categorical_features(df)
    
    print("Sayısal özellikler işleniyor...")
    numerical_features = preprocess_numerical_features(df)
    
    print("Kullanım alanları işleniyor...")
    usage_features = preprocess_usage_areas(df)
    
    print("Özellik seçimi yapılıyor...")
    features = pd.concat([categorical_features, numerical_features, usage_features], axis=1)
    
    print("Veri ön işleme tamamlandı!")
    
    # PCA uygula
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    
    # Farklı küme sayıları için metrikleri hesapla
    n_clusters_range = range(20, 31)
    metrics_kmeans = {'silhouette': [], 'calinski': [], 'davies': [], 'sse': []}
    metrics_ward = {'silhouette': [], 'calinski': [], 'davies': [], 'sse': []}
    
    print("\nFarklı küme sayıları için metrikler hesaplanıyor...")
    for n_clusters in n_clusters_range:
        print(f"\nKüme sayısı: {n_clusters}")
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(features)
        
        # Ward
        ward = AgglomerativeClustering(n_clusters=n_clusters)
        labels_ward = ward.fit_predict(features)
        
        # Metrikleri hesapla
        metrics_kmeans['silhouette'].append(silhouette_score(features, labels_kmeans))
        metrics_kmeans['calinski'].append(calinski_harabasz_score(features, labels_kmeans))
        metrics_kmeans['davies'].append(davies_bouldin_score(features, labels_kmeans))
        metrics_kmeans['sse'].append(kmeans.inertia_)
        
        metrics_ward['silhouette'].append(silhouette_score(features, labels_ward))
        metrics_ward['calinski'].append(calinski_harabasz_score(features, labels_ward))
        metrics_ward['davies'].append(davies_bouldin_score(features, labels_ward))
        metrics_ward['sse'].append(ward.n_clusters_)
    
    # 27 küme için son kümeleme
    print("\n27 küme için son kümeleme yapılıyor...")
    kmeans = KMeans(n_clusters=27, random_state=42)
    labels_kmeans = kmeans.fit_predict(features)
    
    ward = AgglomerativeClustering(n_clusters=27)
    labels_ward = ward.fit_predict(features)
    
    # Küme kalitesi analizi
    analyze_cluster_quality(features, labels_kmeans, labels_ward)
    
    # Kümeleme görselleştirmeleri
    print("\nKümeleme görselleştirmeleri oluşturuluyor...")
    plot_metrics_comparison(metrics_kmeans, metrics_ward, n_clusters_range)
    plot_cluster_distribution(labels_kmeans, labels_ward)
    plot_pca_clusters(X_pca, labels_kmeans, labels_ward)
    plot_optimal_clusters(metrics_kmeans, metrics_ward, n_clusters_range)
    plot_model_performance_metrics(metrics_kmeans, metrics_ward, n_clusters_range)
    
    # 27 küme için son görselleştirmeleri yap
    print("\n27 küme için son görselleştirmeler yapılıyor...")
    visualize_all_metrics(X_pca)
    visualize_dendrogram(X_pca)
    
    # Modelleri kaydet
    print("\nModeller kaydediliyor...")
    scaler = StandardScaler()
    scaler.fit(features)
    save_models(kmeans, ward, scaler, pca)
    
    print("\nAnaliz tamamlandı!")

if __name__ == "__main__":
    main() 