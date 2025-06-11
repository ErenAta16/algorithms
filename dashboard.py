import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import altair as alt
import plotly.figure_factory as ff

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kümeleme Analizi Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("📊 Kümeleme Analizi Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("📑 İçindekiler")
    
    page = st.radio(
        "Sayfalar",
        ["Genel Bakış", "Metrik Analizi", "Kümeleme Sonuçları", "Detaylı Raporlar"],
        label_visibility="collapsed"
    )
    
    # Sidebar'ın altına boşluk ekle
    st.markdown("<br>" * 1, unsafe_allow_html=True)
    
    # Karekod
    try:
        qr_code = Image.open('./frame.png')
        st.image(qr_code, caption="Proje Karekodu", width=250)
    except Exception as e:
        st.warning(f"Karekod görüntüsü yüklenemedi: {str(e)}")

# Veri yükleme fonksiyonları
def load_metrics_report(algorithm):
    try:
        with open(f'reports/{algorithm}_metrik_raporu.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def load_visualization(image_path):
    try:
        return Image.open(image_path)
    except:
        return None

def display_metrics_comparison():
    st.subheader("📊 Metrik Karşılaştırması")
    
    # Bilgi kutusu
    st.info("""
    Bu grafik, farklı küme sayıları için hesaplanan metriklerin karşılaştırmalı 
    analizini göstermektedir. Her metrik, kümeleme kalitesinin farklı bir yönünü 
    ölçer ve birlikte değerlendirildiğinde daha kapsamlı bir analiz sağlar.
    
    En İyi Sonuçlar:
    - Ward Algoritması: Silhouette: 0.2192, SSE: 27.0000
    - K-means Algoritması: Silhouette: 0.2056, SSE: 3359.3992
    """)
    
    detayli_metrik_img = load_visualization('visualizations/metrics/detayli_metrik_analizi.png')
    if detayli_metrik_img:
        st.image(detayli_metrik_img, use_container_width=True)

def display_clustering_results():
    st.subheader("🎯 Kümeleme Sonuçları")
    
    # Bilgi kutusu
    st.info("""
    Bu grafikler, 27 küme için K-means ve Ward algoritmalarının kümeleme sonuçlarını 
    göstermektedir. Ward algoritması daha iyi performans göstermiştir:
    
    Ward Algoritması:
    - Silhouette Skoru: 0.2192
    - SSE: 27.0000
    - Calinski-Harabasz: 16.3164
    - Davies-Bouldin: 1.5130
    
    K-means Algoritması:
    - Silhouette Skoru: 0.2056
    - SSE: 3359.3992
    - Calinski-Harabasz: 15.9301
    - Davies-Bouldin: 1.6088
    """)
    
    kumeleme_sonuclari_img = load_visualization('visualizations/clustering/kumeleme_sonuclari.png')
    if kumeleme_sonuclari_img:
        st.image(kumeleme_sonuclari_img, use_container_width=True)

def display_cluster_distribution():
    st.subheader("📈 Küme Dağılımları")
    
    # Bilgi kutusu
    st.info("""
    Bu grafikler, her iki algoritmanın oluşturduğu kümelerin dağılımını göstermektedir.
    Ward algoritması daha dengeli bir dağılım sağlamıştır.
    """)
    
    kume_dagilimi_img = load_visualization('visualizations/clustering/cluster_distribution.png')
    if kume_dagilimi_img:
        st.image(kume_dagilimi_img, use_container_width=True)

def display_dendrogram():
    st.subheader("🌳 Dendrogram")
    
    # Bilgi kutusu
    st.info("""
    Bu dendrogram, Ward algoritmasının hiyerarşik kümeleme yapısını göstermektedir.
    Optimal küme sayısı 27 olarak belirlenmiştir.
    
    Ward Algoritması Performansı:
    - Silhouette Skoru: 0.2192
    - SSE: 27.0000
    """)
    
    dendrogram_img = load_visualization('visualizations/clustering/dendrogram.png')
    if dendrogram_img:
        st.image(dendrogram_img, use_container_width=True)

def display_pca_visualization():
    st.subheader("🔍 PCA Görselleştirmesi")
    
    # Bilgi kutusu
    st.info("""
    Bu grafikler, veriyi 2 boyuta indirgeyerek (PCA) kümeleme sonuçlarını göstermektedir.
    Ward algoritması daha iyi küme ayrımı sağlamıştır.
    """)
    
    pca_img = load_visualization('visualizations/clustering/pca_clusters.png')
    if pca_img:
        st.image(pca_img, use_container_width=True)

def display_optimal_clusters():
    st.subheader("🎯 Optimal Küme Sayısı Analizi")
    
    # Bilgi kutusu
    st.info("""
    Bu grafikler, farklı küme sayıları için hesaplanan metrikleri göstermektedir.
    27 küme sayısı optimal olarak belirlenmiştir.
    
    En İyi Sonuçlar (27 Küme):
    Ward Algoritması:
    - Silhouette: 0.2192
    - SSE: 27.0000
    - Calinski-Harabasz: 16.3164
    - Davies-Bouldin: 1.5130
    """)
    
    optimal_kume_img = load_visualization('visualizations/metrics/optimal_clusters.png')
    if optimal_kume_img:
        st.image(optimal_kume_img, use_container_width=True)

def display_model_performance():
    st.subheader("📊 Model Performans Analizi")
    
    # Bilgi kutusu
    st.info("""
    Bu grafikler, her iki algoritmanın performans metriklerini karşılaştırmalı olarak 
    göstermektedir. Ward algoritması tüm metriklerde daha iyi performans göstermiştir.
    """)
    
    performans_img = load_visualization('visualizations/metrics/model_performance.png')
    if performans_img:
        st.image(performans_img, use_container_width=True)

def main():
    st.title("Algoritma Kümeleme Analizi")
    
    # Sidebar
    st.sidebar.title("Görselleştirmeler")
    page = st.sidebar.selectbox(
        "Görselleştirme Seçin",
        ["Metrik Karşılaştırması", "Kümeleme Sonuçları", "Küme Dağılımları", 
         "Dendrogram", "PCA Görselleştirmesi", "Optimal Küme Sayısı", "Model Performansı"]
    )
    
    if page == "Metrik Karşılaştırması":
        display_metrics_comparison()
    elif page == "Kümeleme Sonuçları":
        display_clustering_results()
    elif page == "Küme Dağılımları":
        display_cluster_distribution()
    elif page == "Dendrogram":
        display_dendrogram()
    elif page == "PCA Görselleştirmesi":
        display_pca_visualization()
    elif page == "Optimal Küme Sayısı":
        display_optimal_clusters()
    elif page == "Model Performansı":
        display_model_performance()

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Kümeleme Model Analizi</p>
    </div>
""", unsafe_allow_html=True)
