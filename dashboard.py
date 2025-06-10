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

# Sayfa içerikleri
if page == "Genel Bakış":
    st.header("🔍 Proje Genel Bakış")
    
    # Proje bilgileri
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Proje Hakkında")
        st.markdown("""
        Bu proje, algoritmaların özelliklerini ve kullanım alanlarını içeren bir veri seti üzerinde 
        kümeleme analizi yapmayı amaçlamaktadır. K-means ve Ward algoritmaları kullanılarak 
        algoritmaların benzerliklerine göre gruplandırılması sağlanmıştır.
        """)
        
        st.subheader("🎯 Kullanılan Algoritmalar")
        st.markdown("""
        - **K-means Clustering**
          - Hızlı ve ölçeklenebilir
          - Küresel kümeler oluşturur
          - Önceden belirlenmiş küme sayısı gerektirir
        
        - **Ward (Hierarchical) Clustering**
          - Hiyerarşik yapı oluşturur
          - Küme sayısını otomatik belirler
          - Daha detaylı küme analizi sağlar
        """)
    
    with col2:
        st.subheader("📊 Veri Seti Özellikleri")
        metrics = {
            "Toplam Algoritma Sayısı": "300+",
            "Özellik Sayısı": "12",
            "Küme Sayısı": "4-5",
            "Analiz Türü": "Denetimsiz Öğrenme"
        }
        
        for key, value in metrics.items():
            st.metric(key, value)
    
    st.markdown("---")
    
    # Dirsek yöntemi analizi
    st.subheader("📈 Dirsek Yöntemi Analizi")
    
    # Bilgi kutusu
    st.markdown("""
    <div class="info-box">
        <h4>Dirsek Yöntemi Nedir?</h4>
        <p>Dirsek yöntemi, optimal küme sayısını belirlemek için kullanılan bir tekniktir. 
        Bu yöntemde, küme sayısı arttıkça SSE (Sum of Squared Errors) değerindeki azalma 
        grafiğinde bir "dirsek" noktası aranır. Bu nokta, küme sayısının daha fazla 
        artırılmasının SSE'yi önemli ölçüde azaltmayacağını gösterir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grafik ve açıklama
    col1, col2 = st.columns([2, 1])
    
    with col1:
        elbow_img = load_visualization('visualizations/metrics/dirsek_yontemi.png')
        if elbow_img:
            st.image(elbow_img, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 📊 Grafik Yorumu
        
        - **X Ekseni**: Küme sayısı (k)
        - **Y Ekseni**: SSE değeri
        
        #### Önemli Noktalar:
        1. **Dirsek Noktası**: Grafikte belirgin bir eğim değişimi
        2. **Optimal Küme Sayısı**: Dirsek noktasındaki k değeri
        3. **SSE Azalması**: Her k değeri için SSE'deki azalma oranı
        
        #### Yorum:
        - Küme sayısı arttıkça SSE değeri azalır
        - Dirsek noktasından sonra azalma hızı düşer
        - Bu nokta optimal küme sayısını gösterir
        """)

elif page == "Metrik Analizi":
    st.header("📊 Metrik Analizi")
    
    # Metrik seçimi
    metric_type = st.selectbox(
        "Metrik Türü",
        ["Silhouette Skoru", "Calinski-Harabasz Skoru", "Davies-Bouldin İndeksi", "SSE"]
    )
    
    # K-means ve Ward metriklerini yükle
    kmeans_metrics = load_metrics_report('kmeans')
    ward_metrics = load_metrics_report('ward')
    
    if kmeans_metrics and ward_metrics:
        # Metrik kartları
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("K-means Metrikleri")
            metric_key = {
                "Silhouette Skoru": "silhouette",
                "Calinski-Harabasz Skoru": "calinski_harabasz",
                "Davies-Bouldin İndeksi": "davies_bouldin",
                "SSE": "SSE"
            }[metric_type]
            
            value = kmeans_metrics['metrics'][metric_key]
            
            # SSE için özel yorum
            if metric_key == 'SSE':
                interpretation = 'Küçük değerler kompakt kümeleri gösterir'
                explanation = 'Küme içi kareler toplamı (SSW) - Kümeleme kalitesinin bir göstergesidir'
            else:
                interpretation = kmeans_metrics['interpretations'][metric_key]['interpretation']
                explanation = kmeans_metrics['interpretations'][metric_key]['explanation']
            
            st.metric(
                metric_type,
                f"{value:.4f}",
                interpretation
            )
            
            st.info(explanation)
        
        with col2:
            st.subheader("Ward Metrikleri")
            value = ward_metrics['metrics'][metric_key]
            
            # SSE için özel yorum
            if metric_key == 'SSE':
                interpretation = 'Küçük değerler kompakt kümeleri gösterir'
                explanation = 'Küme içi kareler toplamı (SSW) - Kümeleme kalitesinin bir göstergesidir'
            else:
                interpretation = ward_metrics['interpretations'][metric_key]['interpretation']
                explanation = ward_metrics['interpretations'][metric_key]['explanation']
            
            st.metric(
                metric_type,
                f"{value:.4f}",
                interpretation
            )
            
            st.info(explanation)
        
        # Metrik karşılaştırma grafiği
        st.subheader("📈 Metrik Karşılaştırması")
        
        # Bilgi kutusu
        st.info("""
        Bu grafik, farklı küme sayıları için hesaplanan metriklerin karşılaştırmalı 
        analizini göstermektedir. Her metrik, kümeleme kalitesinin farklı bir yönünü 
        ölçer ve birlikte değerlendirildiğinde daha kapsamlı bir analiz sağlar.
        """)
        
        detayli_metrik_img = load_visualization('visualizations/metrics/detayli_metrik_analizi.png')
        if detayli_metrik_img:
            st.image(detayli_metrik_img, use_container_width=True)

elif page == "Kümeleme Sonuçları":
    st.header("🎯 Kümeleme Sonuçları")
    
    # Bilgi kutusu
    st.info("""
    Bu bölümde, K-means ve Ward algoritmalarının kümeleme sonuçları 
    görselleştirilmiştir. Her iki algoritmanın sonuçları karşılaştırmalı olarak 
    sunulmuştur.
    """)
    
    # Algoritma seçimi
    algorithm = st.radio(
        "Algoritma Seçimi",
        ["K-means", "Ward"],
        horizontal=True,
        label_visibility="visible"
    )
    
    if algorithm == "K-means":
        st.subheader("K-means Kümeleme Sonuçları")
        
        # Kümeleme sonuçları
        col1, col2 = st.columns(2)
        
        with col1:
            kmeans_img = load_visualization('visualizations/clustering/kumeleme_sonuclari.png')
            if kmeans_img:
                st.image(kmeans_img, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### 📊 K-means Kümeleme Analizi
            
            #### Önemli Noktalar:
            1. **Küme Merkezleri**: Her kümenin merkez noktası
            2. **Küme Sınırları**: Kümelerin birbirinden ayrılma durumu
            3. **Küme Yoğunluğu**: Noktaların küme merkezine yakınlığı
            
            #### Yorum:
            - Kümeler arası ayrım net mi?
            - Küme içi yoğunluk nasıl?
            - Aykırı değerler var mı?
            """)
        
        st.subheader("K-means Küme Boyutları")
        kmeans_sizes_img = load_visualization('visualizations/clustering/kume_boyutlari.png')
        if kmeans_sizes_img:
            st.image(kmeans_sizes_img, use_container_width=True)
    
    else:  # Ward
        st.subheader("Ward Kümeleme Sonuçları")
        
        # Kümeleme sonuçları
        col1, col2 = st.columns(2)
        
        with col1:
            ward_img = load_visualization('visualizations/clustering/kumeleme_sonuclari.png')
            if ward_img:
                st.image(ward_img, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### 📊 Ward Kümeleme Analizi
            
            #### Önemli Noktalar:
            1. **Hiyerarşik Yapı**: Kümelerin oluşum sırası
            2. **Küme İlişkileri**: Kümeler arası benzerlikler
            3. **Küme Kararlılığı**: Farklı küme sayılarındaki tutarlılık
            
            #### Yorum:
            - Hiyerarşik yapı belirgin mi?
            - Kümeler arası ilişkiler nasıl?
            - Kararlı küme yapıları var mı?
            """)
        
        st.subheader("Ward Küme Boyutları")
        ward_sizes_img = load_visualization('visualizations/clustering/kume_boyutlari.png')
        if ward_sizes_img:
            st.image(ward_sizes_img, use_container_width=True)

elif page == "Detaylı Raporlar":
    st.header("📑 Detaylı Raporlar")
    
    # Bilgi kutusu
    st.info("""
    Bu bölümde, her iki algoritma için detaylı metrik raporları ve yorumlar 
    sunulmaktadır. Her metrik, kümeleme kalitesinin farklı bir yönünü ölçer ve 
    birlikte değerlendirildiğinde daha kapsamlı bir analiz sağlar.
    """)
    
    # Algoritma seçimi
    algorithm = st.radio(
        "Algoritma Seçimi",
        ["K-means", "Ward"],
        horizontal=True,
        label_visibility="visible"
    )
    
    if algorithm == "K-means":
        st.subheader("K-means Algoritması Detaylı Raporu")
        kmeans_metrics = load_metrics_report('kmeans')
        if kmeans_metrics:
            # Metrikler bölümü
            st.write("### 📊 Metrik Değerleri")
            st.write("Aşağıda, K-means algoritmasının kümeleme performansını değerlendiren metriklerin detaylı sonuçları yer almaktadır.")
            
            # Metrikleri iki sütunda göster
            col1, col2 = st.columns(2)
            with col1:
                for metric_name, value in list(kmeans_metrics['metrics'].items())[:2]:
                    st.metric(
                        label=metric_name,
                        value=f"{value:.4f}",
                        delta=None
                    )
            with col2:
                for metric_name, value in list(kmeans_metrics['metrics'].items())[2:]:
                    st.metric(
                        label=metric_name,
                        value=f"{value:.4f}",
                        delta=None
                    )
            
            # Yorumlar bölümü
            st.write("### 📝 Metrik Yorumları ve Açıklamaları")
            st.write("Her bir metriğin detaylı açıklaması ve yorumu aşağıda sunulmaktadır.")
            
            for metric, interpretation in kmeans_metrics['interpretations'].items():
                with st.expander(f"🔍 {metric} Analizi"):
                    st.write("#### Yorum")
                    st.write(interpretation['interpretation'])
                    st.write("#### Detaylı Açıklama")
                    st.write(interpretation['explanation'])
                    st.write("#### Öneriler")
                    if metric == "Silhouette Skoru":
                        st.write("- Yüksek değerler (0.7 üzeri) iyi ayrılmış kümeleri gösterir")
                        st.write("- Düşük değerler küme sayısının yeniden değerlendirilmesi gerektiğini işaret eder")
                    elif metric == "Calinski-Harabasz Skoru":
                        st.write("- Yüksek değerler daha iyi küme ayrımını gösterir")
                        st.write("- Farklı küme sayıları için karşılaştırma yapılabilir")
                    elif metric == "Davies-Bouldin İndeksi":
                        st.write("- Düşük değerler daha iyi küme ayrımını gösterir")
                        st.write("- 1'den küçük değerler tercih edilir")
                    elif metric == "SSE":
                        st.write("- Düşük değerler kompakt kümeleri gösterir")
                        st.write("- Dirsek yöntemi ile optimal küme sayısı belirlenebilir")
    
    else:  # Ward
        st.subheader("Ward Algoritması Detaylı Raporu")
        ward_metrics = load_metrics_report('ward')
        if ward_metrics:
            # Metrikler bölümü
            st.write("### 📊 Metrik Değerleri")
            st.write("Aşağıda, Ward algoritmasının kümeleme performansını değerlendiren metriklerin detaylı sonuçları yer almaktadır.")
            
            # Metrikleri iki sütunda göster
            col1, col2 = st.columns(2)
            with col1:
                for metric_name, value in list(ward_metrics['metrics'].items())[:2]:
                    st.metric(
                        label=metric_name,
                        value=f"{value:.4f}",
                        delta=None
                    )
            with col2:
                for metric_name, value in list(ward_metrics['metrics'].items())[2:]:
                    st.metric(
                        label=metric_name,
                        value=f"{value:.4f}",
                        delta=None
                    )
            
            # Yorumlar bölümü
            st.write("### 📝 Metrik Yorumları ve Açıklamaları")
            st.write("Her bir metriğin detaylı açıklaması ve yorumu aşağıda sunulmaktadır.")
            
            for metric, interpretation in ward_metrics['interpretations'].items():
                with st.expander(f"🔍 {metric} Analizi"):
                    st.write("#### Yorum")
                    st.write(interpretation['interpretation'])
                    st.write("#### Detaylı Açıklama")
                    st.write(interpretation['explanation'])
                    st.write("#### Öneriler")
                    if metric == "Silhouette Skoru":
                        st.write("- Yüksek değerler (0.7 üzeri) iyi ayrılmış kümeleri gösterir")
                        st.write("- Düşük değerler küme sayısının yeniden değerlendirilmesi gerektiğini işaret eder")
                    elif metric == "Calinski-Harabasz Skoru":
                        st.write("- Yüksek değerler daha iyi küme ayrımını gösterir")
                        st.write("- Farklı küme sayıları için karşılaştırma yapılabilir")
                    elif metric == "Davies-Bouldin İndeksi":
                        st.write("- Düşük değerler daha iyi küme ayrımını gösterir")
                        st.write("- 1'den küçük değerler tercih edilir")
                    elif metric == "SSE":
                        st.write("- Düşük değerler kompakt kümeleri gösterir")
                        st.write("- Dirsek yöntemi ile optimal küme sayısı belirlenebilir")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Kümeleme Model Analizi</p>
    </div>
""", unsafe_allow_html=True)
