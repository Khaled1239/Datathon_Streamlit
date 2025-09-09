import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Dashboard Produksi Padi Jawa Timur", layout="wide")

st.title("🌾 Dashboard Analisis Produksi Padi - Jawa Timur")

# ====== Sidebar ======
menu = st.sidebar.radio(
    "Pilih Analisis",
    [
        "Dashboard Overview",
        "Trend Produksi Padi Per Tahun",
        "Hubungan Cuaca dengan Produksi",
        "Distribusi & Outlier Produksi",
        "Analisis Spasial Antar Daerah",
        "Choropleth Maps Jawa Timur",
        "Prediksi Produksi Padi",
        "Prediksi Per Kabupaten"
    ]
)

# ====== Load Data ======
@st.cache_data
def load_data():
    df = pd.read_csv("Limao.csv")
    return df

df = load_data()

# ====== DASHBOARD OVERVIEW ======
if menu == "Dashboard Overview":
    st.subheader("📊 Ringkasan Produksi Padi")

    # KPI Cards
    total_produksi = df["Produksi_Padi_Ton_clean"].sum()
    produksi_rata = df["Produksi_Padi_Ton_clean"].mean()
    kabupaten = df["Kabupaten_Kota"].nunique()
    tahun_terakhir = df["Tahun"].max()
    produksi_terakhir = df[df["Tahun"] == tahun_terakhir]["Produksi_Padi_Ton_clean"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Produksi", f"{total_produksi:,.0f} ton")
    col2.metric("Rata-rata Produksi", f"{produksi_rata:,.0f} ton")
    col3.metric("Jumlah Kabupaten", kabupaten)
    col4.metric(f"Produksi {tahun_terakhir}", f"{produksi_terakhir:,.0f} ton")

    # Trend Produksi mini chart
    st.markdown("### 📈 Trend Produksi Tahunan")
    trend = df.groupby("Tahun")["Produksi_Padi_Ton_clean"].sum().reset_index()
    fig = px.line(trend, x="Tahun", y="Produksi_Padi_Ton_clean", markers=True)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Dua grafik kecil sejajar
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 🌦️ Curah Hujan vs Produksi")
        fig = px.scatter(df, x="Curah_Hujan_mm_clean", y="Produksi_Padi_Ton_clean",
                         color="Kabupaten_Kota", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("### 📦 Distribusi Produksi (Boxplot)")
        fig = px.box(df, y="Produksi_Padi_Ton_clean", points="all")
        st.plotly_chart(fig, use_container_width=True)

# ====== Trend Produksi ======
elif menu == "Trend Produksi Padi Per Tahun":
    st.subheader("📈 Trend Produksi Padi Per Tahun")
    trend = df.groupby("Tahun")["Produksi_Padi_Ton_clean"].sum().reset_index()
    fig = px.line(trend, x="Tahun", y="Produksi_Padi_Ton_clean", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ====== Hubungan Cuaca ======
elif menu == "Hubungan Cuaca dengan Produksi":
    st.subheader("☁️ Hubungan Faktor Cuaca dengan Produksi Padi")
    faktor_map = {
        "Suhu": "Suhu_Rata_C_clean",
        "Curah Hujan": "Curah_Hujan_mm_clean",
        "Kelembapan": "Kelembapan_Persen_clean"
    }
    faktor = st.selectbox("Pilih faktor cuaca:", list(faktor_map.keys()))
    fig = px.scatter(df, x=faktor_map[faktor], y="Produksi_Padi_Ton_clean",
                     color="Kabupaten_Kota", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# ====== Distribusi ======
elif menu == "Distribusi & Outlier Produksi":
    st.subheader("📦 Distribusi Produksi Padi & Outlier")
    fig = px.box(df, y="Produksi_Padi_Ton_clean", points="all")
    st.plotly_chart(fig, use_container_width=True)

# ====== Analisis Spasial ======
elif menu == "Analisis Spasial Antar Daerah":
    st.subheader("🗺️ Analisis Spasial: Pengaruh Cuaca Antar Daerah")
    corr = df.groupby("Kabupaten_Kota")[
        ["Produksi_Padi_Ton_clean", "Suhu_Rata_C_clean", "Curah_Hujan_mm_clean", "Kelembapan_Persen_clean"]
    ].mean().corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

# ====== Choropleth Map ======
elif menu == "Choropleth Maps Jawa Timur":
    st.subheader("🗺️ Choropleth Maps Produksi Padi Jawa Timur")
    import geopandas as gpd
    import json

    # Baca GeoJSON
    gdf = gpd.read_file("Main.geojson")

    # Agregasi data produksi per kabupaten/kota
    df_agg = df.groupby("Kabupaten_Kota")["Produksi_Padi_Ton_clean"].sum().reset_index()

    # Samakan format nama (pastikan ada prefix "Kabupaten"/"Kota")
    df_agg["Kabupaten_Kota"] = df_agg["Kabupaten_Kota"].apply(
        lambda x: "Kabupaten " + x if not x.startswith(("Kabupaten", "Kota")) else x
    )

    # Merge GeoDataFrame dengan data CSV
    gdf = gdf.merge(df_agg, left_on="name", right_on="Kabupaten_Kota", how="left")

    # Konversi ke GeoJSON agar bisa dipakai Plotly
    gjson = json.loads(gdf.to_json())

    st.markdown("""
        ℹ️ **Catatan:** Peta ini menampilkan total produksi padi kumulatif
        untuk tahun **2018 - 2024**, dijumlahkan per kabupaten/kota.
        """)

    # Plot Choropleth Map
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gjson,
        locations=gdf.index,
        color="Produksi_Padi_Ton_clean",
        hover_name="name",
        mapbox_style="carto-positron",
        center={"lat": -7.5, "lon": 112},
        zoom=6,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

# ====== Prediksi Produksi ======
elif menu == "Prediksi Produksi Padi":
    st.subheader("🤖 Prediksi Produksi Padi Tahun 2025")

    # Load hasil prediksi
    pred_df = pd.read_csv("1YPrediction.csv", parse_dates=["Tanggal"])
    pred_df = pred_df.sort_values("Tanggal")

    # Metric Cards - Total Produksi per Model
    total_rf = pred_df["RF_Pred"].sum()
    total_lgbm = pred_df["LGBM_Pred"].sum()
    total_blend = pred_df["Blended_Pred"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest", f"{total_rf:,.0f} ton")
    col2.metric("LightGBM", f"{total_lgbm:,.0f} ton")
    col3.metric("Blended", f"{total_blend:,.0f} ton")

    # Line chart per model
    fig = px.line(
        pred_df,
        x="Tanggal",
        y=["RF_Pred", "LGBM_Pred", "Blended_Pred"],
        markers=True,
        color_discrete_map={
            "RF_Pred": "orange",
            "LGBM_Pred": "green",
            "Blended_Pred": "blue"
        },
        labels={"value": "Produksi (Ton)", "variable": "Model"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel hasil prediksi
    st.markdown("### 📅 Detail Prediksi 2025")
    st.dataframe(pred_df, use_container_width=True)

    # Tabel RMSE & SMAPE (sementara hardcoded)
    st.markdown("### 📊 Perbandingan Kinerja Model")
    eval_df = pd.DataFrame({
        "Kelas Model": [
            "Baseline Klasik", "Spasial Klasik", "Spasial + Iklim",
            "Non-Linear", "Non-Linear", "Non-Linear Hibrida"
        ],
        "Model Spesifik": [
            "STARMA", "GSTARIMA (Invers Jarak)", "GSTARIMAX",
            "Random Forest", "LightGBM", "Blended (RF + LGBM)"
        ],
        "RMSE (Ton)": ["~20,800", "~20,278", "~20,200", "~17,000", "~16,400", "~16,389"],
        "SMAPE (%)": ["~75%", "~73%", "~72%", "~66%", "~56%", "~60%"]
    })
    st.table(eval_df)

    ##################################################################################################################################
    ##################################################################################################################################
    import plotly.express as px
    import plotly.graph_objects as go

    # --- Bersihkan angka dari string ---
    eval_clean = eval_df.copy()
    eval_clean["RMSE (Ton)"] = eval_clean["RMSE (Ton)"].str.replace("~", "").str.replace(",", "").astype(float)
    eval_clean["SMAPE (%)"] = eval_clean["SMAPE (%)"].str.replace("~", "").str.replace("%", "").astype(float)

    # Tambahkan kategori untuk warna
    eval_clean["Kategori"] = eval_clean["Kelas Model"].replace({
        "Baseline Klasik": "Klasik",
        "Spasial Klasik": "Klasik",
        "Spasial + Iklim": "Klasik",
        "Non-Linear": "Non-Linear",
        "Non-Linear Hibrida": "Non-Linear"
    })


    #Bar Chart Terpisah - RMSE
    st.markdown("### 📊 Bar Chart - RMSE per Model")
    fig_rmse = px.bar(
        eval_clean,
        x="Model Spesifik",
        y="RMSE (Ton)",
        color="Kategori",
        text="RMSE (Ton)",
        color_discrete_map={"Klasik": "skyblue", "Non-Linear": "orange"}
    )
    fig_rmse.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Bar Chart - SMAPE
    st.markdown("### 📊 Bar Chart - SMAPE per Model")
    fig_smape = px.bar(
        eval_clean,
        x="Model Spesifik",
        y="SMAPE (%)",
        color="Kategori",
        text="SMAPE (%)",
        color_discrete_map={"Klasik": "skyblue", "Non-Linear": "orange"}
    )
    fig_smape.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    st.plotly_chart(fig_smape, use_container_width=True)

elif menu == "Prediksi Per Kabupaten":
    st.subheader("🤖 Prediksi Produksi Padi per Kabupaten (2025)")

    # Load prediksi per kabupaten
    pred_kab = pd.read_csv("1YPrediction_K.csv", parse_dates=["Tanggal"])

    # Pilih kabupaten
    kabupaten_list = pred_kab["Kabupaten_Kota"].unique()
    kabupaten = st.selectbox("Pilih Kabupaten/Kota:", kabupaten_list)

    df_kab = pred_kab[pred_kab["Kabupaten_Kota"] == kabupaten]

    # Tampilkan metrik total
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest", f"{df_kab['RF_Pred'].sum():,.0f} ton")
    col2.metric("LightGBM", f"{df_kab['LGBM_Pred'].sum():,.0f} ton")
    col3.metric("Blended", f"{df_kab['Blended_Pred'].sum():,.0f} ton")

    # Line chart prediksi per bulan
    fig = px.line(
        df_kab,
        x="Tanggal",
        y=["RF_Pred", "LGBM_Pred", "Blended_Pred"],
        markers=True,
        labels={"value": "Produksi (Ton)", "variable": "Model"},
        title=f"Prediksi Produksi Padi Kabupaten {kabupaten} Tahun 2025"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabel detail
    st.markdown("### 📅 Detail Prediksi")
    st.dataframe(df_kab, use_container_width=True)
