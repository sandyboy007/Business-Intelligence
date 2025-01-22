import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot

# Set page layout
st.set_page_config(
    page_title="Meteorological & Insect Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
file_meteo_csv = r"C:\Users\USER\OneDrive\Desktop\business intiligence\data sets\dati_meteo_storici_(Cicalino1).csv"
file_catture_csv = r"C:\Users\USER\OneDrive\Desktop\business intiligence\data sets\grafico_delle_catture_(Cicalino 1).csv"

meteo_df = pd.read_csv(file_meteo_csv, sep=',', skiprows=2)
catture_df = pd.read_csv(file_catture_csv, sep=',', skiprows=2)

# Rename columns
meteo_df.columns = ['DateTime', 'Media_Temperatura', 'Low_Temp', 'High_Temp', 'Media_Umidita']
catture_df.columns = ['DateTime', 'Numero_Insetti', 'Nuove_Catture', 'Recensito', 'Evento']

# Data cleaning and conversion
meteo_df['Media_Temperatura'] = meteo_df['Media_Temperatura'].astype(str).str.replace(',', '.', regex=False).astype(float)
catture_df['Numero_Insetti'] = catture_df['Numero_Insetti'].fillna(0).astype(int)
catture_df['Nuove_Catture'] = catture_df['Nuove_Catture'].fillna(0).astype(int)

# Convert datetime columns
meteo_df['DateTime'] = pd.to_datetime(meteo_df['DateTime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
catture_df['DateTime'] = pd.to_datetime(catture_df['DateTime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

meteo_df['Date'] = meteo_df['DateTime'].dt.date
catture_df['Date'] = catture_df['DateTime'].dt.date

# Filter captures for 6:00 AM
catture_filtered = catture_df[catture_df['DateTime'].dt.hour == 6][['Date', 'Numero_Insetti', 'Nuove_Catture']]

# Merge datasets
meteo_with_catture = pd.merge(meteo_df, catture_filtered, on='Date', how='left')
meteo_with_catture.fillna(0, inplace=True)

# Convert 'Date' to datetime and set as index for resampling
meteo_with_catture['Date'] = pd.to_datetime(meteo_with_catture['Date'])
meteo_with_catture.set_index('Date', inplace=True)

# Title
st.title("Meteorological and Insect Data Analysis")

# Buttons for interaction
if st.button("Show Dataset Overview"):
    st.header("Dataset Overview")
    st.subheader("Meteorological Data with Insect Captures")
    st.dataframe(meteo_with_catture.head())

if st.button("Show Temperature & Humidity Distribution"):
    st.header("Weather Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature Histogram")
        fig1, ax1 = plt.subplots()
        ax1.hist(meteo_with_catture['Media_Temperatura'], bins=20, edgecolor="black")
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    with col2:
        st.subheader("Humidity Histogram")
        fig2, ax2 = plt.subplots()
        ax2.hist(meteo_with_catture['Media_Umidita'], bins=20, edgecolor="black")
        ax2.set_xlabel("Humidity (%)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

if st.button("Show Autocorrelation of Insect Captures"):
    st.header("Insect Captures Analysis")
    daily_insects = meteo_with_catture['Numero_Insetti'].resample('D').sum()

    st.subheader("Autocorrelation of Insect Captures")
    fig3, ax3 = plt.subplots()
    plot_acf(daily_insects.dropna(), ax=ax3, lags=30)
    st.pyplot(fig3)

if st.button("Show Seasonal Decomposition of Insect Captures"):
    st.header("Seasonal Decomposition")
    daily_insects = meteo_with_catture['Numero_Insetti'].resample('D').sum()

    if len(daily_insects.dropna()) >= 7:
        decomposition = seasonal_decompose(daily_insects, model='additive', period=7)
        fig4 = decomposition.plot()
        st.pyplot(fig4)
    else:
        st.warning("Not enough data points for seasonal decomposition. At least 7 data points are required.")

if st.button("Show Correlation Analysis"):
    st.header("Correlation Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Temperature vs Insect Captures")
        fig5, ax5 = plt.subplots()
        sns.kdeplot(
            x=meteo_with_catture['Media_Temperatura'],
            y=meteo_with_catture['Numero_Insetti'],
            cmap="Blues",
            fill=True,
            ax=ax5
        )
        ax5.set_xlabel("Temperature (°C)")
        ax5.set_ylabel("Insect Captures")
        st.pyplot(fig5)

    with col4:
        st.subheader("Correlation Matrix")
        correlation_matrix = meteo_with_catture[['Media_Temperatura', 'Numero_Insetti']].corr()
        st.dataframe(correlation_matrix)

if st.button("Show Polynomial Regression"):
    st.header("Polynomial Regression")
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(meteo_with_catture[['Media_Temperatura']])
    model = LinearRegression()
    model.fit(X_poly, meteo_with_catture['Numero_Insetti'])
    st.write("Model R² Score:", model.score(X_poly, meteo_with_catture['Numero_Insetti']))
    