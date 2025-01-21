import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
 
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
 
# Streamlit layout
st.title('Meteorological Data and Insect Captures Analysis')
 
# Show data
st.subheader('Weather and Insect Data')
st.write(meteo_with_catture.head())
 
# Histogram of Temperature and Humidity
st.subheader('Temperature and Humidity Distribution')
col1, col2 = st.columns(2)
 
with col1:
    st.write('Temperature Histogram')
    fig1, ax1 = plt.subplots()
    ax1.hist(meteo_with_catture['Media_Temperatura'], bins=20, edgecolor="black")
    st.pyplot(fig1)
 
with col2:
    st.write('Humidity Histogram')
    fig2, ax2 = plt.subplots()
    ax2.hist(meteo_with_catture['Media_Umidita'], bins=20, edgecolor="black")
    st.pyplot(fig2)
 
# Autocorrelation plot of insect captures
st.subheader('Autocorrelation of Insect Captures')
daily_insects = meteo_with_catture['Numero_Insetti'].resample('D').sum()
 
fig3, ax3 = plt.subplots()
autocorrelation_plot(daily_insects, ax=ax3)
plt.title('Autocorrelation of the Number of Insects')
st.pyplot(fig3)
 
# Seasonal decomposition of insect captures
st.subheader('Seasonal Decomposition of Insect Captures')
if len(daily_insects.dropna()) >= 7:
    decomposition = seasonal_decompose(daily_insects, model='additive', period=7)
    fig4 = decomposition.plot()
    st.pyplot(fig4)
 
# Residual analysis
st.subheader('Residual Analysis')
if len(daily_insects.dropna()) >= 7:
    residui = decomposition.resid
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(residui, marker='o', linestyle='-', color='b')
    ax5.axhline(0, color='red', linestyle='--')
    plt.title("Residuals of the Model")
    st.pyplot(fig5)
 
# Q-Q Plot
st.subheader("Q-Q Plot of Residuals")
fig6, ax6 = plt.subplots()
probplot(residui.dropna(), dist="norm", plot=ax6)
st.pyplot(fig6)
 
# Autocorrelation Plot of Residuals
st.subheader("Autocorrelation of Residuals")
fig7, ax7 = plt.subplots()
plot_acf(residui.dropna(), lags=30, ax=ax7)
st.pyplot(fig7)
 
# Partial Autocorrelation Plot
st.subheader("Partial Autocorrelation of Insect Captures")
fig8, ax8 = plt.subplots()
plot_pacf(meteo_with_catture['Numero_Insetti'].dropna(), lags=25, ax=ax8)
st.pyplot(fig8)
 
# Ljung-Box Test for Weekly Seasonality
st.subheader("Ljung-Box Test for Weekly Seasonality")
result = acorr_ljungbox(meteo_with_catture['Numero_Insetti'].dropna(), lags=[7], return_df=True)
st.write(result)
 
# Density Plot: Temperature vs Number of Insects
st.subheader("Density: Temperature vs Number of Insects")
fig9, ax9 = plt.subplots()
sns.kdeplot(x=meteo_with_catture['Media_Temperatura'], y=meteo_with_catture['Numero_Insetti'], cmap="Blues", fill=True, ax=ax9)
st.pyplot(fig9)
 
# Correlation: Temperature vs Number of Insects
st.subheader("Correlation: Temperature vs Number of Insects")
st.write(meteo_with_catture[['Media_Temperatura', 'Numero_Insetti']].corr())
 
# Polynomial Regression: Temperature vs Number of Insects
st.subheader("Polynomial Regression: Temperature vs Number of Insects")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(meteo_with_catture[['Media_Temperatura']])
model = LinearRegression()
model.fit(X_poly, meteo_with_catture['Numero_Insetti'])
st.write("Model Score:", model.score(X_poly, meteo_with_catture['Numero_Insetti']))


