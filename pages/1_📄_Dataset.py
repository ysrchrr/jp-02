import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Dataset",
    page_icon="📄",
)

st.title('📄Dataset Komentar Aplikasi Tokopedia')
st.write('Data Komentar Tokopedia ini diambil dari Google Play Store')
DATA_URL = ('https://muyacho.com/documents/dataset_postprocessing.csv')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Sedang memuat data')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Berhasil mendapatkan data")

# Set a custom index to start from 1
data.index = np.arange(1, len(data) + 1)

st.subheader('Data Komentar')
st.write(data)
data_df = pd.read_csv(DATA_URL)

# Pisahkan data menjadi data ulasan dan label
labels = data_df['label'].tolist()

# Hitung jumlah keseluruhan data dan jumlah label positif, netral, dan negatif
total_data = len(data_df)
positive_count = labels.count('Positif')
neutral_count = labels.count('Netral')
negative_count = labels.count('Negatif')

# Tampilkan jumlah keseluruhan data dan jumlah label positif, netral, dan negatif
st.write('Jumlah Keseluruhan Data:', total_data)
st.write('Positif:', positive_count)
st.write('Netral:', neutral_count)
st.write('Negatif:', negative_count)
