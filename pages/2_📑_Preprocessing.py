import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Preprocessing",
    page_icon="📑",
)

st.title('📑Preprocessing')

DATA_URL = ('https://muyacho.com/documents/dataset_preprocessing.csv')


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

st.subheader('Hasil Preprocessing dari data sebelumnya')
st.write(data)