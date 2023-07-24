import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Data Sheet",
    page_icon="📄",
)

st.title('📄Data Sheet Pages')

DATA_URL = ('https://muyacho.com/documents/data.csv')

@st.cache_data
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

st.subheader('Data Ulasan')
st.write(data)
