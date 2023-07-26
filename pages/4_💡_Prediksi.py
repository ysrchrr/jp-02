import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Buat aplikasi Streamlit
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = naive_bayes.predict(text_vectorized)
    return prediction[0]

st.title('Prediksi')
# Baca file CSV dari direktori lokal
file_path = 'https://muyacho.com/documents/dataset_postprocessing.csv'  # Ganti dengan path ke file CSV Anda
data_df = pd.read_csv(file_path)

# Pisahkan data menjadi data ulasan dan label
responses = data_df['response'].tolist()
labels = data_df['label'].tolist()

# Hitung jumlah keseluruhan data dan jumlah label positif, netral, dan negatif
total_data = len(data_df)
positive_count = labels.count('Positif')
neutral_count = labels.count('Netral')
negative_count = labels.count('Negatif')

# Proses data teks menjadi vektor fitur menggunakan Count Vectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(responses)

# Latih model Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_vectorized, labels)

# Masukkan ulasan produk
st.write('Komentar Aplikasi Tokopedia : ')
user_input = st.text_area('Ulasan produk')

if st.button('Prediksi'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f'Hasil Prediksi: {prediction}')
    else:
        st.write('Masukkan ulasan produk terlebih dahulu')

# Tampilkan jumlah keseluruhan data dan jumlah label positif, netral, dan negatif
st.write('Jumlah Keseluruhan Data:', total_data)
st.write('Positif:', positive_count)
st.write('Netral:', neutral_count)
st.write('Negatif:', negative_count)
