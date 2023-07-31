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
# file_path = 'E:\PJ\skripsi-bagas\coba.csv'  # Ganti dengan path ke file CSV Anda
data_df = pd.read_csv(file_path)

# Pisahkan data menjadi data ulasan dan label
responses = data_df['response'].tolist()
labels = data_df['label'].tolist()



# Proses data teks menjadi vektor fitur menggunakan Count Vectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(responses)

# Latih model Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_vectorized, labels)

# Masukkan ulasan produk
st.write('Komentar Aplikasi Tokopedia : ')
user_input = st.text_area('Komentar')

if st.button('Prediksi'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f'Hasil Prediksi: {prediction}')
    else:
        st.write('Masukkan komentar terlebih dahulu')

