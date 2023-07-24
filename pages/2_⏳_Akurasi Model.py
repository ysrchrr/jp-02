# file_app.py
import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Fungsi untuk melatih model Naive Bayes
def train_naive_bayes(data_df, training_percentage):
    # Pisahkan data menjadi data ulasan dan label
    responses = data_df['response'].tolist()
    labels = data_df['label'].tolist()

    # Hitung jumlah data yang akan digunakan untuk pelatihan
    total_data = len(data_df)
    training_data_count = int(total_data * training_percentage / 100)

    # Ambil data untuk pelatihan sesuai persentase yang ditentukan
    training_responses = responses[:training_data_count]
    training_labels = labels[:training_data_count]

    # Proses data teks menjadi vektor fitur menggunakan Count Vectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(training_responses)

    # Latih model Naive Bayes
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_vectorized, training_labels)

    return naive_bayes, vectorizer, training_data_count

# Fungsi untuk melakukan prediksi dan menghitung akurasi model
def predict_and_evaluate(model, vectorizer, data_df, training_data_count):
    # Ambil data untuk pengujian (data yang tidak digunakan dalam pelatihan)
    testing_responses = data_df['response'][training_data_count:]
    testing_labels = data_df['label'][training_data_count:]

    # Proses data teks uji menjadi vektor fitur menggunakan Count Vectorizer yang sama
    X_test_vectorized = vectorizer.transform(testing_responses)

    # Lakukan prediksi menggunakan model yang telah dilatih
    predictions = model.predict(X_test_vectorized)

    # Hitung akurasi model
    accuracy = accuracy_score(testing_labels, predictions)

    return accuracy

# Mulai aplikasi Streamlit
def main():
    st.title('Aplikasi Analisis Sentimen dengan Naive Bayes')

    # Upload file CSV
    st.write('Unggah file CSV dengan kolom mengandung "response" dan "label":')
    uploaded_file = st.file_uploader('Pilih file CSV', type=['csv'])

    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file)

        # Pilih persentase data yang digunakan untuk pelatihan
        training_percentage = st.selectbox('Pilih Persentase Data untuk Pelatihan:', [90])

        # Latih model Naive Bayes
        model, vectorizer, training_data_count = train_naive_bayes(data_df, training_percentage)

        # Hitung dan tampilkan akurasi model
        accuracy = predict_and_evaluate(model, vectorizer, data_df, training_data_count)
        accuracy_percentage = accuracy * 100
        st.write(f'Akurasi Model dengan Persentase {training_percentage}% Data Pelatihan: {accuracy_percentage:.2f}%')

if __name__ == '__main__':
    main()
