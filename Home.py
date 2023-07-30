import streamlit as st

st.set_page_config(
    page_title="Dashboard",
    page_icon="🏠",
)

st.title('Selamat datang👋')
st.subheader('Website Analis Sentimen Komentar Tokopedia dengan Metode Naive Bayes 📊')

st.markdown(
    """
    ### Tokopedia
    PT Tokopedia merupakan perusahaan perdagangan elektronik atau sering disebut toko daring. Sejak didirikan pada tahun 2009, Tokopedia telah bertransformasi menjadi sebuah unicorn yang berpengaruh tidak hanya di Indonesia tetapi juga di Asia Tenggara.
    [Wikipedia](https://id.wikipedia.org/wiki/Tokopedia)
    ### Klasifikasi Naive Bayes
    Naive Bayes classifier merupakan salah satu metoda pemelajaran mesin yang memanfaatkan perhitungan probabilitas dan statistik yang dikemukakan oleh ilmuwan Inggris Thomas Bayes, yaitu memprediksi probabilitas pada masa depan berdasarkan pengalaman pada masa sebelumnya.
    [Wikipedia](https://id.wikipedia.org/wiki/Naive_Bayes_classifier)
"""
)