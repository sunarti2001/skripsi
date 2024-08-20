import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Membaca dataset yang telah dibuat
df = pd.read_csv('komplen.csv')

# Menampilkan 10 contoh data mentah
print("10 Contoh Data Mentah:")
print(df.head(10))

# Pre-processing function
def preprocess_text(text):
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = word_tokenize(text)
    # Menghapus stop words
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Stemming dan Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    return ' '.join(tokens)

# Menerapkan pre-processing pada data
df['Processed_Complaint'] = df['Complaint'].apply(preprocess_text)

# Menampilkan 10 contoh data setelah pra-pemrosesan
print("\n10 Contoh Data Setelah Pra-pemrosesan:")
print(df[['Complaint', 'Processed_Complaint']].head(10))

# Menggunakan TF-IDF untuk ekstraksi fitur
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Complaint'])
y = df['Category']

# Menampilkan 10 contoh hasil TF-IDF
print("\n10 Contoh Hasil TF-IDF:")
print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).head(10))

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan jumlah data latih dan data uji
print(f"\nJumlah Data Latih: {X_train.shape[0]}")
print(f"Jumlah Data Uji: {X_test.shape[0]}")

# Membuat model Naïve Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Laporan Klasifikasi
class_report = classification_report(y_test, y_pred)

# Menampilkan hasil evaluasi
print(f'\nAkurasi: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)