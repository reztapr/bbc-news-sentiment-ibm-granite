# Sentiment Analysis on BBC News using IBM Granite 3.3 8B Instruct

## Deskripsi Proyek
Proyek ini melakukan analisis sentimen pada teks berita dari **BBC News** menggunakan model **IBM Granite 3.3 8B Instruct** yang diakses melalui **LangChain** dan **Replicate API**.  
Tujuan proyek ini adalah untuk mengklasifikasikan deskripsi berita menjadi dua kategori sentimen utama:
- Positive
- Negative

Dataset yang digunakan berasal dari [BBC News Dataset di Kaggle](https://www.kaggle.com/datasets/pariza/bbc-news-summary), yang berisi kumpulan berita dan deskripsinya dari berbagai kategori seperti *politics, tech, sport, business,* dan *entertainment*.

---

## Teknologi dan Library yang Digunakan
- **LangChain Community** – framework untuk integrasi model LLM.  
- **Replicate API** – menjalankan model IBM Granite 3.3 8B Instruct.  
- **Pandas** – membaca dan mengolah data CSV.  
- **tqdm** – menampilkan progress bar selama proses klasifikasi.  
- **Google Colab** (opsional) – lingkungan eksekusi notebook interaktif.

---

## Instalasi

1. Clone repositori ini:
   git clone https://github.com/username/bbc-news-sentiment-ibm-granite.git
   cd bbc-news-sentiment-ibm-granite

2. Buat dan aktifkan virtual environment:
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows

3. Install semua dependensi:
   pip install -r requirements.txt

4. Siapkan API key dari Replicate dan simpan di environment variable:
   export REPLICATE_API_TOKEN=your_api_key_here

---

## Cara Menjalankan Proyek
Jalankan file Python:
python sentiment_analysis.py
Atau buka bbc_sentiment.ipynb di Google Colab dan jalankan setiap sel secara berurutan.

---

## Output
Proyek ini akan menghasilkan:
Kolom baru sentiment pada dataset dengan label Positive atau Negative.
File hasil klasifikasi dalam format CSV (bbc_sentiment_result.csv).
Ringkasan jumlah berita berdasarkan kategori sentimen.

---

## Struktur Folder
bbc-news-sentiment-ibm-granite/
│
├── data/
│   └── bbc_news.csv
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── bbc_sentiment.ipynb

---

## Author: @reztapr
