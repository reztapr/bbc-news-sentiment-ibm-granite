"""
sentiment_analysis.py
---------------------
Analisis sentimen pada berita BBC menggunakan model IBM Granite 3.3 8B Instruct
melalui LangChain dan Replicate API.
"""

# === 1. Import Library ===
import os
import time
import pandas as pd
from tqdm import tqdm
from langchain_community.llms import Replicate


# === 2. Konfigurasi Token API ===
# Ganti dengan token Replicate milik kamu sendiri
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_token_here"

# Inisialisasi model IBM Granite 3.3 8B Instruct
model = "ibm-granite/granite-3.3-8b-instruct"
llm = Replicate(model=model, replicate_api_token=os.environ["REPLICATE_API_TOKEN"])
print("Model IBM Granite siap digunakan.")


# === 3. Membaca Dataset ===
def load_dataset(file_path: str):
    """Memuat dataset CSV berisi berita BBC."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat. Jumlah data: {len(df)}")
        return df
    except FileNotFoundError:
        print("File tidak ditemukan. Pastikan file bbc_news.csv ada di folder data/.")
        exit()


# === 4. Fungsi Klasifikasi Sentimen ===
def classify_sentiment(text: str) -> str:
    """
    Mengklasifikasikan teks berita menjadi Positive atau Negative.
    Menggunakan model IBM Granite melalui Replicate API.
    """
    prompt = f"""
    Determine if the following news text expresses a positive or negative sentiment.
    Respond only with one word: Positive or Negative.

    Text:
    {text}
    """

    try:
        result = llm.invoke(prompt).strip()
        # Jika model mengembalikan hasil lain seperti "Neutral", ubah ke kategori terdekat
        if "neutral" in result.lower():
            result = "Neutral (detected)"
        elif "positive" not in result.lower() and "negative" not in result.lower():
            result = f"Unclear ({result})"
        return result
    except Exception as e:
        return f"Error: {e}"


# === 5. Proses Klasifikasi Sentimen ===
def analyze_sentiment(df: pd.DataFrame, sample_size: int = 10):
    """
    Melakukan analisis sentimen pada sejumlah data (default: 10 baris pertama).
    """
    results = []
    for text in tqdm(df["description"].head(sample_size), desc="Classifying"):
        sentiment = classify_sentiment(text)
        results.append(sentiment)
        time.sleep(10)  # jeda untuk menghindari batasan rate API

    df_results = df.head(sample_size).copy()
    df_results["predicted_sentiment"] = results
    return df_results


# === 6. Menyimpan Hasil ===
def save_results(df_results: pd.DataFrame, output_path: str = "bbc_sentiment_result.csv"):
    """Menyimpan hasil analisis ke file CSV."""
    df_results.to_csv(output_path, index=False)
    print(f"Hasil analisis disimpan ke {output_path}")


# === 7. Main Program ===
if __name__ == "__main__":
    print("=== Analisis Sentimen BBC News ===")
    data_path = "data/bbc_news.csv"
    df = load_dataset(data_path)

    df_results = analyze_sentiment(df, sample_size=10)
    print("\nContoh hasil klasifikasi:")
    print(df_results[["description", "predicted_sentiment"]].head())

    save_results(df_results)
