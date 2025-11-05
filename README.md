# 📰 TruthLens – AI-Powered Real & Fake News Predictor

**TruthLens** is an **AI-powered web application** that detects whether a news article is **real or fake** using **Natural Language Processing (NLP)** and **Machine Learning**.  
Beyond detection, it also provides a **“Real News Recovery”** feature that fetches **authentic and verified articles** related to the input — helping users uncover the truth.

---

## 🚀 Features

- 🔍 **Fake News Detection:** Classifies news as *real* or *fake* using TF-IDF and Multinomial Naive Bayes.  
- 🧠 **Real News Recovery:** Fetches verified and related articles from trusted sources like **BBC**, **Reuters**, and **The Hindu**.  
- 💬 **Interactive Web Interface:** Built using **Gradio** for a simple and clean user experience.  
- ⚡ **Auto Browser Launch:** Automatically opens in your default browser when executed.  
- 📊 **Performance Display:** Shows model accuracy and prediction confidence.  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python |
| **Libraries** | Gradio, Scikit-learn, Pandas, BeautifulSoup4, Requests |
| **Model** | Multinomial Naive Bayes |
| **Vectorizer** | TF-IDF (n-gram range 1–2) |
| **Dataset** | Real and Fake News Dataset *(not uploaded due to size)* |

---

## 🧠 How It Works

1. The model is trained on a dataset containing labeled **real** and **fake** news articles.  
2. The text input is converted into numeric features using **TF-IDF Vectorization**.  
3. The trained **Naive Bayes** model predicts whether the news is *Real* or *Fake*.  
4. If fake, TruthLens searches for **verified and related articles** from credible news outlets.  
5. The result and related articles are displayed in an elegant **Gradio web interface**.

---

## 🖥️ Running the Project

### 🔧 Requirements
- Python 3.8 or above
- Install dependencies:
  ```bash
  pip install gradio pandas scikit-learn beautifulsoup4 requests

