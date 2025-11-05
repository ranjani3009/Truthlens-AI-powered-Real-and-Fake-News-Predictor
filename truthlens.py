import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# GNews API Key
GNEWS_KEY = "a6739ad5222da7ff2d10d05310242467"
# Train Model
def train_model():
    df = pd.read_csv("C:\\Users\\ranja\\OneDrive\\Desktop\\mini project\\Ranjani\\df.csv", low_memory=False)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    X, y = df["text"].astype(str), df["label"].astype(int)

    tfidf = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = MultinomialNB(alpha=0.1)
    model.fit(tfidf.fit_transform(X_train), y_train)

    acc = accuracy_score(y_test, model.predict(tfidf.transform(X_test)))
    return model, tfidf, acc

model, tfidf, accuracy = train_model()
# Helper Functions
def get_keywords(text, top_n=5):
    words = [w for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(words[:top_n]) or text

def fetch_related_articles(query, max_results=3):
    try:
        url = f"https://news.google.com/rss/search?q={query}+site:bbc.com+OR+site:reuters.com+OR+site:thehindu.com&hl=en-IN&gl=IN&ceid=IN:en"
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")
        articles = []
        for it in items[:max_results]:
            title = it.title.text
            link = it.link.text
            source = it.source.text if it.source else "Trusted Source"
            articles.append(f"🔗 <a href='{link}' target='_blank' style='color:#0047b3;text-decoration:none;'>{title}</a> — <i>{source}</i>")
        return articles
    except Exception as e:
        print("⚠️ Error fetching articles:", e)
        return []
def predict_news(text):
    if not text.strip():
        return "<b>⚠️ Please enter some news text.</b>", ""

    X_input = tfidf.transform([text])
    prob = model.predict_proba(X_input)[0][1]
    label = "REAL" if prob >= 0.65 else "FAKE"

    color = "#00cc66" if label == "REAL" else "#cc0000"
    confidence = f"{prob*100:.2f}%" if label == "REAL" else f"{(1-prob)*100:.2f}%"
    keywords = get_keywords(text)

    related = fetch_related_articles(keywords)
    if not related:
        related = [
            "🔗 <a href='https://www.bbc.com/news' target='_blank'>BBC News</a> — BBC",
            "🔗 <a href='https://www.reuters.com' target='_blank'>Reuters</a> — Reuters"
        ]

    if label == "REAL":
        result_box = f"""
        <div style='background:{color};padding:15px;border-radius:12px;color:white;font-weight:bold;
        text-align:center;font-size:20px;'>✅ This News is REAL ({confidence} confidence)</div>"""
        header = "<h4 style='color:#003366;'>📰 Verified Real News Articles:</h4>"
    else:
        result_box = f"""
        <div style='background:{color};padding:15px;border-radius:12px;color:white;font-weight:bold;
        text-align:center;font-size:20px;'>🚫 This News is FAKE ({confidence} confidence)</div>"""
        header = "<h4 style='color:#003366;'>✅ Related True Verified News Articles:</h4>"

    articles_html = "<br>".join(related)
    return result_box, f"{header}{articles_html}"
# Gradio Interface Design
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:linear-gradient(to right,#a1c4fd,#c2e9fb);
        border-radius:10px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4712/4712108.png' width='90'>
        <h1 style='color:#002b5b; font-weight:900; margin-top:10px;'>TruthLens – AI-Powered Fake or Real News Predictor</h1>
        <p style='color:#333; font-size:15px;'>Detect fake news instantly and explore real, verified recovery articles.</p>
    </div>
    """)

    news_text = gr.Textbox(
        label="📝 Enter News Content:",
        placeholder="Paste or type the news headline or paragraph here...",
        lines=7
    )

    check_button = gr.Button("🔍 Check News", variant="primary")

    result_output = gr.HTML(label="Prediction Result")
    related_output = gr.HTML(label="Verified or Recovery Articles")

    check_button.click(fn=predict_news, inputs=news_text, outputs=[result_output, related_output])
# Automatically open browser & remove extra UI
app.launch(show_api=False, inbrowser=True, favicon_path="https://cdn-icons-png.flaticon.com/512/4712/4712108.png")
