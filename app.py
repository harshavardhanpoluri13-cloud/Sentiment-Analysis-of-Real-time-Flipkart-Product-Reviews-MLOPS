
"""
Streamlit App — Flipkart Sentiment Analysis
Deploy on AWS EC2 | YONEX MAVIS 350 Reviews
"""

import streamlit as st
import joblib
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── NLTK (lazy download on first launch) ─────────────────────────────────────
import nltk
import os

NLTK_READY = False
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    stopwords.words('english')  # trigger exception if not downloaded
    NLTK_READY = True
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_READY = True

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# ── Text Cleaning (must match notebook preprocessing) ────────────────────────
def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'read more', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('sentiment_model.pkl')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Review Sentiment Analyser",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #2874f0, #fb641b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .subtitle { text-align: center; color: #666; font-size: 1rem; margin-bottom: 20px; }
    .result-positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745; padding: 20px; border-radius: 10px;
        font-size: 1.4rem; font-weight: 700; color: #155724; text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545; padding: 20px; border-radius: 10px;
        font-size: 1.4rem; font-weight: 700; color: #721c24; text-align: center;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 15px; text-align: center; border: 1px solid #e0e0e0;
    }
    .stTextArea textarea { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5e/Yonex_logo.svg/320px-Yonex_logo.svg.png",
             width=150)
    st.markdown("### 🏸 About This App")
    st.info(
        "This app classifies customer reviews for the "
        "**YONEX MAVIS 350 Nylon Shuttle** from Flipkart as "
        "**Positive** or **Negative** using a machine learning model."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    try:
        model = load_model()
        vec_name  = type(model.named_steps['vectorizer']).__name__
        mod_name  = type(model.named_steps['model']).__name__
        st.success(f"✅ Model loaded")
        st.markdown(f"- **Vectorizer:** `{vec_name}`")
        st.markdown(f"- **Classifier:** `{mod_name}`")
    except Exception as e:
        st.error(f"⚠️ Model not found: {e}")
        st.warning("Run the notebook first to generate `sentiment_model.pkl`")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + scikit-learn")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏸 Flipkart Review Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YONEX MAVIS 350 Nylon Shuttle · Real-time Sentiment Classification</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Review", "📋 Batch Analysis", "📊 Dataset Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Enter a Customer Review")
        user_review = st.text_area(
            "Paste or type a review below:",
            height=180,
            placeholder="e.g. Great shuttlecock! Perfect for indoor badminton. Very durable and good flight.",
            key="single_review"
        )

        example_reviews = {
            "Select an example...": "",
            "✅ Positive: Great product":
                "Amazing shuttle! Very durable and the flight is perfect. Great value for money.",
            "❌ Negative: Fake product":
                "Complete waste of money. Not original Yonex. Feathers broke after one game.",
            "✅ Positive: Recommended":
                "Good quality shuttlecock. Plays like feather shuttle at much lower price.",
            "❌ Negative: Poor quality":
                "Worst product. Damaged corks inside box. Flipkart delivered fake item.",
        }
        selected = st.selectbox("Or try an example:", list(example_reviews.keys()))
        if selected != "Select an example...":
            user_review = example_reviews[selected]
            st.text_area("Selected review:", value=user_review, height=100, disabled=True, key="ex")

        predict_btn = st.button("🔮 Analyse Sentiment", type="primary", use_container_width=True)

    with col2:
        st.subheader("Result")
        if predict_btn and user_review.strip():
            try:
                model = load_model()
                cleaned = clean_text(user_review)
                prediction = model.predict([cleaned])[0]
                try:
                    proba = model.predict_proba([cleaned])[0]
                    confidence = max(proba) * 100
                    neg_prob = proba[0] * 100
                    pos_prob = proba[1] * 100
                    has_proba = True
                except AttributeError:
                    has_proba = False

                if prediction == 1:
                    st.markdown('<div class="result-positive">😊 POSITIVE<br><small>Customer is satisfied</small></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-negative">😞 NEGATIVE<br><small>Customer is dissatisfied</small></div>',
                                unsafe_allow_html=True)

                st.markdown("")

                if has_proba:
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown("**Probability Breakdown**")
                    prob_df = pd.DataFrame({
                        'Sentiment': ['😞 Negative', '😊 Positive'],
                        'Probability': [neg_prob/100, pos_prob/100]
                    })
                    st.bar_chart(prob_df.set_index('Sentiment'))

                with st.expander("🔎 Cleaned Text"):
                    st.code(cleaned if cleaned else "(empty after cleaning)")

            except FileNotFoundError:
                st.error("❌ `sentiment_model.pkl` not found. Please run the notebook first!")

        elif predict_btn:
            st.warning("⚠️ Please enter a review text.")
        else:
            st.markdown("""
            <div style='text-align:center; color:#999; padding: 40px 0;'>
                <div style='font-size:3rem'>🎯</div>
                <div>Your result will appear here</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Batch Review Analysis")
    st.info("Upload a CSV file with a column named **`Review text`** to classify multiple reviews at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'Review text' not in df_upload.columns:
                st.error("❌ CSV must contain a column named `Review text`")
            else:
                model = load_model()
                with st.spinner("🔄 Analysing reviews..."):
                    df_upload['cleaned'] = df_upload['Review text'].apply(clean_text)
                    df_upload['Predicted Sentiment'] = model.predict(df_upload['cleaned'])
                    df_upload['Sentiment Label'] = df_upload['Predicted Sentiment'].map(
                        {1: '😊 Positive', 0: '😞 Negative'}
                    )

                pos_count = (df_upload['Predicted Sentiment'] == 1).sum()
                neg_count = (df_upload['Predicted Sentiment'] == 0).sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", len(df_upload))
                col2.metric("😊 Positive", pos_count, f"{pos_count/len(df_upload)*100:.1f}%")
                col3.metric("😞 Negative", neg_count, f"-{neg_count/len(df_upload)*100:.1f}%")

                st.dataframe(
                    df_upload[['Review text', 'Sentiment Label']].head(50),
                    use_container_width=True
                )

                csv_out = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv_out,
                                   "sentiment_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Dataset Insights")

    try:
        df_raw = pd.read_csv('data.csv')
        df_raw['sentiment'] = df_raw['Ratings'].apply(
            lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{len(df_raw):,}")
        col2.metric("Avg Rating", f"{df_raw['Ratings'].mean():.2f} ⭐")
        col3.metric("Positive Reviews", f"{(df_raw['Ratings']>=4).sum():,}")
        col4.metric("Negative Reviews", f"{(df_raw['Ratings']<=2).sum():,}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Rating distribution
        rc = df_raw['Ratings'].value_counts().sort_index()
        axes[0].bar(rc.index, rc.values,
                    color=['#e74c3c','#e67e22','#f39c12','#2ecc71','#27ae60'],
                    edgecolor='white')
        axes[0].set_title('Rating Distribution', fontweight='bold')
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Count')

        # Sentiment pie
        sc = df_raw['sentiment'].value_counts()
        colors_pie = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
        axes[1].pie(sc.values,
                    labels=sc.index,
                    colors=[colors_pie.get(s, 'gray') for s in sc.index],
                    autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Sentiment Distribution', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Sample Reviews by Rating")
        rating_filter = st.slider("Filter by Rating", 1, 5, (1, 5))
        filtered = df_raw[
            (df_raw['Ratings'] >= rating_filter[0]) &
            (df_raw['Ratings'] <= rating_filter[1])
        ][['Reviewer Name', 'Ratings', 'Review Title', 'Review text']].head(10)
        st.dataframe(filtered, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ `data.csv` not found in the app directory. Place it alongside `app.py`.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏸 Flipkart Sentiment Analyser · Powered by scikit-learn · Deployed on AWS EC2")
