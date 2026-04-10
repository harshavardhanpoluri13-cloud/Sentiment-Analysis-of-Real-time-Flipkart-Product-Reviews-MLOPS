"""
Streamlit App — Flipkart Sentiment Analysis
Supports: joblib model OR MLflow registered model
Deploy on AWS EC2  |  YONEX MAVIS 350 Reviews
"""

import streamlit as st
import joblib
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    stopwords.words('english')
except LookupError:
    for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

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

@st.cache_resource
def load_model():
    if os.path.exists('sentiment_model.pkl'):
        model = joblib.load('sentiment_model.pkl')
        return model, 'joblib (local)'
    try:
        import mlflow.sklearn
        mlflow_uri  = os.getenv('MLFLOW_TRACKING_URI', 'mlruns')
        model_name  = os.getenv('MLFLOW_MODEL_NAME', 'flipkart_sentiment_classifier')
        model_stage = os.getenv('MLFLOW_MODEL_STAGE', 'Production')
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        model = mlflow.sklearn.load_model(f'models:/{model_name}/{model_stage}')
        return model, f'MLflow ({model_name}/{model_stage})'
    except Exception as e:
        raise FileNotFoundError(f'No model found. Error: {e}')

st.set_page_config(
    page_title='Flipkart Sentiment Analyser',
    page_icon='🏸',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
  .big-title {
    font-size:2.3rem; font-weight:900; text-align:center; padding:8px 0;
    background:linear-gradient(135deg,#2874f0,#fb641b);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .subtitle { text-align:center; color:#888; font-size:1rem; margin-bottom:16px; }
  .res-pos {
    background:linear-gradient(120deg,#d4edda,#c3e6cb); border-left:5px solid #28a745;
    padding:18px; border-radius:12px; font-size:1.5rem; font-weight:800;
    color:#155724; text-align:center; margin:10px 0;
  }
  .res-neg {
    background:linear-gradient(120deg,#f8d7da,#f5c6cb); border-left:5px solid #dc3545;
    padding:18px; border-radius:12px; font-size:1.5rem; font-weight:800;
    color:#721c24; text-align:center; margin:10px 0;
  }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### 🏸 Flipkart Sentiment')
    st.markdown('**Product:** YONEX MAVIS 350 Nylon Shuttle')
    st.markdown('---')
    try:
        model, model_src = load_model()
        st.success('✅ Model loaded')
        vec_name = type(model.named_steps['vectorizer']).__name__
        clf_name = type(model.named_steps['model']).__name__
        st.markdown(f'**Source:** {model_src}')
        st.markdown(f'**Vectorizer:** `{vec_name}`')
        st.markdown(f'**Classifier:** `{clf_name}`')
    except Exception as e:
        st.error(f'Model not loaded: {e}')
        model = None
    st.markdown('---')
    st.markdown('**MLflow UI:**')
    st.code('mlflow ui --host 0.0.0.0 --port 5000')
    st.markdown('**Prefect Dashboard:**')
    st.code('prefect server start')
    st.caption('Built with Streamlit · MLflow · Prefect')

st.markdown('<div class="big-title">🏸 Flipkart Review Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YONEX MAVIS 350 · Real-time Sentiment · MLOps Project</div>', unsafe_allow_html=True)
st.markdown('---')

tab1, tab2, tab3, tab4 = st.tabs(['🔍 Single Review', '📋 Batch Analysis', '📊 Dataset Insights', 'ℹ️ MLflow & Prefect'])

with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader('Enter a Customer Review')
        examples = {
            'Select an example...': '',
            '✅ Great product': 'Amazing shuttle! Very durable and the flight is perfect. Great value for money.',
            '❌ Fake product':  'Complete waste of money. Not original Yonex. Feathers broke after one game.',
            '✅ Good quality':  'Good quality shuttlecock. Plays like feather shuttle at much lower price.',
            '❌ Poor quality':  'Worst product. Damaged corks inside box. Flipkart delivered fake item.',
        }
        ex = st.selectbox('Try an example:', list(examples.keys()))
        default_text = examples[ex] if ex != 'Select an example...' else ''
        user_review  = st.text_area('Or type your review:', value=default_text, height=160,
                                    placeholder='Type a product review here...')
        run_btn = st.button('🔮 Analyse Sentiment', type='primary', use_container_width=True)

    with col2:
        st.subheader('Result')
        if run_btn and user_review.strip():
            if model is None:
                st.error('Model not loaded. Check sidebar.')
            else:
                cleaned = clean_text(user_review)
                pred    = model.predict([cleaned])[0]
                if pred == 1:
                    st.markdown('<div class="res-pos">😊 POSITIVE<br><small>Customer is satisfied</small></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="res-neg">😞 NEGATIVE<br><small>Customer is dissatisfied</small></div>',
                                unsafe_allow_html=True)
                st.markdown('')
                try:
                    proba = model.predict_proba([cleaned])[0]
                    c1, c2 = st.columns(2)
                    c1.metric('😞 Negative', f'{proba[0]*100:.1f}%')
                    c2.metric('😊 Positive', f'{proba[1]*100:.1f}%')
                    prob_df = pd.DataFrame({'Sentiment': ['Negative', 'Positive'],
                                            'Confidence': [proba[0], proba[1]]})
                    st.bar_chart(prob_df.set_index('Sentiment'))
                except AttributeError:
                    pass
                with st.expander('🔎 Cleaned Text'):
                    st.code(cleaned or '(empty after cleaning)')
        elif run_btn:
            st.warning('Please enter a review.')
        else:
            st.markdown("<div style='text-align:center;color:#bbb;padding:50px 0'><div style='font-size:3.5rem'>🎯</div><div>Result appears here</div></div>",
                        unsafe_allow_html=True)

with tab2:
    st.subheader('📋 Batch Review Classification')
    st.info('Upload a CSV with a **`Review text`** column.')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded and model:
        df_up = pd.read_csv(uploaded)
        if 'Review text' not in df_up.columns:
            st.error('CSV must contain a `Review text` column.')
        else:
            with st.spinner('Classifying...'):
                df_up['cleaned']   = df_up['Review text'].apply(clean_text)
                df_up['Sentiment'] = model.predict(df_up['cleaned'])
                df_up['Label']     = df_up['Sentiment'].map({1: '😊 Positive', 0: '😞 Negative'})
            pos = (df_up['Sentiment'] == 1).sum()
            neg = (df_up['Sentiment'] == 0).sum()
            c1, c2, c3 = st.columns(3)
            c1.metric('Total', len(df_up))
            c2.metric('😊 Positive', pos, f'{pos/len(df_up)*100:.1f}%')
            c3.metric('😞 Negative', neg, f'{neg/len(df_up)*100:.1f}%')
            st.dataframe(df_up[['Review text', 'Label']].head(50), use_container_width=True)
            st.download_button('⬇️ Download Results', df_up.to_csv(index=False).encode(),
                               'sentiment_results.csv', 'text/csv')

with tab3:
    st.subheader('📊 Dataset Overview')
    try:
        df_raw = pd.read_csv('data.csv')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Reviews',    f'{len(df_raw):,}')
        c2.metric('Avg Rating',       f'{df_raw["Ratings"].mean():.2f} ⭐')
        c3.metric('Positive (≥4)',    f'{(df_raw["Ratings"]>=4).sum():,}')
        c4.metric('Negative (≤2)',    f'{(df_raw["Ratings"]<=2).sum():,}')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        rc = df_raw['Ratings'].value_counts().sort_index()
        axes[0].bar(rc.index, rc.values,
                    color=['#e74c3c','#e67e22','#f39c12','#2ecc71','#27ae60'], edgecolor='white')
        axes[0].set_title('Rating Distribution', fontweight='bold')
        df_raw['lbl'] = df_raw['Ratings'].apply(lambda x: 'Positive' if x>=4 else ('Negative' if x<=2 else 'Neutral'))
        sc = df_raw['lbl'].value_counts()
        axes[1].pie(sc.values, labels=sc.index,
                    colors=['#2ecc71','#e74c3c','#f39c12'], autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Sentiment Distribution', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except FileNotFoundError:
        st.warning('Place `data.csv` in the same directory as `app.py`.')

with tab4:
    st.subheader('ℹ️ MLflow & Prefect Quick Reference')
    st.markdown("""
### 🧪 Start MLflow UI
```bash
# In your project directory on EC2:
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://<EC2-IP>:5000
```
**What you'll see:**
- All 10+ experiment runs with custom names
- Metric comparison charts (F1, accuracy, AUC)
- Hyperparameter parallel coordinates plot
- Artifacts: confusion matrix PNGs, classification reports
- Model Registry: `flipkart_sentiment_classifier` → Production stage

---
### 🔄 Start Prefect Dashboard
```bash
# Terminal 1
prefect server start
# Open: http://<EC2-IP>:4200

# Terminal 2
python prefect_deploy.py   # creates daily schedule

# Terminal 3
prefect agent start -q default   # starts executing scheduled runs
```

---
### 📦 Load Production Model from Registry
```python
import mlflow.sklearn, mlflow
mlflow.set_tracking_uri('http://<EC2-IP>:5000')
model = mlflow.sklearn.load_model('models:/flipkart_sentiment_classifier/Production')
pred  = model.predict(['your review text here'])
```

---
### 🔁 Retrain Model via Prefect Flow
```bash
python sentiment_flow.py   # manual run
# Or let the daily cron schedule pick it up automatically
```
    """)

st.markdown('---')
st.caption('🏸 Flipkart Sentiment Analyser · MLflow + Prefect + Streamlit · AWS EC2')
