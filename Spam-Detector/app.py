import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import time
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentinella | Spam AI", layout="wide", initial_sidebar_state="expanded")

st.title("📧 Email Spam Detection System")
st.markdown("This application uses **NLP** and **Machine Learning** to classify messages as Ham or Spam.")

# --- 1. CORE LOGIC ---
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = {'i', 'me', 'my', 'the', 'is', 'in', 'it', 'to', 'for', 'of', 'and', 'a', 'this', 'that'}
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

@st.cache_resource
def load_and_train():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['message'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(),
        'SVM (Calibrated)': CalibratedClassifierCV(LinearSVC(random_state=42))
    }

    trained_models = {}
    metrics = []
    cms = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        cms[name] = confusion_matrix(y_test, y_pred)
        metrics.append({
            'Model': name, 'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred), 'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    return tfidf, trained_models, pd.DataFrame(metrics), df, cms

tfidf, trained_models, metrics_df, original_df, confusion_matrices = load_and_train()

# --- 2. PDF GENERATOR FUNCTION ---
def create_pdf_report(filename, total, spam_count, ham_count, conclusion):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sentinella AI - Batch Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Source File: {filename}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"- Total Lines Processed: {total}", ln=True)
    pdf.cell(200, 10, txt=f"- Spam Messages Found: {spam_count}", ln=True)
    pdf.cell(200, 10, txt=f"- Ham Messages Found: {ham_count}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    color = (255, 0, 0) if conclusion == "SPAM" else (0, 128, 0)
    pdf.set_text_color(*color)
    pdf.cell(200, 10, txt=f"FINAL CONCLUSION: THE FILE IS LIKELY {conclusion}", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. UI TABS ---
tab1, tab2, tab3 = st.tabs(["✨ Live Predictor", "📁 Batch Upload", "📊 Analytics"])

with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Live Analysis")
    user_input = st.text_area("Type message below...", placeholder="Check if this email is safe...", height=150)
    
    if user_input:
        selected_model = st.selectbox("Model", list(trained_models.keys()), key="live_model")
        cleaned_msg = clean_text(user_input)
        vec_msg = tfidf.transform([cleaned_msg])
        probs = trained_models[selected_model].predict_proba(vec_msg)[0]
        
        is_spam = probs[1] > probs[0]
        color = "#FF4B4B" if is_spam else "#00CC96"
        label = "🚨 SPAM DETECTED" if is_spam else "✅ LEGITIMATE (HAM)"
        
        st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
        
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Ham Probability", f"{probs[0]:.2%}")
        col_p2.metric("Spam Probability", f"{probs[1]:.2%}")
        st.progress(float(probs[1]))
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Batch File Processing")
    uploaded_file = st.file_uploader("Upload .txt file", type="txt")
    
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8").splitlines()
        batch_results = []
        selected_m = st.selectbox("Select Model", list(trained_models.keys()), key="batch_model")
        
        with st.spinner('Analyzing...'):
            for line in content:
                if line.strip():
                    vec = tfidf.transform([clean_text(line)])
                    p = trained_models[selected_m].predict_proba(vec)[0]
                    batch_results.append("Spam" if p[1] > p[0] else "Ham")
        
        counts = pd.Series(batch_results).value_counts()
        spam_n = counts.get("Spam", 0)
        ham_n = counts.get("Ham", 0)
        final_conclusion = "SPAM" if spam_n > ham_n else "HAM"
        
        pdf_bytes = create_pdf_report(uploaded_file.name, len(batch_results), spam_n, ham_n, final_conclusion)
        
        st.success("Analysis Complete!")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Spam_Report_{datetime.now().strftime('%H%M%S')}.pdf",
            mime="application/pdf"
        )
        st.dataframe(pd.DataFrame({"Message": [l for l in content if l.strip()], "Result": batch_results}), use_container_width=True)

with tab3:
    st.subheader("Deep Model Analytics")
    
    # 1. Performance Comparison Bar Chart
    st.markdown("#### Comprehensive Performance Comparison")
    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')
    melted = metrics_df.melt(id_vars='Model')
    sns.barplot(data=melted, x='variable', y='value', hue='Model', palette='viridis')
    plt.ylim(0.8, 1.05)
    st.pyplot(fig_metrics)

    # 2. Dataset Distribution Pie Chart (RESIZED TO BE SMALLER)
    st.markdown("#### Dataset Label Distribution")
    fig_pie, ax_pie = plt.subplots(figsize=(3, 3)) # Reduced size from 6x6 to 3x3
    plt.style.use('dark_background')
    original_df['label'].value_counts().plot(
        kind='pie', 
        autopct='%1.1f%%', 
        colors=['#03DAC6', '#BB86FC'], 
        explode=[0.05, 0.05], 
        startangle=90, 
        ax=ax_pie,
        textprops={'color':"w", 'fontsize': 8} # Smaller font for smaller chart
    )
    ax_pie.set_ylabel('')
    fig_pie.patch.set_facecolor('#1E1E1E')
    plt.tight_layout()
    st.pyplot(fig_pie)

    # 3. Confusion Matrix Gallery
    st.markdown("#### Confusion Matrices (Error Analysis)")
    cols = st.columns(3)
    for i, name in enumerate(confusion_matrices.keys()):
        with cols[i]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            sns.heatmap(confusion_matrices[name], annot=True, fmt='d', cmap='Purples', cbar=False)
            ax_cm.set_title(f"{name}", fontsize=10)
            ax_cm.set_xlabel("Predicted", fontsize=8)
            ax_cm.set_ylabel("Actual", fontsize=8)
            st.pyplot(fig_cm)