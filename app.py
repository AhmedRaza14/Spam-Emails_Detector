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
    # Load dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['message'].apply(clean_text)
    
    # Feature extraction
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label_num']
    
    # EXPLICIT TRAIN-TEST SPLIT (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,  # 20% for testing
        train_size=0.8,  # 80% for training
        stratify=y,      # Maintain class distribution
        random_state=42
    )
    
    # Store split information for visualization
    split_info = {
        'Total Samples': len(df),
        'Training Samples': X_train.shape[0],
        'Test Samples': X_test.shape[0],
        'Training %': 80,
        'Test %': 20
    }
    
    # Models to train
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (Calibrated)': CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000))
    }
    
    trained_models = {}
    test_metrics = []
    confusion_matrices = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions on test set only
        y_test_pred = model.predict(X_test)
        
        # Store confusion matrices for test set
        confusion_matrices[name] = confusion_matrix(y_test, y_test_pred)
        
        # Test metrics only
        test_metrics.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred),
            'Recall': recall_score(y_test, y_test_pred),
            'F1-Score': f1_score(y_test, y_test_pred)
        })
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(test_metrics)
    
    return tfidf, trained_models, metrics_df, df, confusion_matrices, split_info

# Load and train models
tfidf, trained_models, metrics_df, original_df, confusion_matrices, split_info = load_and_train()

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
    st.subheader("Model Performance Analytics - Test Set Results")
    
    # 1. TRAIN-TEST SPLIT INFORMATION
    st.markdown("#### 📊 Dataset Split (80% Train, 20% Test)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", split_info['Total Samples'])
    with col2:
        st.metric("Training Set", f"{split_info['Training Samples']} ({split_info['Training %']}%)")
    with col3:
        st.metric("Test Set", f"{split_info['Test Samples']} ({split_info['Test %']}%)")
    
    st.markdown("---")
    
    # 2. MODEL PERFORMANCE BAR CHART (TEST SET ONLY)
    st.markdown("#### 🎯 Model Performance on Test Set")
    
    # Create bar chart for test set metrics
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    
    x = np.arange(len(metrics_df['Model']))
    width = 0.2
    
    # Plot bars for each metric
    ax_bar.bar(x - width*1.5, metrics_df['Accuracy'], width, label='Accuracy', color='#03DAC6')
    ax_bar.bar(x - width/2, metrics_df['Precision'], width, label='Precision', color='#BB86FC')
    ax_bar.bar(x + width/2, metrics_df['Recall'], width, label='Recall', color='#FFB74D')
    ax_bar.bar(x + width*1.5, metrics_df['F1-Score'], width, label='F1-Score', color='#EF5350')
    
    ax_bar.set_xlabel('Models')
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('Model Performance Comparison on Test Set', fontsize=13, pad=15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
    ax_bar.legend(loc='lower right')
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig_bar)
    
    st.markdown("---")
    
    # 3. PERFORMANCE METRICS TABLE (TEST SET ONLY)
    st.markdown("#### 📋 Detailed Performance Metrics (Test Set)")
    
    # Add ranking column
    metrics_display = metrics_df.copy()
    metrics_display['Accuracy'] = metrics_display['Accuracy'].apply(lambda x: f"{x:.4f}")
    metrics_display['Precision'] = metrics_display['Precision'].apply(lambda x: f"{x:.4f}")
    metrics_display['Recall'] = metrics_display['Recall'].apply(lambda x: f"{x:.4f}")
    metrics_display['F1-Score'] = metrics_display['F1-Score'].apply(lambda x: f"{x:.4f}")
    
    # Highlight best model
    st.dataframe(metrics_display, use_container_width=True)
    
    st.markdown("---")
    
    # 4. BEST MODEL RECOMMENDATION
    st.markdown("#### 🏆 Best Model Recommendation")
    
    # Find best model based on test accuracy
    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = metrics_df['Accuracy'].max()
    best_f1 = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'F1-Score']
    
    st.success(f"""
    **🏆 {best_model}** is the best performing model!
    - **Accuracy**: {best_accuracy:.2%}
    - **F1-Score**: {best_f1:.4f}
    - **Test Set Size**: {split_info['Test Samples']} samples
    """)
    
    st.markdown("---")
    
    # 5. CONFUSION MATRICES SIDE BY SIDE (TEST SET ONLY)
    st.markdown("#### 🔍 Confusion Matrices")
    
    # Display confusion matrices in a grid (3 in a row)
    model_names = list(confusion_matrices.keys())
    
    # Create 3 columns for the 3 models
    cols = st.columns(3)
    
    for idx, (col, model_name) in enumerate(zip(cols, model_names)):
        with col:
            st.markdown(f"**{model_name}**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            
            # Choose colormap based on model
            cmap = 'Greens' if model_name == 'Naive Bayes' else ('Blues' if model_name == 'Logistic Regression' else 'Purples')
            
            sns.heatmap(confusion_matrices[model_name], annot=True, fmt='d', 
                       cmap=cmap, cbar=False, ax=ax_cm,
                       annot_kws={'size': 12})
            
            # ax_cm.set_title(f"Test Set", fontsize=10, pad=10)
            ax_cm.set_xlabel("Predicted", fontsize=9)
            ax_cm.set_ylabel("Actual", fontsize=9)
            
            # Set tick labels
            ax_cm.set_xticklabels(['HAM', 'SPAM'], fontsize=8)
            ax_cm.set_yticklabels(['HAM', 'SPAM'], fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig_cm)
            
            # Calculate and display basic metrics from confusion matrix
            cm = confusion_matrices[model_name]
            tn, fp, fn, tp = cm.ravel()
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("True Negatives", tn)
                st.metric("False Positives", fp)
            with col_metric2:
                st.metric("True Positives", tp)
                st.metric("False Negatives", fn)
    
    st.markdown("---")
    

    
    # 7. ORIGINAL DATASET DISTRIBUTION
    st.markdown("#### 📊 Original Dataset Distribution")
    
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    plt.style.use('dark_background')
    
    # Create pie chart
    counts = original_df['label'].value_counts()
    colors_pie = ['#03DAC6', '#BB86FC']
    explode = [0.05, 0.05]
    
    wedges, texts, autotexts = ax_pie.pie(counts, labels=counts.index, autopct='%1.1f%%',
                                          colors=colors_pie, explode=explode, 
                                          startangle=90, textprops={'fontsize': 11})
    
    # Style the text
    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax_pie.set_title('Full Dataset Label Distribution', fontsize=13, pad=20, color='white')
    ax_pie.set_ylabel('')
    fig_pie.patch.set_facecolor('#1E1E1E')
    ax_pie.legend(wedges, [f'HAM: {counts["ham"]}', f'SPAM: {counts["spam"]}'],
                 title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    st.pyplot(fig_pie)