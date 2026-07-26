import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Aegis Mail | AI Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark styling CSS
st.markdown("""
<style>
    /* Dark Theme Core overrides */
    .stApp {
        background-color: #0F0F13;
        color: #CDD6F4;
    }
    
    /* Main container cards */
    .main-card {
        background-color: #161622;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #232334;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    
    /* Subheadings and title */
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #89B4FA 0%, #CBA6F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #A6ADC8;
        margin-bottom: 30px;
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #89b4fa 0%, #b4befe 100%) !important;
        color: #11111b !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(137, 180, 250, 0.4);
    }
    
    /* Alert styles */
    .spam-alert {
        background-color: rgba(243, 139, 168, 0.15);
        border: 1px solid #f38ba8;
        border-radius: 8px;
        color: #f38ba8;
        padding: 20px;
        margin: 15px 0;
        font-weight: 600;
        text-align: center;
        font-size: 1.5rem;
    }
    
    .ham-alert {
        background-color: rgba(166, 227, 161, 0.15);
        border: 1px solid #a6e3a1;
        border-radius: 8px;
        color: #a6e3a1;
        padding: 20px;
        margin: 15px 0;
        font-weight: 600;
        text-align: center;
        font-size: 1.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #11111B;
        border-right: 1px solid #1E1E2E;
    }
    
    /* Metric Card styling */
    .metric-box {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #89B4FA;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A6ADC8;
    }
</style>
""", unsafe_allow_html=True)

# Try importing the predictor
predictor_loaded = False
try:
    from predict import SpamPredictor
    predictor = SpamPredictor()
    predictor_loaded = True
except Exception as e:
    predictor_error = str(e)

# App Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #CBA6F7;'>🛡️ Aegis Mail</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #A6ADC8; font-size: 0.85rem; margin-bottom: 25px;'>Advanced AI Spam Classification</p>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Menu",
    ["📧 Single Email Predictor", "📁 Batch CSV Predictor", "📊 Model Analytics & EDA"]
)

# Sidebar metadata / details
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
if predictor_loaded:
    st.sidebar.success("Model Status: **Active**")
    st.sidebar.info(f"Classifier: `{type(predictor.model).__name__}`")
    st.sidebar.info(f"Vectorizer: `{type(predictor.vectorizer).__name__}`")
else:
    st.sidebar.error("Model Status: **Offline**")
    st.sidebar.warning("Please run `python train.py` first to train and serialize models.")

# Main Application Layout
st.markdown("<div class='app-title'>Aegis Mail Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Real-time spam filtering and exploratory analytics using Scikit-Learn</div>", unsafe_allow_html=True)

if not predictor_loaded:
    st.error(f"⚠️ Model Initialization Failed! Error Details: {predictor_error}")
    st.info("💡 **Instructions to resolve:** Run the ML training script in your terminal to train and serialize the model files:\n`python train.py`")
else:
    # ----------------------------------------------------
    # PAGE 1: SINGLE EMAIL PREDICTOR
    # ----------------------------------------------------
    if menu == "📧 Single Email Predictor":
        st.markdown("### 📧 Single Message Classification")
        st.write("Analyze individual email or SMS messages below to detect if they contain spam characteristics.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='main-card'>", unsafe_allow_html=True)
            email_text = st.text_area(
                "Email / SMS Message Content:",
                placeholder="Type or paste your message content here...",
                height=220
            )
            
            # Predict Button
            predict_btn = st.button("⚡ Classify Message")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            if predict_btn and email_text.strip():
                with st.spinner("Analyzing text patterns..."):
                    result = predictor.predict(email_text)
                    
                    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='text-align: center; margin-bottom: 10px;'>Classification Result</h4>", unsafe_allow_html=True)
                    
                    # Custom result alerts
                    if result["prediction"] == "Spam":
                        st.markdown(f"<div class='spam-alert'>⚠️ SPAM DETECTED</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='ham-alert'>🛡️ NOT SPAM (HAM)</div>", unsafe_allow_html=True)
                    
                    # Confidence Meter
                    conf_pct = result["confidence"] * 100
                    st.markdown(f"<p style='text-align: center; font-size: 0.95rem; margin-bottom: 5px; color: #A6ADC8;'>Prediction Confidence: <b>{conf_pct:.2f}%</b></p>", unsafe_allow_html=True)
                    st.progress(result["confidence"])
                    
                    st.markdown("---")
                    
                    # Message Stats
                    st.write("**Message Statistics:**")
                    c_len = len(email_text)
                    w_cnt = len(email_text.split())
                    
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{c_len}</div>
                            <div class='metric-label'>Characters</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_s2:
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{w_cnt}</div>
                            <div class='metric-label'>Words</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='main-card' style='text-align: center; padding: 50px 20px;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 3rem; color: #45475A;'>📥</p>", unsafe_allow_html=True)
                st.markdown("<p style='color: #A6ADC8;'>Enter email content on the left and click <b>Classify Message</b> to view classification analytics.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
    # ----------------------------------------------------
    # PAGE 2: BATCH CSV PREDICTOR
    # ----------------------------------------------------
    elif menu == "📁 Batch CSV Predictor":
        st.markdown("### 📁 Batch CSV Classification")
        st.write("Upload a CSV file containing multiple messages to process them simultaneously.")
        
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df_upload = pd.read_csv(uploaded_file, encoding="utf-8")
                st.success("File uploaded successfully!")
                
                # Column selection
                columns = df_upload.columns.tolist()
                text_col = st.selectbox("Select the column containing the message text:", columns)
                
                if st.button("🚀 Process Batch"):
                    with st.spinner("Analyzing batch datasets..."):
                        texts = df_upload[text_col].astype(str).tolist()
                        
                        # Run predictions
                        preds = predictor.predict_batch(texts)
                        
                        # Populate outputs
                        df_upload["Prediction"] = [p["prediction"] for p in preds]
                        df_upload["Confidence"] = [f"{p['confidence'] * 100:.2f}%" for p in preds]
                        
                        # Calculate statistics
                        total_rows = len(df_upload)
                        spam_rows = sum(1 for p in preds if p["prediction"] == "Spam")
                        ham_rows = total_rows - spam_rows
                        spam_ratio = (spam_rows / total_rows) * 100
                        
                        st.markdown("---")
                        st.markdown("#### Batch Summary Metrics")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.markdown(f"<div class='metric-box'><div class='metric-value'>{total_rows}</div><div class='metric-label'>Total Processed</div></div>", unsafe_allow_html=True)
                        with col_m2:
                            st.markdown(f"<div class='metric-box'><div class='metric-value' style='color: #f38ba8;'>{spam_rows}</div><div class='metric-label'>Spam Detected</div></div>", unsafe_allow_html=True)
                        with col_m3:
                            st.markdown(f"<div class='metric-box'><div class='metric-value' style='color: #a6e3a1;'>{ham_rows}</div><div class='metric-label'>Not Spam (Ham)</div></div>", unsafe_allow_html=True)
                        with col_m4:
                            st.markdown(f"<div class='metric-box'><div class='metric-value'>{spam_ratio:.1f}%</div><div class='metric-label'>Spam Percentage</div></div>", unsafe_allow_html=True)
                            
                        st.markdown("---")
                        st.markdown("#### Classification Results Table")
                        st.dataframe(df_upload[[text_col, "Prediction", "Confidence"]].head(100), use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        df_upload.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Full Results as CSV",
                            data=csv_data,
                            file_name="spam_predictions_output.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
        else:
            st.markdown("<p style='color: #A6ADC8; text-align: center; padding: 40px;'>Please upload a CSV file with at least one text column. A sample CSV should have a row of text like: <i>\"Free coupon! Win now!\"</i></p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # ----------------------------------------------------
    # PAGE 3: MODEL ANALYTICS & EDA
    # ----------------------------------------------------
    elif menu == "📊 Model Analytics & EDA":
        st.markdown("### 📊 Dataset Analytics & Model Evaluation")
        st.write("Explore dataset insights, distributions, and comparative evaluation graphs generated during model training.")
        
        tab1, tab2 = st.tabs(["📈 Dataset Exploratory Analysis", "🏆 Model Performance & Selection"])
        
        with tab1:
            st.markdown("#### Dataset Insights (SMS Spam Collection Dataset)")
            
            col_eda1, col_eda2 = st.columns(2)
            
            with col_eda1:
                # Spam vs Ham Counts
                fig_path_counts = os.path.join("screenshots", "spam_vs_ham.png")
                if os.path.exists(fig_path_counts):
                    st.image(fig_path_counts, caption="Distribution of Spam vs Ham messages in the dataset.", use_container_width=True)
                else:
                    st.info("Count distribution plot not generated. Please run `python train.py` first.")
                    
                # Most Common words plot
                fig_path_common = os.path.join("screenshots", "most_common_words.png")
                if os.path.exists(fig_path_common):
                    st.image(fig_path_common, caption="Top 20 most frequent words in Ham vs Spam messages after preprocessing.", use_container_width=True)
                else:
                    st.info("Word frequency comparison plot not generated. Please run `python train.py` first.")
                    
            with col_eda2:
                # Message Length Histogram
                fig_path_len = os.path.join("screenshots", "message_length_histogram.png")
                if os.path.exists(fig_path_len):
                    st.image(fig_path_len, caption="Comparison of message length distribution (Ham vs Spam). Ham messages are usually shorter.", use_container_width=True)
                else:
                    st.info("Message length histogram not generated. Please run `python train.py` first.")
                    
                # Correlation Heatmap
                fig_path_corr = os.path.join("screenshots", "correlation_heatmap.png")
                if os.path.exists(fig_path_corr):
                    st.image(fig_path_corr, caption="Correlation heatmap of engineered text features.", use_container_width=True)
                else:
                    st.info("Correlation heatmap plot not generated. Please run `python train.py` first.")
            
            # Word clouds row
            st.markdown("#### Word Clouds")
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                fig_path_wc_ham = os.path.join("screenshots", "wordcloud_ham.png")
                if os.path.exists(fig_path_wc_ham):
                    st.image(fig_path_wc_ham, caption="Ham Word Cloud (Common Legitimate Terms)", use_container_width=True)
            with col_wc2:
                fig_path_wc_spam = os.path.join("screenshots", "wordcloud_spam.png")
                if os.path.exists(fig_path_wc_spam):
                    st.image(fig_path_wc_spam, caption="Spam Word Cloud (Common Spam Terms)", use_container_width=True)
                    
        with tab2:
            st.markdown("#### Machine Learning Model Comparison")
            st.write("We evaluated 4 distinct models on the TF-IDF feature space (Naive Bayes was evaluated on both Count Vectorizer and TF-IDF). The results show how each algorithm performs across different evaluation parameters.")
            
            # Display evaluation table
            res_csv_path = os.path.join("models", "model_comparison_results.csv")
            if os.path.exists(res_csv_path):
                df_res_tbl = pd.read_csv(res_csv_path)
                st.markdown("##### Detailed Metric Comparison Matrix")
                st.dataframe(df_res_tbl.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#2b5040"), use_container_width=True)
            else:
                st.info("Results CSV not found. Please train models first.")
                
            col_eval1, col_eval2 = st.columns(2)
            
            with col_eval1:
                # Comparison bar plot
                fig_path_comp = os.path.join("screenshots", "model_comparison.png")
                if os.path.exists(fig_path_comp):
                    st.image(fig_path_comp, caption="Accuracy, Precision, Recall, and F1-Score comparison for all models.", use_container_width=True)
                else:
                    st.info("Model comparison plot not found. Please train models first.")
                    
                # ROC Curves plot
                fig_path_roc = os.path.join("screenshots", "roc_curve_all.png")
                if os.path.exists(fig_path_roc):
                    st.image(fig_path_roc, caption="Receiver Operating Characteristic (ROC) Curves for evaluated models.", use_container_width=True)
                else:
                    st.info("ROC Curves plot not found. Please train models first.")
                    
            with col_eval2:
                # Best Model Confusion Matrix
                fig_path_cm = os.path.join("screenshots", "confusion_matrix_best.png")
                if os.path.exists(fig_path_cm):
                    st.image(fig_path_cm, caption=f"Confusion Matrix for the Selected Best Model.", use_container_width=True)
                else:
                    st.info("Confusion Matrix plot not found. Please train models first.")
                    
                st.markdown("<div class='main-card'>", unsafe_allow_html=True)
                st.markdown("##### 📌 Selected Model Selection Logic")
                st.write("""
                **Why Precision matters most here:**
                In email filtration systems, false positives are critical. If a legitimate email (Ham) is wrongly classified as Spam (False Positive), the user might miss important correspondence (e.g., job offer, verification codes). 
                
                If a Spam email slips into the inbox (False Negative), it is merely an annoyance. Hence, we prioritize models with high **Precision** over high **Recall**.
                
                The model with the highest Precision (and F1-score as tiebreaker) was serialized and loaded for live predictions.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
