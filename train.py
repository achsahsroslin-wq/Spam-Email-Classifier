import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from wordcloud import WordCloud
from collections import Counter
import re
import nltk

# Import text preprocessing
from preprocess import preprocess_text

# Set style for Seaborn/Matplotlib for modern premium look
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1E1E2E',
    'axes.facecolor': '#1E1E2E',
    'text.color': '#CDD6F4',
    'axes.labelcolor': '#CDD6F4',
    'xtick.color': '#CDD6F4',
    'ytick.color': '#CDD6F4',
    'axes.titlecolor': '#CDD6F4',
    'grid.color': '#313244',
    'font.family': 'sans-serif'
})

DATA_URL = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
DATA_DIR = "data"
MODEL_DIR = "models"
SCREENSHOT_DIR = "screenshots"
NOTEBOOK_DIR = "notebook"
DATA_PATH = os.path.join(DATA_DIR, "spam.csv")

def setup_directories():
    """Create project directories if they do not exist."""
    print("Setting up project directories...")
    for directory in [DATA_DIR, MODEL_DIR, SCREENSHOT_DIR, NOTEBOOK_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")

def download_dataset():
    """Download the SMS Spam Collection dataset from Github."""
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return
    
    print(f"Downloading dataset from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"Dataset downloaded successfully and saved to {DATA_PATH}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Create a tiny dummy dataset if download fails, so execution can proceed
        print("Creating a fallback dummy dataset...")
        dummy_data = {
            "v1": ["ham", "spam", "ham", "spam", "ham", "spam"] * 20,
            "v2": [
                "Hey, how are you? Are we still meeting today?",
                "WINNER!! You have won a 1-week free membership to our services. Claim now!",
                "Can you send me the files by tonight?",
                "URGENT: Your mobile number has won a £2000 bonus. Call 09061104276 to claim.",
                "Let's grab a coffee tomorrow afternoon.",
                "FREE Ringtone! Reply to this message to claim your free ringtone."
            ] * 20,
            "Unnamed: 2": [np.nan] * 120,
            "Unnamed: 3": [np.nan] * 120,
            "Unnamed: 4": [np.nan] * 120
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(DATA_PATH, index=False, encoding="latin-1")
        print("Fallback dummy dataset created successfully.")

def load_and_clean_data():
    """Load, clean and prepare the dataset."""
    print("Loading and cleaning dataset...")
    # Load dataset with latin-1 encoding
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    
    # 1. Drop unused columns
    cols_to_drop = [col for col in df.columns if col.startswith("Unnamed")]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        
    # 2. Rename columns
    df.columns = ["label", "message"]
    
    # 3. Remove duplicates
    initial_shape = df.shape
    df.drop_duplicates(keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Removed duplicates: {initial_shape[0] - df.shape[0]} rows dropped. Remaining: {df.shape[0]} rows.")
    
    # 4. Handle missing values
    df.dropna(subset=["message", "label"], inplace=True)
    
    # 5. Convert labels to binary (ham=0, spam=1)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    
    return df

def run_eda(df):
    """Perform Exploratory Data Analysis and save visualization graphs."""
    print("Performing Exploratory Data Analysis...")
    
    # Calculate email text features
    df["length"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))
    df["sent_count"] = df["message"].apply(lambda x: len(re.split(r'[.!?]+', x)) - 1)
    
    # 1. Spam vs Ham Count
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="label", data=df, palette=["#30b090", "#e05060"])
    plt.title("Distribution of Spam vs Ham Messages", fontsize=14, fontweight="bold")
    plt.xlabel("Message Type")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "spam_vs_ham.png"), facecolor='#1E1E2E')
    plt.close()
    
    # 2. Message Length Comparison (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df["label_num"] == 0]["length"], bins=50, color="#30b090", label="Ham", kde=True, alpha=0.6)
    sns.histplot(df[df["label_num"] == 1]["length"], bins=50, color="#e05060", label="Spam", kde=True, alpha=0.6)
    plt.xlim(0, 300) # Limit x-axis to zoom in on typical lengths
    plt.title("Distribution of Message Length (Characters)", fontsize=14, fontweight="bold")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "message_length_histogram.png"), facecolor='#1E1E2E')
    plt.close()
    
    # 3. Correlation Heatmap of Text Features
    plt.figure(figsize=(6, 5))
    corr = df[["label_num", "length", "word_count", "sent_count"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, cbar=True, annot_kws={"weight": "bold"})
    plt.title("Correlation Heatmap of Text Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "correlation_heatmap.png"), facecolor='#1E1E2E')
    plt.close()
    
    # 4. Word Clouds
    # Preprocess all messages for generating clean word clouds
    print("Preprocessing text for word clouds and common words analysis...")
    df["processed_message"] = df["message"].apply(preprocess_text)
    
    spam_words = " ".join(df[df["label_num"] == 1]["processed_message"])
    ham_words = " ".join(df[df["label_num"] == 0]["processed_message"])
    
    # Generate Spam Word Cloud
    if spam_words.strip():
        wc_spam = WordCloud(width=800, height=400, background_color="#1E1E2E", colormap="Reds", max_words=100).generate(spam_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_spam, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Common Words in Spam Messages", fontsize=16, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOT_DIR, "wordcloud_spam.png"), facecolor='#1E1E2E')
        plt.close()
        
    # Generate Ham Word Cloud
    if ham_words.strip():
        wc_ham = WordCloud(width=800, height=400, background_color="#1E1E2E", colormap="Greens", max_words=100).generate(ham_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_ham, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Common Words in Ham Messages", fontsize=16, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOT_DIR, "wordcloud_ham.png"), facecolor='#1E1E2E')
        plt.close()
        
    # 5. Most Common Words Bar Chart (Spam vs Ham)
    spam_token_counts = Counter(spam_words.split())
    ham_token_counts = Counter(ham_words.split())
    
    spam_common = pd.DataFrame(spam_token_counts.most_common(20), columns=["Word", "Count"])
    ham_common = pd.DataFrame(ham_token_counts.most_common(20), columns=["Word", "Count"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x="Count", y="Word", data=ham_common, ax=axes[0], palette="viridis")
    axes[0].set_title("Top 20 Words in Ham Messages", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")
    
    sns.barplot(x="Count", y="Word", data=spam_common, ax=axes[1], palette="flare")
    axes[1].set_title("Top 20 Words in Spam Messages", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("")
    
    plt.suptitle("Most Common Words Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "most_common_words.png"), facecolor='#1E1E2E')
    plt.close()
    
    print("EDA Visualizations saved successfully in 'screenshots/'")
    return df

def train_and_compare_models(df):
    """Train multiple models, compare their performance and save the best model."""
    print("Starting Model Training and Evaluation Pipeline...")
    
    X = df["processed_message"]
    y = df["label_num"]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Engineering: TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Feature Engineering: CountVectorizer (for Naive Bayes comparison)
    print("Fitting CountVectorizer (for comparison)...")
    count_vec = CountVectorizer(max_features=5000, min_df=2)
    X_train_count = count_vec.fit_transform(X_train)
    X_test_count = count_vec.transform(X_test)
    
    # Define models
    models = {
        "Multinomial Naive Bayes (TF-IDF)": (MultinomialNB(), X_train_tfidf, X_test_tfidf),
        "Multinomial Naive Bayes (Count)": (MultinomialNB(), X_train_count, X_test_count),
        "Linear SVM (TF-IDF)": (LinearSVC(random_state=42, dual=False), X_train_tfidf, X_test_tfidf),
        "Logistic Regression (TF-IDF)": (LogisticRegression(random_state=42, max_iter=1000), X_train_tfidf, X_test_tfidf),
        "Random Forest (TF-IDF)": (RandomForestClassifier(random_state=42, n_estimators=100), X_train_tfidf, X_test_tfidf)
    }
    
    results = []
    trained_models = {}
    
    # Evaluation
    for name, (model, X_tr, X_te) in models.items():
        print(f"Training model: {name}...")
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC AUC if probability or decision function is available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_te)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr = None, None
            roc_auc = np.nan
            
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "FPR": fpr,
            "TPR": tpr,
            "cm": confusion_matrix(y_test, y_pred)
        })
        
        trained_models[name] = model
        print(f"Results for {name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Plot Model Comparison metrics
    plt.figure(figsize=(12, 6))
    df_melted = pd.melt(df_results, id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="magma")
    plt.title("Model Comparison Metrics", fontsize=15, fontweight="bold")
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0.8, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "model_comparison.png"), facecolor='#1E1E2E')
    plt.close()
    
    # Plot ROC Curves for all models
    plt.figure(figsize=(8, 6))
    for res in results:
        if res["FPR"] is not None and res["TPR"] is not None:
            plt.plot(res["FPR"], res["TPR"], lw=2, label=f'{res["Model"]} (AUC = {res["ROC-AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "roc_curve_all.png"), facecolor='#1E1E2E')
    plt.close()
    
    # Select the best model based on Precision (crucial for spam detection to avoid False Positives)
    # If Precisions are equal, we choose based on F1-Score
    best_row = df_results.sort_values(by=["Precision", "F1-Score"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    
    print("\n" + "="*50)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Accuracy:  {best_row['Accuracy']:.4f}")
    print(f"Precision: {best_row['Precision']:.4f}  <-- Prioritized to minimize False Positives")
    print(f"Recall:    {best_row['Recall']:.4f}")
    print(f"F1-Score:  {best_row['F1-Score']:.4f}")
    print("="*50)
    
    # Plot Confusion Matrix of the Best Model
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_row["cm"], annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Ham (0)", "Spam (1)"], yticklabels=["Ham (0)", "Spam (1)"],
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(SCREENSHOT_DIR, "confusion_matrix_best.png"), facecolor='#1E1E2E')
    plt.close()
    
    # Save the selected model, vectorizer, and results summary
    print("\nSaving best model and vectorizers...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_model.pkl"))
    
    # We must save the vectorizer that corresponds to the best model.
    # Note: "Multinomial Naive Bayes (Count)" uses CountVectorizer, others use TF-IDF.
    if "Count" in best_model_name:
        joblib.dump(count_vec, os.path.join(MODEL_DIR, "vectorizer.pkl"))
        print("CountVectorizer saved as vectorizer.pkl")
    else:
        joblib.dump(tfidf, os.path.join(MODEL_DIR, "vectorizer.pkl"))
        print("TF-IDF Vectorizer saved as vectorizer.pkl")
        
    # Save results as a csv file
    df_summary = df_results.drop(columns=["FPR", "TPR", "cm"])
    df_summary.to_csv(os.path.join(MODEL_DIR, "model_comparison_results.csv"), index=False)
    print("Model comparison results saved to models/model_comparison_results.csv")
    
    print("\nML Training Pipeline executed successfully!")

if __name__ == "__main__":
    setup_directories()
    download_dataset()
    df = load_and_clean_data()
    df = run_eda(df)
    train_and_compare_models(df)
