import os
import joblib
import numpy as np
from preprocess import preprocess_text

MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

class SpamPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and vectorizer from disk."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Model or Vectorizer not found. Please run 'python train.py' first to train and save models."
            )
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        
    def predict(self, text):
        """
        Predict whether a text message is Spam or Not Spam.
        
        Parameters:
        - text (str): The raw text of the message.
        
        Returns:
        - dict: Containing prediction, label_num, confidence, and model_name.
        """
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict class
        pred = self.model.predict(vectorized)[0]
        
        # Calculate confidence score (probability)
        confidence = 0.5
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vectorized)[0]
            # Use probability of predicted class
            confidence = proba[pred]
        elif hasattr(self.model, "decision_function"):
            decision_score = self.model.decision_function(vectorized)[0]
            # Calibrate decision score into pseudo-probability using sigmoid function
            pseudo_prob = 1 / (1 + np.exp(-decision_score))
            # Pseudo-prob is for class 1 (Spam). If prediction is 0, confidence is 1 - pseudo_prob
            if pred == 1:
                confidence = pseudo_prob
            else:
                confidence = 1 - pseudo_prob
                
        label = "Spam" if pred == 1 else "Not Spam"
        
        return {
            "text": text,
            "prediction": label,
            "label_num": int(pred),
            "confidence": float(confidence),
            "model_name": type(self.model).__name__
        }

    def predict_batch(self, texts):
        """
        Predict a batch of text messages.
        
        Parameters:
        - texts (list of str): List of raw text messages.
        
        Returns:
        - list of dict: Predictions for each message.
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# Helper function for quick testing
if __name__ == "__main__":
    try:
        predictor = SpamPredictor()
        
        # Test ham
        test_ham = "Hey, are we still going for dinner tonight? Let me know."
        res1 = predictor.predict(test_ham)
        print(f"\nText: '{res1['text']}'")
        print(f"Prediction: {res1['prediction']} (Confidence: {res1['confidence'] * 100:.2f}%)")
        
        # Test spam
        test_spam = "Congratulations! You've won a $1,000 Walmart Gift Card! Click here to claim your prize now!"
        res2 = predictor.predict(test_spam)
        print(f"\nText: '{res2['text']}'")
        print(f"Prediction: {res2['prediction']} (Confidence: {res2['confidence'] * 100:.2f}%)")
        
    except FileNotFoundError as e:
        print(e)
