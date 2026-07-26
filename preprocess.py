import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded automatically
def download_nltk_resources():
    for resource in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download NLTK resource {resource}: {e}")

download_nltk_resources()

def preprocess_text(text):
    """
    Cleans and preprocesses the raw text of a message:
    1. Converts to lowercase.
    2. Tokenizes into words.
    3. Removes punctuation and special characters.
    4. Removes stopwords.
    5. Stems words using PorterStemmer.
    
    Parameters:
    - text (str): The raw input message.
    
    Returns:
    - str: The preprocessed, tokenized, and stemmed text as a space-separated string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Tokenization (using NLTK word_tokenize with fallback to simple split)
    try:
        words = nltk.word_tokenize(text)
    except Exception:
        # Strip simple punctuation first and split by whitespace
        for p in string.punctuation:
            text = text.replace(p, " ")
        words = text.split()
        
    # Get English stopwords and punctuation set
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback empty set if stopwords download failed
        stop_words = set()
        
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    
    # 3. Remove punctuation, stopwords, and apply Stemming
    cleaned_words = []
    for word in words:
        # Strip internal punctuation
        clean_word = "".join(char for char in word if char not in punctuation)
        # Filter out empty strings and stopwords
        if clean_word and clean_word not in stop_words:
            # 5. Stemming
            try:
                stemmed = stemmer.stem(clean_word)
                cleaned_words.append(stemmed)
            except Exception:
                cleaned_words.append(clean_word)
            
    return " ".join(cleaned_words)

if __name__ == "__main__":
    test_text = "WINNER!! You have won a 1-week free membership to our services! Claim now at http://win.com."
    print("Original Text:", test_text)
    print("Preprocessed Text:", preprocess_text(test_text))
