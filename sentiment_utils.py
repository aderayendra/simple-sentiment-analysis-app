import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
def setup_nltk():
    """Download NLTK resources if not already present."""
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        nltk.download(resource, quiet=True)

# Initialize tools
setup_nltk()
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def preprocess_text(text):
    """Preprocess the input text for sentiment analysis."""
    if not isinstance(text, str):
        text = str(text)

    # Lowercasing
    text = text.lower()

    # Removing Whitespace
    text = " ".join(text.split())

    # Removing Numbers & Special Characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Removing Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords Removal & Stemming
    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]

    return ' '.join(cleaned_tokens)
