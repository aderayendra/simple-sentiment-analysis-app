import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

# --- 1. CONFIGURATION & SETUP ---
# Updated to match your uploaded file names
INPUT_FILE = "dataset/imdb_dataset.csv"
OUTPUT_FILE = "dataset/dataset.csv"


def setup_nltk():
    """Download NLTK resources if not already present."""
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        nltk.download(resource, quiet=True)


# --- 2. PREPROCESSING LOGIC ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercasing
    text = text.lower()

    # 2. Removing Whitespace
    text = " ".join(text.split())

    # 3. Removing Numbers & Special Characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Removing Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Stopwords Removal & Stemming
    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]

    return ' '.join(cleaned_tokens)


# Initialize NLTK and tools
setup_nltk()
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# --- 3. MAIN WORKFLOW ---
try:
    # Load the CSV data
    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {INPUT_FILE}")

    # Remove empty columns that may exist in the CSV structure
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Apply preprocessing to the 'comment' column
    print("Preprocessing comments, please wait...")
    # We keep the 'label' column and create 'processed_comment'
    df['comment'] = df['comment'].apply(preprocess_text)

    # Reorder columns to have label next to the processed text
    df = df[['comment', 'label']]

    # Save to a new CSV file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Processed data saved to {OUTPUT_FILE}")

    # Display preview
    print("\nPreview of cleaned data:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
