import pandas as pd
from sentiment_utils import preprocess_text

# --- 1. CONFIGURATION & SETUP ---
# Updated to match your uploaded file names
INPUT_FILE = "dataset/imdb_dataset.csv"
OUTPUT_FILE = "dataset/dataset.csv"



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
