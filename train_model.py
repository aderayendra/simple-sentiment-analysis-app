import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Configuration
DATASET_PATH = r'D:\Projects\sentiment-analysis-app\dataset\dataset.csv'
MODEL_DIR = r'D:\Projects\sentiment-analysis-app\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

def train_model():
    # 1. Load the dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Preprocess data
    # Drop rows with missing values in 'comment' or 'label'
    df = df.dropna(subset=['comment', 'label'])
    
    # The dataset has columns 'comment' and 'label'
    X = df['comment']
    y = df['label']
    
    # Vectorize text data
    # Optimization: Added ngrams (bigrams), sublinear_tf, and tuned min_df/max_features
    print("Vectorizing text with optimized TfidfVectorizer (ngrams range 1,2)...")
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=15000, 
        ngram_range=(1, 2), 
        min_df=3, 
        max_df=0.8,
        sublinear_tf=True
    )
    X_vectorized = vectorizer.fit_transform(X)
    
    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # 4. Train a Decision Tree Classifier with Hyperparameter Tuning
    print("Training Decision Tree model with RandomizedSearchCV for hyperparameter tuning...")
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'ccp_alpha': [0.0, 0.001, 0.01]
    }
    
    base_clf = DecisionTreeClassifier(random_state=42)
    # Using n_iter=20 to keep it fast while still exploring better params
    random_search = RandomizedSearchCV(base_clf, param_distributions=param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X_train, y_train)
    
    clf = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")
    
    # 5. Evaluate the model
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the model and vectorizer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    
    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Training complete!")

if __name__ == "__main__":
    train_model()
