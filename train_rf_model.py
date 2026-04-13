import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Configuration
DATASET_PATH = r'D:\Projects\sentiment-analysis-app\dataset\dataset.csv'
MODEL_DIR = r'D:\Projects\sentiment-analysis-app\model'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_sentiment_model.joblib')
# Vectorizer can be shared or separate, but for RF we'll use the same settings as DT for consistency in preprocessing
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'rf_tfidf_vectorizer.joblib')


def train_rf_model():
    # 1. Load the dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # 2. Preprocess data
    df = df.dropna(subset=['comment', 'label'])
    X = df['comment']
    y = df['label']

    # Vectorize text data
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

    # 4. Train a Random Forest Classifier with Hyperparameter Tuning
    print("Training Random Forest model with RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    base_clf = RandomForestClassifier(random_state=42)
    # Using n_iter=10 to keep it relatively fast
    random_search = RandomizedSearchCV(base_clf, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy',
                                       n_jobs=-1, verbose=1, random_state=42)
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

    print("Random Forest Training complete!")


if __name__ == "__main__":
    train_rf_model()
