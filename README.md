### Sentiment Analysis Web Application

This project is a **Sentiment Analysis Web Application** that classifies text as **Positive** or **Negative**. It uses Machine Learning models (Decision Tree and Random Forest) trained with **scikit-learn** and a web interface built with **Django**.

---

#### 1. Project Structure
The project is organized into the following components:

```text
D:\PROJECTS\SENTIMENT-ANALYSIS-APP
├── dataset\
│   ├── imdb_dataset.csv        # Raw data source
│   └── dataset.csv             # Processed data for training
├── model\
│   ├── sentiment_model.joblib      # Trained Decision Tree model
│   ├── tfidf_vectorizer.joblib     # TF-IDF vectorizer (Decision Tree)
│   ├── rf_sentiment_model.joblib   # Trained Random Forest model
│   └── rf_tfidf_vectorizer.joblib  # TF-IDF vectorizer (Random Forest)
├── sentiment\                  # Django App containing web logic
│   ├── templates\sentiment\
│   │   └── index.html          # Web page (HTML/CSS)
│   ├── urls.py                 # App-specific URL routing
│   └── views.py                # Logic for predictions and fallback
├── sentiment_analysis_app\     # Main Django project configuration
│   ├── settings.py             # Global project settings
│   └── urls.py                 # Main URL routing
├── preprocess_dataset.py       # Script to clean and prepare data
├── sentiment_utils.py          # Shared utility for text preprocessing
├── train_model.py              # Script to train the Decision Tree model
├── train_rf_model.py           # Script to train the Random Forest model
├── manage.py                   # Django management script
└── README.md                   # Project documentation
```

---

#### 2. Workflow

**A. Data Preparation & Utility (`preprocess_dataset.py`, `sentiment_utils.py`)**
1.  **Preprocessing Logic:** `sentiment_utils.py` contains the shared `preprocess_text()` function used in both training and real-time inference.
2.  **Dataset Processing:** `preprocess_dataset.py` reads `imdb_dataset.csv`, cleans it using the shared utility, and saves it as `dataset.csv`.

**B. Training Phase (`train_model.py`, `train_rf_model.py`)**
1.  **Loading Data:** Scripts read the processed `dataset.csv`.
2.  **Vectorization:** `TfidfVectorizer` converts text into numerical features (TF-IDF).
3.  **Model Training:** Different classifiers (Decision Tree or Random Forest) are trained on numerical values.
4.  **Saving:** The models and their respective vectorizers are saved in the `model/` directory.

**C. Web App Phase (Django)**
1.  **Initialization:** `sentiment/views.py` loads available models and vectorizers upon server startup.
2.  **User Input:** Text is entered via the web interface (`index.html`).
3.  **Prediction:**
    -   The `api_sentiment` view in `views.py` handles requests (JSON or Form data).
    -   It uses the shared `preprocess_text()` from `sentiment_utils.py`.
    -   Predictions are made using the selected model (Decision Tree or Random Forest).
4.  **Display:** The result is returned and displayed with color-coding (Green for Positive, Red for Negative).

---

#### 3. Key Files
*   **`preprocess_dataset.py`**: Prepares the raw data for model training.
*   **`sentiment_utils.py`**: Centralized text cleaning logic (lowercasing, punctuation removal, stemming).
*   **`train_model.py` / `train_rf_model.py`**: Training scripts for building the models.
*   **`sentiment/views.py`**: The interface between the user request and the ML prediction.
*   **`sentiment/templates/sentiment/index.html`**: The interactive web frontend.
