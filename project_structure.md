### Project Structure and Functionality

The project is a **Sentiment Analysis Web Application** that uses a Machine Learning model (Decision Tree) to classify text as positive, negative, or neutral. It is built using **Django** for the web interface and **scikit-learn** for the machine learning logic.

---

#### 1. Project Structure
The project is organized into three main parts: the **Machine Learning** component, the **Django Web App**, and the **Dataset/Model storage**.

```text
D:\PROJECTS\SENTIMENT-ANALYSIS-APP
├── dataset\
│   └── dataset.csv             # The raw data used for training
├── model\
│   ├── sentiment_model.joblib   # The trained Decision Tree model
│   └── tfidf_vectorizer.joblib # The saved TF-IDF vectorizer
├── sentiment\                  # The Django "App" containing web logic
│   ├── templates\sentiment\
│   │   └── index.html          # The web page (HTML/CSS)
│   ├── urls.py                 # App-specific URL routing
│   └── views.py                # Logic for handling requests and predictions
├── sentiment_analysis_app\     # Main Django project configuration
│   ├── settings.py             # Global project settings
│   └── urls.py                 # Main URL routing
├── train.py                    # Script to train the model from the dataset
└── manage.py                   # Django management script
```

---

#### 2. How It Works

The workflow is divided into two phases: **Training** and **Inference (Web App)**.

**A. Training Phase (`train.py`)**
1.  **Loading Data:** It reads `dataset.csv` which contains comments and their corresponding labels (positive, negative, neutral).
2.  **Vectorization:** Computers cannot understand text directly, so it uses `TfidfVectorizer` to convert words into numerical values (TF-IDF features).
3.  **Model Training:** It trains a `DecisionTreeClassifier` on these numerical values.
4.  **Saving:** The trained model and the vectorizer are saved as `.joblib` files in the `model/` directory so they can be reused without retraining.

**B. Web App Phase (Django)**
1.  **Initialization:** When the server starts, `sentiment/views.py` loads the saved model and vectorizer into memory.
2.  **User Input:** You enter text into the text area on the web page (`index.html`).
3.  **Prediction:**
    -   The `index` view in `views.py` receives the text via a POST request.
    -   It uses the loaded **vectorizer** to transform your text into the same numerical format the model expects.
    -   The **model** then predicts the sentiment label.
4.  **Display:** The result is sent back to the browser and displayed with color-coding (Green for Positive, Red for Negative, Yellow for Neutral).

---

#### 3. Key Files
*   **`train.py`**: Your "factory" script that builds the brain of the app.
*   **`sentiment/views.py`**: The "bridge" that connects the web interface to the machine learning model.
*   **`sentiment/templates/sentiment/index.html`**: The "face" of the application that users interact with.