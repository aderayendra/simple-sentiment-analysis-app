from django.shortcuts import render
import joblib
import os
from django.conf import settings
from sentiment_utils import preprocess_text

# Paths to models and vectorizers
DT_MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'sentiment_model.joblib')
DT_VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'model', 'tfidf_vectorizer.joblib')
RF_MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'rf_sentiment_model.joblib')
RF_VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'model', 'rf_tfidf_vectorizer.joblib')

# Global storage for models and vectorizers
MODELS = {}
VECTORIZERS = {}

def load_models():
    """Load both models and vectorizers if they exist."""
    # Decision Tree
    try:
        if os.path.exists(DT_MODEL_PATH) and os.path.exists(DT_VECTORIZER_PATH):
            MODELS['dt'] = joblib.load(DT_MODEL_PATH)
            VECTORIZERS['dt'] = joblib.load(DT_VECTORIZER_PATH)
            print("Decision Tree model loaded.")
        else:
            print("Decision Tree files not found.")
    except Exception as e:
        print(f"Error loading Decision Tree model: {e}")

    # Random Forest
    try:
        if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_VECTORIZER_PATH):
            MODELS['rf'] = joblib.load(RF_MODEL_PATH)
            VECTORIZERS['rf'] = joblib.load(RF_VECTORIZER_PATH)
            print("Random Forest model loaded.")
        else:
            print("Random Forest files not found.")
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")


# Load models once when the server starts
load_models()


def index(request):
    return render(request, 'sentiment/index.html')


from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt  # For simplicity in an API context, but can also use CSRF tokens for web requests
def api_sentiment(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed', 'status': 'error'}, status=405)

    try:
        # Handle both JSON and Form data
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            comment = data.get('comment', '')
            model_type = data.get('model_type', 'dt')  # Default to 'dt'
        else:
            comment = request.POST.get('comment', '')
            model_type = request.POST.get('model_type', 'dt')

        comment = preprocess_text(comment)

        # Validate model choice
        if model_type not in MODELS:
            # If chosen model is not loaded, fallback to what's available
            model_type = 'dt' if 'dt' in MODELS else ('rf' if 'rf' in MODELS else None)

        if model_type and comment and model_type in MODELS and model_type in VECTORIZERS:
            model = MODELS[model_type]
            vectorizer = VECTORIZERS[model_type]

            vectorized_comment = vectorizer.transform([comment])
            prediction = model.predict(vectorized_comment)[0]
            return JsonResponse({
                'sentiment': prediction,
                'model_used': 'Decision Tree' if model_type == 'dt' else 'Random Forest',
                'status': 'success'
            })
        elif not model_type:
            return JsonResponse({'error': 'No models available', 'status': 'error'}, status=500)
        elif not comment:
            return JsonResponse({'error': 'Comment missing', 'status': 'error'}, status=400)
        else:
            return JsonResponse({'error': f'Model {model_type} not loaded', 'status': 'error'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e), 'status': 'error'}, status=500)


