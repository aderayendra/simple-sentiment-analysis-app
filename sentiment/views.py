from django.shortcuts import render
import joblib
import os
from django.conf import settings

# Paths to the model and vectorizer
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'sentiment_model.joblib')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'model', 'tfidf_vectorizer.joblib')

# Load model and vectorizer once when the server starts
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        model = None
        vectorizer = None
        print("Model or Vectorizer file not found.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None


def index(request):
    return render(request, 'sentiment/index.html')


from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt  # For simplicity in an API context, but can also use CSRF tokens for web requests
def api_sentiment(request):
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                comment = data.get('comment', '')
            else:
                comment = request.POST.get('comment', '')

            if comment and model and vectorizer:
                vectorized_comment = vectorizer.transform([comment])
                prediction = model.predict(vectorized_comment)[0]
                return JsonResponse({
                    'sentiment': prediction,
                    'status': 'success'
                })
            elif not (model and vectorizer):
                return JsonResponse({'error': 'Model not loaded', 'status': 'error'}, status=500)
            else:
                return JsonResponse({'error': 'Comment missing', 'status': 'error'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e), 'status': 'error'}, status=500)

    return JsonResponse({'error': 'Only POST allowed', 'status': 'error'}, status=405)
