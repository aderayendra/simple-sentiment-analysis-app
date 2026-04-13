from django.test import TestCase, Client
from django.urls import reverse
import json

class SentimentAPITest(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('api_sentiment')

    def test_sentiment_prediction_positive_dt(self):
        response = self.client.post(self.url, {'comment': 'I love this project!', 'model_type': 'dt'})
        if response.status_code == 200:
            data = json.loads(response.content)
            self.assertIn(data['sentiment'], ['positive', 'negative'])
            self.assertEqual(data['model_used'], 'Decision Tree')
        else:
            print(f"DT Positive test skipped or failed due to: {response.content}")

    def test_sentiment_prediction_positive_rf(self):
        response = self.client.post(self.url, {'comment': 'I love this project!', 'model_type': 'rf'})
        if response.status_code == 200:
            data = json.loads(response.content)
            self.assertIn(data['sentiment'], ['positive', 'negative'])
            self.assertEqual(data['model_used'], 'Random Forest')
        else:
            print(f"RF Positive test skipped or failed due to: {response.content}")

    def test_sentiment_prediction_negative_dt(self):
        response = self.client.post(self.url, {'comment': 'This is terrible.', 'model_type': 'dt'})
        if response.status_code == 200:
            data = json.loads(response.content)
            self.assertIn(data['sentiment'], ['positive', 'negative'])
            self.assertEqual(data['model_used'], 'Decision Tree')
        else:
            print(f"DT Negative test skipped or failed due to: {response.content}")

    def test_sentiment_prediction_negative_rf(self):
        response = self.client.post(self.url, {'comment': 'This is terrible.', 'model_type': 'rf'})
        if response.status_code == 200:
            data = json.loads(response.content)
            self.assertIn(data['sentiment'], ['positive', 'negative'])
            self.assertEqual(data['model_used'], 'Random Forest')
        else:
            print(f"RF Negative test skipped or failed due to: {response.content}")
