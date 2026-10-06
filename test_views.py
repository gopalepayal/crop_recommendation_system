import os
import django
from django.test import Client

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_recommendation_system.settings')
django.setup()

def run_tests():
    c = Client()

    urls = [
        '/',
        '/overview/',
        '/crop-analysis/',
        '/environmental-analysis/',
        '/model-performance/',
        '/recommendation/',
        '/dataset-explorer/',
    ]

    print("--- 1. Testing GET Endpoints (Landing Page & Dashboard Pages) ---")
    for url in urls:
        response = c.get(url)
        assert response.status_code == 200, f"Failed GET {url}"
        print(f"GET {url} - SUCCESS (200)")

    print("\n--- 2. Testing Landing Page Content Requirements ---")
    resp_landing = c.get('/')
    html_landing = resp_landing.content.decode('utf-8')

    assert 'Smart Crop Recommendation System' in html_landing, "Hero main heading missing"
    assert 'Make Better Crop Decisions with Soil &amp; Environmental Data' in html_landing or 'Make Better Crop Decisions with Soil & Environmental Data' in html_landing, "Hero subtitle missing"
    assert 'Enter soil and environmental conditions to receive an ML-based crop recommendation.' in html_landing, "Hero text missing"
    assert 'How It Works' in html_landing, "How It Works section missing"
    assert 'Soil-Based Analysis' in html_landing, "Soil feature card missing"
    assert 'Environmental Analysis' in html_landing, "Environmental feature card missing"
    assert 'ML-Based Recommendation' in html_landing, "ML feature card missing"
    assert '2,200+' in html_landing, "2200+ stat missing"
    assert '22' in html_landing, "22 stat missing"
    assert '7' in html_landing, "7 stat missing"
    assert '5' in html_landing, "5 stat missing"
    assert 'Powered by Machine Learning' in html_landing, "Powered by ML section missing"
    assert 'Random Forest' in html_landing, "Best model Random Forest missing"
    assert '99.55%' in html_landing, "Accuracy 99.55% missing"
    assert 'Ready to Explore Crop Recommendations?' in html_landing, "CTA heading missing"
    assert 'Start Recommendation' in html_landing, "CTA button missing"
    print("Landing Page Content & Stats Verification - SUCCESS")

    print("\n--- 3. Testing Recommendation Form & Prediction Flow ---")
    valid_data = {
        'n': '99',
        'p': '36',
        'k': '20',
        'temp': '20.579819',
        'humidity': '65.345839',
        'ph': '6.671086',
        'rainfall': '78.346045'
    }
    response = c.post('/recommendation/', valid_data)
    assert response.status_code == 200, "Valid POST /recommendation/ failed"
    html_rec = response.content.decode('utf-8')
    assert 'Maize' in html_rec or 'maize' in html_rec, "Predicted crop Maize missing"
    assert 'Model Confidence:' in html_rec or 'Confidence:' in html_rec, "Confidence badge missing"
    assert 'Top 3 Recommended Crops' in html_rec, "Top 3 recommendations missing"
    print("Recommendation POST Flow - SUCCESS")

    print("\n==========================================")
    print(" ALL LANDING PAGE & DASHBOARD TESTS PASSED! ")
    print("==========================================")

if __name__ == '__main__':
    run_tests()
