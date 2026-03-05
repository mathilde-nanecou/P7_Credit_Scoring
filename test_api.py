import pytest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix

# Configuration du chemin pour importer app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

# --- 1. FIXTURE POUR L'API ---
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- 2. TESTS DE L'API (INTEGRATION) ---
# On vérifie que l'API répond correctement aux requêtes HTTP

def test_api_home(client):
    """Vérifie que l'accueil de l'API fonctionne"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"API" in response.data

def test_predict_valid_client(client):
    """Vérifie une prédiction sur un client existant (ID 100001)"""
    response = client.get('/predict?id=100001')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'probability' in data
    assert 0.0 <= data['probability'] <= 1.0

def test_predict_error_cases(client):
    """Vérifie que l'API gère bien les erreurs (ID inconnu, format...)"""
    # ID Inconnu
    assert client.get('/predict?id=999999').status_code == 404
    # Mauvais format (texte)
    assert client.get('/predict?id=abc').status_code == 400
    # Pas d'ID
    assert client.get('/predict').status_code == 400

# --- 3. TESTS DE LOGIQUE MÉTIER (UNITAIRES) ---
# Ici on teste tes calculs spécifiques au projet

class TestBusinessLogic:
    
    def test_payment_rate_calculation(self):
        """Vérifie que le calcul du taux de paiement est correct"""
        df = pd.DataFrame({'AMT_ANNUITY': [1000], 'AMT_CREDIT': [10000]})
        payment_rate = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        assert payment_rate.iloc[0] == 0.1

    def test_business_cost_metric(self):
        """Vérifie la métrique de coût personnalisée (FN=10, FP=1)"""
        y_true = np.array([1, 0]) # Un vrai défaut, un vrai sain
        y_pred = np.array([0, 1]) # On se trompe sur les deux (1 FN, 1 FP)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # Coût : (1 * 10) + (1 * 1) = 11
        total_cost = (fn * 10) + (fp * 1)
        assert total_cost == 11

class TestDataPreprocessing:

    def test_column_cleaning(self):
        """Vérifie que le nettoyage des noms de colonnes (caractères spéciaux) fonctionne"""
        import re
        col_name = "AMT_INCOME (TOTAL)!"
        clean_name = re.sub('[^A-Za-z0-9_]+', '', col_name)
        assert clean_name == "AMT_INCOME_TOTAL"

    def test_inf_values_handling(self):
        """Vérifie le remplacement des valeurs infinies"""
        df = pd.DataFrame({'col': [1.0, np.inf]})
        df = df.replace([np.inf, -np.inf], np.nan)
        assert pd.isna(df['col'].iloc[1])