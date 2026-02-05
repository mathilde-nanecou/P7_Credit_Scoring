import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app import app
from sklearn.metrics import confusion_matrix


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"API" in response.data


def test_predict_valid_client(client):
    response = client.get('/predict?id=103497')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['client_id'] == 103497
    assert 'probability' in data
    assert 'decision' in data
    assert 'threshold' in data


def test_predict_unknown_client(client):
    response = client.get('/predict?id=99999999999')
    assert response.status_code == 404
    assert b"non trouv" in response.data.lower()


def test_predict_no_id(client):
    response = client.get('/predict')
    assert response.status_code == 400


def test_predict_bad_id_format(client):
    response = client.get('/predict?id=banane')
    assert response.status_code == 400


def test_predict_probability_range(client):
    response = client.get('/predict?id=103497')
    data = response.get_json()
    
    assert 0.0 <= data['probability'] <= 1.0


def test_predict_decision_values(client):
    response = client.get('/predict?id=103497')
    data = response.get_json()
    
    assert data['decision'] in ['Accordé', 'Refusé']


def test_predict_threshold_consistency(client):
    response = client.get('/predict?id=103497')
    data = response.get_json()
    
    if data['probability'] > data['threshold']:
        assert data['decision'] == 'Refusé'
    else:
        assert data['decision'] == 'Accordé'


class TestOneHotEncoder:
    
    def test_one_hot_encoder_basic(self):
        df = pd.DataFrame({
            'num_col': [1, 2, 3],
            'cat_col': ['A', 'B', 'A']
        })
        
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df_encoded = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)
        new_columns = [c for c in df_encoded.columns if c not in original_columns]
        
        assert 'cat_col_A' in df_encoded.columns
        assert 'cat_col_B' in df_encoded.columns
        assert len(new_columns) > 0
    
    def test_one_hot_encoder_no_categorical(self):
        df = pd.DataFrame({
            'num_col1': [1, 2, 3],
            'num_col2': [4.0, 5.0, 6.0]
        })
        
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        assert len(categorical_columns) == 0
    
    def test_one_hot_encoder_with_nan(self):
        df = pd.DataFrame({
            'cat_col': ['A', 'B', None]
        })
        
        df_encoded = pd.get_dummies(df, columns=['cat_col'], dummy_na=True)
        
        assert 'cat_col_nan' in df_encoded.columns


class TestBusinessCostMetric:
    
    def test_business_cost_all_correct(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * 10) + (fp * 1)
        
        assert total_cost == 0
    
    def test_business_cost_all_fn(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * 10) + (fp * 1)
        
        assert total_cost == 40
    
    def test_business_cost_all_fp(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * 10) + (fp * 1)
        
        assert total_cost == 4
    
    def test_business_cost_mixed(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * 10) + (fp * 1)
        
        assert fn == 2
        assert fp == 1
        assert total_cost == 21
    
    def test_business_cost_normalized(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * 10) + (fp * 1)
        normalized_cost = total_cost / len(y_true)
        
        assert normalized_cost == 11 / 4


class TestFeatureEngineering:
    
    def test_payment_rate_calculation(self):
        df = pd.DataFrame({
            'AMT_ANNUITY': [1000, 2000, 3000],
            'AMT_CREDIT': [10000, 10000, 15000]
        })
        
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        
        assert df['PAYMENT_RATE'].iloc[0] == 0.1
        assert df['PAYMENT_RATE'].iloc[1] == 0.2
        assert df['PAYMENT_RATE'].iloc[2] == 0.2
    
    def test_income_credit_perc_calculation(self):
        df = pd.DataFrame({
            'AMT_INCOME_TOTAL': [50000, 100000],
            'AMT_CREDIT': [200000, 200000]
        })
        
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        
        assert df['INCOME_CREDIT_PERC'].iloc[0] == 0.25
        assert df['INCOME_CREDIT_PERC'].iloc[1] == 0.5
    
    def test_income_per_person_calculation(self):
        df = pd.DataFrame({
            'AMT_INCOME_TOTAL': [60000, 120000],
            'CNT_FAM_MEMBERS': [2, 4]
        })
        
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        
        assert df['INCOME_PER_PERSON'].iloc[0] == 30000
        assert df['INCOME_PER_PERSON'].iloc[1] == 30000
    
    def test_days_employed_anomaly_replacement(self):
        df = pd.DataFrame({
            'DAYS_EMPLOYED': [365243, -500, -1000]
        })
        
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        
        assert pd.isna(df['DAYS_EMPLOYED'].iloc[0])
        assert df['DAYS_EMPLOYED'].iloc[1] == -500


class TestInstallmentsFeatures:
    
    def test_dpd_calculation(self):
        df = pd.DataFrame({
            'DAYS_ENTRY_PAYMENT': [-5, -10, -15],
            'DAYS_INSTALMENT': [-10, -10, -10]
        })
        
        df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
        
        assert df['DPD'].iloc[0] == 5
        assert df['DPD'].iloc[1] == 0
        assert df['DPD'].iloc[2] == 0
    
    def test_dbd_calculation(self):
        df = pd.DataFrame({
            'DAYS_ENTRY_PAYMENT': [-15, -10, -5],
            'DAYS_INSTALMENT': [-10, -10, -10]
        })
        
        df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
        df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)
        
        assert df['DBD'].iloc[0] == 5
        assert df['DBD'].iloc[1] == 0
        assert df['DBD'].iloc[2] == 0
    
    def test_payment_diff_calculation(self):
        df = pd.DataFrame({
            'AMT_INSTALMENT': [1000, 1000, 1000],
            'AMT_PAYMENT': [1000, 800, 1200]
        })
        
        df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
        
        assert df['PAYMENT_DIFF'].iloc[0] == 0
        assert df['PAYMENT_DIFF'].iloc[1] == 200
        assert df['PAYMENT_DIFF'].iloc[2] == -200


class TestDataCleaning:
    
    def test_column_name_cleaning(self):
        import re
        
        df = pd.DataFrame({
            'col (test)': [1, 2],
            'col:special': [3, 4],
            'col normal': [5, 6]
        })
        
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        
        assert 'coltest' in df.columns
        assert 'colspecial' in df.columns
        assert 'colnormal' in df.columns
    
    def test_infinite_values_replacement(self):
        df = pd.DataFrame({
            'col1': [1.0, np.inf, -np.inf, 4.0]
        })
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        assert pd.isna(df['col1'].iloc[1])
        assert pd.isna(df['col1'].iloc[2])
        assert df['col1'].iloc[0] == 1.0
        assert df['col1'].iloc[3] == 4.0


class TestThresholdOptimization:
    
    def test_threshold_search(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.9])
        
        thresholds = np.arange(0.1, 1.0, 0.1)
        costs = []
        
        for thr in thresholds:
            y_pred = (y_prob > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_cost = (fn * 10) + (fp * 1)
            costs.append(total_cost)
        
        min_cost = min(costs)
        best_threshold = thresholds[costs.index(min_cost)]
        
        assert min_cost >= 0
        assert 0.1 <= best_threshold <= 0.9
    
    def test_threshold_extreme_low(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        
        threshold = 0.01
        y_pred = (y_prob > threshold).astype(int)
        
        assert y_pred[0] == 1
        assert y_pred[1] == 1
    
    def test_threshold_extreme_high(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        
        threshold = 0.99
        y_pred = (y_prob > threshold).astype(int)
        
        assert y_pred[0] == 0
        assert y_pred[1] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
