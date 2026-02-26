import pickle
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configuration des chemins
MODEL_PATH = 'model_lgbm.pkl'
DATA_PATH = 'data/application_test.csv'

# =========================================================
# 1. CHARGEMENT
# =========================================================
print("⏳ Démarrage de l'API...")

# A. Chargement du Modèle
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Erreur: Le fichier modèle {MODEL_PATH} est introuvable.")
    model = None

# B. Chargement des Données
try:
    # CORRECTION ICI : index_col=0 force Pandas à utiliser la première colonne comme ID
    df = pd.read_csv(DATA_PATH, index_col=0)
    
    # Nettoyage
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
        
    print(f"✅ Données clients chargées ({df.shape[0]} entrées)")
    # Affiche les 5 premiers IDs pour que tu puisses tester
    print(f"👉 IDs disponibles pour le test : {df.index[:5].tolist()}")

except FileNotFoundError:
    print(f"❌ Erreur: Le fichier de données {DATA_PATH} est introuvable.")
    df = None


# =========================================================
# 2. ROUTES
# =========================================================

@app.route('/')
def index():
    return "<h1>API de Scoring Crédit active.</h1><p>Utilisez /predict?id=XXXXXX</p>"

@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.args.get('id')

    # Validation
    if not client_id:
        return jsonify({'error': 'Paramètre "id" manquant.'}), 400

    try:
        client_id = int(client_id)
    except ValueError:
        return jsonify({'error': 'L\'ID doit être un entier.'}), 400

    if model is None or df is None:
        return jsonify({'error': 'Le modèle ou les données ne sont pas chargés.'}), 500

    # Vérification présence ID
    if client_id not in df.index:
        return jsonify({'error': f'Client ID {client_id} non trouvé dans la base de test.'}), 404

    # Prédiction
    try:
        client_data = df.loc[[client_id]]
        
        proba = model.predict_proba(client_data)[:, 1][0]
        
        threshold = 0.50
        decision = "Refusé" if proba > threshold else "Accordé"

        return jsonify({
            'client_id': client_id,
            'probability': round(float(proba), 3),
            'decision': decision,
            'threshold': threshold
        })

    except Exception as e:
        return jsonify({'error': f'Erreur de prédiction : {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)