import pickle
import pandas as pd
from flask import Flask, jsonify, request
import os

app = Flask(__name__)

MODEL_PATH = 'model_lgbm.pkl'
DATA_PATH = 'data/application_test.csv'

# =========================================================
# 1. CHARGEMENT
# =========================================================
print("⏳ Démarrage de l'API...")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé")
except Exception as e:
    print(f"❌ Erreur modèle: {e}")
    model = None

try:
    # On charge sans index_col pour garder SK_ID_CURR comme une colonne normale
    df = pd.read_csv(DATA_PATH)
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    print(f"✅ Données chargées : {df.shape}")
except Exception as e:
    print(f"❌ Erreur données: {e}")
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
    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400

    try:
        id_int = int(client_id)
        # On cherche le client
        client_row = df[df['SK_ID_CURR'] == id_int]
    except Exception:
        return jsonify({"error": "ID invalide"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client {client_id} non trouvé"}), 404

    try:
        # Nettoyage des colonnes texte
        client_data_clean = client_row.select_dtypes(exclude=['object'])
        
        # Suppression de l'ID pour ne pas polluer la prédiction
        if 'SK_ID_CURR' in client_data_clean.columns:
            client_data_clean = client_data_clean.drop(columns=['SK_ID_CURR'])

        # Prédiction
        probability = model.predict_proba(client_data_clean)[0][1]
        
        # Seuil métier (0.5 ou celui que tu as optimisé)
        threshold = 0.5
        decision = "Refusé" if probability > threshold else "Accordé"

        return jsonify({
            "status": "success",
            "client_id": id_int,
            "probability": float(probability),
            "decision": decision,
            "threshold": threshold
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Indispensable pour Render : utiliser le port défini par l'environnement
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)