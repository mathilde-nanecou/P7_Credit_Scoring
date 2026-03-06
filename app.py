import pickle
import pandas as pd
from flask import Flask, jsonify, request
import os

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
    print(f"✅ Modèle chargé")
except Exception as e:
    print(f"❌ Erreur modèle: {e}")
    model = None

# B. Chargement des Données
try:
    df = pd.read_csv(DATA_PATH)
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    print(f"✅ Données clients chargées ({df.shape[0]} entrées)")
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
        client_row = df[df['SK_ID_CURR'] == id_int]
    except ValueError:
        return jsonify({"error": "ID client doit être un nombre"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client ID {client_id} non trouvé"}), 404

    try:
        # 1. NETTOYAGE : Suppression des colonnes texte
        client_data_clean = client_row.select_dtypes(exclude=['object'])
        
        # 2. ALIGNEMENT : On force les données à avoir les 261 colonnes du modèle
        # On récupère la liste des colonnes attendues directement depuis le modèle chargé
        expected_features = model.feature_name_
        
        # .reindex() va garder les bonnes colonnes et mettre 0 si une colonne manque
        client_data_final = client_data_clean.reindex(columns=expected_features, fill_value=0)

        # 3. PRÉDICTION : On utilise maintenant le DataFrame filtré à 261 colonnes
        probability = model.predict_proba(client_data_final)[0][1]
        
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
        # En cas d'erreur, on affiche le message pour comprendre si besoin
        return jsonify({"error": f"Erreur de prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)