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
    
    # 1. On récupère la ligne du client
    client_data = df[df['SK_ID_CURR'] == int(client_id)]
    
    if client_data.empty:
        return jsonify({"error": "Client non trouvé"}), 404

    try:
        # 2. NETTOYAGE : On ne garde que les colonnes numériques
        # Le modèle plante car il voit des colonnes "objet" (texte)
        client_data_clean = client_data.select_dtypes(exclude=['object'])
        
        # 3. On enlève aussi l'ID qui ne doit pas servir à la prédiction
        if 'SK_ID_CURR' in client_data_clean.columns:
            client_data_clean = client_data_clean.drop(columns=['SK_ID_CURR'])

        # 4. Prédiction
        probability = model.predict_proba(client_data_clean)[0][1]
        
        # ... la suite de ton code (seuil, décision, etc.) ...
        return jsonify({
            "probability": float(probability),
            "decision": "Refusé" if probability > 0.5 else "Accordé",
            "threshold": 0.5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)