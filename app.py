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
    # 1. Récupération de l'ID depuis l'URL (ex: ?id=100001)
    client_id = request.args.get('id')
    
    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400

    # 2. Recherche du client dans le fichier application_test.csv
    try:
        # On force la conversion en entier pour la comparaison
        id_int = int(client_id)
        client_row = df[df['SK_ID_CURR'] == id_int]
    except ValueError:
        return jsonify({"error": "ID client doit être un nombre"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client ID {client_id} non trouvé dans la base de test."}), 404

    try:
        # 3. NETTOYAGE CRUCIAL : On prépare les données pour le modèle
        # On ne garde que les colonnes numériques (int, float, bool)
        client_data_clean = client_row.select_dtypes(exclude=['object'])
        
        # On supprime l'ID car il ne doit pas être une "feature" pour la prédiction
        if 'SK_ID_CURR' in client_data_clean.columns:
            client_data_clean = client_data_clean.drop(columns=['SK_ID_CURR'])

        # 4. CALCUL DE LA PRÉDICTION
        # model.predict_proba renvoie [[proba_classe_0, proba_classe_1]]
        # On récupère la probabilité de défaut (classe 1)
        probability = model.predict_proba(client_data_clean)[0][1]
        
        # Définition du seuil (à ajuster selon tes besoins métier)
        threshold = 0.5
        decision = "Refusé" if probability > threshold else "Accordé"

        # 5. RETOUR DES RÉSULTATS EN JSON
        return jsonify({
            "status": "success",
            "client_id": id_int,
            "probability": float(probability),
            "decision": decision,
            "threshold": threshold
        })

    except Exception as e:
        # En cas d'erreur interne (modèle manquant, colonnes incompatibles, etc.)
        return jsonify({"error": f"Erreur de prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)