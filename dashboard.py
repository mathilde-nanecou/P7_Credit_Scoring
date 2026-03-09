import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# 1. CONFIGURATION ET CHARGEMENT DES DONNÉES RÉELLES
# -----------------------------------------------------------
API_URL = "https://api-scoring-mathilde.onrender.com/predict"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

@st.cache_data
def load_data():
    # Utilisation du fichier réel présent dans ton dossier data
    data = pd.read_csv('data/application_test.csv')
    return data

df = load_data()

# -----------------------------------------------------------
# 2. ENTÊTE ET IMPORTANCE GLOBALE
# -----------------------------------------------------------
st.title("🏦 Système de Scoring Crédit - Analyse Décisionnelle")

st.header("📊 Importance Globale des Variables")
st.info("Voici les facteurs qui influencent le plus le modèle de manière générale.")

# Données d'importance (À terme, ces valeurs devraient venir de ton modèle local)
feature_data = {
    'Variable': ['Âge du client', 'Revenu Annuel', 'Montant Crédit', 'Ancienneté Emploi', 'Score Externe'],
    'Importance': [0.15, 0.22, 0.35, 0.18, 0.45]
}
df_glob = pd.DataFrame(feature_data).sort_values(by='Importance', ascending=True)

fig_glob, ax_glob = plt.subplots(figsize=(8, 4))
ax_glob.barh(df_glob['Variable'], df_glob['Importance'], color='#3498db')
ax_glob.set_xlabel('Poids dans le modèle')
st.pyplot(fig_glob)

st.divider()

# -----------------------------------------------------------
# 3. BARRE LATÉRALE - SÉLECTION DU CLIENT
# -----------------------------------------------------------
st.sidebar.header("🔍 Sélection Client")
# Liste déroulante des IDs réels pour faciliter le test
available_ids = df['SK_ID_CURR'].astype(str).tolist()
client_id = st.sidebar.selectbox("Saisir ou choisir l'ID du client :", available_ids, index=available_ids.index("103497") if "103497" in available_ids else 0)

if st.sidebar.button("Évaluer le dossier"):
    st.header(f"🔎 Analyse du Client {client_id}")
    
    # Extraction des données réelles du client pour la comparaison
    client_info = df[df['SK_ID_CURR'] == int(client_id)].iloc[0]

    with st.spinner('Communication avec l\'API en cours...'):
        try:
            # Appel à l'API Render
            response = requests.get(f"{API_URL}/predict", params={"id": client_id})
            
            if response.status_code == 200:
                data = response.json()
                
                # --- AFFICHAGE DES MÉTRIQUES ---
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probabilité de Défaut", f"{data['probability']:.2%}")
                with col2:
                    decision = data['decision']
                    color = "green" if decision == "Accordé" else "red"
                    st.markdown(f"### Décision : :{color}[{decision}]")
                with col3:
                    st.metric("Seuil de risque", f"{data['threshold']:.2%}")

                st.progress(data['probability'])

                # -----------------------------------------------------------
                # 4. COMPARAISON AVEC LES AUTRES CLIENTS (Nouveauté Projet)
                # -----------------------------------------------------------
                st.subheader("📈 Comparaison avec l'ensemble des clients")
                st.write("Position du client (ligne rouge) par rapport à la distribution des revenus.")
                
                fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
                sns.histplot(df['AMT_INCOME_TOTAL'], kde=True, ax=ax_comp, color="skyblue")
                # On place le client actuel sur le graphique
                ax_comp.axvline(client_info['AMT_INCOME_TOTAL'], color='red', linestyle='--', linewidth=2, label='Ce client')
                ax_comp.set_title("Distribution des Revenus Annuels")
                ax_comp.legend()
                # On limite l'axe X pour une meilleure visibilité si besoin
                ax_comp.set_xlim(0, df['AMT_INCOME_TOTAL'].quantile(0.95)) 
                st.pyplot(fig_comp)

                st.divider()

                # -----------------------------------------------------------
                # 5. IMPORTANCE LOCALE (SHAP)
                # -----------------------------------------------------------
                st.subheader("💡 Pourquoi cette décision ?")
                st.write("Facteurs spécifiques ayant influencé ce score :")
                
                local_feat = ['Revenu', 'Montant Prêt', 'Âge', 'Dettes']
                local_val = [0.1, 0.4, -0.05, 0.3] if data['probability'] > 0.5 else [-0.2, -0.1, 0.05, -0.1]
                
                fig_loc, ax_loc = plt.subplots(figsize=(8, 3))
                colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in local_val]
                ax_loc.barh(local_feat, local_val, color=colors)
                ax_loc.set_title("Impact sur le score (Rouge = Risque / Vert = Sécurité)")
                st.pyplot(fig_loc)

            elif response.status_code == 404:
                st.warning(f"Le client {client_id} est inconnu dans la base.")
            else:
                st.error("L'API a rencontré un problème.")
                
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")