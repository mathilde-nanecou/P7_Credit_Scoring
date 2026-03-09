# 🏦 Système de Scoring Crédit - "Prêt à dépenser"

![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![Flask](https://img.shields.io/badge/flask-2.0-black.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.20-red.svg) ![Render](https://img.shields.io/badge/deployment-live-green.svg) ![MLFlow](https://img.shields.io/badge/MLFlow-tracking-orange.svg)

## 📌 Présentation du Projet
Ce projet a été réalisé pour la société financière **"Prêt à dépenser"**, qui propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. 

**Mission :** Développer un algorithme de classification automatique pour calculer la probabilité de faillite d'un client et fournir une décision (Accord/Refus) transparente et explicable.



---

## 🏗️ Architecture du Système
L'application repose sur une architecture découplée pour garantir la flexibilité et la mise à l'échelle :

* **Backend (API Rest) :** Construit avec **Flask**, déployé sur **Render**. Il gère la logique de prédiction et l'accès au modèle.
* **Frontend (Dashboard) :** Interface **Streamlit** permettant aux chargés de relation client d'analyser les dossiers de manière interactive.
* **Modèle :** Algorithme **LightGBM** optimisé.

---

## 📊 Méthodologie Data Science

### 1. Préparation des Données & Feature Engineering
* Utilisation de sources de données variées (données comportementales, historiques bancaires).
* Nettoyage et agrégation des tables (`application`, `bureau`, `previous_application`).
* Traitement des valeurs manquantes et des outliers (valeurs infinies).

### 2. Modélisation et Optimisation
* **Algorithme :** LightGBM pour sa performance sur les grands jeux de données et sa gestion native des valeurs manquantes.
* **Gestion du déséquilibre :** Application de la stratégie `class_weight='balanced'` pour compenser la faible proportion de clients en défaut (Target=1).
* **Tracking :** Utilisation de **MLFlow** pour comparer les performances des différents modèles (Baseline, RandomForest, LightGBM).

### 3. Score Métier (Fonction de Coût)
Pour ce projet, une erreur de prédiction n'a pas le même coût selon sa nature :
* **Faux Négatif (FN) :** Prédire qu'un client va rembourser alors qu'il fait défaut (coût élevé : 10).
* **Faux Positif (FP) :** Prédire qu'un client fera défaut alors qu'il est sain (coût faible : 1).
Le seuil de classification a été optimisé pour minimiser ce coût métier global.

---

## 🧪 Qualité du Code et MLOps

### 🛠️ Tests Unitaires (PyTest)
Une suite de tests automatisés assure la fiabilité du code avant chaque déploiement :
* **Logique Métier :** Vérification des calculs de ratios financiers (ex: *Payment Rate*).
* **Prétraitement :** Validation du nettoyage des caractères spéciaux dans les noms de colonnes.
* **API :** Tests d'intégration sur les routes `/` et `/predict` (gestion des IDs valides, invalides et manquants).

### 🔄 CI/CD (GitHub Actions)
Mise en place d'un pipeline d'intégration continue :
1.  Déclenchement automatique à chaque `push`.
2.  Installation de l'environnement virtuel.
3.  Exécution des tests unitaires.
4.  Déploiement automatique vers **Render** si les tests sont validés.



### 📈 Analyse du Data Drift
Analyse de la stabilité du modèle entre l'entraînement et la production via la librairie **Evidently AI**. Un rapport HTML a été généré pour surveiller le glissement des variables critiques (ex: `AMT_INCOME_TOTAL`, `EXT_SOURCE_1`).

---

## 🖥️ Utilisation du Dashboard

Le dashboard Streamlit permet :
1.  **Visualisation Globale :** Importance des variables au niveau du modèle complet.
2.  **Analyse Locale :** Probabilité de défaut pour un client spécifique avec jauge de risque.
3.  **Interprétabilité :** Graphiques SHAP simplifiés expliquant l'impact de chaque variable (revenu, âge, etc.) sur la décision finale.
4.  **Comparaison :** Positionnement du client par rapport à l'ensemble de la base de données.

---

## ⚙️ Installation Locale

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/votre-repo/projet-scoring-credit.git](https://github.com/votre-repo/projet-scoring-credit.git)
    ```
2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Lancer le Dashboard :**
    ```bash
    streamlit run dashboard.py
    ```

---

## 🔗 Liens et Livrables
* **API Live :** [https://api-scoring-mathilde.onrender.com/](https://api-scoring-mathilde.onrender.com/)
* **Modèle :** `model_lgbm.pkl`
* **Note Méthodologique :** Disponible dans le dossier `docs/`.

---
*Projet réalisé par Mathilde Nanécou - Parcours Data Scientist @ OpenClassrooms*