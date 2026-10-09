# Analyse de Sentiment (NLP) - Textes en Français

Ce projet est une pipeline complète de Machine Learning & Deep Learning (MLOps) visant à classifier des critiques de films (Allociné) en sentiments positifs ou négatifs.

## Architecture du projet
* **Données** : 100 000 critiques issues du dataset Allociné (Hugging Face `datasets`).
* **Baseline ML** : TF-IDF + Logistic Regression (Scikit-Learn).
* **Deep Learning** : Fine-tuning de **CamemBERT** avec PyTorch et l'API `transformers`.
* **MLOps** : Suivi des expériences, métriques et modèles via **MLflow**.
* **Déploiement** : Interface web interactive avec **Streamlit**.

## ⚙️ Installation locale