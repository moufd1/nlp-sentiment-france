import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Configuration de la page web
st.set_page_config(page_title="Analyse de Sentiment", page_icon="🍿")

st.title("IA d'Analyse de Sentiment")
st.markdown("Cette application devine si une critique de film est **positive** ou **négative**.")

# Entraînement et mise en cache du modèle pour la démo
@st.cache_resource
def load_model():
    dataset = load_dataset("tblard/allocine")
    df_train = pd.DataFrame(dataset['train']).sample(5000, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline.fit(df_train['review'], df_train['label'])
    return pipeline

with st.spinner("Chargement de l'Intelligence Artificielle..."):
    model = load_model()

# Zone de texte pour l'utilisateur
user_input = st.text_area(
    "✍️ Écrivez votre propre critique de film (en français) :", 
    "Le scénario est incroyable, les acteurs jouent parfaitement. Un vrai chef-d'œuvre !"
)

# Bouton de prédiction
if st.button("Analyser le sentiment"):
    if user_input.strip() != "":
        # Prédiction
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]
        
        # Affichage du résultat
        if prediction == 1:
            st.success(f"🟢 L'IA pense que c'est POSITIF (Confiance : {proba[1]:.2%})")
        else:
            st.error(f"🔴 L'IA pense que c'est NÉGATIF (Confiance : {proba[0]:.2%})")
        
        st.markdown("---")
        st.caption("Modèle : TF-IDF + Régression Logistique | Données : Allociné")
    else:
        st.warning("Veuillez entrer du texte.")