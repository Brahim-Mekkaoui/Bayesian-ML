import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score

import streamlit as st

st.set_page_config(
    page_title="Credit Scoring Bayésien",
    page_icon="💳",
    layout="wide",
)

st.title("Interface interactive de scoring de crédit")
st.markdown(
    "Testez un profil client en temps réel et obtenez une estimation de risque de défaut basée sur un modèle Naïve Bayes gaussien."
)

@st.cache_data
def charger_donnees(path: str):
    data = pd.read_csv(path)
    data.drop("ID", axis=1, inplace=True)
    return data

@st.cache_data
def entrainer_modele(data: pd.DataFrame):
    X = data.drop("default.payment.next.month", axis=1)
    y = data["default.payment.next.month"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    performance = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "class_ratio_defaut": y.mean(),
    }
    return model, performance, X.columns.tolist(), X.median().to_dict()

def predire(_model, colonnes, profil: dict):
    profil_df = pd.DataFrame([profil], columns=colonnes)
    classe = _model.predict(profil_df)[0]
    proba = _model.predict_proba(profil_df)[0, 1]
    return classe, proba

# Chargement et entraînement
path_data = "UCI_Credit_Card.csv"
try:
    data = charger_donnees(path_data)
except FileNotFoundError:
    st.error(f"Fichier non trouvé : {path_data}. Vérifiez que le CSV est dans le dossier du script.")
    st.stop()

model, performance, colonnes, valeurs_par_defaut = entrainer_modele(data)

with st.sidebar:
    st.header("Profil du client")
    sex = st.selectbox("SEX", options=[1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme", index=0)
    education = st.selectbox(
        "EDUCATION",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate school",
            2: "University",
            3: "High school",
            4: "Other",
        }[x],
        index=min(int(valeurs_par_defaut["EDUCATION"]) - 1, 3),
    )
    marriage = st.selectbox(
        "MARRIAGE",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Inconnu",
            1: "Marié",
            2: "Célibataire",
            3: "Autre",
        }[x],
        index=min(int(valeurs_par_defaut["MARRIAGE"]), 3),
    )
    age = st.slider("AGE", min_value=18, max_value=90, value=int(valeurs_par_defaut["AGE"]))
    limit_bal = st.slider(
        "LIMIT_BAL", min_value=10000, max_value=1000000, value=int(valeurs_par_defaut["LIMIT_BAL"]), step=1000
    )
    bill_amt1 = st.slider(
        "BILL_AMT1", min_value=0, max_value=200000, value=int(valeurs_par_defaut["BILL_AMT1"]), step=1000
    )
    pay_amt1 = st.slider(
        "PAY_AMT1", min_value=0, max_value=200000, value=int(valeurs_par_defaut["PAY_AMT1"]), step=1000
    )

    st.subheader("Historique de paiement")
    pay_0 = st.slider("PAY_0", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_0"]), step=1)
    pay_2 = st.slider("PAY_2", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_2"]), step=1)
    pay_3 = st.slider("PAY_3", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_3"]), step=1)
    pay_4 = st.slider("PAY_4", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_4"]), step=1)
    pay_5 = st.slider("PAY_5", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_5"]), step=1)
    pay_6 = st.slider("PAY_6", min_value=-2, max_value=8, value=int(valeurs_par_defaut["PAY_6"]), step=1)

st.header("Résumé du modèle")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{performance['accuracy']:.2%}")
col2.metric("ROC AUC", f"{performance['roc_auc']:.3f}")
col3.metric("Part des défauts", f"{performance['class_ratio_defaut']:.2%}")

profil = {
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    "LIMIT_BAL": limit_bal,
    "PAY_0": pay_0,
    "PAY_2": pay_2,
    "PAY_3": pay_3,
    "PAY_4": pay_4,
    "PAY_5": pay_5,
    "PAY_6": pay_6,
    "BILL_AMT1": bill_amt1,
    "PAY_AMT1": pay_amt1,
    # Utilisation des valeurs par défaut pour les autres colonnes
}
for colonne in colonnes:
    if colonne not in profil:
        profil[colonne] = valeurs_par_defaut[colonne]

if st.button("Évaluer le profil"):
    classe, proba = predire(model, colonnes, profil)
    niveau = "Élevé" if proba >= 0.5 else "Faible"

    st.subheader("Résultat prédictif")
    st.write(f"**Probabilité de défaut estimée :** {proba:.2%}")
    st.write(f"**Prédiction :** {'Défaut' if classe == 1 else 'Pas de défaut'}")
    st.write(f"**Niveau de risque :** {niveau}")

    st.markdown("---")
    st.write(
        "Le modèle utilisé est un GaussianNB. Les variables de paiement retardé (`PAY_*`) ont un fort impact sur le score de risque."
    )
else:
    st.info("Modifiez les valeurs du profil client dans la barre latérale puis cliquez sur 'Évaluer le profil'.")

st.sidebar.markdown("---")
st.sidebar.write(
    "Ce prototype réentraîne le modèle depuis les données CSV et permet d’explorer facilement l’impact des variables sociodémographiques et de l’historique de paiement."
)
