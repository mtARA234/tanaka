
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ===============================
# FINGERPRINT GENERATORS
# ===============================
FP_SIZE = 2048
RADIUS = 2
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan.GetFingerprint(mol))

def combine_fp(drug_smiles, excipient_smiles):
    fp1 = smiles_to_fp(drug_smiles)
    fp2 = smiles_to_fp(excipient_smiles)
    if fp1 is None or fp2 is None:
        return None
    return np.concatenate([fp1, fp2])  # 2048 + 2048 = 4096 features

# ===============================
# LOAD MODELS
# ===============================
# IC50 classifiers/regressors per target
targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]
clf_models = {t: joblib.load(f"{t}_clf.pkl") for t in targets}
reg_models = {t: joblib.load(f"{t}_reg.pkl") for t in targets}

# Toxicity model
tox_model = joblib.load("tox_model.pkl")

# Compatibility model
compat_model = joblib.load("compatibility_model.pkl")

# ===============================
# APP UI
# ===============================
st.title("🧪 AI Drug Discovery Platform")

smiles_input = st.text_input("Enter Drug SMILES:")
targets_selected = st.multiselect("Select targets to predict IC50:", targets)
predict_tox = st.checkbox("Predict Toxicity?")
predict_comp = st.checkbox("Predict Drug-Excipient Compatibility?")

excipient_options = ["Lactose", "Mannitol", "Starch", "Sucrose"]
selected_excipient = None
if predict_comp:
    selected_excipient = st.selectbox("Select Excipient:", excipient_options)

if st.button("Run Prediction"):

    if not smiles_input:
        st.error("Please enter a SMILES string!")
    else:
        fp = smiles_to_fp(smiles_input)
        if fp is None:
            st.error("Invalid SMILES!")
        else:
            st.subheader("📊 IC50 Predictions")
            results_ic50 = {}
            for t in targets_selected:
                clf = clf_models[t]
                reg = reg_models[t]

                # Classification
                active_prob = clf.predict_proba(fp.reshape(1, -1))[0][1]
                is_active = active_prob > 0.5

                # Regression (only if predicted active)
                if is_active:
                    pic50 = reg.predict(fp.reshape(1, -1))[0]
                    ic50 = 10 ** (-pic50) * 1e9  # convert from molar to nM
                    results_ic50[t] = {
                        "Active": True,
                        "pIC50": round(pic50,3),
                        "IC50(nM)": round(ic50,2),
                        "Confidence": round(active_prob,2)
                    }
                else:
                    results_ic50[t] = {
                        "Active": False,
                        "Confidence": round(active_prob,2)
                    }

            st.json(results_ic50)

            # ===============================
            # Toxicity
            # ===============================
            if predict_tox:
                tox_pred = tox_model.predict(fp.reshape(1,-1))[0]
                st.subheader("⚠️ Toxicity Prediction")
                st.write("Toxic" if tox_pred else "Non-toxic")

            # ===============================
            # Compatibility
            # ===============================
            if predict_comp:
                comp_fp = combine_fp(smiles_input, selected_excipient)
                if comp_fp is None:
                    st.error("Invalid SMILES for drug or excipient")
                else:
                    comp_pred = compat_model.predict(comp_fp.reshape(1,-1))[0]
                    st.subheader("🧩 Drug-Excipient Compatibility")
                    st.write("Compatible" if comp_pred else "Incompatible")

                  