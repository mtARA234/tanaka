
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import shap
import matplotlib.pyplot as plt

# ===============================
# Setup Morgan Fingerprint
# ===============================
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(morgan.GetFingerprint(mol))
    return None

def calculate_druglikeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    # Lipinski Rule of 5
    lipinski = all([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
    return {
        'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba, 'TPSA': tpsa, 'Lipinski': lipinski
    }

# ===============================
# Load Models
# ===============================
targets = ["SERT","DAT","D2","D3","D4","5HT1A","5HT6","5HT7"]
models = {}
for t in targets:
    models[t] = {
        'clf': joblib.load(f'{t}_clf.pkl'),
        'reg': joblib.load(f'{t}_reg.pkl')
    }

# Toxicity & Compatibility
tox_model = joblib.load('tox_model.pkl')
comp_model = joblib.load('compatibility_model.pkl')

# ===============================
# Prediction Functions
# ===============================
def predict_ic50(smiles, selected_targets):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return None
    results = {}
    for t in selected_targets:
        clf = models[t]['clf']
        reg = models[t]['reg']
        prob_active = clf.predict_proba(fp.reshape(1,-1))[0][1]
        active = prob_active > 0.5
        if active:
            pIC50 = reg.predict(fp.reshape(1,-1))[0]
            IC50 = 10**(-pIC50) * 1e9  # nM
            results[t] = {'Active': True, 'pIC50': pIC50, 'IC50': IC50, 'Confidence': prob_active}
        else:
            results[t] = {'Active': False, 'Confidence': prob_active}
    return results

def predict_toxicity(smiles):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return None
    pred = tox_model.predict(fp.reshape(1,-1))[0]
    return 'Toxic' if pred == 1 else 'Non-Toxic'

def predict_compatibility(smiles, excipient):
    fp = smiles_to_fp(smiles)
    # Here we assume excipient encoded as categorical (needs preprocessing if multiple)
    X = np.append(fp, hash(excipient)%1000)  # simple encoding
    pred = comp_model.predict(X.reshape(1,-1))[0]
    return 'Compatible' if pred == 1 else 'Incompatible'

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title='AI Drug Discovery', layout='wide')
st.title('🧪 AI Drug Discovery SaaS Platform')

smiles_input = st.text_input('Enter Drug SMILES')
selected_targets = st.multiselect('Select Targets', targets, default=['SERT','DAT'])

toxicity_toggle = st.checkbox('Predict Toxicity', value=True)
compat_toggle = st.checkbox('Check API-Excipient Compatibility', value=True)
excipient_input = None
if compat_toggle:
    excipient_input = st.text_input('Enter Excipient')

batch_file = st.file_uploader('Or upload CSV (column: SMILES) for batch prediction', type=['csv'])

if st.button('Predict'):
    if batch_file:
        df = pd.read_csv(batch_file)
        all_results = []
        for smi in df['SMILES']:
            res = predict_ic50(smi, selected_targets)
            tox = predict_toxicity(smi) if toxicity_toggle else None
            comp = predict_compatibility(smi, excipient_input) if compat_toggle else None
            druglik = calculate_druglikeness(smi)
            all_results.append({'SMILES': smi, 'IC50': res, 'Toxicity': tox, 'Compatibility': comp, **(druglik or {})})
        st.write(pd.DataFrame(all_results))
    else:
        if not smiles_input:
            st.warning('Please provide a SMILES string')
        else:
            res = predict_ic50(smiles_input, selected_targets)
            tox = predict_toxicity(smiles_input) if toxicity_toggle else None
            comp = predict_compatibility(smiles_input, excipient_input) if compat_toggle else None
            druglik = calculate_druglikeness(smiles_input)
            st.write('### IC50 Predictions')
            st.write(res)
            if toxicity_toggle:
                st.write('### Toxicity Prediction')
                st.write(tox)
            if compat_toggle:
                st.write('### API-Excipient Compatibility')
                st.write(comp)
            if druglik:
                st.write('### Drug-Likeness')
                st.write(druglik)

# Optional: SHAP explainability
# Add shap plots for one target
if st.checkbox('Show Explainability for SERT'):
    sample_fp = smiles_to_fp(smiles_input).reshape(1,-1)
    explainer = shap.TreeExplainer(models['SERT']['reg'])
    shap_values = explainer.shap_values(sample_fp)
    st.write('SHAP summary for SERT regression')
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_fp, show=False)
    st.pyplot(fig)
    