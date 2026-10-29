"""Interactive transaction anomaly-detection learning project.

Author: Moe Htet Ar Kar (Phoe Cho)

The model is trained on synthetic data. An anomaly is not proof of fraud.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title = "Financial Anomaly Monitor", page_icon = "💹", layout = "wide")
st.markdown("<style>" + """.stApp{background:linear-gradient(145deg,#0a1515 0%,#102625 55%,#0a1718 100%);color:#e8fff8;} [data-testid='stSidebar']{background:#102321;} 
[data-testid='stMetric']{background:#15332e;border:1px solid #2a6555;padding:14px;border-radius:12px;} .stButton>button{background:#48d6a2;
color:#09231b;border:0;border-radius:9px;font-weight:700;} h1,h2,h3{color:#eafff7;}""" + "</style>", unsafe_allow_html = True)

st.title("💹 Financial Risk & Anomaly Monitor")
st.caption("Transaction feature review · Unsupervised Isolation Forest")
st.warning("This demo uses synthetic data. An unusual transaction is not confirmed fraud, and a normal result does not prove legitimacy.")

FEATURES = ["Transaction Amount ($)", "Transactions in Last Hour", "Account Age (Days)", "Distance from Home (km)"]


@st.cache_resource
def train_anomaly_model():

    """Fit an Isolation Forest using reproducible synthetic examples."""
    rng = np.random.default_rng(42)
    sample_count = 1000
    training = pd.DataFrame(
        {
            FEATURES[0]: rng.exponential(scale = 300.0, size = sample_count),
            FEATURES[1]: rng.poisson(lam = 3, size = sample_count),
            FEATURES[2]: rng.integers(30, 3650, size = sample_count),
            FEATURES[3]: rng.exponential(scale = 20.0, size = sample_count),
        }
    )
    model = IsolationForest(contamination = 0.05, random_state = 42)
    model.fit(training[FEATURES])
    return model


model = train_anomaly_model()
st.sidebar.header("Transaction parameters")
amount = st.sidebar.number_input("Transaction amount ($)", min_value = 1.0, max_value = 100000.0, value = 250.0, step = 25.0)
velocity = st.sidebar.slider("Transactions in the last hour", 1, 50, 2)
account_age = st.sidebar.slider("Account age (days)", 1, 3650, 365)
geo_distance = st.sidebar.slider("Distance from home (km)", 0, 5000, 10)

transaction = pd.DataFrame(
    [[amount, velocity, account_age, geo_distance]],
    columns=FEATURES,
)

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Transaction feature review")
    st.dataframe(transaction, use_container_width = True, hide_index = True)
    st.caption("The values above are passed to the model in the same feature order used during training.")
with right:
    st.subheader("Model verdict")
    if st.button("Evaluate transaction", use_container_width = True):
        prediction = int(model.predict(transaction[FEATURES])[0])
        decision_score = float(model.decision_function(transaction[FEATURES])[0])
        st.metric("Decision-function score", f"{decision_score:.4f}")

        if prediction == -1:
            st.error("ANOMALY FLAGGED — model considers this row unusual.")
            st.write("Suggested demo workflow: send the example for human review. This does not establish fraud.")
        else:
            st.success("NO ANOMALY FLAG — model did not mark this row as unusual.")
            st.write("This result does not prove the transaction is legitimate or risk-free.")
        with st.expander("How to read the score"):
            st.write("Isolation Forest predicts -1 for an outlier and 1 for an inlier. Lower decision-function values generally indicate more anomalous observations.")
            st.write("Model training source: 1,000 synthetic transaction examples; no fraud labels are used.")
    else:
        st.info("Adjust the transaction parameters and click **Evaluate transaction**.")

st.divider()
st.caption("Educational fintech ML prototype · Synthetic data · Not a production fraud detector")
