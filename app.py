import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
}

.stButton>button {
    width:100%;
    border-radius:10px;
    height:3em;
    font-size:16px;
}

.card {
    background-color:#1c1f26;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
model = joblib.load("hardware_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

knn_model = joblib.load("knn_recommender.pkl")
tfidf_rec = joblib.load("tfidf_recommender.pkl")
df = joblib.load("hardware_dataset.pkl")

encoder = joblib.load("category_encoder.pkl")

# -----------------------------
# CREATE PRODUCT NAME
# -----------------------------
df["product_name"] = (
    df["CPU"]
    .fillna(df["GPU"])
    .fillna(df["motherBoard"])
    .fillna(df["Ram"])
    .fillna(df["SSD"])
    .fillna(df["PowerSupply"])
    .fillna(df["cabinates"])
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<p class="title">💻 PC Hardware Recommendation System</p>', unsafe_allow_html=True)

# -----------------------------
# SELECT HARDWARE
# -----------------------------
hardware = st.selectbox(
    "Select Hardware",
    df["product_name"].dropna().unique()
)

# -----------------------------
# CATEGORY PREDICTION
# -----------------------------
if st.button("Predict Category"):

    X = tfidf.transform([hardware])

    prediction = model.predict(X)

    category = encoder.inverse_transform(prediction)

    st.success(f"💻 Predicted Category: {category[0]}")

# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
# -----------------------------
# RECOMMENDATION SYSTEM
# -----------------------------
if st.button("Recommend Similar Hardware"):

    st.subheader("⭐ Recommended Hardware")

    df_unique = df.drop_duplicates(subset="product_name").reset_index(drop=True)

    X_all = tfidf_rec.transform(df_unique["product_name"])

    idx = df_unique[df_unique["product_name"] == hardware].index[0]

    distances, indices = knn_model.kneighbors(X_all[idx])

    recommended = []

    for i in indices[0]:

        name = df_unique.iloc[i]["product_name"]
        price = df_unique.iloc[i]["price"]

        if name != hardware and name not in [r[0] for r in recommended]:
            recommended.append((name, price))

    for item, price in recommended[:5]:

        st.markdown(f"""
        <div class="card">
        💻 <b>{item}</b><br>
        💰 Price: ₹{price}
        </div>
        """, unsafe_allow_html=True)
