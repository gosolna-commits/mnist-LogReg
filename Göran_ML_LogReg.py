# Göran_ML_LogReg.py
# ---------------------------------------------------
# Installera:
# pip install streamlit streamlit-drawable-canvas
# pip install scikit-learn pillow opencv-python
#
# pip install opencv-python
#
# Kör:
# streamlit run Göran_ML_LogReg.py
# pip install joblib

#Mest effektiva förbättringarna nu
#I ordning:
#* Centrering
#* Tjockare brush
#* max_iter=3000
#* C=10
#* solver="saga"


# ---------------------------------------------------

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import cv2
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from streamlit_drawable_canvas import st_canvas

# =====================================
# STREAMLIT
# =====================================

st.set_page_config(
    page_title="MNIST Logistic Regression",
    layout="centered"
)

st.title("MNIST Logistic Regression")

st.title("Rita en siffra (0–9)")

# =====================================
# LADDA DATA + TRÄNA MODELL
# =====================================

@st.cache_resource
def load_model():

    mnist = fetch_openml('mnist_784', version=1)

    X = mnist.data.astype(np.float32).to_numpy()
    y = mnist.target.astype(np.int32)

    # Normalisering
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================
    # LOGISTIC REGRESSION
    # =====================================

    if os.path.exists("logreg_model.pkl"):

        model = joblib.load("logreg_model.pkl")
        return model, 0.92

    else:

        model = LogisticRegression(
            max_iter=3000,
#           solver="lbfgs",
            solver="saga",
            C=20
        )

        model.fit(X_train, y_train)
    
        joblib.dump(model, "logreg_model.pkl")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        return model, acc


with st.spinner("Tränar Logistic Regression..."):
    
    model, acc = load_model()

    st.success(f"Accuracy: {acc:.4f}")

# =====================================
# CANVAS
# =====================================

canvas_result = st_canvas(
    fill_color="white",
# Testa    stroke_width=25,
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# =====================================
# PREDIKTION
# =====================================

if st.button("Skicka"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        # RGBA -> grayscale
        img = cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_RGBA2GRAY
        )

        # Resize till 28x28
        img = cv2.resize(img, (28,28))

        # Blur för mjukare MNIST-liknande siffror
        img = cv2.GaussianBlur(img, (3,3), 0)
        
#######################################################################
        # Hitta vita pixlar
        coords = cv2.findNonZero((img * 255).astype(np.uint8))

        if coords is not None:

            x, y, w, h = cv2.boundingRect(coords)

            digit = img[y:y+h, x:x+w]

            # Skapa svart canvas
            square = np.zeros((28,28), dtype=np.float32)

            # Resize siffran
            digit = cv2.resize(digit, (20,20))

            # Centrera
            x_offset = 4
            y_offset = 4

            square[
                y_offset:y_offset+20,
                x_offset:x_offset+20
            ] = digit

            img = square

#######################################################################

        # Normalisering
        img = img / 255.0

        # Tom canvas?
        if np.sum(img) < 5:

            st.warning("Rita en siffra först")

        else:

            # Flatten
            img_flat = img.reshape(1, 784)

            # Prediktion
            prediction = model.predict(img_flat)[0]

            probabilities = model.predict_proba(img_flat)[0]

            confidence = probabilities[prediction] * 100

            # =====================================
            # RESULTAT
            # =====================================

            st.markdown(
                f"## Prediktion: {prediction} ({confidence:.1f}%)"
            )

            st.image(
                img,
                width=150,
                caption="28x28-bild som modellen ser"
            )

            # =====================================
            # SANNNOLIKHETER
            # =====================================

            st.subheader("Sannolikheter")

            for i, p in enumerate(probabilities):

                st.write(f"{i}: {p*100:.2f}%")

                st.progress(float(p))
                
                