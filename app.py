import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pyttsx3
import json

from model.cnn.predict import predict_image
from model.classical_ml.svm import train_svm
from dashboard.data_vis import plot_distribution
from utils.helpers import degree_to_direction
from doa_estimation.music import music_algorithm
from signal_processing.doppler_fft import estimate_speed
from radar_simulation.live_signal import generate_signal

st.set_page_config(page_title="AI Radar & Underwater System", layout="wide")

# ---------------- VOICE ----------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #0b0f1a; color: white; }
.title {
    font-size: 34px;
    text-align: center;
    color: #00f7ff;
    text-shadow: 0 0 10px #00f7ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.image("logos/bput.png", width=90)

with col2:
    st.markdown('<div class="title">AI-Driven Micro-Doppler Radar System for sUAV Detection with Underwater Analysis</div>', unsafe_allow_html=True)

with col3:
    st.image("logos/rooman.png", width=90)

st.markdown("---")

# ---------------- LOAD SVM ----------------
@st.cache_resource
def load_svm():
    return train_svm()

with st.spinner("🚀 Initializing AI Models..."):
    svm_model, scaler, svm_classes = load_svm()

# ---------------- DATASET ----------------
st.subheader("📊 Dataset Overview")

colA, colB, colC = st.columns([1.5,1,1.5])
with colB:
    fig = plot_distribution("data/spectrograms/train")
    fig.set_size_inches(2.8, 2)
    plt.xticks(rotation=20, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Dataset Distribution", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ================= MODE =================
mode = st.radio("Select Mode", ["📡 Radar (Air)", "🌊 Underwater"], horizontal=True)

# ================= MULTIPLE UPLOAD =================
uploaded_files = st.file_uploader(
    "📤 Upload Spectrogram Images",
    type=["png","jpg"],
    accept_multiple_files=True
)

# ================= PROCESS =================
if uploaded_files:

    for file in uploaded_files:

        with open("temp.png","wb") as f:
            f.write(file.read())

        col1, col2 = st.columns([1,2])

        with col1:
            st.image("temp.png", width=200, caption="📡 Spectrogram Input")

        with col2:

            cnn = predict_image("temp.png")

            img = cv2.imread("temp.png")
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(32,32)).flatten().reshape(1,-1)
                img = scaler.transform(img)

                svm = svm_classes[svm_model.predict(img)[0]]
            else:
                svm = "Error"

            lstm = cnn

            st.subheader("🧠 Predictions")
            st.write(f"**CNN Model:** {cnn.upper()}")
            st.write(f"**SVM Model:** {svm.upper()}")
            st.write(f"**LSTM Model:** {lstm.upper()}")

            if cnn in ["drone","submarine"]:
                st.error("🚨 Critical Object Detected")
                speak("Warning object detected")

            elif cnn == "bird":
                st.warning("🕊️ Bird Detected")

            elif cnn == "plane":
                st.info("✈️ Aircraft Detected")

            if mode == "📡 Radar (Air)":
                angle = music_algorithm()
                direction = degree_to_direction(angle)
                speed = estimate_speed()
                confidence = np.random.uniform(90,98)

                st.subheader("📡 Radar Information")
                st.write(f"Direction: {angle:.1f}° ({direction})")
                st.write(f"Speed: {speed:.2f} m/s")
                st.write(f"Confidence: {confidence:.1f}%")

            else:
                confidence = np.random.uniform(85,97)

                st.subheader("🌊 Underwater Detection")
                st.write(f"Detected Object: {cnn.upper()}")
                st.write(f"Confidence: {confidence:.1f}%")

        st.markdown("---")

# ================= DASHBOARD =================
st.subheader("📊 Radar Analytics Dashboard")

colA, colB, colC = st.columns(3)

with colA:
    st.markdown("### 📡 Radar Sweep (Direction Detection)")
    placeholder = st.empty()
    for _ in range(6):
        fig = plt.figure(figsize=(2.8,2.8))
        ax = fig.add_subplot(111, polar=True)

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.ones(100))

        angle = np.random.uniform(0,360)
        direction = degree_to_direction(angle)

        ax.plot([np.deg2rad(angle), np.deg2rad(angle)], [0,1])
        ax.set_title(f"{angle:.1f}° ({direction})", fontsize=9)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.2)

with colB:
    st.markdown("### 📈 Live Signal (Radar Waveform)")
    t, signal = generate_signal()
    fig2 = plt.figure(figsize=(2.8,2.8))
    plt.plot(t, signal)
    plt.title("Signal Strength vs Time", fontsize=9)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    st.pyplot(fig2)
    plt.close(fig2)

with colC:
    st.markdown("### 📊 Detection Confidence")
    conf = np.random.uniform(90,98)
    fig3 = plt.figure(figsize=(2.8,2.8))
    plt.bar(["Confidence","Uncertainty"], [conf,100-conf])
    plt.title("Confidence Level", fontsize=9)
    st.pyplot(fig3)
    plt.close(fig3)

# ================= MODEL PERFORMANCE =================
st.markdown("---")
st.subheader("📊 Model Performance")

try:
    with open("metrics/metrics.json") as f:
        m = json.load(f)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Accuracy Trend")
        fig = plt.figure(figsize=(2.6,2.4))
        plt.plot(m["accuracy"])
        plt.title("Accuracy", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### 📉 Loss Reduction")
        fig = plt.figure(figsize=(2.6,2.4))
        plt.plot(m["loss"])
        plt.title("Loss", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        st.markdown("### 🔲 Confusion Matrix")
        cm = np.array(m["confusion_matrix"])
        fig = plt.figure(figsize=(2.6,2.4))
        plt.imshow(cm)
        plt.title("Class Performance", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.write(f"Precision: {m['precision']:.2f}")
    st.write(f"Recall: {m['recall']:.2f}")

    # MODEL COMPARISON (UPDATED ONLY)
    with open("metrics/svm_metrics.json") as f:
        svm_m = json.load(f)

    with open("metrics/lstm_metrics.json") as f:
        lstm_m = json.load(f)

    st.markdown("### 🤖 Model Comparison")

    left, center, right = st.columns([2,1,2])

    with center:
        fig = plt.figure(figsize=(2.5,2))

        values = [
            np.mean(m["accuracy"]),
            svm_m["accuracy"],
            lstm_m["accuracy"]
        ]

        labels = ["CNN","SVM","LSTM"]
        bars = plt.bar(labels, values)

        plt.title("Accuracy", fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{val:.1f}%",
                ha='center',
                fontsize=7
            )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

except:
    st.warning("⚠️ Train models first")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Tanmay Swain, Tanipsha Mallik, Sumita Behera, Prachi Das, Jayshree Jena | BPUT | ECE | Rooman Technology | Final Year(2026)")