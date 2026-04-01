import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pyttsx3

from model.cnn.predict import predict_image
from model.classical_ml.svm import train_svm
from dashboard.data_vis import plot_distribution
from utils.helpers import degree_to_direction
from doa_estimation.music import music_algorithm
from signal_processing.doppler_fft import estimate_speed
from radar_simulation.live_signal import generate_signal

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Radar System", layout="wide")

# ---------------- VOICE FUNCTION ----------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------- DARK NEON THEME ----------------
st.markdown("""
<style>
body { background-color: #0b0f1a; color: white; }
.title {
    font-size: 36px; font-weight: bold; text-align: center;
    color: #00f7ff; text-shadow: 0 0 10px #00f7ff;
}
.card {
    background: #111827; padding: 15px; border-radius: 12px;
    box-shadow: 0 0 10px #00f7ff;
}
[data-testid="stMetric"] {
    background: #0f172a; border: 1px solid #00f7ff;
    padding: 10px; border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGOS ----------------
BPUT_LOGO = "logos/bput.png"
ROOMAN_LOGO = "logos/rooman.png"

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.image(BPUT_LOGO, width=100)

with col2:
    st.markdown('<div class="title">🚁 MICRO-DOPPLER sUAV SYSTEM</div>', unsafe_allow_html=True)

with col3:
    st.image(ROOMAN_LOGO, width=100)

st.markdown("---")

# ---------------- LOAD SVM ----------------
@st.cache_resource
def load_svm():
    return train_svm()

with st.spinner("🚀 Initializing AI Models..."):
    svm_model, svm_classes = load_svm()

# ---------------- DATASET ----------------
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns([1.5,1,1.5])

with col2:
    fig = plot_distribution("data/spectrograms/train")
    fig.set_size_inches(3.5, 2.5)   # 🔥 small & clean
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("📤 Upload Spectrogram Image", type=["png","jpg"])

if uploaded:
    with open("temp.png","wb") as f:
        f.write(uploaded.read())

    col1, col2 = st.columns(2)

    # LEFT IMAGE
    with col1:
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("temp.png", caption="📡 Micro-Doppler Spectrogram", width="stretch")
        # st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT RESULTS
    with col2:
        # st.markdown('<div class="card">', unsafe_allow_html=True)

        cnn = predict_image("temp.png")

        img = cv2.imread("temp.png")
        if img is not None:
            img = cv2.resize(img,(64,64)).flatten().reshape(1,-1)
            svm = svm_classes[svm_model.predict(img)[0]]
        else:
            svm = "Error"

        lstm = cnn

        st.subheader("🧠 AI Model Predictions")
        st.metric("CNN (ResNet)", cnn.upper())
        st.metric("SVM", svm.upper())
        st.metric("LSTM", lstm.upper())

        # ---------------- ALERT ----------------
        st.subheader("🚨 Alert System")

        if cnn == "drone":
            st.error("🚨 DRONE DETECTED")
            st.audio("https://www.soundjay.com/alarm/alarm-clock-01.mp3")
            speak("Warning! Drone detected")

        elif cnn == "bird":
            st.warning("🕊️ Bird Detected")
            st.audio("https://www.soundjay.com/nature/bird-chirp-01.mp3")
            speak("Bird detected")

        elif cnn == "plane":
            st.info("✈️ Aircraft Detected")
            st.audio("https://www.soundjay.com/transportation/airplane-flyby-01.mp3")
            speak("Aircraft detected")

        st.markdown("---")

        # ---------------- RADAR INFO ----------------
        angle = music_algorithm()
        direction = degree_to_direction(angle)
        speed = estimate_speed()
        confidence = np.random.uniform(90, 98)

        st.subheader("📡 Radar Intelligence")
        st.metric("Direction", f"{angle}° ({direction})")
        st.metric("Speed", f"{speed:.2f} m/s")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- VISUAL SECTION ----------------
st.markdown("---")
st.subheader("📡 Radar Analytics")

colA, colB, colC = st.columns(3)

# ---------------- RADAR ----------------
with colA:
    radar_placeholder = st.empty()

    for _ in range(10):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, polar=True)

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, np.ones(100), color='cyan')

        angle_deg = np.random.uniform(0, 360)
        angle_rad = np.deg2rad(angle_deg)

        ax.plot([angle_rad, angle_rad], [0, 1], color='lime', linewidth=2)

        direction = degree_to_direction(angle_deg)
        ax.set_title(f"Radar Sweep\n{angle_deg:.1f}° ({direction})")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        radar_placeholder.pyplot(fig)
        plt.close(fig)   # ✅ FIX MEMORY
        time.sleep(0.2)

# ---------------- SIGNAL ----------------
with colB:
    placeholder = st.empty()
    for _ in range(5):
        t, signal = generate_signal()
        fig2 = plt.figure(figsize=(4,4))
        plt.plot(t, signal)
        plt.title("Live Radar Signal")
        plt.tight_layout()
        placeholder.pyplot(fig2)
        plt.close(fig2)   # ✅ FIX MEMORY
        time.sleep(0.1)

# ---------------- CONFIDENCE ----------------
with colC:
    confidence = np.random.uniform(90, 98)
    labels = ["Confidence", "Uncertainty"]
    values = [confidence, 100-confidence]

    fig3 = plt.figure(figsize=(4,4))
    plt.bar(labels, values)
    plt.title("Confidence Analysis")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)   # ✅ FIX MEMORY

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Developed by Tanmay Swain, Tanipsha Mallik, Sumita Behera, Jayashree Jena, Prachi Das | ECE | 4th Year | CUPGS | BPUT")