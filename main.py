# app.py
import streamlit as st
import numpy as np
from scipy.signal import hilbert, butter, lfilter
from PIL import Image
import soundfile as sf
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Modulation Simulator", layout="wide")

# ---------------- Helper Functions ----------------

def time_axis(N, fs):
    return np.arange(N)/fs

def awgn(signal, snr_db, seed=None):
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_pow = sig_pow / snr_linear
    noise = rng.normal(0, np.sqrt(noise_pow/2), size=signal.shape)
    return signal + noise

def envelope_detect(signal):
    analytic = hilbert(signal)
    env = np.abs(analytic)
    b, a = butter(4, 0.05)
    return lfilter(b, a, env)

def fm_demodulate(signal, fs):
    analytic = hilbert(signal)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    return np.concatenate(([inst_freq[0]], inst_freq))

def recover_image(signal, shape):
    arr = np.clip(signal, -1, 1)
    arr = ((arr + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr.reshape((shape[1], shape[0])))

def coherent_demodulate(signal, fc, fs):
    t = time_axis(len(signal), fs)
    carrier = np.cos(2 * np.pi * fc * t)
    demod = signal * carrier
    b, a = butter(4, 0.05)
    return lfilter(b, a, demod)

def array_to_audio_bytes(audio_array, sample_rate):
    audio_array = audio_array.astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format='WAV')
    return buf.getvalue()

# ---------------- File Loading ----------------

@st.cache_data
def load_audio(file, target_fs=48000):
    data, fs = sf.read(file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if fs != target_fs:
        from scipy.signal import resample
        n = int(len(data) * target_fs / fs)
        data = resample(data, n)
        fs = target_fs
    data = data / np.max(np.abs(data))
    return data, fs

@st.cache_data
def load_image(file):
    img = Image.open(file).convert('L')
    arr = np.array(img).astype(float)
    arr_norm = (arr - 127.5) / 127.5
    return arr_norm, img.size

# ---------------- Modulation Functions ----------------

def am_modulate(msg, fc, fs, depth=0.7):
    t = time_axis(len(msg), fs)
    return (1 + depth * msg) * np.cos(2 * np.pi * fc * t)

def dsb_sc_modulate(msg, fc, fs):
    t = time_axis(len(msg), fs)
    return msg * np.cos(2 * np.pi * fc * t)

def ssb_modulate(msg, fc, fs, side='usb'):
    t = time_axis(len(msg), fs)
    m_hat = np.imag(hilbert(msg))
    wc = 2 * np.pi * fc
    if side == 'usb':
        return msg * np.cos(wc * t) - m_hat * np.sin(wc * t)
    else:
        return msg * np.cos(wc * t) + m_hat * np.sin(wc * t)

def fm_modulate(msg, fc, fs, kf=5.0):
    t = time_axis(len(msg), fs)
    dt = 1 / fs
    integral = np.cumsum(msg) * dt
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral)

# ---------------- Streamlit UI ----------------

st.title("📡 Real-Time Modulation & Demodulation Simulator")

input_type = st.radio("Select Input Type", ["Audio", "Image"])
uploaded_file = st.file_uploader("Upload Audio/Image", type=["wav", "mp3", "jpg", "png"])

modulations = st.multiselect("Select Modulations", ['AM', 'DSB-SC', 'SSB', 'FM'])

snr_db = st.slider("SNR (dB)", 0, 40, 20)
carrier = st.number_input("Carrier Frequency (Hz)", value=5000)
fs = st.number_input("Sampling Frequency", value=48000)

if uploaded_file is not None and modulations:

    # Load file
    if input_type == "Audio":
        msg, fs_actual = load_audio(uploaded_file, target_fs=fs)
        msg_flat = msg
        st.subheader("Original Audio")
        st.audio(array_to_audio_bytes(msg, fs), format='audio/wav')
        st.markdown("<br><br>", unsafe_allow_html=True)
    else:
        msg, shape = load_image(uploaded_file)
        msg_flat = msg.flatten()
        st.subheader("Original Image")
        st.image(recover_image(msg_flat, shape), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Original Image Waveform (Flattened)")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(msg_flat, color='blue')
        ax.set_title("Flattened Pixel Intensity")
        st.pyplot(fig)
        st.markdown("<br><br>", unsafe_allow_html=True)

    # Tab layout for each modulation
    mod_tabs = st.tabs(modulations)

    for i, mod in enumerate(modulations):
        with mod_tabs[i]:
            # Modulate
            if mod == 'AM':
                tx = am_modulate(msg_flat, carrier, fs)
            elif mod == 'DSB-SC':
                tx = dsb_sc_modulate(msg_flat, carrier, fs)
            elif mod == 'SSB':
                tx = ssb_modulate(msg_flat, carrier, fs)
            elif mod == 'FM':
                tx = fm_modulate(msg_flat, carrier, fs)

            # Add noise
            tx_noisy = awgn(tx, snr_db)

            # Demodulate
            if mod == 'AM':
                rec = envelope_detect(tx_noisy)
                rec = 2 * rec / np.max(rec) - 1
            elif mod in ['DSB-SC', 'SSB']:
                rec = coherent_demodulate(tx_noisy, carrier, fs)
                rec = rec / np.max(np.abs(rec))
            elif mod == 'FM':
                rec = fm_demodulate(tx_noisy, fs)
                rec = rec / np.max(np.abs(rec))

            # Display modulated + demodulated plots
            st.subheader(f"{mod} Modulated + Recovered Signals")

            fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            axs[0].plot(tx_noisy, color='red')
            axs[0].set_title("Noisy Modulated Signal")
            axs[1].plot(rec, color='green')
            axs[1].set_title("Recovered Signal")
            st.pyplot(fig)
            st.markdown("<br>", unsafe_allow_html=True)

            # Audio or Image output
            if input_type == "Audio":
                st.audio(array_to_audio_bytes(rec, fs), format='audio/wav')
            else:
                img_rec = recover_image(rec, shape)
                st.image(img_rec, caption=f"{mod} Recovered Image", use_container_width=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
