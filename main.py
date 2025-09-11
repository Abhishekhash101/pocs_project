import streamlit as st
import numpy as np
from scipy.signal import hilbert, butter, lfilter, resample
from PIL import Image
import soundfile as sf
import matplotlib.pyplot as plt
import io

# Imports for added features
from streamlit_drawable_canvas import st_canvas
from skimage.metrics import structural_similarity as ssim
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import numexpr as ne

st.set_page_config(page_title="Advanced Modulation Simulator", layout="wide")

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
    inst_freq -= np.mean(inst_freq) # Remove DC offset
    return np.concatenate(([inst_freq[0]], inst_freq))

def recover_image(signal, shape):
    arr = np.clip(signal, -1, 1)
    arr = ((arr + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr.reshape((shape[1], shape[0])))

def coherent_demodulate(signal, fc, fs, phase_offset_deg=0):
    t = time_axis(len(signal), fs)
    phase_offset_rad = np.deg2rad(phase_offset_deg)
    carrier = np.cos(2 * np.pi * fc * t + phase_offset_rad)
    demod = signal * carrier
    b, a = butter(4, 0.05)
    return lfilter(b, a, demod)

def array_to_audio_bytes(audio_array, sample_rate):
    audio_array = audio_array.astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format='WAV')
    return buf.getvalue()

def plot_spectrum(ax, signal, fs, title, carrier_freq=None):
    N = len(signal)
    if N == 0: return
    yf = np.fft.fft(signal * np.hanning(N))
    xf = np.fft.fftfreq(N, 1 / fs)
    ax.plot(np.fft.fftshift(xf), 20 * np.log10(np.abs(np.fft.fftshift(yf)) + 1e-9))
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)
    zoom_range = fs / 4
    if carrier_freq:
        ax.set_xlim(carrier_freq - zoom_range, carrier_freq + zoom_range)
    else:
        ax.set_xlim(-zoom_range, zoom_range)

def calculate_signal_metrics(original_signal, recovered_signal, original_2d_shape=None):
    min_len = min(len(original_signal), len(recovered_signal))
    original = original_signal[:min_len].astype(np.float64)
    recovered = recovered_signal[:min_len].astype(np.float64)

    # Correlation Accuracy
    if np.std(original) < 1e-6 or np.std(recovered) < 1e-6:
        correlation = 0.0
    else:
        correlation = np.corrcoef(original, recovered)[0, 1]
    accuracy_percentage = correlation * 100

    # MSE and PSNR
    mse = np.mean((original - recovered) ** 2)
    psnr = 20 * np.log10(2.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    # SSIM (only for 2D image data)
    ssim_score = None
    if original_2d_shape is not None and original.size > 1:
        original_2d = ((original.reshape(original_2d_shape[1], original_2d_shape[0]) + 1) * 127.5).astype(np.uint8)
        recovered_2d = ((recovered.reshape(original_2d_shape[1], original_2d_shape[0]) + 1) * 127.5).astype(np.uint8)
        ssim_score = ssim(original_2d, recovered_2d, data_range=255)
    
    return accuracy_percentage, mse, psnr, ssim_score

def estimate_bandwidth(signal, fs, threshold=0.995):
    N = len(signal)
    if N == 0: return 0
    yf = np.fft.fft(signal * np.hanning(N))
    psd = np.abs(yf[:N//2])**2
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    if total_power == 0: return 0
    try:
        idx = np.where(cumulative_power >= total_power * threshold)[0][0]
        return freqs[idx]
    except IndexError:
        return fs / 2

# ---------------- File Loading ----------------

@st.cache_data
def load_audio(file, target_fs):
    data, fs_orig = sf.read(file)
    if data.ndim > 1: data = data.mean(axis=1)
    if fs_orig != target_fs:
        data = resample(data, int(len(data) * target_fs / fs_orig))
    if np.max(np.abs(data)) > 0: data /= np.max(np.abs(data))
    return data

@st.cache_data
def load_image(file):
    img = Image.open(file).convert('L')
    arr = np.array(img).astype(float)
    return (arr - 127.5) / 127.5, img.size

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

st.title("📡 Real-Time Interactive Modulation Simulator")

input_type = st.radio("Select Input Type", ["Audio", "Image", "Draw Signal", "Live Audio", "Live Camera", "Equation"])
uploaded_file = st.file_uploader("Upload Audio/Image File", type=["wav", "mp3", "jpg", "png"])

modulations = st.multiselect("Select Modulations", ['AM', 'DSB-SC', 'SSB', 'FM'])
snr_db = st.slider("SNR (dB)", -10, 40, 20)
carrier = st.number_input("Carrier Frequency (Hz)", value=5000)
fs = st.number_input("Sampling Frequency", value=48000)

msg_flat = None
original_image_shape = None
msg_for_metrics = None

# --- Input Handling ---
if input_type == "Live Audio":
    st.subheader("🎤 Record Your Voice")
    audio_info = mic_recorder(start_prompt="▶️ Start Recording", stop_prompt="⏹️ Stop Recording", key='recorder')
    if audio_info:
        audio_bytes = audio_info['bytes']
        st.audio(audio_bytes)
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            data, samplerate = sf.read(wav_buffer)
            if data.ndim > 1: data = data.mean(axis=1)
            if samplerate != fs: data = resample(data, int(len(data) * fs / samplerate))
            if np.max(np.abs(data)) > 0: msg_flat = data / np.max(np.abs(data))
        except Exception as e:
            st.error(f"Could not process audio. Make sure FFmpeg is installed. Error: {e}")

elif input_type == "Live Camera":
    st.subheader("📸 Capture from Webcam")
    camera_photo = st.camera_input("Take a picture")
    if camera_photo:
        msg, shape = load_image(camera_photo)
        msg_flat = msg.flatten()
        original_image_shape = shape
        st.image(camera_photo, caption="Captured Image")

elif input_type == "Draw Signal":
    st.subheader("✍️ Draw Your Custom Waveform")
    st.info("Draw a single, continuous line from left to right. Then click the button below to process it.")
    canvas_result = st_canvas(stroke_width=5, stroke_color="#000000", background_color="#EEEEEE", height=200, width=700, drawing_mode="freedraw", key="canvas")

    if st.button("Process Drawing", key="process_drawing"):
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, 3]) > 0:
            drawn_image = canvas_result.image_data.astype(np.float32)[:, :, 3]
            height, width = drawn_image.shape
            raw_signal = np.array([np.sum(drawn_image[:, x] * np.arange(height)) / np.sum(drawn_image[:, x]) if np.sum(drawn_image[:, x]) > 0 else height / 2.0 for x in range(width)])
            raw_signal = height - raw_signal
            if np.max(raw_signal) != np.min(raw_signal):
                signal_centered = raw_signal - np.mean(raw_signal)
                msg_flat = signal_centered / np.max(np.abs(signal_centered))
            else:
                msg_flat = np.zeros_like(raw_signal)
            st.session_state.drawn_signal = msg_flat
            st.success("Drawing processed successfully!")

    if 'drawn_signal' in st.session_state:
        msg_flat = st.session_state.drawn_signal
        st.subheader("Your Processed Waveform")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(msg_flat, color='blue')
        ax.set_title("Processed Message Signal m(t)")
        ax.grid(True)
        st.pyplot(fig)

elif input_type == "Equation":
    st.subheader("🧪 Define Signal with an Equation")
    st.info("Define `m(t)` using NumPy. Use `t` for time. Examples: `np.sin(2*np.pi*100*t)`, `np.sign(np.sin(2*np.pi*10*t))`")
    if 'equation_str' not in st.session_state:
        st.session_state.equation_str = "np.sin(2 * np.pi * 100 * t)"
    equation_str = st.text_input("Enter your equation:", st.session_state.equation_str)
    st.session_state.equation_str = equation_str
    duration = st.number_input("Signal Duration (s)", 0.1, 10.0, 1.0, 0.1)

    if st.button("Generate Signal", key="generate_signal"):
        try:
            N = int(duration * fs)
            t = np.arange(N) / fs
            generated_signal = ne.evaluate(equation_str, local_dict={'np': np, 't': t})
            if not isinstance(generated_signal, np.ndarray):
                st.error("Equation did not produce a valid signal. Please check the expression.")
            else:
                if np.max(np.abs(generated_signal)) > 0:
                    msg_flat = generated_signal / np.max(np.abs(generated_signal))
                else:
                    msg_flat = generated_signal
                st.session_state.equation_signal = msg_flat
                st.success("Signal generated successfully!")
        except Exception as e:
            st.error(f"Invalid equation or syntax error: {e}")

    if 'equation_signal' in st.session_state:
        msg_flat = st.session_state.equation_signal
        st.subheader("Generated Waveform")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(np.arange(len(msg_flat))/fs, msg_flat, color='blue')
        ax.set_title("Generated Message Signal m(t)")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        st.pyplot(fig)

elif uploaded_file is not None:
    if input_type == "Audio":
        msg_flat = load_audio(uploaded_file, fs)
        st.subheader("Original Audio")
        st.audio(array_to_audio_bytes(msg_flat, fs))
    elif input_type == "Image":
        msg, shape = load_image(uploaded_file)
        msg_flat = msg.flatten()
        original_image_shape = shape
        st.subheader("Original Image")
        st.image(recover_image(msg_flat, shape), use_column_width=True)

if msg_flat is not None:
    msg_for_metrics = msg_flat.copy()
    if input_type in ["Image", "Live Camera"]:
        bandwidth_hz = estimate_bandwidth(msg_flat, fs)
        suggested_fc = int(np.ceil(bandwidth_hz * 5 / 1000) * 1000) if bandwidth_hz > 0 else 5000
        suggested_fs = int(np.ceil(2.5 * (suggested_fc + bandwidth_hz) / 1000) * 1000) if suggested_fc > 0 else 48000
        st.info(f"💡 **Suggestion for Clear Image:** Est. Bandwidth: `{bandwidth_hz:,.0f} Hz`. Try $f_c \approx$ `{suggested_fc:,} Hz` and $f_s >$ `{suggested_fs:,} Hz`.")

# --- Main Processing Loop ---
if msg_flat is not None and modulations:
    mod_tabs = st.tabs(modulations)
    for i, mod in enumerate(modulations):
        with mod_tabs[i]:
            phase_offset_deg = 0
            if mod in ['DSB-SC', 'SSB']:
                phase_offset_deg = st.slider("Coherent Demodulator Phase Offset (°)", -180, 180, 0, key=f"phase_{mod}")

            # Modulate, Add Noise, Demodulate
            if mod == 'AM': tx = am_modulate(msg_flat, carrier, fs)
            elif mod == 'DSB-SC': tx = dsb_sc_modulate(msg_flat, carrier, fs)
            elif mod == 'SSB': tx = ssb_modulate(msg_flat, carrier, fs)
            elif mod == 'FM': tx = fm_modulate(msg_flat, carrier, fs)
            tx_noisy = awgn(tx, snr_db)
            if mod == 'AM': rec = envelope_detect(tx_noisy)
            elif mod in ['DSB-SC', 'SSB']: rec = coherent_demodulate(tx_noisy, carrier, fs, phase_offset_deg)
            elif mod == 'FM': rec = fm_demodulate(tx_noisy, fs)
            if np.max(np.abs(rec)) > 0: rec /= np.max(np.abs(rec))

            # --- Display Results ---
            st.subheader("Signal Analysis")
            fig, axs = plt.subplots(2, 2, figsize=(14, 8))
            axs[0, 0].plot(time_axis(len(tx_noisy), fs), tx_noisy, color='red'); axs[0, 0].set_title("Noisy Modulated Signal (Time)")
            axs[0, 1].plot(time_axis(len(rec), fs), rec, color='green'); axs[0, 1].set_title("Recovered Signal (Time)")
            plot_spectrum(axs[1, 0], tx_noisy, fs, "Modulated Spectrum", carrier_freq=carrier)
            plot_spectrum(axs[1, 1], rec, fs, "Recovered Spectrum")
            plt.tight_layout(); st.pyplot(fig)

            st.subheader("Output & Quality Metrics")
            accuracy, mse, psnr, ssim_val = calculate_signal_metrics(msg_for_metrics, rec, original_image_shape)
            
            cols = st.columns(4)
            cols[0].metric("🎯 Accuracy", f"{accuracy:.2f}%")
            cols[1].metric("PSNR (dB)", f"{psnr:.2f}")
            cols[2].metric("MSE", f"{mse:.4f}", delta_color="inverse")
            if ssim_val is not None:
                cols[3].metric("SSIM", f"{ssim_val:.4f}")
            
            if input_type in ["Audio", "Live Audio"]:
                st.audio(array_to_audio_bytes(rec, fs))
            elif input_type in ["Image", "Live Camera"] and original_image_shape:
                st.image(recover_image(rec, original_image_shape), caption=f"{mod} Recovered Image")

else:
    st.info("Select an input type and at least one modulation scheme to begin.")
