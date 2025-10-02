import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from scipy.signal import find_peaks
from PIL import Image
from st_audiorec import st_audiorec

# -------------------------------
# Helper functions
# -------------------------------
def safe_eval(func_str, t):
    """Safely evaluate continuous functions like sin(t), cos(t), exp(-t), sqrt(t) using NumPy."""
    allowed_names = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp,
        'log': lambda x: np.log(np.clip(x, 1e-9, None)),  # avoid log(0) or log(negative)
        'sqrt': lambda x: np.sqrt(np.clip(x, 0, None)),   # avoid sqrt of negative numbers
        'abs': np.abs, 
        't': t, 
        'pi': np.pi, 
        'e': np.e,
        'floor': np.floor,
        'ceil': np.ceil,
        'sign': np.sign,
        'heaviside': lambda x: np.heaviside(x, 0),
        'mod': np.mod
    }
    return eval(func_str, {"__builtins__": {}}, allowed_names)

def parse_discrete_input(sig_str):
    """Parse input like [1,2,3,4] or 1,2,3,4 into a numpy array."""
    sig_str = sig_str.strip()              # remove outer spaces
    sig_str = sig_str.strip("[]")          # remove brackets if present
    # split and clean each number
    values = [x.strip() for x in sig_str.split(",") if x.strip() != ""]
    return np.array([float(x) for x in values])

def autocorrelation(signal):
    return np.correlate(signal, signal, mode='full')

def crosscorrelation(sig1, sig2):
    return np.correlate(sig1, sig2, mode='full')

def detect_periodicity(auto, lags=None, tol=0.25, min_side_peaks=2):
    """
    Improved periodicity detection:
    - Uses peaks if enough exist
    - Falls back to FFT frequency analysis for short windows
    """
    auto = np.asarray(auto)
    if auto.size == 0:
        return "Non-periodic"

    # normalize safely
    auto_norm = auto / (np.max(np.abs(auto)) + 1e-9)

    # Try a few prominence levels to find peaks
    prominences = [0.3, 0.2, 0.1, 0.05, 0.02]
    peaks = np.array([], dtype=int)
    for p in prominences:
        peaks, _ = find_peaks(auto_norm, prominence=p)
        if peaks.size > 0:
            break

    center_idx = int(np.argmax(auto_norm))
    side_peaks = [pk for pk in peaks if abs(pk - center_idx) > 1]

    if len(side_peaks) < min_side_peaks:
        # fallback: use FFT magnitude to detect dominant frequency
        fft_vals = np.fft.fft(auto_norm)
        fft_freqs = np.fft.fftfreq(len(auto_norm), d=1.0)
        mag = np.abs(fft_vals[:len(auto_norm)//2])
        if np.max(mag) > 0.3:  # heuristic threshold
            return "Periodic"
        else:
            return "Non-periodic"

    return "Periodic"



def analyze_autocorr(sig, auto, lags):
    """Analysis for single signal autocorrelation."""
    energy = auto[len(auto)//2]
    periodic = detect_periodicity(auto, lags)
    spread = np.sum(np.abs(auto) > 0.05 * np.max(auto))

    return {
        "Energy": float(energy),
        "Periodicity": periodic,
        "Effective Spread": int(spread)
    }

def analyze_crosscorr(sig1, sig2, cross, lags_cross):
    """Analysis for cross correlation between two signals."""
    max_corr = np.max(cross)
    norm_corr = max_corr / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-9)
    lag_at_max = lags_cross[np.argmax(cross)]

    return {
        "Max Correlation": float(max_corr),
        "Normalized Similarity": float(norm_corr),
        "Relative Delay": int(lag_at_max)
    }

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Signal Correlation App", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📈 Input & Plots", "🧮 Analysis", "🎵 Custom Audio Input", "🎤 Real-Time Audio Input"])

# -------------------------------
# Home Page
# -------------------------------
if page == "🏠 Home":
    st.title("📊 Auto & Cross Correlation of Signals")
    st.markdown("""
    Welcome to the **Signal Correlation App**!   

    ### What’s Inside?
    - 📌 **Auto-correlation** → Measures how similar a signal is to a delayed version of itself. Reveals periodicity, signal energy, and effective spread.""")

    image = Image.open("ct_auto.png")
    st.image(image, caption="Auto-correlation of CT signals", width=500)

    image = Image.open("dt_auto.png")
    st.image(image, caption="Auto-correlation of DT signals", width=500)

    st.markdown("""
    - 📌 **Cross-correlation** → Measures similarity between two different signals. Reveals alignment, maximum correlation, relative delay, and normalized similarity.  
    """)

    image = Image.open("ct_cross.png")
    st.image(image, caption="Cross-correlation of CT signals", width=500)

    image = Image.open("dt_cross.png")
    st.image(image, caption="Cross-correlation of DT signals", width=500)

    st.markdown("""
    ### Why use this app?
    - Visualize how signals are related.  
    - Detect hidden patterns in periodic and non-periodic signals.  
    - Explore continuous & discrete signals interactively.  

    👉 Click **Input & Plots** in the sidebar to get started!
    """)


# -------------------------------
# Input & Plots
# -------------------------------
elif page == "📈 Input & Plots":
    st.title("📈 Input Signals & Correlation Plots")

    corr_type = st.radio("-->Select Correlation Type:", ["Auto-correlation", "Cross-correlation"])
    signal_type = st.radio("--> Select Signal Type:", ["Continuous", "Discrete"])

    if signal_type == "Continuous":
        from scipy.integrate import quad

        if corr_type == "Auto-correlation":
            st.subheader("Enter Continuous-Time Function of t")
            func = st.text_input("Signal Function (in terms of t):", " ")

            t_min, t_max = st.slider("⏱️ Time Range (t)", -10.0, 10.0, (-5.0, 5.0))
            amp_scale = st.slider("🔊 Amplitude Scale", 0.1, 5.0, 1.0)

            if st.button("Compute Auto-correlation"):
                start_time=time.time()
                try:
                    sig_func = lambda t: amp_scale * safe_eval(func, t)

                    taus = np.linspace(t_min, t_max, 100)  # lag values
                    auto_vals = []

                    for tau in taus:
                        integrand = lambda tt: sig_func(tt) * sig_func(tt + tau)
                        val, _ = quad(integrand, t_min, t_max, limit=200)
                        auto_vals.append(val)

                    auto_vals = np.array(auto_vals)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(taus, auto_vals, color='blue')
                    ax.set_title("Auto-correlation of Signal (Continuous-Time, Integration)")
                    ax.set_xlabel("τ (Lag)")
                    ax.set_ylabel("Correlation")
                    ax.grid(True)
                    st.pyplot(fig)

                    end_time = time.time()  # end timer
                    st.success(f"⏱ Time taken to compute: {end_time - start_time:.4f} seconds")

                    # Show values under graph
                    st.markdown("**Computed Auto-correlation Values (Integration):**")
                    st.write(dict(zip(taus.tolist(), auto_vals.tolist())))

                    st.session_state["results"] = ("auto_ct", auto_vals, taus)

                except Exception as e:
                    st.error(f"Error in evaluating function: {e}")

        else:  # Cross-correlation
            st.subheader("Enter Continuous-Time Functions of t")
            func1 = st.text_input("Signal 1 Function (in terms of t):", " ")
            func2 = st.text_input("Signal 2 Function (in terms of t):", " ")

            t_min, t_max = st.slider("⏱️ Time Range (t)", -10.0, 10.0, (-5.0, 5.0))
            amp_scale = st.slider("🔊 Amplitude Scale", 0.1, 5.0, 1.0)

            if st.button("Compute Cross-correlation"):
                start_time=time.time()
                try:
                    sig1_func = lambda t: amp_scale * safe_eval(func1, t)
                    sig2_func = lambda t: amp_scale * safe_eval(func2, t)

                    taus = np.linspace(t_min, t_max, 100)  # lag values
                    cross_vals = []

                    for tau in taus:
                        integrand = lambda tt: sig1_func(tt) * sig2_func(tt + tau)
                        val, _ = quad(integrand, t_min, t_max, limit=200)
                        cross_vals.append(val)

                    cross_vals = np.array(cross_vals)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(taus, cross_vals, color='red')
                    ax.set_title("Cross-correlation of Signals (Continuous-Time, Integration)")
                    ax.set_xlabel("τ (Lag)")
                    ax.set_ylabel("Correlation")
                    ax.grid(True)
                    st.pyplot(fig)

                    end_time = time.time()  # end timer
                    st.success(f"⏱ Time taken to compute: {end_time - start_time:.4f} seconds")

                    # Show values under graph
                    st.markdown("**Computed Cross-correlation Values (Integration):**")
                    st.write(dict(zip(taus.tolist(), cross_vals.tolist())))

                    st.session_state["results"] = ("cross_ct", cross_vals, taus)

                except Exception as e:
                    st.error(f"Error in evaluating functions: {e}")

    else:  # Discrete
        if corr_type == "Auto-correlation":
            st.subheader("Enter Discrete-Time Signal (Comma-separated numbers)")
            sig_str = st.text_input("Signal:", " ")
            amp_scale = st.slider("🔊 Amplitude Scale", 0.1, 5.0, 1.0)
            max_len = st.slider("📏 Max Sequence Length", 5, 50, 20)

            if st.button("Compute Auto-correlation"):
                start_time=time.time()
                try:
                    sig = parse_discrete_input(sig_str)[:max_len] * amp_scale
                    auto = autocorrelation(sig)
                    lags = np.arange(-len(sig)+1, len(sig))

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.stem(lags, auto, basefmt=" ")
                    ax.set_title("Auto-correlation of Signal")
                    ax.set_xlabel("Lag")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
                    st.pyplot(fig)

                    end_time = time.time()  # end timer
                    st.success(f"⏱ Time taken to compute: {end_time - start_time:.4f} seconds")

                    # 🔹 Show numerical values under the graph
                    st.markdown("**Computed Auto-correlation Values:**")
                    st.write(dict(zip(lags.tolist(), auto.tolist())))

                    st.session_state["results"] = ("auto", sig, auto, lags)

                except Exception as e:
                    st.error(f"Error: {e}")

        else:  # Cross-correlation
            st.subheader("Enter Discrete-Time Signals (Comma-separated numbers)")
            sig1_str = st.text_input("Signal 1:", " ")
            sig2_str = st.text_input("Signal 2:", " ")
            amp_scale = st.slider("🔊 Amplitude Scale", 0.1, 5.0, 1.0)
            max_len = st.slider("📏 Max Sequence Length", 5, 50, 20)

            if st.button("Compute Cross-correlation"):
                start_time=time.time()
                try:
                    sig1 = parse_discrete_input(sig1_str)[:max_len] * amp_scale
                    sig2 = parse_discrete_input(sig2_str)[:max_len] * amp_scale

                    cross = crosscorrelation(sig1, sig2)
                    lags_cross = np.arange(-len(sig2)+1, len(sig1))

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.stem(lags_cross, cross, basefmt=" ")
                    ax.set_title("Cross-correlation of Signals")
                    ax.set_xlabel("Lag")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True)
                    st.pyplot(fig)

                    end_time = time.time()  # end timer
                    st.success(f"⏱ Time taken to compute: {end_time - start_time:.4f} seconds")

                    # 🔹 Show numerical values under the graph
                    st.markdown("**Computed Cross-correlation Values:**")
                    st.write(dict(zip(lags_cross.tolist(), cross.tolist())))

                    st.session_state["results"] = ("cross", sig1, sig2, cross, lags_cross)

                except Exception as e:
                    st.error(f"Error: {e}")


# -------------------------------
# Analysis Page
# -------------------------------
elif page == "🧮 Analysis":
    st.title("🧮 Analysis of Results")

    if "results" not in st.session_state:
        st.warning("⚠️ Please go to **Input & Plots** and compute correlations first.")
    else:
        res = st.session_state["results"]

        # ---------------- Auto-correlation ----------------
        if res[0] in ["auto", "auto_ct"]:
            if res[0] == "auto":  # discrete
                _, sig, auto, lags = res
            else:  # continuous
                _, auto_vals, lags = res
                sig = None  
                auto = auto_vals

            # Energy
            energy = auto[len(auto)//2]

            # ✅ Improved periodicity detection
            # Use robust detector without inconclusive based on time window
            lags_int = np.arange(len(auto)) if np.issubdtype(lags.dtype, np.floating) else lags
            periodic = detect_periodicity(auto, lags_int)

            # Effective spread
            spread = np.sum(np.abs(auto) > 0.05 * np.max(auto))

            analysis_auto = {
                "Energy": float(energy),
                "Periodicity": periodic,
                "Effective Spread": int(spread)
            }

            st.subheader("🔍 Analysis of Auto-correlation")
            st.json(analysis_auto)

        # ---------------- Cross-correlation ----------------
        else:  # cross or cross_ct
            if res[0] == "cross":  # discrete
                _, sig1, sig2, cross, lags_cross = res
            else:  # continuous
                _, cross_vals, lags_cross = res
                cross = cross_vals
                sig1 = sig2 = np.ones_like(cross)  # dummy for normalization

            # Max correlation
            max_corr = np.max(cross)
            # Normalized similarity
            norm_corr = max_corr / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-9)
            # Lag at max
            lag_at_max = lags_cross[np.argmax(cross)]

            analysis_cross = {
                "Max Correlation": float(max_corr),
                "Normalized Similarity": float(norm_corr),
                "Relative Delay (Lag at Max)": float(lag_at_max)
            }

            st.subheader("🔍 Analysis of Cross-correlation")
            st.json(analysis_cross)


# -------------------------------
# Custom Audio Input
# -------------------------------
elif page == "🎵 Custom Audio Input":
    st.title("🎵 Audio Signal Correlation")
    
    corr_type = st.radio("Select Correlation Type:", ["Auto-correlation", "Cross-correlation"])
    
    # Upload audio files
    audio_file1 = st.file_uploader("Upload WAV Audio File 1", type=["wav"])
    
    if corr_type == "Cross-correlation":
        audio_file2 = st.file_uploader("Upload WAV Audio File 2", type=["wav"])
    else:
        audio_file2 = None

    if audio_file1 is not None and (corr_type == "Auto-correlation" or audio_file2 is not None):
        from scipy.io import wavfile
        fs1, y1 = wavfile.read(audio_file1)
        if len(y1.shape) > 1:  # convert stereo to mono
            y1 = y1.mean(axis=1)
        
        if audio_file2 is not None:
            fs2, y2 = wavfile.read(audio_file2)
            if len(y2.shape) > 1:
                y2 = y2.mean(axis=1)

        # Slider for downsampling
        downsample_factor = st.slider("📉 Downsample Factor", 1, 5000, 10)

        # Slider for selecting a time window
        t_max_audio = len(y1)/fs1
        t_start, t_end = st.slider("Select Time Window (seconds)", 0.0, t_max_audio, (0.0, t_max_audio))
        start_sample = int(t_start * fs1)
        end_sample = int(t_end * fs1)
        y1_window = y1[start_sample:end_sample:downsample_factor]

        if audio_file2 is not None:
            t_max_audio2 = len(y2)/fs2
            start_sample2 = int(t_start * fs2)
            end_sample2 = int(t_end * fs2)
            y2_window = y2[start_sample2:end_sample2:downsample_factor]

        st.audio(audio_file1, format="audio/wav")
        if audio_file2 is not None:
            st.audio(audio_file2, format="audio/wav")
        
        if st.button("Compute Correlation"):
            start_time = time.time()
            
            if corr_type == "Auto-correlation":
                # Auto-correlation
                auto = np.correlate(y1_window, y1_window, mode='full')
                lags = np.arange(-len(y1_window)+1, len(y1_window))
                
                fig, ax = plt.subplots(figsize=(10,4))
                ax.stem(lags, auto, basefmt=" ")
                ax.set_title("Audio Auto-correlation")
                ax.set_xlabel("Lag (samples)")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)
                
                # Analysis
                energy = np.sum(y1_window**2)
                peaks, _ = find_peaks(auto, height=0.5*np.max(auto))
                periodicity = "Periodic" if len(peaks) >= 4 else "Non-periodic"
                effective_spread = np.sum(np.abs(auto) > 0.05*np.max(auto))
                
                st.subheader("🔍 Analysis")
                st.write({
                    "Energy": float(energy),
                    "Periodicity": periodicity,
                    "Effective Spread": int(effective_spread)
                })
            
            else:
                # Cross-correlation
                cross = np.correlate(y1_window, y2_window, mode='full')
                lags_cross = np.arange(-len(y2_window)+1, len(y1_window))
                
                fig, ax = plt.subplots(figsize=(10,4))
                ax.stem(lags_cross, cross, basefmt=" ")
                ax.set_title("Audio Cross-correlation")
                ax.set_xlabel("Lag (samples)")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)
                
                # Analysis
                max_corr = np.max(cross)
                norm_corr = max_corr / (np.linalg.norm(y1_window) * np.linalg.norm(y2_window) + 1e-9)
                lag_at_max = lags_cross[np.argmax(cross)]
                
                st.subheader("🔍 Analysis")
                st.write({
                    "Max Correlation": float(max_corr),
                    "Normalized Similarity": float(norm_corr),
                    "Relative Delay (samples)": int(lag_at_max)
                })
            
            end_time = time.time()
            st.info(f"⏱️ Time Taken: {end_time - start_time:.3f} seconds")

# -------------------------------
# Real-Time Audio Input
# -------------------------------
elif page == "🎤 Real-Time Audio Input":
    st.title("🎤 Record Real-Time Audio & Correlation")

    from st_audiorec import st_audiorec  # ✅ ensure import is here

    corr_type = st.radio("Select Correlation Type:", ["Auto-correlation", "Cross-correlation"])

    # Sliders
    downsample_factor = st.slider("📉 Downsample Factor", 1, 5000, 5)
    t_window = st.slider("Select Time Window (seconds)", 1, 10, 5)

    # --- Record Audio ---
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")

        if st.button("Compute Correlation"):  # ✅ only compute when button is clicked
            import io
            import soundfile as sf

            start_time = time.time()

            # Read audio data
            y1, fs = sf.read(io.BytesIO(wav_audio_data))

            if len(y1.shape) > 1:  # convert stereo → mono
                y1 = y1.mean(axis=1)

            # Use only last t_window seconds
            n_samples = int(t_window * fs)
            if len(y1) > n_samples:
                y1 = y1[-n_samples:]

            # Downsample
            y1_window = y1[::downsample_factor]

            if corr_type == "Auto-correlation":
                auto = np.correlate(y1_window, y1_window, mode="full")
                lags = np.arange(-len(y1_window) + 1, len(y1_window))

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.stem(lags, auto, basefmt=" ")
                ax.set_title("Real-Time Audio Auto-correlation")
                ax.set_xlabel("Lag (samples)")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                st.pyplot(fig)

                # Analysis
                energy = np.sum(y1_window ** 2)
                peaks, _ = find_peaks(auto, height=0.5 * np.max(auto))
                periodicity = "Periodic" if len(peaks) >= 4 else "Non-periodic"
                effective_spread = np.sum(np.abs(auto) > 0.05 * np.max(auto))

                st.subheader("🔍 Analysis")
                st.write({
                    "Energy": float(energy),
                    "Periodicity": periodicity,
                    "Effective Spread": int(effective_spread),
                })

            else:
                st.warning("🎤 For Cross-correlation, record a second audio stream feature can be added later.")

            end_time = time.time()  # ⬅️ end timer here
            st.info(f"⏱️ Time Taken: {end_time - start_time:.3f} seconds")