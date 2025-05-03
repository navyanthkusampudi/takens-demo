import io
import numpy as np
import sounddevice as sd
import wavio
import warnings
from scipy.io.wavfile import WavFileWarning
from scipy.io import wavfile
from scipy.signal import spectrogram as spgram
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

# suppress non‑data chunk warnings from scipy wavfile
warnings.filterwarnings("ignore", category=WavFileWarning)

# ——————————————————————————————————————————————————————————————————————
# Recording via sounddevice + wavio
def record_python(duration=5, sr=44100):
    """
    Record `duration` seconds from the default microphone at sample rate `sr`.
    Returns WAV‐encoded bytes (mono, 16‑bit).
    """
    st.info(f"Recording {duration:.1f}s at {sr} Hz…")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    buf = io.BytesIO()
    # wavio expects shape (nsamples, nchannels)
    wavio.write(buf, audio, sr, sampwidth=2)
    st.success("Recording complete!")
    return buf.getvalue()

# ——————————————————————————————————————————————————————————————————————
# Your existing processing functions
def load_audio(audio_bytes):
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    y = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
    y /= np.max(np.abs(y))
    return sr, y

def compute_spectrogram(y, sr, nperseg=1024, noverlap=512):
    f, t, Sxx = spgram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db

def compute_embedding(y, tau, m):
    N = len(y) - (m - 1) * tau
    if N <= 0:
        return None
    return np.column_stack([y[i : i + N] for i in range(0, m * tau, tau)])

def plot_full_timeseries(y, sr, t0, t1):
    time = np.arange(len(y)) / sr
    y_win = y[int(t0 * sr) : int(t1 * sr)]
    time_win = np.arange(int(t0 * sr), int(t1 * sr)) / sr
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time, y, color="lightgray", linewidth=1)
    ax.plot(time_win, y_win, color="red", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title="Full Time Series with Selected Window")
    return fig

def plot_full_spectrogram(f, t, Sxx_db, t0, t1):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(t, f, Sxx_db, cmap="gray", shading="gouraud")
    ax.axvspan(t0, t1, color="red", alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
           title="Full Spectrogram with Selected Window")
    return fig

def plot_window_scatter(y_win, sr):
    time_win = np.arange(len(y_win)) / sr
    fig = px.scatter(
        x=time_win, y=y_win,
        labels={"x": "Time (s)", "y": "Amplitude"},
        title="Selected Window Time Series"
    )
    fig.update_traces(marker=dict(size=4))
    return fig

def plot_embedding_2d(X, tau):
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1],
        labels={"x": "y(t)", "y": f"y(t+{tau})"},
        title="2D Takens Embedding"
    )
    fig.update_layout(width=600, height=600)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_embedding_3d(X, time_axis):
    fig = go.Figure(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=time_axis,
        mode="markers",
        marker=dict(size=4, color=time_axis, colorscale="Viridis", showscale=False)
    ))
    fig.update_layout(
        title="3D Takens Embedding (with Time)",
        scene=dict(
            xaxis_title="y(t)",
            yaxis_title="y(t+τ)",
            zaxis_title="Time (s)"
        ),
        width=600, height=600,
        margin=dict(t=40, b=40)
    )
    return fig

# ——————————————————————————————————————————————————————————————————————
def main():
    st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
    st.title("Interactive Takens’ Time‑Delay Embedding")

    st.sidebar.header("Audio Source")
    source = st.sidebar.radio("Choose Source", ("Upload WAV", "Record (Python)"))

    audio_bytes = None
    if source == "Upload WAV":
        uploaded = st.sidebar.file_uploader("Upload a WAV file", type=["wav"])
        if uploaded:
            audio_bytes = uploaded.read()
    else:
        duration = st.sidebar.slider("Duration (s)", 1.0, 30.0, 5.0, step=1.0)
        sr_rec = st.sidebar.selectbox("Record SR (Hz)", [8000, 16000, 44100, 48000], index=2)
        if st.sidebar.button("Start Recording"):
            audio_bytes = record_python(duration=duration, sr=sr_rec)

    if not audio_bytes:
        st.info("Please upload or record audio to proceed.")
        return

    # play & process
    st.audio(audio_bytes, format="audio/wav")
    sr, y_full = load_audio(audio_bytes)

    # window / embedding params
    duration_full = len(y_full) / sr
    default_end = min(duration_full, 1.0)
    st.sidebar.header("Select data window")
    t0, t1 = st.sidebar.slider("Window (s)", 0.0, duration_full, (0.0, default_end), step=0.01)
    i0, i1 = int(t0 * sr), int(t1 * sr)
    y_win = y_full[i0:i1]
    time_win = np.arange(i0, i1) / sr

    st.sidebar.header("Embedding parameters")
    tau = st.sidebar.slider("Delay τ (samples)", 1, 100, 10, 1)
    m = st.sidebar.slider("Dimension m", 2, 20, 3, 1)
    show_3d = st.sidebar.checkbox("Show 3D embedding", False)

    # compute spectrogram
    f, t_spec, Sxx_db = compute_spectrogram(y_full, sr)

    # render
    st.pyplot(plot_full_timeseries(y_full, sr, t0, t1))
    st.pyplot(plot_full_spectrogram(f, t_spec, Sxx_db, t0, t1))
    st.plotly_chart(plot_window_scatter(y_win, sr), use_container_width=True)

    # embedding
    X = compute_embedding(y_win, tau, m)
    if X is None:
        st.error(f"Segment too short for m={m}, τ={tau} (need > {(m-1)*tau} samples).")
        return

    st.plotly_chart(plot_embedding_2d(X, tau), use_container_width=False)
    if show_3d:
        if m >= 3:
            st.plotly_chart(plot_embedding_3d(X, time_win[: len(X)]), use_container_width=False)
        else:
            st.warning("Need m ≥ 3 for 3D embedding.")

if __name__ == "__main__":
    main()
