import io

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram as spgram
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- Audio Handling ---

def load_wav(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    """
    Load WAV from bytes and return sample rate and normalized mono waveform.
    """
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    y = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
    y /= np.max(np.abs(y))
    return sr, y


def record_from_mic() -> bytes | None:
    """
    Capture microphone input via WebRTC and return WAV bytes.
    """
    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
    )
    if ctx.audio_receiver:
        frames = ctx.audio_receiver.get_frames()
        if frames:
            sr = frames[0].sample_rate
            arr = np.concatenate([f.to_ndarray()[0] for f in frames])
            buf = io.BytesIO()
            wavfile.write(buf, sr, arr)
            return buf.getvalue()
    return None


def get_audio_source() -> tuple[int, np.ndarray] | None:
    """
    Sidebar choice: upload WAV or record from mic.
    Returns (sr, waveform) or None.
    """
    st.sidebar.header("Audio Source")
    source = st.sidebar.radio("Choose Source", ("Upload WAV", "Record from mic"))

    if source == "Record from mic":
        st.sidebar.warning("⚠️ Remember to select your microphone input device!")
        audio_bytes = record_from_mic()
        if audio_bytes:
            return load_wav(audio_bytes)
        else:
            return None

    uploaded = st.sidebar.file_uploader("Upload WAV file", type=["wav"])
    if uploaded:
        return load_wav(uploaded.read())
    return None

# --- Signal Processing ---

def compute_spectrogram(y: np.ndarray, sr: int,
                        nperseg: int = 1024, noverlap: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dB-scaled spectrogram of the signal.
    """
    f, t, Sxx = spgram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def compute_embedding(y: np.ndarray, tau: int, m: int) -> np.ndarray | None:
    """
    Build Takens embedding matrix of dimension m and delay tau.
    """
    N = len(y) - (m - 1) * tau
    if N <= 0:
        return None
    return np.column_stack([y[i:i+N] for i in range(0, m * tau, tau)])

# --- Plotting ---

def plot_waveform_window(y: np.ndarray, sr: int, window: tuple[float, float]) -> plt.Figure:
    t0, t1 = window
    times = np.arange(len(y)) / sr
    i0, i1 = int(t0 * sr), int(t1 * sr)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, y, color="lightgray", linewidth=1)
    ax.plot(times[i0:i1], y[i0:i1], color="red", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title="Waveform with Selected Window")
    return fig


def plot_spectrogram_window(f: np.ndarray, t: np.ndarray, Sxx_db: np.ndarray,
                            window: tuple[float, float]) -> plt.Figure:
    t0, t1 = window
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(t, f, Sxx_db, cmap="gray", shading="gouraud")
    ax.axvspan(t0, t1, color="red", alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
           title="Spectrogram with Selected Window")
    return fig


def plot_scatter_segment(y: np.ndarray, sr: int) -> go.Figure:
    times = np.arange(len(y)) / sr
    fig = px.scatter(x=times, y=y,
                     labels={"x": "Time (s)", "y": "Amplitude"},
                     title="Selected Segment Waveform")
    fig.update_traces(marker=dict(size=4))
    return fig


def plot_embedding_2d(X: np.ndarray, tau: int) -> go.Figure:
    fig = px.scatter(x=X[:, 0], y=X[:, 1],
                     labels={"x": "y(t)", "y": f"y(t+{tau})"},
                     title="2D Takens Embedding")
    fig.update_layout(width=600, height=600)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_embedding_3d(X: np.ndarray, times: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=times[:len(X)],
        mode="markers",
        marker=dict(size=4, color=times[:len(X)], colorscale="Viridis", showscale=False)
    ))
    fig.update_layout(scene=dict(
        xaxis_title="y(t)", yaxis_title="y(t+τ)", zaxis_title="Time (s)"
    ), width=600, height=600)
    return fig

# --- UI Controls ---

def select_window(duration: float) -> tuple[float, float]:
    st.sidebar.header("Select Window (s)")
    default = duration * 0.1
    return st.sidebar.slider("Window Range", 0.0, duration, (0.0, default), step=0.01)


def embedding_controls() -> tuple[int, int, bool]:
    st.sidebar.header("Embedding Parameters")
    tau = st.sidebar.slider("Delay τ (samples)", 1, 50, 10)
    m = st.sidebar.slider("Embedding Dimension m", 2, 20, 3)
    show3d = st.sidebar.checkbox("Show 3D Embedding", False)
    return tau, m, show3d

# --- Main Application ---

def main():
    st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
    st.title("Interactive Takens’ Time-Delay Embedding")

    audio_data = get_audio_source()
    if audio_data is None:
        st.info("Please upload or record audio to proceed.")
        return
    sr, y = audio_data
    duration = len(y) / sr

    # Playback
    buf = io.BytesIO()
    wavfile.write(buf, sr, y)
    st.audio(buf.getvalue(), format="audio/wav")

    # Window & embedding
    t0, t1 = select_window(duration)
    tau, m, show3d = embedding_controls()
    i0, i1 = int(t0*sr), int(t1*sr)
    y_seg = y[i0:i1]

    f, t_spec, Sxx_db = compute_spectrogram(y, sr)

    # Display
    st.pyplot(plot_waveform_window(y, sr, (t0, t1)))
    st.pyplot(plot_spectrogram_window(f, t_spec, Sxx_db, (t0, t1)))
    st.plotly_chart(plot_scatter_segment(y_seg, sr), use_container_width=True)

    X = compute_embedding(y_seg, tau, m)
    if X is None:
        st.error("Window too short for chosen τ and m.")
        return

    st.plotly_chart(plot_embedding_2d(X, tau), use_container_width=False)
    if show3d:
        times = np.arange(i0, i1) / sr
        st.plotly_chart(plot_embedding_3d(X, times), use_container_width=False)


if __name__ == '__main__':
    main()