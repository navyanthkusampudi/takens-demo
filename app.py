import io

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram as spgram
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# --- Data Loading & Recording ---

def load_wav(audio_bytes: bytes):
    """
    Load WAV from bytes and return sample rate and normalized mono signal.
    """
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    # Stereo to mono
    y = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
    y /= np.max(np.abs(y))
    return sr, y


def record_from_mic() -> bytes | None:
    """
    Capture audio via WebRTC and return WAV bytes when available.
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


# --- Signal Processing ---

def compute_spectrogram(y: np.ndarray, sr: int,
                        nperseg: int = 1024, noverlap: int = 512):
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
    return np.column_stack([y[i:i+N] for i in range(0, m*tau, tau)])


# --- Plotting Utilities ---

def plot_waveform(y: np.ndarray, sr: int, t0: float, t1: float) -> plt.Figure:
    """
    Plot full waveform (gray) with selected window highlighted (red).
    """
    times = np.arange(len(y)) / sr
    i0, i1 = int(t0*sr), int(t1*sr)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, y, color="lightgray")
    ax.plot(times[i0:i1], y[i0:i1], color="red")
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title="Waveform with Selected Window")
    return fig


def plot_spectrogram(f, t, Sxx_db, t0: float, t1: float) -> plt.Figure:
    """
    Plot full spectrogram (gray-white) with time window overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(t, f, Sxx_db, cmap="gray", shading="gouraud")
    ax.axvspan(t0, t1, color="red", alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
           title="Spectrogram with Selected Window")
    return fig


def plot_selected_scatter(y_win: np.ndarray, sr: int) -> go.Figure:
    """
    Plot selected window as Plotly scatter with marker size=4.
    """
    times = np.arange(len(y_win)) / sr
    fig = px.scatter(x=times, y=y_win,
                     labels={"x": "Time (s)", "y": "Amplitude"},
                     title="Selected Segment Waveform")
    fig.update_traces(marker=dict(size=4))
    return fig


def plot_embedding_2d(X: np.ndarray, tau: int) -> go.Figure:
    """
    2D Takens embedding (first two coords) in a 600x600 square.
    """
    fig = px.scatter(x=X[:,0], y=X[:,1],
                     labels={"x": "y(t)", "y": f"y(t+{tau})"},
                     title="2D Takens Embedding")
    fig.update_layout(width=600, height=600)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_embedding_3d(X: np.ndarray, times: np.ndarray) -> go.Figure:
    """
    3D embedding: x=y(t), y=y(t+τ), z=time.
    """
    fig = go.Figure(go.Scatter3d(
        x=X[:,0], y=X[:,1], z=times[:len(X)],
        mode="markers",
        marker=dict(size=4, color=times[:len(X)],
                    colorscale="Viridis", showscale=False)
    ))
    fig.update_layout(scene=dict(
        xaxis_title="y(t)", yaxis_title="y(t+τ)", zaxis_title="Time (s)"
    ), width=600, height=600)
    return fig


# --- Streamlit App Workflow ---
def main():
    st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
    inject_css()
    st.title("Interactive Takens’ Time-Delay Embedding")

    audio_bytes = get_audio_source()
    if not audio_bytes:
        st.info("Please upload or record audio to proceed.")
        return

    st.audio(audio_bytes, format="audio/wav")
    sr, y_full = load_wav(audio_bytes)

    # Sidebar controls
    t0, t1 = get_window(len(y_full)/sr)
    tau, m, show_3d = get_embedding_params()

    # Spectrogram once
    f, t_spec, Sxx_db = compute_spectrogram(y_full, sr)

    # Display plots
    st.pyplot(plot_waveform(y_full, sr, t0, t1))
    st.pyplot(plot_spectrogram(f, t_spec, Sxx_db, t0, t1))
    st.plotly_chart(plot_selected_scatter(y_full[int(t0*sr):int(t1*sr)], sr), use_container_width=True)

    X = compute_embedding(y_full[int(t0*sr):int(t1*sr)], tau, m)
    if X is None:
        st.error("Segment too short for given m/τ.")
        return

    st.plotly_chart(plot_embedding_2d(X, tau), use_container_width=False)
    if show_3d:
        st.plotly_chart(plot_embedding_3d(X, np.arange(int(t0*sr), int(t1*sr))/sr), use_container_width=False)


# --- UI Helper Functions ---
def inject_css():
    st.markdown('''
    <style>
      .webrtc-ui select, .webrtc-ui option, .webrtc-ui label {
        color: #000!important; background: #fff!important;
      }
    </style>
    ''', unsafe_allow_html=True)


def get_audio_source() -> bytes | None:
    st.sidebar.header("Audio Source")
    source = st.sidebar.radio("Choose Source", ("Upload WAV", "Record from mic"))
    if source == "Record from mic":
        st.sidebar.warning("⚠️ Remember to select your microphone input.")
        return record_from_mic()
    uploaded = st.sidebar.file_uploader("Upload WAV file", type=["wav"])
    return uploaded.read() if uploaded else None


def get_window(duration: float) -> tuple[float, float]:
    st.sidebar.header("Select Data Window (s)")
    default_end = duration / 10
    return st.sidebar.slider("Window", 0.0, duration, (0.0, default_end), step=0.01)


def get_embedding_params() -> tuple[int, int, bool]:
    st.sidebar.header("Embedding Parameters")
    tau = st.sidebar.slider("Delay τ (samples)", 1, 50, 10)
    m = st.sidebar.slider("Dimension m", 2, 20, 3)
    show_3d = st.sidebar.checkbox("Show 3D Embedding", False)
    return tau, m, show_3d


if __name__ == "__main__":
    main()
