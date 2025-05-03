import io

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram as spgram
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


def load_audio(audio_bytes):
    """
    Read WAV from raw bytes and return sample rate and normalized mono signal.
    """
    sr, data = wavfile.read(io.BytesIO(audio_bytes))
    # Convert stereo to mono if needed
    y = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
    # Normalize amplitude
    y /= np.max(np.abs(y))
    return sr, y


def compute_spectrogram(y, sr, nperseg=1024, noverlap=512):
    """
    Compute spectrogram (in dB) of signal y.
    Returns frequencies, times, and dB-scaled spectrogram.
    """
    f, t, Sxx = spgram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def compute_embedding(y, tau, m):
    """
    Build Takens time-delay embedding matrix of dimension m and delay tau.
    Returns array of shape (N, m), or None if window too short.
    """
    N = len(y) - (m - 1) * tau
    if N <= 0:
        return None
    return np.column_stack([y[i : i + N] for i in range(0, m * tau, tau)])


def plot_full_timeseries(y, sr, t0, t1):
    """
    Plot full time series in gray with selected window overlayed in red.
    """
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
    """
    Plot full spectrogram in gray-scale with a red transparent span for the selected window.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(t, f, Sxx_db, cmap="gray", shading="gouraud")
    ax.axvspan(t0, t1, color="red", alpha=0.3)
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
           title="Full Spectrogram with Selected Window")
    return fig


def plot_window_scatter(y_win, sr):
    """
    Plot selected window as a Plotly scatter time series.
    """
    time_win = np.arange(len(y_win)) / sr
    fig = px.scatter(
        x=time_win, y=y_win,
        labels={"x": "Time (s)", "y": "Amplitude"},
        title="Selected Window Time Series"
    )
    return fig


def plot_embedding_2d(X, tau):
    """
    Plot 2D Takens embedding as a 600x600 square with equal scales.
    """
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1],
        labels={"x": "y(t)", "y": f"y(t+{tau})"},
        title="2D Takens Embedding"
    )
    fig.update_layout(width=600, height=600)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_embedding_3d(X, tau):
    """
    Plot 3D Takens embedding with first three coordinates.
    """
    fig = go.Figure(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers", marker=dict(size=3, color="blue")
    ))
    fig.update_layout(
        title="3D Takens Embedding",
        scene=dict(
            xaxis_title="y(t)",
            yaxis_title=f"y(t+{tau})",
            zaxis_title=f"y(t+{2*tau})"
        ),
        width=600, height=600,
        margin=dict(t=40, b=40)
    )
    return fig


def main():
    st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
    st.title("Interactive Takens’ Time-Delay Embedding")

    # Upload and play audio
    uploaded = st.file_uploader("Upload a WAV file (mono or stereo)", type=["wav"])
    if not uploaded:
        st.info("Please upload a WAV file to begin.")
        return

    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")
    sr, y_full = load_audio(audio_bytes)

    # Sidebar controls
    duration = len(y_full) / sr
    st.sidebar.header("Select data window")
    default_end = duration / 10
    t0, t1 = st.sidebar.slider(
        "Window (s)", 0.0, float(duration), (0.0, float(default_end)), step=0.01
    )
    i0, i1 = int(t0 * sr), int(t1 * sr)
    y_win = y_full[i0:i1]

    st.sidebar.header("Embedding parameters")
    tau = st.sidebar.slider("Delay τ (samples)", 1, 50, 10, 1)
    m = st.sidebar.slider("Dimension m", 2, 20, 3, 1)
    show_3d = st.sidebar.checkbox("Show 3D embedding", False)

    # Precompute spectrogram
    f, t_spec, Sxx_db = compute_spectrogram(y_full, sr)

    # Plots
    st.pyplot(plot_full_timeseries(y_full, sr, t0, t1))
    st.pyplot(plot_full_spectrogram(f, t_spec, Sxx_db, t0, t1))
    st.plotly_chart(plot_window_scatter(y_win, sr), use_container_width=True)

    # Embedding
    X = compute_embedding(y_win, tau, m)
    if X is None:
        st.error(f"Segment too short for m={m}, τ={tau} (need > {(m-1)*tau} samples).")
        return
    st.plotly_chart(plot_embedding_2d(X, tau), use_container_width=False)
    if show_3d:
        if m >= 3:
            st.plotly_chart(plot_embedding_3d(X, tau), use_container_width=False)
        else:
            st.warning("Need m ≥ 3 for 3D embedding.")

if __name__ == "__main__":
    main()
