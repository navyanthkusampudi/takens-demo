import io

import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
st.title("Interactive Takens’ Time-Delay Embedding")

# --- 1) Upload & read ---
uploaded = st.file_uploader("Upload a WAV file (mono or stereo)", type=["wav"])
if not uploaded:
    st.info("Please upload a WAV file to begin.")
    st.stop()

bytes_wav = io.BytesIO(uploaded.read())
sr, data = wavfile.read(bytes_wav)
# stereo → mono
y_full = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
y_full /= np.max(np.abs(y_full))

# --- 2) Pre-compute spectrogram on the full signal ---
f, t_spec, Sxx = spectrogram(y_full, fs=sr, nperseg=1024, noverlap=512)
# power → decibels
Sxx_db = 10 * np.log10(Sxx + 1e-10)

# --- 3) Sidebar: select a window in seconds ---
duration = len(y_full) / sr
st.sidebar.header("Select data window")
t0, t1 = st.sidebar.slider(
    "Window (sec)",
    min_value=0.0,
    max_value=float(duration),
    value=(0.0, float(duration)),
    step=0.01,
)
i0, i1 = int(t0 * sr), int(t1 * sr)
y_win = y_full[i0:i1]
actual_dur = (i1 - i0) / sr
st.sidebar.write(f"Selected segment: {actual_dur:.2f} s")

# --- 4) Sidebar: embedding parameters ---
st.sidebar.header("Embedding parameters")
τ = st.sidebar.slider("Delay τ (samples)", 1, 50, 10, 1)
m = st.sidebar.slider("Dimension m", 2, 20, 3, 1)
show_3d = st.sidebar.checkbox("Show 3D embedding", value=True)

# --- 5) Plot full time series + highlight window ---
time_full = np.arange(len(y_full)) / sr
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=time_full, y=y_full,
    mode="lines",
    line=dict(color="lightgray"),
    name="Full signal"
))
fig_ts.add_trace(go.Scatter(
    x=time_full[i0:i1], y=y_full[i0:i1],
    mode="lines",
    line=dict(color="red"),
    name="Selected window"
))
fig_ts.update_layout(
    title="Full Time Series with Selected Window",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    showlegend=False,
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_ts, use_container_width=True)

# --- 6) Plot full spectrogram + highlight window ---
fig_spec = go.Figure()
fig_spec.add_trace(go.Heatmap(
    z=Sxx_db,
    x=t_spec,
    y=f,
    colorscale=[[0, "white"], [1, "black"]],
    showscale=False,
    zsmooth="best"
))
# overlay rectangle
fig_spec.update_layout(
    title="Full Spectrogram with Selected Window",
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    shapes=[
        dict(
            type="rect",
            x0=t0, x1=t1,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor="red",
            opacity=0.2,
            layer="above",
            line_width=0,
        )
    ],
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig_spec, use_container_width=True)

# --- 7) Takens embedding on the windowed data ---
N = len(y_win) - (m - 1) * τ
if N <= 0:
    st.error(
        f"Segment too short for m={m}, τ={τ} "
        f"(need > {(m-1)*τ} samples)."
    )
    st.stop()

# build embedding matrix
X = np.column_stack([y_win[i : i + N] for i in range(0, m*τ, τ)])

# 2D scatter
fig2d = px.scatter(
    x=X[:, 0],
    y=X[:, 1],
    labels={"x": "y(t)", "y": f"y(t+{τ})"},
    title="2D Takens Embedding"
)
st.plotly_chart(fig2d, use_container_width=True)

# optional 3D
if show_3d:
    if m >= 3:
        trace3d = go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers", marker=dict(size=3, color="blue")
        )
        layout3d = dict(
            scene=dict(
                xaxis_title="y(t)",
                yaxis_title=f"y(t+{τ})",
                zaxis_title=f"y(t+{2*τ})",
            ),
            title="3D Takens Embedding",
            margin=dict(t=40, b=40)
        )
        st.plotly_chart({"data":[trace3d], "layout":layout3d}, use_container_width=True)
    else:
        st.warning("Need m ≥ 3 to show the 3D plot.")
