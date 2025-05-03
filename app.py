import io
import numpy as np
import streamlit as st
from scipy.io import wavfile
import plotly.express as px
from plotly.graph_objs import Scatter3d

st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
st.title("Interactive Takens’ Time-Delay Embedding")

uploaded = st.file_uploader("Upload a WAV file (1–5 s, mono or stereo)", type=["wav"])
if not uploaded:
    st.info("Please upload a WAV file to begin.")
    st.stop()

# Read and normalize
bytes_wav = io.BytesIO(uploaded.read())
sr, data = wavfile.read(bytes_wav)
y = data.mean(axis=1) if data.ndim > 1 else data.astype(float)
y = y / np.max(np.abs(y))

st.sidebar.header("Embedding parameters")
τ = st.sidebar.slider("Delay τ (samples)", 1, 50, 10, 1)
m = st.sidebar.slider("Dimension m", 2, 20, 3, 1)
do_3d = st.sidebar.checkbox("Show 3D embedding", True)

N = len(y) - (m - 1) * τ
if N <= 0:
    st.error("Signal too short for these parameters. Try smaller τ or m.")
    st.stop()

X = np.column_stack([y[i : i + N] for i in range(0, m * τ, τ)])

# 2D plot
fig2d = px.scatter(
    x=X[:, 0], y=X[:, 1],
    labels={"x": "y(t)", "y": f"y(t+{τ})"},
    title="2D Takens Embedding"
)
st.plotly_chart(fig2d, use_container_width=True)

# 3D plot
if do_3d and m >= 3:
    fig3d = Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers", marker=dict(size=3)
    )
    layout3d = dict(
        scene=dict(
            xaxis_title="y(t)",
            yaxis_title=f"y(t+{τ})",
            zaxis_title=f"y(t+{2*τ})"
        ),
        title="3D Takens Embedding"
    )
    st.plotly_chart({"data": [fig3d], "layout": layout3d}, use_container_width=True)
elif do_3d:
    st.warning("m must be ≥ 3 to show a 3D embedding.")