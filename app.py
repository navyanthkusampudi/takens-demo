import io

import numpy as np
import streamlit as st
from scipy.io import wavfile
import plotly.express as px
from plotly.graph_objs import Scatter3d

st.set_page_config(page_title="Takens Embedding Demo", layout="wide")
st.title("Interactive Takens’ Time-Delay Embedding")

# 1. Upload
uploaded = st.file_uploader("Upload a WAV file (mono or stereo)", type=["wav"])
if not uploaded:
    st.info("Please upload a WAV file to begin.")
    st.stop()

# 2. Read & normalize
bytes_wav = io.BytesIO(uploaded.read())
sr, data = wavfile.read(bytes_wav)
# stereo → mono
y = data.mean(axis=1).astype(float) if data.ndim > 1 else data.astype(float)
y /= np.max(np.abs(y))

# 3. Let user pick a time window
duration = len(y) / sr
st.sidebar.header("Select data window")
t0, t1 = st.sidebar.slider(
    "Window (seconds)",
    min_value=0.0,
    max_value=float(duration)/10.0,
    value=(0.0, float(duration)),
    step=0.01,
)
i0, i1 = int(t0 * sr), int(t1 * sr)
y = y[i0:i1]
actual_dur = (i1 - i0) / sr
st.sidebar.write(f"Selected: {actual_dur:.2f} s segment")

# 4. Embedding parameters
st.sidebar.header("Embedding parameters")
τ = st.sidebar.slider("Delay τ (samples)", 1, 50, 10, 1)
m = st.sidebar.slider("Dimension m", 2, 20, 3, 1)
show_3d = st.sidebar.checkbox("Show 3D (first 3 coords)", value=True)

# 5. Check window length
N = len(y) - (m - 1) * τ
if N <= 0:
    st.error(
        f"Segment too short for m={m} and τ={τ} (need > {(m-1)*τ} samples). "
        "Try a longer window or smaller m/τ."
    )
    st.stop()

# 6. Build embedding
X = np.column_stack([y[i : i + N] for i in range(0, m * τ, τ)])

# 7. 2D plot
fig2d = px.scatter(
    x=X[:, 0],
    y=X[:, 1],
    labels={"x": "y(t)", "y": f"y(t+{τ})"},
    title="2D Takens Embedding",
)
st.plotly_chart(fig2d, use_container_width=True)

# 8. Optional 3D
if show_3d:
    if m >= 3:
        fig3d = Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(size=3),
        )
        layout3d = dict(
            scene=dict(
                xaxis_title="y(t)",
                yaxis_title=f"y(t+{τ})",
                zaxis_title=f"y(t+{2*τ})",
            ),
            title="3D Takens Embedding",
        )
        st.plotly_chart({"data": [fig3d], "layout": layout3d}, use_container_width=True)
    else:
        st.warning("Need m ≥ 3 to show a 3D plot.")
