from __future__ import annotations
import io
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neuro-Learn", layout="wide", page_icon="🧠")
st.title("🧠 Neuro-Learn — Content Engagement Analyzer")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mock_mode = st.toggle("Mock mode (no GPU needed)", value=True)
    window_sec = st.slider("Window size (seconds)", 1, 30, 5)
    api_url = st.text_input("API URL", value=API_URL)

# ── Upload + analyze ─────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload video or audio file",
    type=["mp4", "mp3", "wav"],
)

if uploaded_file and st.button("▶ Analyze", type="primary"):
    with st.spinner("Running inference… (this may take a while in real mode)"):
        try:
            response = requests.post(
                f"{api_url}/analyze",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                data={"mock": str(mock_mode).lower(), "window_sec": str(window_sec)},
                timeout=600,
            )
            if response.status_code == 200:
                st.session_state["result"] = response.json()
                st.session_state["uploaded_bytes"] = uploaded_file.getvalue()
                st.session_state["uploaded_name"] = uploaded_file.name
            else:
                st.error(f"API error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at {api_url}. Is it running?")

# ── Results ───────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    uploaded_bytes = st.session_state["uploaded_bytes"]
    uploaded_name = st.session_state["uploaded_name"]

    st.metric("Overall Engagement Score", f"{result['overall_score']} / 100")

    # ── Engagement curve ─────────────────────────────────────────────────
    df = pd.DataFrame(result["timeseries"])
    fig = go.Figure()

    for seg in result["flagged_segments"]:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="red", opacity=0.15, line_width=0,
            annotation_text="⚠", annotation_position="top left",
        )

    fig.add_trace(go.Scatter(
        x=df["t_start"], y=df["score"],
        mode="lines+markers",
        line=dict(color="#4C9EFF", width=2),
        marker=dict(
            color=["#FF4444" if f else "#4C9EFF" for f in df["flagged"]],
            size=8,
        ),
        name="Engagement",
    ))
    fig.update_layout(
        title="Engagement Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Score (0–100)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Timeline slider ───────────────────────────────────────────────────
    max_t = max(1, int(result["duration_sec"]) - 1)
    t = st.slider("⏱ Timeline", 0, max_t, 0, key="timeline")

    # ── Bottom panels ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Media Player")
        suffix = Path(uploaded_name).suffix.lower()
        tmp_media = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_media.write(uploaded_bytes)
        tmp_media.close()
        if suffix == ".mp4":
            st.video(tmp_media.name, start_time=t)
        else:
            st.audio(tmp_media.name, start_time=t)

    with col2:
        st.subheader("Brain Activation")
        gif_url = f"{api_url}{result['brain_animation_path']}"
        gif_resp = requests.get(gif_url, timeout=30)
        if gif_resp.status_code == 200:
            reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            try:
                n_frames = reader.get_length()
            except Exception:
                n_frames = len(list(reader))
                reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            frame_idx = min(t, n_frames - 1)
            frame = reader.get_data(frame_idx)
            st.image(frame, caption=f"Brain activation at t = {t}s", use_container_width=True)
        else:
            st.warning("Brain animation not available.")

    # ── Flagged segments table ────────────────────────────────────────────
    if result["flagged_segments"]:
        st.subheader("⚠ Flagged Segments — Low Engagement")
        flagged_df = pd.DataFrame(result["flagged_segments"])
        flagged_df.columns = ["Start (s)", "End (s)", "Score", "Suggestion"]
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    else:
        st.success("No low-engagement segments detected.")
