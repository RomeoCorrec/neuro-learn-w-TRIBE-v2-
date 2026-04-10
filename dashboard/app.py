from __future__ import annotations
import io
import os
import tempfile
import threading
import time
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neuro-Learn", layout="wide", page_icon="🧠")

# ── Design system ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:          #030B1A;
    --bg-card:     #071428;
    --bg-panel:    #0A1B35;
    --cyan:        #00D4FF;
    --cyan-dim:    rgba(0, 212, 255, 0.10);
    --cyan-glow:   rgba(0, 212, 255, 0.30);
    --purple:      #6E4EDB;
    --amber:       #FFB347;
    --red:         #FF4B4B;
    --text:        #C8DFF5;
    --text-dim:    #3D6080;
    --border:      rgba(0, 212, 255, 0.13);
    --font-d:      'Syne', sans-serif;
    --font-m:      'DM Mono', monospace;
}

/* ── Base ── */
html, body { background: var(--bg) !important; }

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 30%, rgba(0, 80, 200, 0.07) 0%, transparent 100%),
        radial-gradient(ellipse 50% 60% at 85% 70%, rgba(110, 78, 219, 0.06) 0%, transparent 100%);
    color: var(--text) !important;
    font-family: var(--font-m) !important;
}

/* Scanlines overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0, 212, 255, 0.012) 3px,
        rgba(0, 212, 255, 0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #040D1E !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: var(--font-m) !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] h2 {
    font-family: var(--font-d) !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    color: var(--cyan) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: var(--font-d) !important;
    letter-spacing: -0.02em;
}

/* ── Custom title block ── */
.nl-title-bar {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.nl-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 14px var(--cyan), 0 0 28px rgba(0,212,255,0.4);
    flex-shrink: 0;
    animation: nl-pulse 2.4s ease-in-out infinite;
}
@keyframes nl-pulse {
    0%,100% { box-shadow: 0 0 14px var(--cyan), 0 0 28px rgba(0,212,255,0.4); }
    50%      { box-shadow: 0 0 4px  var(--cyan), 0 0 8px  rgba(0,212,255,0.2); }
}
.nl-title {
    font-family: var(--font-d);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0;
}
.nl-title em { color: var(--cyan); font-style: normal; }
.nl-sub {
    font-family: var(--font-m);
    font-size: 0.62rem;
    color: var(--text-dim);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.nl-tag {
    margin-left: auto;
    font-family: var(--font-m);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 0.3rem 0.65rem;
    border-radius: 2px;
}

/* ── Section labels ── */
.nl-section {
    font-family: var(--font-m);
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem;
}
.nl-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Primary button ── */
.stButton > button {
    font-family: var(--font-m) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-dim) !important;
    border-radius: 2px !important;
    transition: all 0.18s ease !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 16px var(--cyan-glow) !important;
    background: var(--cyan-dim) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--cyan) !important;
    color: var(--bg) !important;
    border-color: var(--cyan) !important;
    font-weight: 500 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: #33DEFF !important;
    box-shadow: 0 0 24px var(--cyan-glow) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(0,212,255,0.2) !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    background: var(--cyan-dim) !important;
}
[data-testid="stFileUploader"] * {
    font-family: var(--font-m) !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
}
[data-testid="stFileUploaderFileName"] {
    color: var(--text) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    height: 6px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--cyan) 0%, var(--purple) 100%) !important;
    box-shadow: 0 0 10px var(--cyan-glow) !important;
    border-radius: 2px !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1.25rem 1.5rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, var(--cyan), var(--purple));
    box-shadow: 0 0 12px var(--cyan);
}
[data-testid="stMetric"] label,
[data-testid="stMetricLabel"] * {
    font-family: var(--font-m) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] * {
    font-family: var(--font-d) !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
    letter-spacing: -0.04em !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .st-bx,
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: var(--border) !important;
}
[data-testid="stSlider"] [aria-label] {
    color: var(--text) !important;
    font-family: var(--font-m) !important;
}
[data-testid="stSlider"] [data-baseweb="thumb"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 8px var(--cyan-glow) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background: var(--cyan) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--font-m) !important;
    font-size: 0.78rem !important;
    color: var(--text) !important;
}

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] * {
    font-family: var(--font-m) !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.08em !important;
}

/* ── Toggle ── */
.stToggle p, .stToggle label {
    font-family: var(--font-m) !important;
    font-size: 0.78rem !important;
    color: var(--text) !important;
}

/* ── Text input ── */
.stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: var(--font-m) !important;
    font-size: 0.78rem !important;
}
.stTextInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px var(--cyan-dim) !important;
}
.stTextInput label {
    font-family: var(--font-m) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* ── Image ── */
.stImage img {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Selectbox / other inputs ── */
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    font-family: var(--font-m) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nl-title-bar">
    <div class="nl-dot"></div>
    <div>
        <div class="nl-title">NEURO<em>·</em>LEARN</div>
        <div class="nl-sub">Content Engagement Analyzer &nbsp;·&nbsp; Neural Signal Mapping</div>
    </div>
    <div class="nl-tag">v0.1.0</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mock_mode = st.toggle("Mock mode (no GPU needed)", value=True)
    window_sec = st.slider("Window size (seconds)", 1, 30, 5)
    api_url = st.text_input("API URL", value=API_URL)

# ── Upload + analyze ──────────────────────────────────────────────────────
st.markdown('<div class="nl-section">Input</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload video or audio file",
    type=["mp4", "mp3", "wav"],
    label_visibility="collapsed",
)

STAGE_LABELS = {
    "uploading":      "Uploading file…",
    "loading":        "Loading and converting audio…",
    "inference":      "Running brain encoding model…",
    "roi_extraction": "Extracting brain regions…",
    "scoring":        "Computing engagement scores…",
    "animation":      "Generating brain animation…",
    "done":           "Done.",
    "idle":           "",
}

if uploaded_file and st.button("▶  Analyze", type="primary"):
    result_container: dict = {}
    file_bytes = uploaded_file.getvalue()

    def _run_analyze() -> None:
        try:
            resp = requests.post(
                f"{api_url}/analyze",
                files={"file": (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)},
                data={"mock": str(mock_mode).lower(), "window_sec": str(window_sec)},
                timeout=600,
            )
            result_container["response"] = resp
        except Exception as exc:
            result_container["error"] = exc

    thread = threading.Thread(target=_run_analyze, daemon=True)
    thread.start()

    progress_bar = st.progress(0)
    stage_label = st.empty()

    while thread.is_alive():
        try:
            prog = requests.get(f"{api_url}/progress", timeout=2).json()
            pct = prog.get("pct", 0)
            stage = prog.get("stage", "idle")
            progress_bar.progress(pct)
            stage_label.caption(STAGE_LABELS.get(stage, stage))
        except Exception:
            pass
        time.sleep(0.5)

    thread.join()
    progress_bar.progress(100)
    stage_label.empty()

    if "error" in result_container:
        exc = result_container["error"]
        if isinstance(exc, requests.exceptions.ConnectionError):
            st.error(f"Cannot reach API at {api_url}. Is it running?")
        else:
            st.error(f"Request failed: {exc}")
    else:
        response = result_container["response"]
        if response.status_code == 200:
            st.session_state["result"] = response.json()
            st.session_state["uploaded_bytes"] = file_bytes
            st.session_state["uploaded_name"] = uploaded_file.name
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text or "Unknown error"
            st.error(f"API error {response.status_code}: {detail}")

# ── Results ───────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    uploaded_bytes = st.session_state["uploaded_bytes"]
    uploaded_name = st.session_state["uploaded_name"]

    st.markdown('<div class="nl-section">Engagement Score</div>', unsafe_allow_html=True)
    st.metric("Overall Engagement Score", f"{result['overall_score']} / 100")

    # ── Engagement curve ──────────────────────────────────────────────────
    st.markdown('<div class="nl-section">Neural Signal Timeline</div>', unsafe_allow_html=True)
    df = pd.DataFrame(result["timeseries"])
    fig = go.Figure()

    for seg in result["flagged_segments"]:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="rgba(255, 75, 75, 0.08)",
            line=dict(color="rgba(255,75,75,0.35)", width=1, dash="dot"),
            annotation_text="⚠",
            annotation_position="top left",
            annotation=dict(
                font=dict(color="#FF4B4B", size=10, family="DM Mono"),
            ),
        )

    fig.add_trace(go.Scatter(
        x=df["t_start"], y=df["score"],
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.05)",
        marker=dict(
            color=["#FF4B4B" if f else "#00D4FF" for f in df["flagged"]],
            size=7,
            line=dict(color="rgba(0,0,0,0.4)", width=1),
        ),
        name="Engagement",
        hovertemplate="<b>t = %{x}s</b><br>Score: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,40,0.6)",
        xaxis=dict(
            title="Time (s)",
            gridcolor="rgba(0,212,255,0.07)",
            color="#3D6080",
            tickfont=dict(family="DM Mono", size=10, color="#3D6080"),
            linecolor="rgba(0,212,255,0.13)",
            title_font=dict(family="DM Mono", size=10, color="#3D6080"),
        ),
        yaxis=dict(
            title="Score",
            range=[0, 100],
            gridcolor="rgba(0,212,255,0.07)",
            color="#3D6080",
            tickfont=dict(family="DM Mono", size=10, color="#3D6080"),
            linecolor="rgba(0,212,255,0.13)",
            title_font=dict(family="DM Mono", size=10, color="#3D6080"),
        ),
        height=260,
        margin=dict(l=45, r=20, t=15, b=45),
        font=dict(family="DM Mono"),
        legend=dict(
            font=dict(family="DM Mono", size=10, color="#3D6080"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#071428",
            bordercolor="#00D4FF",
            font=dict(family="DM Mono", size=11, color="#C8DFF5"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Timeline slider ────────────────────────────────────────────────────
    max_t = max(1, int(result["duration_sec"]) - 1)
    t = st.slider("Timeline", 0, max_t, 0, key="timeline")

    # ── Bottom panels ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="nl-section">Media</div>', unsafe_allow_html=True)
        suffix = Path(uploaded_name).suffix.lower()
        tmp_media = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_media.write(uploaded_bytes)
        tmp_media.close()
        if suffix == ".mp4":
            st.video(tmp_media.name, start_time=t)
        else:
            st.audio(tmp_media.name, start_time=t)

    with col2:
        st.markdown('<div class="nl-section">Brain Activation</div>', unsafe_allow_html=True)
        gif_url = f"{api_url}{result['brain_animation_path']}"
        gif_resp = requests.get(gif_url, timeout=30)
        if gif_resp.status_code == 200:
            reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            try:
                n_frames = reader.get_length()
            except Exception:
                n_frames = len(list(reader))
                reader = imageio.get_reader(io.BytesIO(gif_resp.content))
            frame_idx = min(int(t * n_frames / max(1, int(result["duration_sec"]))), n_frames - 1)
            frame = reader.get_data(frame_idx)
            st.image(frame, caption=f"t = {t}s", use_container_width=True)
        else:
            st.warning("Brain animation not available.")

    # ── Flagged segments ───────────────────────────────────────────────────
    st.markdown('<div class="nl-section">Flagged Segments</div>', unsafe_allow_html=True)
    if result["flagged_segments"]:
        flagged_df = pd.DataFrame(result["flagged_segments"])
        flagged_df.columns = ["Start (s)", "End (s)", "Score", "Suggestion"]
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    else:
        st.success("No low-engagement segments detected.")
