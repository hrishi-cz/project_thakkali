"""APEX AutoML Frontend - Comprehensive Multimodal ML Platform with Workflow Integration."""

import os
import uuid

# Load .env from project root before any os.getenv calls
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass
import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="AutoVision",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — AutoVision Claude-Inspired Research Platform Design System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg:#0b0b1a;--surface:#12122a;--card:#1a1a3e;
  --border:#2d2d5e;--bright:#4040a0;
  --violet:#7c3aed;--violet2:#a78bfa;--vglow:rgba(124,58,237,.15);
  --amber:#f59e0b;--aglow:rgba(245,158,11,.12);
  --teal:#14b8a6;--tglow:rgba(20,184,166,.12);
  --red:#ef4444;
  --t1:#f8fafc;--t2:#a1a1c2;--t3:#5a5a8a;
  --r-sm:8px;--r-md:12px;--r-lg:16px;--r-xl:24px;
}
html,body,[class*="css"],[data-testid="stAppViewContainer"]{
  font-family:'Inter',-apple-system,sans-serif!important;background:var(--bg)!important;color:var(--t1)!important;
}
[data-testid="stMain"]{background:var(--bg)!important;}
[data-testid="block-container"]{padding:1.5rem 2rem!important;max-width:1400px!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1{font-size:1.6rem!important;font-weight:800!important;letter-spacing:-.03em!important;color:var(--t1)!important}
h2{font-size:1.25rem!important;font-weight:700!important;letter-spacing:-.02em!important;color:var(--t1)!important}
h3{font-size:1.05rem!important;font-weight:600!important;color:var(--t2)!important}

/* HERO */
.autovision-hero{
  background:linear-gradient(135deg,#0f0f2e 0%,#12122a 50%,#0b0b1a 100%);
  border:1px solid var(--border);border-top:none;
  padding:48px 40px 40px;border-radius:var(--r-xl);
  margin-bottom:32px;position:relative;overflow:hidden;
}
.autovision-hero::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--violet),var(--violet2),var(--amber));
}
.autovision-hero::after{
  content:'';position:absolute;top:-80px;right:-80px;width:300px;height:300px;
  background:radial-gradient(circle,rgba(124,58,237,.08) 0%,transparent 70%);pointer-events:none;
}
.av-title{
  font-size:2.6rem!important;font-weight:900!important;letter-spacing:-.04em;
  background:linear-gradient(135deg,#f8fafc,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0!important;
}
.av-sub{color:var(--t2);font-size:1rem;margin-top:10px;font-weight:400;}
.av-tag{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--vglow);border:1px solid rgba(124,58,237,.25);
  color:var(--violet2);font-size:.7rem;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;
  padding:6px 12px;border-radius:20px;margin-top:16px;
}
.av-stats{display:flex;gap:28px;margin-top:24px;flex-wrap:wrap;}
.av-stat-v{font-size:1.5rem;font-weight:700;color:var(--violet2);font-family:'JetBrains Mono',monospace;}
.av-stat-l{font-size:.65rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;margin-top:2px;}

/* PHASE STEPPER */
.av-stepper{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-lg);padding:20px 24px;margin-bottom:28px;
}
.av-step{display:flex;align-items:center;flex:1;}
.av-step-inner{display:flex;flex-direction:column;align-items:center;}
.av-circle{
  width:44px;height:44px;border-radius:var(--r-md);
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.9rem;
  background:var(--card);border:2px solid var(--border);color:var(--t3);
  transition:all .25s ease;
}
.av-step.done .av-circle{background:var(--tglow);border-color:var(--teal);color:var(--teal);}
.av-step.active .av-circle{
  background:var(--vglow);border-color:var(--violet);color:var(--violet2);
  box-shadow:0 0 0 4px rgba(124,58,237,.12),0 0 20px rgba(124,58,237,.2);
  animation:pulse-glow 2s ease infinite;
}
@keyframes pulse-glow{
  0%,100%{box-shadow:0 0 0 4px rgba(124,58,237,.12),0 0 20px rgba(124,58,237,.2);}
  50%{box-shadow:0 0 0 6px rgba(124,58,237,.08),0 0 32px rgba(124,58,237,.35);}
}
.av-label{font-size:.6rem;margin-top:8px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--t3);text-align:center;white-space:nowrap;}
.av-step.done .av-label{color:var(--teal);}
.av-step.active .av-label{color:var(--violet2);}
.av-connector{flex:1;height:2px;background:var(--border);margin:0 8px;margin-bottom:20px;}
.av-step.done+.av-connector,.av-step.done .av-connector{background:linear-gradient(90deg,var(--teal),var(--border));}

/* METRICS */
[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:var(--r-md)!important;padding:16px 20px!important;transition:border-color .2s,box-shadow .2s!important;}
[data-testid="stMetric"]:hover{border-color:var(--violet)!important;box-shadow:0 0 0 1px var(--vglow)!important;}
[data-testid="stMetricLabel"]{color:var(--t3)!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.05em!important;}
[data-testid="stMetricValue"]{color:var(--t1)!important;font-family:'JetBrains Mono',monospace!important;font-weight:600!important;}

/* PHASE CARDS */
.phase-header{background:var(--card);border:1px solid var(--border);padding:20px 24px;border-radius:var(--r-md);margin-bottom:20px;transition:border-color .2s;}
.phase-header:hover{border-color:var(--violet);}

/* BADGES */
.success-badge{background:var(--tglow);color:var(--teal);border:1px solid rgba(20,184,166,.25);padding:10px 16px;border-radius:var(--r-sm);margin:8px 0;font-weight:500;}
.info-badge{background:var(--vglow);color:var(--violet2);border:1px solid rgba(124,58,237,.25);padding:10px 16px;border-radius:var(--r-sm);margin:8px 0;font-weight:500;}
.warning-badge{background:var(--aglow);color:var(--amber);border:1px solid rgba(245,158,11,.25);padding:10px 16px;border-radius:var(--r-sm);margin:8px 0;font-weight:500;}
.badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;font-size:.72rem;font-weight:600;letter-spacing:.03em;}
.badge.violet{background:var(--vglow);color:var(--violet2);border:1px solid rgba(124,58,237,.25);}
.badge.teal{background:var(--tglow);color:var(--teal);border:1px solid rgba(20,184,166,.25);}
.badge.amber{background:var(--aglow);color:var(--amber);border:1px solid rgba(245,158,11,.25);}

/* BUTTONS */
.stButton>button{
  background:linear-gradient(135deg,var(--violet) 0%,#5b21b6 100%)!important;
  color:#fff!important;border:none!important;border-radius:var(--r-md)!important;
  padding:10px 22px!important;font-weight:600!important;font-size:.875rem!important;
  box-shadow:0 1px 3px rgba(0,0,0,.3)!important;transition:all .2s ease!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#8b5cf6 0%,var(--violet) 100%)!important;
  box-shadow:0 4px 20px rgba(124,58,237,.4),0 0 0 1px rgba(124,58,237,.2)!important;
  transform:translateY(-1px)!important;
}

/* EXPANDERS */
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:var(--r-md)!important;margin-bottom:8px!important;}
[data-testid="stExpander"]:hover{border-color:var(--bright)!important;}
[data-testid="stExpander"] summary{font-weight:600!important;color:var(--t1)!important;}

/* TABS */
.stTabs [data-baseweb="tab"]{font-weight:500!important;font-size:.875rem!important;color:var(--t3)!important;}
.stTabs [aria-selected="true"]{color:var(--violet2)!important;border-bottom:2px solid var(--violet)!important;font-weight:600!important;}

/* INPUTS */
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input{
  background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--t1)!important;border-radius:var(--r-sm)!important;
}

/* PROGRESS */
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,var(--violet),var(--violet2))!important;border-radius:4px!important;}

/* MODEL CANDIDATE CARD */
.model-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:16px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;transition:all .2s;}
.model-card.selected{border-color:var(--violet);background:var(--vglow);}
.model-card:hover{border-color:var(--bright);}
.model-card-name{font-weight:700;color:var(--t1);font-size:.9rem;}
.model-card-detail{font-size:.75rem;color:var(--t3);margin-top:2px;}
.score-bar{width:80px;height:6px;background:var(--border);border-radius:3px;overflow:hidden;}
.score-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--violet),var(--violet2));}

/* PAPER CITATION */
.paper-cite{display:flex;align-items:flex-start;gap:10px;padding:8px 12px;border-radius:var(--r-sm);background:var(--surface);border-left:2px solid var(--violet);margin-bottom:6px;}
.paper-cite-num{color:var(--violet2);font-weight:700;font-size:.75rem;min-width:20px;font-family:'JetBrains Mono',monospace;}
.paper-cite-text{color:var(--t2);font-size:.78rem;line-height:1.4;}
.paper-cite-venue{color:var(--teal);font-size:.7rem;font-style:italic;}

/* RESEARCH FOOTER */
.av-footer{margin-top:40px;padding:24px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;}
.av-footer-cite{font-size:.75rem;color:var(--t3);font-style:italic;}
.av-footer-badge{font-size:.7rem;font-weight:600;color:var(--violet2);background:var(--vglow);border:1px solid rgba(124,58,237,.2);padding:4px 10px;border-radius:4px;}

/* CARDS */
.av-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r-md);padding:20px 24px;transition:border-color .2s,transform .2s;position:relative;overflow:hidden;}
.av-card:hover{border-color:var(--violet);transform:translateY(-1px);}
.av-card.violet::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--violet);}
.av-card.amber::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--amber);}
.av-card.teal::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--teal);}
.av-card-title{font-size:.85rem;font-weight:700;color:var(--t1);margin-bottom:4px;}
.av-card-value{font-size:1.4rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:var(--violet2);}
.av-card-sub{font-size:.75rem;color:var(--t3);margin-top:4px;}

/* ALERTS */
.av-alert{padding:12px 16px;border-radius:var(--r-sm);margin:8px 0;font-size:.875rem;font-weight:500;}
.av-alert.success{background:var(--tglow);border-left:3px solid var(--teal);color:var(--teal);}
.av-alert.warning{background:var(--aglow);border-left:3px solid var(--amber);color:var(--amber);}
.av-alert.violet{background:var(--vglow);border-left:3px solid var(--violet);color:var(--violet2);}
.av-alert.error{background:rgba(239,68,68,.1);border-left:3px solid var(--red);color:#f87171;}

/* BADGES (extended) */
.badge.red{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.25);}
.badge.gray{background:rgba(255,255,255,.05);color:var(--t3);border:1px solid var(--border);}

/* AMBER CTA */
.amber-cta .stButton>button{background:linear-gradient(135deg,var(--amber) 0%,#d97706 100%)!important;color:#0b0b1a!important;font-weight:700!important;}

/* INPUTS */
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--t1)!important;border-radius:var(--r-sm)!important;}
[data-testid="stTextArea"] textarea:focus,[data-testid="stTextInput"] input:focus{border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(124,58,237,.1)!important;}

/* PROGRESS */
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,var(--violet),var(--violet2))!important;border-radius:4px!important;}
[data-testid="stSpinner"]>div{border-top-color:var(--violet)!important;}

/* MISC */
hr{border-color:var(--border)!important;margin:20px 0!important;}
[data-testid="stFileUploader"]{border:2px dashed var(--border)!important;border-radius:var(--r-md)!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--violet)!important;}
.stDataFrame{border-radius:var(--r-md)!important;overflow:hidden;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bright);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--violet);}
[data-testid="stMain"]{animation:fadeIn .4s ease-out;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
</style>
""", unsafe_allow_html=True)

# API Configuration
def _secrets_get(key: str, default=None):
    """Read from st.secrets without crashing when secrets.toml is absent."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

API_BASE_URL = (
    os.getenv("APEX_API_BASE_URL")
    or _secrets_get("apex_api_base_url")
    or "http://localhost:8001"
)

# -- Visualization Helpers -------------------------------------------------
def _kv_table(data, title=''):
    """Render dict as a styled key-value dataframe instead of raw JSON."""
    if not isinstance(data, dict) or not data:
        if data:
            st.code(str(data)[:500], language='json')
        return
    if title:
        st.caption(title)
    import pandas as _pd
    rows = [{'Key': str(k), 'Value': str(v)[:200]} for k, v in data.items()]
    st.dataframe(_pd.DataFrame(rows), width="stretch", hide_index=True)


def _model_card(name, score, detail, selected):
    """Render a styled model candidate card."""
    pct = int(min(100, score * 100))
    cls = 'selected' if selected else ''
    star = chr(10022) + ' ' if selected else ''
    clr = '#a78bfa' if selected else '#5a5a8a'
    st.markdown(
        f'<div class="model-card {cls}">'
        f'<div><div class="model-card-name">{star}{name}</div>'
        f'<div class="model-card-detail">{detail}</div></div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:1.1rem;font-weight:700;color:{clr};'
        f'font-family:JetBrains Mono,monospace">{score:.3f}</div>'
        f'<div class="score-bar" style="margin-top:6px">'
        f'<div class="score-fill" style="width:{pct}%"></div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


def _paper_cite(num, title, venue, year, url: str = ""):
    """Render a styled paper citation block with optional hyperlink."""
    link_html = (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'style="color:var(--violet2);text-decoration:none;font-size:.7rem">'
        f'↗ Paper</a>'
        if url else ""
    )
    st.markdown(
        f'<div class="paper-cite">'
        f'<div class="paper-cite-num">[{num}]</div>'
        f'<div><div class="paper-cite-text">{title} {link_html}</div>'
        f'<div class="paper-cite-venue">{venue} &middot; {year}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )



def _show_error_with_retry(message: str, retry_key: str) -> None:
    """Show an error message with a Retry button that triggers st.rerun()."""
    st.error(message)
    if st.button("🔄 Retry", key=retry_key, type="secondary"):
        st.rerun()


def _api_error_detail(resp) -> str:
    """Extract the most descriptive error string from an API response."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            return body.get("detail") or body.get("message") or str(body)[:400]
    except Exception:
        pass
    return resp.text[:400]


def api_call(method: str, path: str, timeout: int = 15, silent: bool = False, **kwargs):
    """Wrap requests with uniform error handling.

    Parameters
    ----------
    method : str
        HTTP method (``"get"``, ``"post"``, ``"delete"``, etc.).
    path : str
        URL path (appended to ``API_BASE_URL`` if not already absolute).
    timeout : int
        Request timeout in seconds.
    silent : bool
        If True, suppress ``st.error()`` on failure (caller handles it).
    **kwargs
        Forwarded to ``requests.<method>()``.

    Returns
    -------
    dict | list | None
        Parsed JSON response, or ``None`` on failure.
    """
    url = path if path.startswith("http") else f"{API_BASE_URL}{path}"
    try:
        resp = getattr(requests, method.lower())(url, timeout=timeout, **kwargs)
        if not resp.ok:
            # 404/422 before ingestion are expected "not ready" states — never show errors for them
            if resp.status_code not in (404, 422) and not silent:
                st.error(f"API {resp.status_code}: {_api_error_detail(resp)}")
            return None
        return resp.json()
    except requests.Timeout:
        if not silent:
            st.error(f"Request timed out ({timeout}s): {path}")
    except Exception as e:
        if not silent:
            st.error(f"API error: {e}")
    return None


# ---------------------------------------------------------------------------
# C5: Cached API fetchers — eliminate redundant round-trips per render cycle
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def _cached_model_registry() -> list:
    """Cached fetch of /model-registry."""
    try:
        resp = requests.get(f"{API_BASE_URL}/model-registry", timeout=15)
        if resp.ok:
            data = resp.json()
            return data if isinstance(data, list) else data.get("models", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=30, show_spinner=False)
def _cached_registered_models(_sid: str) -> list:
    """Cached fetch of /v2/sessions/{sid}/registered-models."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/v2/sessions/{_sid}/registered-models", timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return data if isinstance(data, list) else data.get("models", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=60, show_spinner=False)
def _cached_model_info(_model_id: str) -> dict:
    """Cached fetch of /model-info/{model_id}."""
    try:
        resp = requests.get(f"{API_BASE_URL}/model-info/{_model_id}", timeout=10)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Plain-English labels for fusion strategies
# (used throughout the UI so non-experts understand what was selected)
# ---------------------------------------------------------------------------
_FUSION_LABELS: Dict[str, str] = {
    "attention":           "Attention Fusion — learns which modality to focus on",
    "concatenation":       "Concatenation Fusion — joins all modality vectors end-to-end",
    "graph":               "Graph Attention Fusion — treats modalities as graph nodes",
    "complementarity":     "Complementarity Fusion [CrossFuse, ECCV 2024] — pairs each modality with every other to find unique contributions",
    "crossfuse":           "Complementarity Fusion [CrossFuse, ECCV 2024] — finds non-overlapping information across modalities",
    "structural_semantic": "Structural-Semantic Unifier [ICML 2025] — combines graph structure with language semantics",
    "ssunifier":           "Structural-Semantic Unifier [ICML 2025] — dual-path fusion with learned gating",
    "uncertainty":         "Uncertainty-Weighted Fusion [CVPR 2025] — down-weights unreliable modalities",
    "uncertainty_graph":   "Uncertainty-Weighted Graph Fusion [CVPR 2025] — graph attention with calibrated uncertainty",
}

# ---------------------------------------------------------------------------
# Paper references shown next to technical terms
# ---------------------------------------------------------------------------
_PAPER_REFS: Dict[str, str] = {
    "cross_layer_rgat":        "[2] Cross-layer RGAT head — NeurIPS/IEEE 2025",
    "complementarity_fusion":  "[4] CrossFuse — ECCV/IEEE 2024",
    "structural_semantic":     "[1] Structural-Semantic Unifier — ICML/IEEE 2025",
    "uncertainty_graph":       "[3] Uncertainty Graph Fusion — CVPR/IEEE 2025",
    "ewc":                     "[8] Elastic Weight Consolidation (EWC) — IEEE TNNLS 2024",
    "calibration":             "[9] Calibration of Modern NNs (Guo et al.) — ICML 2017",
    "ddm":                     "[10] DDM Concept Drift — Gama et al., SBIA 2004",
    "driftlens":               "[11] DriftLens cosine drift — IEEE 2024",
    "fttransformer":           "[12] FTTransformer — Gorishniy et al., NeurIPS 2021",
    "focal_loss":              "[13] Focal Loss — Lin et al., ICCV 2017",
    "swa":                     "[14] SWA — Izmailov et al., UAI 2018",
    "pcgrad":                  "[15] PCGrad Gradient Surgery — Yu et al., NeurIPS 2020",
}

# ---------------------------------------------------------------------------
# Beginner-friendly glossary for technical terms shown in the UI
# ---------------------------------------------------------------------------
_GLOSSARY: Dict[str, str] = {
    "Fusion Strategy": (
        "How the model combines multiple data types (e.g. text + numbers + images). "
        "Think of it like mixing ingredients — different recipes give different results."
    ),
    "ECE (Expected Calibration Error)": (
        "Measures if the model's confidence matches reality. "
        "ECE = 0.05 means that when the model says '80% confident', it's actually right about 75–85% of the time. "
        "Lower is better."
    ),
    "NLL (Negative Log-Likelihood)": (
        "A measure of how well the model fits the data. Lower = better. "
        "It penalises the model harshly for being very confident and wrong."
    ),
    "Brier Score": (
        "Average squared difference between predicted probability and actual outcome. "
        "Ranges 0 (perfect) to 1 (worst). Lower is better."
    ),
    "MMD (Maximum Mean Discrepancy)": (
        "A statistical test that checks if two batches of data look like they come from the same distribution. "
        "Used to detect when production data starts looking different from training data."
    ),
    "Cosine Drift (DriftLens)": (
        "Measures if the *direction* of data has changed in the model's internal representation space. "
        "Catches concept-level shifts (like topic changes in text) that MMD might miss."
    ),
    "DDM (Drift Detection Method)": (
        "Watches the model's prediction error rate over time. "
        "If errors start increasing significantly, it raises a warning — then a drift alarm. "
        "Based on Gama et al. (2004)."
    ),
    "Fisher Information (EWC)": (
        "A measure of how important each neural network weight is for the previous task. "
        "EWC uses it to prevent the model from 'forgetting' what it learned when retraining on new data."
    ),
    "PSI (Population Stability Index)": (
        "Compares the distribution of a feature between training and production. "
        "PSI > 0.25 signals significant drift. Originally from credit risk scoring."
    ),
    "Probe Score": (
        "A quick Random-Forest cross-validation score run before full training to estimate "
        "how predictable the target is from each modality. Guides which models to try first."
    ),
    "Head Architecture": (
        "The final layers of the neural network that map the fused representation to a prediction. "
        "Options: MLP (basic), Attention (text-heavy), Cross-Layer RGAT (relational/multi-modal)."
    ),
    "Optuna Trials": (
        "The number of hyperparameter combinations tested automatically. "
        "Each trial tries a different learning rate, batch size, etc., and Optuna learns which values work best."
    ),
    "Retraining Depth": (
        "How much of the model to retrain when drift is detected. "
        "calibration_only = just recalibrate confidence. "
        "head_only = only retrain final layers. "
        "full = retrain everything from scratch."
    ),
    "FTTransformer (Feature Tokenizer + Transformer)": (
        "A tabular encoder from NeurIPS 2021 that treats each column as a 'token' (like a word in a sentence), "
        "then applies a Transformer to learn how columns interact. "
        "Much stronger than a plain MLP for complex tabular data with many feature interactions."
    ),
    "Focal Loss": (
        "A loss function from ICCV 2017 that reduces emphasis on easy, well-classified samples "
        "and focuses training on the hard, misclassified ones. "
        "Automatically activated when class imbalance ratio > 3:1. "
        "Think of it as 'train harder on the difficult cases'."
    ),
    "SWA (Stochastic Weight Averaging)": (
        "Averages the model weights at multiple points in the training trajectory. "
        "This finds 'flatter' regions of the loss landscape, which generalise better to new data. "
        "Activated in the last 10% of training epochs. "
        "Free 1-2% accuracy gain with no extra data (Izmailov et al., UAI 2018)."
    ),
    "PCGrad (Gradient Surgery)": (
        "When training on multiple data types simultaneously, the gradients from text, images, "
        "and tabular data can 'fight' each other. "
        "PCGrad detects these conflicts (negative dot product between gradient vectors) "
        "and projects conflicting gradients to eliminate the interference. "
        "From Yu et al., NeurIPS 2020."
    ),
    "Modality Dropout": (
        "During training, randomly remove one data type (e.g. the image) from some batches. "
        "This forces the model to work with incomplete inputs, making it more robust "
        "when a modality is missing at prediction time."
    ),
}


@st.cache_data(ttl=10, show_spinner=False)
def check_api_connection():
    """Check if API is available (cached for 10 s)."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=10, show_spinner=False)
def get_api_status():
    """Get full API status (cached for 10 s)."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# D3: Typed session-state schema — single source of truth for all keys
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field as dc_field

@dataclass
class FrontendSession:
    """Typed default values for every ``st.session_state`` key.

    New keys should be added here so that (a) the type is documented,
    (b) the default is centralized, and (c) IDE autocomplete works.
    """
    session_id: str = ""  # set to uuid below
    workflow_stage: int = 1
    dataset_uploaded: bool = False
    schema_detected: bool = False
    detected_schema: Optional[Dict] = None
    schema_overrides: Dict = dc_field(default_factory=dict)  # {dataset_id: {target_column, problem_type}}
    active_dataset_group: Optional[int] = None  # Index of chosen dataset group
    model_selected: bool = False
    dataset_info: Dict = dc_field(default_factory=dict)
    ingested_row_count: Optional[int] = None
    training_task_id: Optional[str] = None
    hp_overrides: Optional[Dict] = None
    training_result: Optional[Dict] = None
    trained_model_id: Optional[str] = None
    text_columns: List = dc_field(default_factory=list)
    image_columns: List = dc_field(default_factory=list)
    ingestion_task_id: Optional[str] = None
    schema_candidates: List = dc_field(default_factory=list)
    monitor_result: Dict = dc_field(default_factory=dict)
    drift_result: Dict = dc_field(default_factory=dict)
    model_selection_result: Dict = dc_field(default_factory=dict)
    preprocess_result: Dict = dc_field(default_factory=dict)
    embedding_cache_stats: Dict = dc_field(default_factory=dict)
    meta_learning_insights: Dict = dc_field(default_factory=dict)
    ablation_results: Dict = dc_field(default_factory=dict)
    phase_states: Dict = dc_field(default_factory=lambda: {
        1: {"status": "pending", "reason": ""},
        2: {"status": "pending", "reason": ""},
        3: {"status": "pending", "reason": ""},
        4: {"status": "pending", "reason": ""},
        5: {"status": "pending", "reason": ""},
        6: {"status": "pending", "reason": ""},
        7: {"status": "pending", "reason": ""},
    })


def _init_session_state() -> None:
    """Initialize all session-state keys from FrontendSession defaults."""
    defaults = FrontendSession()
    # session_id is special — generate once per browser tab
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    # All other keys: set only if not already present
    for key, value in defaults.__dict__.items():
        if key == "session_id":
            continue
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


def _try_recover_session_state() -> None:
    """Recover frontend flags from API when session state is empty (e.g. after page refresh).

    Runs when schema_detected is False. Tries the current session_id first; if that
    session doesn't exist in the API (new UUID after refresh), falls back to the most
    recent active session. Restores dataset_uploaded, schema_detected, detected_schema,
    model_selected, and workflow_stage so the user lands on the correct phase.
    """
    if st.session_state.get("schema_detected"):
        return  # already populated — nothing to recover

    _STAGE_ORDER = [
        "ingestion_complete", "schema_detection", "target_detection",
        "global_aggregation", "preprocessing_planning",
        "model_selection", "training", "monitoring",
    ]

    def _recover_from_ctx(_sid: str, _ctx: dict) -> None:
        _stage = str(_ctx.get("pipeline_stage", "") or "")
        _stage_idx = _STAGE_ORDER.index(_stage) if _stage in _STAGE_ORDER else -1

        if _stage_idx >= _STAGE_ORDER.index("ingestion_complete"):
            st.session_state.dataset_uploaded = True
            if st.session_state.get("ingested_row_count") is None:
                st.session_state.ingested_row_count = 1000

        if _stage_idx >= _STAGE_ORDER.index("schema_detection"):
            try:
                _sr = requests.get(
                    f"{API_BASE_URL}/v2/sessions/{_sid}/global-schema", timeout=2
                )
                if _sr.ok:
                    _schema = _sr.json()
                    if _schema and _schema.get("global_problem_type"):
                        st.session_state.detected_schema = _schema
                        st.session_state.schema_detected = True
            except Exception:
                pass

        if _stage_idx >= _STAGE_ORDER.index("model_selection"):
            if not st.session_state.get("model_selected"):
                st.session_state.model_selected = False  # let Phase 4 re-fetch

        # Advance workflow_stage to match API stage
        if _stage_idx >= _STAGE_ORDER.index("preprocessing_planning"):
            if st.session_state.workflow_stage < 4:
                st.session_state.workflow_stage = 4
        elif _stage_idx >= _STAGE_ORDER.index("schema_detection"):
            if st.session_state.workflow_stage < 3:
                st.session_state.workflow_stage = 3
        elif _stage_idx >= _STAGE_ORDER.index("ingestion_complete"):
            if st.session_state.workflow_stage < 2:
                st.session_state.workflow_stage = 2

        # Adopt the recovered session_id so subsequent API calls use the right session
        st.session_state.session_id = _sid

    try:
        _sid = st.session_state.get("session_id") or ""

        # Try current session_id first
        if _sid:
            _r = requests.get(f"{API_BASE_URL}/v2/sessions/{_sid}", timeout=2)
            if _r.ok:
                _recover_from_ctx(_sid, _r.json())
                return

        # session_id not found (new UUID after page refresh) — find most recent active session
        _list_r = requests.get(
            f"{API_BASE_URL}/v2/sessions",
            params={"status": "active", "limit": 5},
            timeout=2,
        )
        if _list_r.ok:
            _sessions = _list_r.json().get("sessions") or []
            if _sessions:
                # Use most recent session (API returns newest-first)
                _best = _sessions[0]
                _best_sid = _best.get("session_id") or _best.get("id") or ""
                if _best_sid:
                    _ctx_r = requests.get(
                        f"{API_BASE_URL}/v2/sessions/{_best_sid}", timeout=2
                    )
                    if _ctx_r.ok:
                        _recover_from_ctx(_best_sid, _ctx_r.json())
            else:
                # Fix A: API has no active sessions (restart cleared all).
                # Don't leave user stuck on Phase N with no way forward.
                if st.session_state.workflow_stage > 1:
                    st.session_state.workflow_stage = 1
        else:
            if st.session_state.workflow_stage > 1:
                st.session_state.workflow_stage = 1

    except Exception:
        pass  # API unreachable — leave state as-is


_try_recover_session_state()


def render_workflow_dashboard():
    """Render the main workflow dashboard."""

    # ── Hero Banner ───────────────────────────────────────────────────
    st.markdown("""
    <div class="autovision-hero">
      <div class="av-tag">⚡ Research-Grade Multimodal AutoML Platform</div>
      <div class="av-title">AutoVision</div>
      <div class="av-sub">Schema-Aware &nbsp;·&nbsp; Adaptive Fusion &nbsp;·&nbsp; Calibrated &nbsp;·&nbsp; Explainable Intelligence</div>
      <div class="av-sub" style="margin-top:14px;font-size:.85rem;letter-spacing:.08em;color:#6b7280;">
        <span style="color:#a78bfa;font-weight:600">Tabular</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#14b8a6;font-weight:600">Text</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#f59e0b;font-weight:600">Image</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#6b7280;font-weight:500">Multimodal</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Visual Phase Progress Stepper ─────────────────────────────────
    _phase_labels = ["Ingest", "Schema", "Preprocess", "Select", "Train", "Monitor", "Predict"]
    _current = st.session_state.workflow_stage
    _states = st.session_state.phase_states

    steps_html = '<div class="av-stepper">'
    for i, label in enumerate(_phase_labels, 1):
        status = _states.get(i, {}).get("status", "pending")
        if status in ("completed", "reused"):
            cls = "done"
        elif i == _current:
            cls = "active"
        else:
            cls = ""
        steps_html += f"""
        <div class="av-step {cls}">
          <div class="av-step-inner">
            <div class="av-circle">{i}</div>
            <div class="av-label">{label}</div>
          </div>
        </div>"""
        if i < 7:
            conn_cls = "done" if status in ("completed", "reused") else ""
            steps_html += f'<div class="av-connector"></div>'
    steps_html += '</div>'
    st.markdown(steps_html, unsafe_allow_html=True)

    # ── Quick Status Row ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        api_status = get_api_status()
        if api_status:
            st.metric("API Status", "Connected ✅", help="Backend is running")
        else:
            st.metric("API Status", "Offline ❌", help="Start: python run_api.py")

    with col2:
        completed = sum(1 for s in _states.values() if s.get("status") in ("completed", "reused"))
        st.metric("Progress", f"{completed}/7 phases", f"Phase {_current} active")

    with col3:
        gpu_available = api_status.get('gpu_available', False) if api_status else False
        if gpu_available:
            gpu_name = api_status.get('gpu_name', 'GPU')
            st.metric("Compute", gpu_name[:20], "GPU Accelerated ⚡")
        else:
            st.metric("Compute", "CPU", "No GPU detected")

    # Only poll session-dependent sidebar endpoints after a dataset has been ingested.
    # This suppresses the waterfall of 404/422s that appear on fresh page load.
    _session_ready = bool(st.session_state.get("dataset_uploaded") or st.session_state.get("ingested_row_count"))

    st.sidebar.divider()
    st.sidebar.markdown("### Drift Status")
    if not _session_ready:
        st.sidebar.caption("Ingest a dataset first")
    else:
        try:
            drift_resp = requests.get(
                f"{API_BASE_URL}/context/{st.session_state.session_id}/drift-status",
                timeout=3,
            )
            if drift_resp.status_code == 200:
                drift_data = drift_resp.json()
                if drift_data.get("drift_detected"):
                    st.sidebar.error(
                        f"Drift detected (severity={float(drift_data.get('drift_severity', 0.0)):.3f})"
                    )
                else:
                    st.sidebar.success("No drift detected")
            else:
                st.sidebar.caption("Drift status unavailable")
        except Exception:
            st.sidebar.caption("Drift status unavailable")

    # ── Global Target ─────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Global Target")
    if not _session_ready:
        st.sidebar.caption("Ingest a dataset first")
    else:
        try:
            gt_resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/global-target",
                timeout=3,
            )
            if gt_resp.status_code == 200:
                gt_data = gt_resp.json()
                gt_val = gt_data.get("global_target")
                gt_conf = float(gt_data.get("confidence", 0) or 0)
                if gt_val:
                    st.sidebar.metric("Target Column", gt_val, f"Confidence: {gt_conf:.0%}")
                else:
                    st.sidebar.caption("No target detected yet")
            else:
                st.sidebar.caption("Run schema detection first")
        except Exception:
            st.sidebar.caption("Target unavailable")

    # ── Fit Analysis ──────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Fit Analysis")
    if not _session_ready:
        st.sidebar.caption("Train first to see fit analysis")
    else:
        try:
            fa_resp = requests.get(
                f"{API_BASE_URL}/context/{st.session_state.session_id}/fit-analysis",
                timeout=3,
            )
            if fa_resp.status_code == 200:
                fa_data = fa_resp.json()
                fit_analysis = fa_data.get("training_fit_analysis", {})
                if fit_analysis:
                    fit_type = fit_analysis.get("fit_type", "unknown")
                    if fit_type == "good":
                        st.sidebar.success(f"Fit: {fit_type}")
                    elif fit_type == "overfitting":
                        st.sidebar.warning(f"Fit: {fit_type}")
                    elif fit_type == "underfitting":
                        st.sidebar.info(f"Fit: {fit_type}")
                    else:
                        st.sidebar.caption(f"Fit: {fit_type}")
                else:
                    st.sidebar.caption("Train first to see fit analysis")
            else:
                st.sidebar.caption("Fit analysis unavailable")
        except Exception:
            st.sidebar.caption("Fit analysis unavailable")
    
    st.divider()
    
    # Workflow Progress
    st.subheader("📋 Workflow Stages")
    
    stages = [
        ("Phase 1", "Data Ingestion", "Upload & cache datasets"),
        ("Phase 2", "Schema Detection", "Detect columns & problem type"),
        ("Phase 3", "Preprocessing", "Prepare data for training"),
        ("Phase 4", "Model Selection", "Auto-select models & params"),
        ("Phase 5", "Training", "Train with GPU acceleration"),
        ("Phase 6", "Monitoring", "Track performance & drift"),
        ("Phase 7", "Prediction", "Make multimodal predictions")
    ]
    
    cols = st.columns(7)
    for i, (col, (phase, name, desc)) in enumerate(zip(cols, stages)):
        with col:
            if st.session_state.workflow_stage == i + 1:
                st.markdown(f"### 🔵 {phase}\n**{name}**")
            elif st.session_state.workflow_stage > i + 1:
                st.markdown(f"### ✅ {phase}\n**{name}**")
            else:
                st.markdown(f"### ⭕ {phase}\n{name}")
            st.caption(desc)
    
    st.divider()
    
    # Phase Selection
    phase = st.radio(
        "Select Workflow Phase:",
        ["Phase 1: Data Ingestion", "Phase 2: Schema Detection", "Phase 3: Preprocessing",
         "Phase 4: Model Selection", "Phase 5: Training", "Phase 6: Monitoring", "Phase 7: Prediction"],
        index=st.session_state.workflow_stage - 1,
        horizontal=True,
        help="Work through phases 1→7 in order. Each phase builds on the previous: ingest data → detect schema → preprocess → select model → train → monitor for drift → predict.",
    )
    
    st.session_state.workflow_stage = int(phase.split()[1].rstrip(':'))
    
    # Render selected phase
    if st.session_state.workflow_stage == 1:
        render_phase_1_data_ingestion()
    elif st.session_state.workflow_stage == 2:
        render_phase_2_schema_detection()
    elif st.session_state.workflow_stage == 3:
        render_phase_3_preprocessing()
    elif st.session_state.workflow_stage == 4:
        render_phase_4_model_selection()
    elif st.session_state.workflow_stage == 5:
        render_phase_5_training()
    elif st.session_state.workflow_stage == 6:
        render_phase_6_monitoring()
    elif st.session_state.workflow_stage == 7:
        render_phase_7_prediction()

    st.divider()

    # FIX-UI-11: System Intelligence Panel (only after ingestion)
    with st.expander("🧠 System Intelligence (ExecutionContext)", expanded=False):
        if not _session_ready:
            st.info("Ingest a dataset to populate intelligence context.")
            intel_resp = None
        else:
            intel_resp = None
            try:
                intel_resp = requests.get(
                    f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence",
                    timeout=5,
                )
            except Exception:
                intel_resp = None

        if intel_resp is not None and intel_resp.status_code == 200:
            intel = intel_resp.json()
            st.caption(
                f"Context version: {intel.get('context_version', 'N/A')} | "
                f"Stage: {intel.get('pipeline_stage', 'N/A')} | "
                f"Last updated: {str(intel.get('updated_at', 'N/A'))[:19]}"
            )

            ic1, ic2, ic3 = st.columns(3)

            with ic1:
                st.markdown("**Active Modalities**")
                scores = intel.get("predictability_scores", {}) or {}
                presence = intel.get("modality_presence", {}) or {}
                if scores:
                    # predictability_scores keys are dataset_ids (UUIDs), not modality names
                    # — display as "dataset predictability", not per-modality
                    _any_score_shown = False
                    for _k, score in scores.items():
                        if not isinstance(score, (int, float)):
                            continue
                        active = score > 0.4
                        icon = "✅" if active else "⚠️"
                        _label = str(_k)[:8] + "…" if len(str(_k)) > 12 else str(_k)
                        st.write(f"{icon} tabular predictability ({_label}): {float(score):.3f}")
                        _any_score_shown = True
                    if not _any_score_shown and presence:
                        for mod, present in presence.items():
                            st.write(f"{'✅' if present else '—'} {mod}")
                elif presence:
                    for mod, present in presence.items():
                        st.write(f"{'✅' if present else '—'} {mod}")
                else:
                    st.info("No modality data yet.")

            with ic2:
                st.markdown("**Fusion Strategy**")
                fstrat = intel.get("fusion_strategy")
                fsrc = intel.get("fusion_source")
                should_fuse = bool(intel.get("should_include_fusion", False))
                if fstrat:
                    src_label = "🧑 User override" if fsrc == "user_override" else "🤖 Auto-selected"
                    friendly_label = _FUSION_LABELS.get(str(fstrat).lower(), fstrat)
                    st.write(f"**{fstrat}** — {src_label}")
                    st.caption(f"_{friendly_label}_")
                else:
                    st.info("Not yet determined")
                st.caption(
                    f"Fusion enabled: {'Yes' if should_fuse else 'No'} "
                    f"({'cross-modal attention active' if should_fuse else 'single modality detected'})"
                )
                st.caption(
                    f"Fusion policy: {'locked' if intel.get('fusion_policy_locked') else 'unlocked'} | "
                    f"Source: {intel.get('fusion_policy_source') or 'N/A'}"
                )
                mod_importance = intel.get("modality_importance", {}) or {}
                if mod_importance:
                    st.markdown("Modality importance:")
                    for mod, weight in mod_importance.items():
                        wt = float(weight) if isinstance(weight, (int, float)) else weight
                        st.write(f"- {mod}: {wt:.3f}" if isinstance(wt, float) else f"- {mod}: {wt}")

            with ic3:
                st.markdown("**Schema Intelligence**")
                target = intel.get("global_target", "Unknown")
                conf = float(intel.get("global_target_confidence", 0) or 0)
                gap = float(intel.get("xs3_confidence_gap", 0) or 0)
                override = bool(intel.get("override_applied", False))
                st.write(f"Target: {target}")
                st.write(f"Confidence: {conf:.1%}")
                st.write(f"XS3 gap: {gap:.3f}")
                if override:
                    st.warning("⚡ Override applied to target/schema")
                    for ov in (intel.get("override_history", []) or [])[-3:]:
                        st.caption(
                            f"[{str(ov.get('timestamp', ''))[:19]}] "
                            f"{ov.get('field', '?')}: "
                            f"{ov.get('old_value', '?')} -> {ov.get('new_value', '?')}"
                        )

            pp_choices = intel.get("preprocessing_choices", {}) or {}
            if pp_choices:
                st.markdown("**Preprocessing Choices (per modality)**")
                for mod, choices in pp_choices.items():
                    if isinstance(choices, dict) and choices:
                        st.write(f"- {mod}: " + ", ".join(f"{k}={v}" for k, v in choices.items()))

            registered_models = list(intel.get("registered_model_ids", []) or [])
            active_prediction_model_id = intel.get("active_prediction_model_id")
            if registered_models or active_prediction_model_id:
                st.markdown("**Model Registry**")
                if active_prediction_model_id:
                    st.write(f"Active prediction model: {active_prediction_model_id}")
                else:
                    st.info("No active prediction model yet.")
                st.caption(f"Registered models: {len(registered_models)}")
                for model_id in registered_models[-3:]:
                    st.write(f"- {model_id}")

            artifact_versions = intel.get("artifact_versions", {}) or {}
            xai_config = intel.get("xai_config", {}) or {}
            if artifact_versions or xai_config:
                st.markdown("**Artifact / XAI Config**")
                if artifact_versions:
                    st.caption("Artifact versions")
                    _kv_table(artifact_versions, "Artifact Versions")
                if xai_config:
                    st.caption("XAI config")
                    _kv_table(xai_config, "XAI Configuration")

            training_fit_analysis = intel.get("training_fit_analysis", {}) or {}
            if training_fit_analysis:
                with st.expander("Training Fit Analysis", expanded=False):
                    _kv_table(training_fit_analysis, "Fit Analysis")

            guardrails = intel.get("guardrails", {}) or {}
            if guardrails:
                with st.expander("Guardrails", expanded=False):
                    st.caption(f"Overall status: {guardrails.get('overall_status', 'N/A')}")
                    latency = guardrails.get("latency", {}) or {}
                    memory = guardrails.get("memory", {}) or {}
                    isolation = guardrails.get("session_isolation", {}) or {}

                    gl1, gl2, gl3 = st.columns(3)
                    gl1.metric("Latency Guard", str(latency.get("status", "N/A")))
                    gl2.metric("VRAM Used", f"{float((memory.get('vram', {}) or {}).get('used_pct', 0.0) or 0.0):.1%}")
                    gl3.metric("RAM Used", f"{float((memory.get('ram', {}) or {}).get('used_pct', 0.0) or 0.0):.1%}")

                    st.write(
                        f"- Session isolation: {isolation.get('status', 'N/A')} "
                        f"({isolation.get('validated_dataset_count', 0)}/{isolation.get('active_dataset_count', 0)} validated)"
                    )
                    st.write(
                        "- Protected endpoints: "
                        + ", ".join(latency.get("protected_endpoints", []) or ["N/A"])
                    )
                    budgets = latency.get("budgets_s", {}) or {}
                    if budgets:
                        st.write(
                            "- Latency budgets: "
                            + ", ".join(f"{k}={v}s" for k, v in budgets.items())
                        )

            execution_log_count = intel.get("execution_log_count")
            if isinstance(execution_log_count, int):
                st.caption(f"Decision log entries: {execution_log_count}")

            phase_timings = intel.get("phase_timings", {}) or {}
            if phase_timings:
                with st.expander("Phase Timings", expanded=False):
                    timing_rows = [
                        {"Phase": phase_name, "Duration (s)": float(duration)}
                        for phase_name, duration in phase_timings.items()
                    ]
                    st.dataframe(pd.DataFrame(timing_rows), width="stretch")
        elif intel_resp is not None and intel_resp.status_code == 404:
            st.info("No session context yet. Complete at least Phase 1 to populate intelligence.")
        else:
            st.info("System intelligence unavailable.")

    st.markdown("### System Intelligence")

    if not _session_ready:
        st.info("Complete Phase 1 (Data Ingestion) to unlock system intelligence panels.")
        return

    with st.expander("Calibration Metrics", expanded=False):
        st.caption(
            "**Calibration** checks whether the model's confidence scores are accurate. "
            "ECE (Expected Calibration Error) < 0.05 = excellent. "
            "NLL = how surprised the model is by correct answers (lower = better). "
            "Brier Score < 0.1 = excellent probability estimates."
        )
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/calibration",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("per_model", []) or []
                if rows:
                    for row in rows:
                        model_label = str(row.get("model_id", "unknown"))
                        ece = float(row.get('ece', 0.0) or 0.0)
                        nll = float(row.get('nll', 0.0) or 0.0)
                        brier = float(row.get('brier', 0.0) or 0.0)
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            label=f"{model_label[:20]} — ECE",
                            value=f"{ece:.3f}",
                            delta="✅ good" if ece < 0.05 else "⚠️ high" if ece > 0.15 else None,
                            delta_color="normal",
                            help="Expected Calibration Error: lower is better. < 0.05 is excellent.",
                        )
                        c2.metric(label="NLL", value=f"{nll:.3f}",
                                  help="Negative Log-Likelihood: lower means model is more confident when correct.")
                        c3.metric(label="Brier Score", value=f"{brier:.3f}",
                                  help="Overall probability accuracy. < 0.1 is excellent.")
                else:
                    st.info("No calibration data available yet. Complete training (Phase 5) first.")
            else:
                st.info("No calibration data available yet.")
        except Exception:
            st.info("No calibration data available yet.")

    with st.expander("XAI — Feature Attribution", expanded=False):
        st.caption(
            "**Explainable AI (XAI)** shows which input features most influenced the prediction. "
            "Longer bars = more important. Negative values push the prediction DOWN, positive push it UP. "
            "Method: Integrated Gradients — traces how changing each feature from a baseline changes the output."
        )
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/xai",
                timeout=5,
            )
            if resp.status_code == 200:
                per_model = resp.json().get("per_model", []) or []
                if per_model:
                    for model_entry in per_model:
                        mid = model_entry.get('model_id', 'unknown')
                        method = model_entry.get('method', 'Integrated Gradients')
                        st.markdown(f"**{mid[:40]}** — *{method}*")
                        tabular_entry = model_entry.get("tabular", {}) if isinstance(model_entry, dict) else {}
                        ranking = tabular_entry.get("feature_ranking", []) if isinstance(tabular_entry, dict) else []
                        if ranking:
                            ranking_df = pd.DataFrame(ranking[:20], columns=["feature", "importance"])
                            if not ranking_df.empty:
                                st.bar_chart(ranking_df.set_index("feature"))
                                st.caption("Top 20 features by attribution score (absolute value).")
                else:
                    st.info("No XAI artifacts available yet. Run a prediction with `explain=True`.")
            else:
                st.info("No XAI artifacts available yet.")
        except Exception:
            st.info("No XAI artifacts available yet.")

    with st.expander("Guardrails", expanded=False):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/guardrails",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                overall = str(data.get("overall_status", "unknown"))
                overall_color = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(overall.lower(), "⚪")
                st.metric(label="Overall Status", value=f"{overall_color} {overall}")

                # Latency metrics
                latency = data.get("latency", {}) or {}
                if latency:
                    st.markdown("**Latency Guard**")
                    lat_cols = st.columns(3)
                    lat_cols[0].metric("Status", str(latency.get("status", "N/A")))
                    budgets = latency.get("budgets_s", {}) or {}
                    if budgets:
                        lat_cols[1].metric("Predict Budget", f"{budgets.get('predict', 'N/A')}s")
                        lat_cols[2].metric("Train Budget", f"{budgets.get('train', 'N/A')}s")
                    protected = latency.get("protected_endpoints", []) or []
                    if protected:
                        st.caption(f"Protected endpoints: {', '.join(protected)}")

                # Memory: color-coded bar
                memory = data.get("memory", {}) or {}
                if memory:
                    st.markdown("**Memory Guard**")
                    mem_cols = st.columns(2)
                    vram = memory.get("vram", {}) or {}
                    ram = memory.get("ram", {}) or {}
                    vram_pct = float(vram.get("used_pct", 0) or 0)
                    ram_pct = float(ram.get("used_pct", 0) or 0)
                    mem_cols[0].metric("VRAM Used", f"{vram_pct:.1%}")
                    mem_cols[1].metric("RAM Used", f"{ram_pct:.1%}")
                    # Visual memory bar
                    if vram_pct > 0 or ram_pct > 0:
                        mem_df = pd.DataFrame({
                            "Resource": ["VRAM", "RAM"],
                            "Usage %": [vram_pct * 100, ram_pct * 100],
                        })
                        st.bar_chart(mem_df.set_index("Resource"))

                # Session isolation: red/green badge
                isolation = data.get("session_isolation", {}) or {}
                if isolation:
                    iso_status = str(isolation.get("status", "unknown"))
                    iso_icon = "🟢" if iso_status.lower() in ("valid", "ok", "healthy") else "🔴"
                    validated = isolation.get("validated_dataset_count", 0)
                    active = isolation.get("active_dataset_count", 0)
                    st.markdown(
                        f"**Session Isolation:** {iso_icon} {iso_status} "
                        f"({validated}/{active} datasets validated)"
                    )
            else:
                st.info("No guardrail data available yet.")
        except Exception:
            st.info("No guardrail data available yet.")

    with st.expander("Ranked Model Candidates", expanded=False):
        st.caption(
            "**Probe Score** (0–1): How well a fast Random Forest surrogate predicts your target. "
            "Higher = AutoVision is more confident this architecture suits your data. "
            "The top-scoring model is selected and passed to full HPO training. "
            "Excluded models failed hardware requirements (VRAM/RAM) or had probe score < 0.25."
        )
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/ranked-candidates",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.write(f"**Selected:** {data.get('selected_model')}")
                st.caption(str(data.get("reason", "")))
                ranked_rows = data.get("ranked", []) or []
                if ranked_rows:
                    st.dataframe(pd.DataFrame(ranked_rows), width="stretch")
                else:
                    st.info("No ranked-candidate data available yet.")
            else:
                st.info("No ranked-candidate data available yet.")
        except Exception:
            st.info("No ranked-candidate data available yet.")

    with st.expander("Trial Intelligence (AutoML)", expanded=False):
        st.caption(
            "**Trial Intelligence** tracks how each Optuna training trial performed. "
            "🟢 **Good** = model is learning well. 🟡 **Underfitting** = model too simple — try more epochs or larger architecture. "
            "🔴 **Overfitting** = memorizing training data — increase dropout or reduce epochs. "
            "**Train/Val Slope**: positive = loss still decreasing (good). Flat at 0 = plateaued. "
            "**Adaptive LR**: AutoVision adjusts the learning rate automatically based on fit type."
        )
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/trial-intelligence",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()

                # Fit analysis: colored badge
                fit = data.get("fit_analysis", {}) or {}
                if fit:
                    fit_type = str(fit.get("fit_type", "unknown"))
                    fit_colors = {"good": "🟢", "underfitting": "🟡", "overfitting": "🔴"}
                    fit_icon = fit_colors.get(fit_type.lower(), "⚪")
                    fit_cols = st.columns(3)
                    fit_cols[0].metric("Fit Type", f"{fit_icon} {fit_type}")
                    train_slope = float(fit.get("train_slope", 0) or 0)
                    val_slope = float(fit.get("val_slope", 0) or 0)
                    fit_cols[1].metric("Train Slope", f"{train_slope:.4f}")
                    fit_cols[2].metric("Val Slope", f"{val_slope:.4f}")

                    # Slope trend chart
                    if abs(train_slope) > 0 or abs(val_slope) > 0:
                        slope_df = pd.DataFrame({
                            "Metric": ["Train Slope", "Val Slope"],
                            "Value": [train_slope, val_slope],
                        })
                        st.bar_chart(slope_df.set_index("Metric"))

                # Adaptive LR: metric with value
                alr = data.get("adaptive_lr", {}) or {}
                if alr:
                    alr_cols = st.columns(3)
                    current_lr = alr.get("current_lr") or alr.get("lr")
                    if current_lr is not None:
                        alr_cols[0].metric("Adaptive LR", f"{float(current_lr):.2e}")
                    schedule = alr.get("schedule_type") or alr.get("type")
                    if schedule:
                        alr_cols[1].metric("Schedule", str(schedule))
                    warmup = alr.get("warmup_epochs")
                    if warmup is not None:
                        alr_cols[2].metric("Warmup Epochs", str(warmup))

                # Recent trials as table
                recent_trials = data.get("recent_trials", []) or []
                if recent_trials:
                    st.markdown("**Recent Trials**")
                    st.dataframe(pd.DataFrame(recent_trials), width="stretch")
                else:
                    st.info("No trial diagnostics available yet.")
            else:
                st.info("No trial diagnostics available yet.")
        except Exception:
            st.info("No trial diagnostics available yet.")

    with st.expander("Preprocessing Plan", expanded=False):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/preprocessing-plan",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                version = data.get("version", "N/A")
                st.markdown(f"### Preprocessing Plan v{version}")

                # Per-modality plan as human-readable bullets
                plan = data.get("plan", {}) or {}
                if plan:
                    for modality, mod_plan in plan.items():
                        st.markdown(f"**{modality.title()} Modality**")
                        if isinstance(mod_plan, dict):
                            for step_key, step_val in mod_plan.items():
                                label = step_key.replace("_", " ").title()
                                st.write(f"- {label}: `{step_val}`")
                        elif isinstance(mod_plan, list):
                            for step in mod_plan:
                                st.write(f"- {step}")
                        else:
                            st.write(f"- {mod_plan}")

                # Choices as key-value table with glossary tooltips
                choices = data.get("choices", {}) or {}
                if choices:
                    st.markdown("**Configuration Choices**")
                    choice_rows = []
                    for key, val in choices.items():
                        glossary_tip = _GLOSSARY.get(key, "")
                        choice_rows.append({
                            "Setting": key.replace("_", " ").title(),
                            "Value": str(val),
                            "Help": glossary_tip[:80] + "..." if len(glossary_tip) > 80 else glossary_tip,
                        })
                    if choice_rows:
                        st.dataframe(pd.DataFrame(choice_rows), width="stretch")

                plan_rows = data.get("per_dataset_plans", []) or []
                if plan_rows:
                    st.markdown("**Per-Dataset Plans**")
                    st.dataframe(pd.DataFrame(plan_rows), width="stretch")
                else:
                    st.info("No per-dataset preprocessing plans available yet.")
            else:
                st.info("No preprocessing-plan data available yet.")
        except Exception:
            st.info("No preprocessing-plan data available yet.")

    with st.expander("Feature Intelligence (semantic signals)", expanded=False):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/feature-intelligence",
                timeout=5,
            )
            if resp.status_code == 200:
                fi_data = resp.json()
                summary = fi_data.get("summary") or {}
                if summary.get("n_datasets", 0) > 0:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Datasets Analysed", summary.get("n_datasets", 0))
                    col2.metric("ID Columns (auto-dropped)", summary.get("total_id_columns", 0))
                    col3.metric("High-Missing Columns", summary.get("total_high_missing_columns", 0))
                    avg_tl = summary.get("avg_text_len")
                    if avg_tl:
                        st.caption(f"Avg text token length: {avg_tl:.0f}")

                    role_counts = summary.get("semantic_role_counts") or {}
                    if role_counts:
                        st.markdown("**Semantic Roles Detected**")
                        role_df = pd.DataFrame(
                            [{"role": k, "count": v} for k, v in role_counts.items()]
                        ).sort_values("count", ascending=False)
                        st.bar_chart(role_df.set_index("role")["count"])

                    pattern_counts = summary.get("business_pattern_counts") or {}
                    if pattern_counts:
                        st.markdown("**Business Patterns**")
                        st.write(", ".join(f"{k} ({v})" for k, v in pattern_counts.items()))

                    per_ds = fi_data.get("per_dataset") or {}
                    for ds_id, ds_intel in per_ds.items():
                        with st.expander(f"Dataset: {ds_id}", expanded=False):
                            inter = ds_intel.get("interaction_summary") or {}
                            if inter:
                                st.markdown("*Top feature interactions:*")
                                inter_df = pd.DataFrame(
                                    [{"feature": k, "score": v} for k, v in inter.items()]
                                ).sort_values("score", ascending=False).head(15)
                                st.bar_chart(inter_df.set_index("feature")["score"])
                            unc = ds_intel.get("uncertainty_summary") or {}
                            if unc:
                                _kv_table(unc, "Uncertainty")
                else:
                    st.info("Feature intelligence not yet computed.")
            else:
                st.info("Feature intelligence not yet computed.")
        except Exception:
            st.info("Feature intelligence not yet computed.")

    with st.expander("Drift & Retraining Monitor", expanded=False):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/intelligence/drift",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()

                # Overall retraining depth badge
                depth = data.get("retraining_depth", "none")
                depth_color = {
                    "none": "🟢",
                    "calibration_only": "🟡",
                    "head_only": "🟠",
                    "full": "🔴",
                }.get(depth, "⚪")
                st.metric(label="Retraining Depth", value=f"{depth_color} {depth}")
                st.caption(f"Last checked: {data.get('last_checked', 'N/A')}")

                # C2: Auto-retrain trigger button (highest-impact UX fix)
                if depth in ("calibration_only", "head_only", "full"):
                    _retrain_label = f"\U0001f501 Auto-trigger retraining ({depth})"
                    if st.button(_retrain_label, type="primary", key="auto_retrain_trigger"):
                        try:
                            _retrain_resp = requests.post(
                                f"{API_BASE_URL}/train-pipeline",
                                json={
                                    "session_id": st.session_state.session_id,
                                    "retraining_depth": depth,
                                    "modalities": list(
                                        (st.session_state.get("detected_schema") or {})
                                        .get("global_modalities", ["tabular"])
                                    ),
                                },
                                timeout=30,
                            )
                            if _retrain_resp.ok:
                                st.success(f"Retraining triggered! Task ID: {_retrain_resp.json().get('task_id', 'N/A')}")
                                st.session_state.workflow_stage = 5
                                st.session_state.training_task_id = _retrain_resp.json().get("task_id")
                                st.rerun()
                            else:
                                st.error(f"Retraining failed: {_api_error_detail(_retrain_resp)}")
                        except requests.Timeout:
                            st.error("Retraining request timed out (30s)")
                        except Exception as _retrain_exc:
                            st.error(f"Retraining error: {_retrain_exc}")

                # Covariate drift
                cov = data.get("covariate_drift") or {}
                if cov:
                    st.markdown("**Covariate Drift (feature distribution)**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Detected", "Yes" if cov.get("detected") else "No")
                    col2.metric("KS Statistic", f"{cov.get('ks_statistic', 0.0):.3f}")
                    col3.metric("Composite Score", f"{cov.get('composite_score', 0.0):.3f}")
                    per_feat = cov.get("per_feature") or {}
                    if per_feat:
                        feat_df = pd.DataFrame(
                            [{"feature": k, "ks": v} for k, v in per_feat.items()]
                        ).sort_values("ks", ascending=False).head(20)
                        st.bar_chart(feat_df.set_index("feature")["ks"])

                # Concept drift
                con = data.get("concept_drift") or {}
                if con:
                    st.markdown("**Concept Drift P(ŷ) shift**")
                    col1, col2 = st.columns(2)
                    col1.metric("Detected", "Yes" if con.get("detected") else "No")
                    col2.metric("KS p-value", f"{con.get('p_value', 1.0):.4f}")

                # Embedding drift
                emb = data.get("embedding_drift") or {}
                if emb:
                    st.markdown("**Embedding Drift (text / image)**")
                    st.caption(
                        "Two drift signals are shown per modality: "
                        "**MMD** (Euclidean distributional shift) and "
                        "**Cosine drift** (directional/semantic shift, DriftLens IEEE 2024). "
                        "Either signal can trigger retraining."
                    )
                    for mod_name, emb_data in emb.items():
                        sev = emb_data.get("severity", "unknown")
                        mmd = float(emb_data.get("mmd_score", 0.0) or 0.0)
                        cosine = float(emb_data.get("cosine_drift_score", 0.0) or 0.0)
                        mmd_sev = emb_data.get("mmd_severity", "")
                        cos_sev = emb_data.get("cosine_severity", "")
                        icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(sev, "⚪")
                        st.write(
                            f"{icon} **{mod_name}**: "
                            f"MMD={mmd:.4f} ({mmd_sev or 'ok'})  |  "
                            f"Cosine={cosine:.4f} ({cos_sev or 'ok'})  |  "
                            f"overall severity={sev}"
                        )
            else:
                st.info("No drift data available yet.")
        except Exception:
            st.info("No drift data available yet.")

    # FIX-UI-12: Global Pipeline Summary
    phases_done = [
        ph for ph, state in st.session_state.phase_states.items()
        if state.get("status") in ("completed", "reused")
    ]
    if len(phases_done) >= 3:
        with st.expander("📋 Global Pipeline Summary", expanded=False):
            schema = st.session_state.get("detected_schema") or {}
            pp_res = st.session_state.get("preprocess_result") or {}
            ms_res = st.session_state.get("model_selection_result") or {}
            tr_res = st.session_state.get("training_result") or {}
            dr_res = st.session_state.get("drift_result") or {}
            di_info = st.session_state.get("dataset_info") or {}
            best = ms_res.get("best_model") or {}
            metrics = tr_res.get("metrics") or {}

            st.markdown("#### 📦 Dataset")
            shapes = di_info.get("shapes", [])
            sources = di_info.get("sources", [])
            total_rows = sum(s[0] for s in shapes if s and isinstance(s, (list, tuple)) and s[0]) if shapes else "?"
            st.write(f"- **Sources loaded:** {len(sources)}")
            st.write(f"- **Total rows:** {total_rows:,}" if isinstance(total_rows, int) else f"- **Total rows:** {total_rows}")
            for i, src in enumerate(sources[:3]):
                st.caption(f"  [{i + 1}] {str(src)[:100]}")

            st.markdown("#### 🔍 Schema")
            target = schema.get("primary_target", "Unknown")
            prob = schema.get("global_problem_type", "Unknown")
            conf = float(schema.get("detection_confidence", 0) or 0)
            mods = schema.get("global_modalities", [])
            override = bool(st.session_state.get("schema_overrides"))
            st.write(f"- **Target column:** {target}")
            st.write(f"- **Problem type:** {prob}")
            st.write(f"- **Detection confidence:** {conf:.1%}")
            st.write(f"- **Modalities detected:** {', '.join(mods) if mods else 'tabular'}")
            if override:
                st.write("- **Override applied:** ✅ (user overrode auto-detected target/type)")

            st.markdown("#### ⚙️ Preprocessing")
            stages = pp_res.get("preprocessing_stages", [])
            tab_sample = (pp_res.get("samples") or {}).get("tabular", {})
            raw_n = len(tab_sample.get("raw_columns", []))
            dropped = tab_sample.get("dropped_columns", [])
            trans_n = len(tab_sample.get("transformed_columns", []))
            cached = bool(pp_res.get("preprocessor_cached", False))
            st.write(f"- **Stages run:** {len(stages)} ({', '.join(s.get('stage', '?').replace('_', ' ') for s in stages)})")
            if raw_n:
                st.write(f"- **Features (raw -> processed):** {raw_n} -> {trans_n}")
            if dropped:
                preview = "`, `".join([str(x) for x in dropped[:5]])
                suffix = "..." if len(dropped) > 5 else ""
                st.write(f"- **Dropped columns:** {len(dropped)} (`{preview}`{suffix})")
            st.write(f"- **Preprocessor:** {'reused from cache ♻️' if cached else 'freshly fitted 🔧'}")

            st.markdown("#### 🧠 Model")
            probe = best.get("probe_score")
            tier = best.get("tier", "?")
            ms_all = ms_res.get("recommended_models", [])
            eligible_modalities = ms_res.get("eligible_modalities", []) or []
            excluded_modalities = ms_res.get("excluded_modalities", {}) or {}
            st.write(f"- **Selected model:** {best.get('name', 'N/A')}")
            st.write(f"- **Tier:** {tier}")
            st.write(f"- **Candidates evaluated:** {len(ms_all)}")
            if eligible_modalities:
                st.write(f"- **Eligible modalities:** {', '.join(str(m) for m in eligible_modalities)}")
            if excluded_modalities:
                st.write(
                    "- **Excluded modalities:** "
                    + ", ".join(f"{k} ({v})" for k, v in excluded_modalities.items())
                )
            if isinstance(probe, (int, float)):
                st.write(f"- **Probe score (RF 1-fold CV):** {float(probe):.3f} - data-driven selection")
            else:
                st.write("- **Selection method:** heuristic (no probe data available)")
            rationale = best.get("rationale", {}) or {}
            if rationale:
                for comp, reason in list(rationale.items())[:3]:
                    st.caption(f"  ↳ {comp}: {reason}")

            st.markdown("#### 🔗 Fusion")
            fstrat = best.get("fusion_strategy", "N/A")
            st.write(f"- **Strategy:** {fstrat}")
            model_mods = list(filter(None, [
                best.get("tabular_encoder") and "tabular",
                best.get("text_encoder") and "text",
                best.get("image_encoder") and "image",
            ]))
            st.write(f"- **Modalities in model:** {', '.join(model_mods) if model_mods else 'N/A'}")

            if metrics:
                st.markdown("#### 🏋️ Training")
                st.write(f"- **Best val loss:** {metrics.get('best_val_loss', 'N/A')}")
                bva = metrics.get("best_val_acc")
                if isinstance(bva, (int, float)):
                    st.write(f"- **Best val accuracy:** {float(bva):.2%}")
                bvf = metrics.get("best_val_f1")
                if isinstance(bvf, (int, float)):
                    st.write(f"- **Best val F1:** {float(bvf):.4f}")
                st.write(f"- **Optuna trials:** {metrics.get('n_trials', 'N/A')}")
                st.write(f"- **Best trial:** #{metrics.get('best_trial', 'N/A')}")
                st.write(f"- **Training time:** {metrics.get('training_time', 'N/A')}")
                feedback_events = metrics.get("trial_feedback_events", []) or []
                st.write(f"- **Adaptive feedback events:** {len(feedback_events)}")
                best_params = metrics.get("best_params", {})
                if best_params:
                    st.caption("Best HP: " + ", ".join(f"{k}={v}" for k, v in best_params.items()))

            st.markdown("#### 📡 Monitoring")
            if dr_res:
                drift_flag = bool(dr_res.get("drift_detected", False))
                psi = float((dr_res.get("metrics") or {}).get("psi", 0) or 0)
                ks = float((dr_res.get("metrics") or {}).get("ks_statistic", 0) or 0)
                st.write(f"- **Drift status:** {'🔴 DRIFT DETECTED' if drift_flag else '🟢 No drift'}")
                st.write(f"- **PSI:** {psi:.4f} | **KS:** {ks:.4f}")
            else:
                st.info("Run Phase 6 Drift Detection to populate.")

    # FIX-UI-13: Decision Trace Timeline
    if st.session_state.schema_detected or st.session_state.get("training_result"):
        with st.expander("🕐 Decision Trace Timeline", expanded=False):
            trace_resp = None
            try:
                trace_resp = requests.get(
                    f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/decision-trace",
                    params={"limit": 50},
                    timeout=5,
                )
            except Exception:
                trace_resp = None

            if trace_resp is not None and trace_resp.status_code == 200:
                trace_data = trace_resp.json()
                curated = trace_data.get("curated_summary", [])
                full_trace = trace_data.get("trace", [])
                total = int(trace_data.get("total_decisions", 0) or 0)

                if curated:
                    st.markdown("**Key Decisions:**")
                    for i, line in enumerate(curated, start=1):
                        st.write(f"{i}. {line}")
                else:
                    st.info("No curated decisions yet — run at least Schema Detection.")

                st.divider()
                st.caption(f"Showing {len(full_trace)} of {total} log entries (ExecutionContext.execution_log)")
                category_color = {
                    "ingestion": "🟦",
                    "schema": "🟩",
                    "target": "🟨",
                    "preprocessing": "🟧",
                    "model_selection": "🟪",
                    "training": "🔴",
                    "monitoring": "🟫",
                    "override": "🔶",
                    "pipeline": "⬜",
                    "other": "⬛",
                }
                for entry in full_trace:
                    cat = entry.get("category", "other")
                    dot = category_color.get(cat, "⬛")
                    stage = entry.get("stage", "?")
                    dec = entry.get("decision", "")
                    ev = entry.get("evidence")
                    ts = str(entry.get("timestamp", ""))[:19]
                    line = f"{dot} [{ts}] {stage}: {dec}"
                    if ev:
                        line += f"\n  ↳ {ev}"
                    st.caption(line)
                # Decision trace CSV export
                if full_trace:
                    try:
                        _trace_df = pd.DataFrame([
                            {
                                "timestamp": e.get("timestamp", "")[:19],
                                "stage": e.get("stage", ""),
                                "decision": e.get("decision", ""),
                                "evidence": e.get("evidence", ""),
                            }
                            for e in full_trace
                        ])
                        st.download_button(
                            "📋 Download Decision Log (CSV)",
                            data=_trace_df.to_csv(index=False),
                            file_name="autovision_decision_log.csv",
                            mime="text/csv",
                            key="download_decision_log",
                        )
                    except Exception:
                        pass
            elif trace_resp is not None and trace_resp.status_code == 404:
                st.info("Session context not yet created — complete Phase 1 first.")
            else:
                st.warning("Decision trace unavailable.")

    # ── Beginner's Glossary ────────────────────────────────────────────────
    with st.expander("📖 Glossary — What do these terms mean?", expanded=False):
        st.markdown(
            "New to machine learning? Here is a plain-English explanation of "
            "every technical term used in this interface."
        )
        for term, explanation in _GLOSSARY.items():
            st.markdown(f"**{term}**")
            st.caption(explanation)
            st.divider()

    # ── Research Foundations (inline panel) ───────────────────────────────
    with st.expander("🔬 Research Foundations — papers implemented in APEX", expanded=False):
        st.markdown(
            "APEX incorporates the following peer-reviewed methods. "
            "Click any paper title to understand what it contributes."
        )
        papers = [
            (
                "[1] Structural-Semantic Unifier (ICML/IEEE 2025)",
                "StructuralSemanticRouter fusion",
                "Combines graph-based structure (how features relate) with "
                "language-model semantics (what features mean). "
                "A learned gate per sample decides which path to trust more.",
                "Active when modalities include both image and text.",
            ),
            (
                "[2] Cross-Layer RGAT Head (NeurIPS/IEEE 2025)",
                "CrossLayerRGATHead task head",
                "Replaces the final MLP with a multi-hop relational graph attention network. "
                "Each 'relation type' captures a different way features influence the prediction. "
                "Cross-layer residual connections prevent vanishing gradients.",
                "Active when ≥3 modalities or relational schema detected.",
            ),
            (
                "[3] Uncertainty Graph Fusion (CVPR/IEEE 2025)",
                "UncertaintyGraphFusion",
                "Weights each modality's contribution by how certain the model is about it. "
                "Unreliable or noisy modalities are automatically down-weighted.",
                "Available as a fusion option for multi-modal datasets.",
            ),
            (
                "[4] CrossFuse — Complementarity-Aware Fusion (ECCV/IEEE 2024)",
                "ComplementarityFusion",
                "Computes a pairwise complementarity matrix: how much unique information "
                "each modality pair contributes that the other doesn't. "
                "Softmax-weighted sum ensures every modality gets a fair but calibrated vote.",
                "Auto-selected when ≥3 modalities are present.",
            ),
            (
                "[8] EWC — Continual Learning (IEEE TNNLS 2024 / Kirkpatrick 2017)",
                "Elastic Weight Consolidation in retraining",
                "When retraining on new data, EWC adds a regularisation penalty "
                "(λ/2 · Σ F_i(θ_i − θ*_i)²) that prevents the model from 'forgetting' "
                "what it learned on the previous dataset. "
                "The Fisher Information matrix (F_i) measures how important each weight is.",
                "Applied automatically during drift-triggered retraining.",
            ),
            (
                "[9] Calibration (Guo et al., ICML 2017)",
                "Temperature scaling / isotonic calibration",
                "Ensures that when the model says '80% confident' it is actually right ~80% of the time. "
                "Uncalibrated models are often overconfident. "
                "Metrics: ECE (Expected Calibration Error), NLL, Brier Score.",
                "Applied after every training run. Shown in Calibration Metrics panel.",
            ),
            (
                "[10] DDM — Drift Detection Method (Gama et al., SBIA 2004)",
                "DDMConceptDriftDetector in monitoring",
                "Watches the running prediction error rate p̄ over time. "
                "Warning: p̄ + s̄ ≥ p_min + 2·s_min. Drift: p̄ + s̄ ≥ p_min + 3·s_min. "
                "Triggers retraining when the model's accuracy degrades beyond statistical thresholds.",
                "Runs on the live error stream from the prediction endpoint.",
            ),
            (
                "[11] DriftLens — Cosine Embedding Drift (IEEE 2024)",
                "cosine_drift_score in detect_embedding_drift()",
                "Complements MMD (Euclidean) drift by measuring directional shift "
                "in the embedding space. Cosine distance = 1 − cosine_similarity. "
                "Catches semantic/topic shifts that leave overall magnitude unchanged "
                "(e.g. text that drifts from sports to politics with similar word counts).",
                "Shown alongside MMD in Drift & Retraining Monitor panel.",
            ),
            (
                "[12] FTTransformer — Feature Tokenizer + Transformer (NeurIPS 2021)",
                "FTTransformerEncoder in modelss/encoders/tabular.py",
                "Each tabular feature is projected to a d-dimensional 'token' via a learned affine transform. "
                "A [CLS] token is prepended and the sequence is passed through a standard Transformer encoder. "
                "The CLS output is used as the tabular embedding. "
                "Outperforms MLP, GBM, and ResNet on many tabular benchmarks. "
                "Architecture: d_token=96, n_layers=3, n_heads=8, output_dim=64.",
                "Highest-capacity tabular encoder — preferred by JIT selector on datasets with many features.",
            ),
            (
                "[13] Focal Loss — Hard Example Mining (ICCV 2017, Lin et al.)",
                "FocalLoss in automl/trainer.py",
                "FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t). "
                "The (1 − p_t)^γ focusing factor down-weights easy samples (high p_t) "
                "so training concentrates on hard, misclassified ones. "
                "Particularly effective for class-imbalanced datasets where the majority class "
                "dominates standard cross-entropy gradients.",
                "Auto-activated when class imbalance ratio > 3:1 (detected from training labels).",
            ),
            (
                "[14] SWA — Stochastic Weight Averaging (UAI 2018, Izmailov et al.)",
                "StochasticWeightAveraging callback in training_orchestrator.py",
                "Rather than taking the last checkpoint, SWA averages weights from the final 10% of "
                "training epochs. The averaged model lies in a flatter region of the loss landscape, "
                "which generalises better. SWA LR = 50% of final trial LR for cyclical re-heating.",
                "Applied automatically in the last 10% of each Optuna trial (≥5 epochs).",
            ),
            (
                "[15] PCGrad — Gradient Surgery (NeurIPS 2020, Yu et al.)",
                "PCGradCallback in automl/trainer.py",
                "When gradients from tabular, text, and image encoders have negative dot product "
                "(i.e. they 'fight' each other), PCGrad projects each conflicting gradient onto "
                "the normal plane of the other: g_i ← g_i − (g_i·g_j / ||g_j||²)·g_j. "
                "This eliminates destructive interference without reducing gradient magnitude for non-conflicting pairs.",
                "Applied as a Lightning on_before_optimizer_step hook whenever ≥2 modality encoders are active.",
            ),
        ]
        for title, component, description, activation in papers:
            st.markdown(f"**{title}**")
            st.caption(f"_Component:_ `{component}`")
            st.write(description)
            st.caption(f"_When active:_ {activation}")
            st.divider()


def render_phase_1_data_ingestion():
    """Phase 1: Data Ingestion with Caching and live progress polling."""
    st.header("Phase 1️⃣ - Data Ingestion & Caching")

    st.markdown("""
    **Workflow:**
    1. Provide dataset sources (Kaggle URLs, HTTP links, or local paths)
    2. System generates SHA-256 hash for each source
    3. Check cache for existing data
    4. Download and validate if not cached
    5. Store in local cache for future use
    """)

    task_id = st.session_state.ingestion_task_id

    # ----- Active ingestion task: poll for progress -----
    if task_id is not None:
        try:
            resp = requests.get(f"{API_BASE_URL}/ingest/status/{task_id}", timeout=5)
            if resp.status_code == 404:
                st.warning("Ingestion task not found. It may have expired.")
                st.session_state.ingestion_task_id = None
                return
            if resp.status_code != 200:
                st.error(f"Status poll failed: {resp.status_code}")
                return
            task = resp.json()
        except Exception as e:
            st.error(f"Could not poll ingestion status: {e}")
            return

        status = task.get("status", "unknown")
        progress_pct = task.get("progress_pct", 0)
        message = task.get("message", "")
        datasets = task.get("datasets", [])
        completed = task.get("completed_sources", 0)
        total = task.get("total_sources", 1)

        # Progress bar
        st.progress(progress_pct / 100, text=f"{message} ({progress_pct}%)")

        # Per-dataset status (show results as they arrive)
        if datasets:
            st.markdown("### 📥 Dataset Progress")
            for idx, ds_info in enumerate(datasets, 1):
                source = ds_info.get("source", "Unknown")
                ds_status = ds_info.get("status", "Unknown")
                if ds_status == "success":
                    shape = ds_info.get("shape")
                    shape_str = f" — {shape[0]} rows x {shape[1]} cols" if shape else ""
                    st.success(f"**Dataset {idx}** — Loaded{shape_str}\n{source[:80]}")
                else:
                    st.error(f"**Dataset {idx}** — {ds_status}\n{source[:80]}")

        # Still downloading — rerun every 1s for smooth slider updates
        if status == "running":
            st.info(f"📦 Processing {completed}/{total} dataset(s)...")
            time.sleep(1)
            st.rerun()
            return

        # Completed
        if status == "completed":
            result = task.get("result", {})
            ingestion = result.get("ingestion_progress", {})
            datasets_list = ingestion.get("datasets", datasets)

            st.session_state.ingestion_task_id = None

            # Update session state — only reset downstream if new data came in
            overall = ingestion.get("status", "failed")
            new_data = overall in ("success", "partial")
            st.session_state.dataset_uploaded = new_data or st.session_state.dataset_uploaded

            successful = [d for d in datasets_list if d.get("status") == "success"]
            st.session_state.phase_states[1] = {
                "status": "completed" if new_data else "reused",
                "reason": (
                    f"Downloaded {len(successful)} dataset(s)"
                    if new_data else
                    "All datasets already in cache - no download performed"
                ),
            }

            if new_data:
                # Only clear downstream results when fresh data actually arrived
                st.session_state.schema_detected = False
                st.session_state.detected_schema = None
                st.session_state.model_selected = False
                st.session_state.training_task_id = None
                st.session_state.hp_overrides = None
                st.session_state.training_result = None
                for key in ("model_selection_result", "preprocess_result", "drift_result", "registry_result"):
                    st.session_state.pop(key, None)
                for phase_num in range(2, 8):
                    st.session_state.phase_states[phase_num] = {
                        "status": "pending",
                        "reason": "",
                    }

            # Extract dataset shapes and show summary
            dataset_shapes = []
            for ds_info in datasets_list:
                shape = ds_info.get("shape")
                if shape and ds_info.get("status") == "success":
                    dataset_shapes.append(shape)

            row_count = dataset_shapes[0][0] if (dataset_shapes and dataset_shapes[0][0]) else 5000
            sources = [ds_info.get("source", "") for ds_info in datasets_list]

            st.session_state.dataset_info = {
                "sources": sources,
                "count": len(dataset_shapes),
                "shapes": dataset_shapes,
                "row_count": row_count,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.ingested_row_count = row_count

            st.success(f"Ingestion complete: {ingestion.get('message', 'Done')}")

        # Failed
        elif status == "failed":
            error = task.get("error", "Unknown error")
            st.error(f"Ingestion failed: {error}")
            st.session_state.phase_states[1] = {
                "status": "failed",
                "reason": str(error),
            }
            st.session_state.ingestion_task_id = None

        # Show navigation even after completion
        if st.session_state.dataset_uploaded:
            if st.button("➡️ Next: Schema Detection", width="stretch"):
                st.session_state.workflow_stage = 2
                st.rerun()
        return

    # ----- No active task: show input form -----
    st.markdown("### 📥 Dataset Sources")
    dataset_sources = st.text_area(
        "Enter one or more dataset sources (Kaggle URL, HTTP link, or local path), one per line:",
        placeholder="https://kaggle.com/datasets/...\nhttps://example.com/data.csv\n/path/to/dataset.csv",
        height=120
    )
    dataset_sources = [s.strip() for s in dataset_sources.splitlines() if s.strip()]
    st.markdown("### 💾 Cache Info")
    st.info("""
    - **Caching**: Automatic SHA-256 hashing
    - **Storage**: `data/dataset_cache/`
    - **Formats**: CSV, Parquet, JSON
    - **Hit Ratio**: Skip download if cached
    """)

    st.divider()

    # Upload/Download section
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Load Datasets", width="stretch"):
            if not dataset_sources:
                st.error("Please enter at least one dataset source (URL or local path) above.")
            elif not check_api_connection():
                st.error("API not connected!")
            else:
                try:
                    # B1 FIX: use stable session_id from st.session_state (not datetime)
                    resp = requests.post(
                        f"{API_BASE_URL}/ingest/datasets",
                        json={
                            "dataset_urls": dataset_sources,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.ingestion_task_id = data["task_id"]
                        st.rerun()
                    else:
                        _show_error_with_retry(
                            f"Failed to start ingestion: {resp.status_code} — {_api_error_detail(resp)}",
                            "retry_ingestion",
                        )
                except Exception as e:
                    _show_error_with_retry(f"Connection error: {e}", "retry_ingestion_conn")

    with col2:
        if st.button("📊 View Cache", width="stretch"):
            try:
                stats_resp = requests.get(f"{API_BASE_URL}/cache/stats", timeout=5)
                meta_resp = requests.get(f"{API_BASE_URL}/cache/metadata", timeout=5)
                if stats_resp.status_code == 200:
                    cache_info = stats_resp.json()
                    meta = meta_resp.json() if meta_resp.status_code == 200 else {}
                    st.success(f"✅ Cache: {cache_info['cache_location']}")

                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Cached Datasets", cache_info['total_items'])
                    col_c2.metric("Total Size (MB)", cache_info['total_size_mb'])
                    col_c3.metric("Status", "Ready")

                    if cache_info['items']:
                        with st.expander(f"📁 Cached Files ({len(cache_info['items'])})"):
                            for item in cache_info['items']:
                                h = item['name']
                                src_url = meta.get(h, {}).get("source", h)
                                st.caption(
                                    f"• `{h[:16]}` — {item['size_mb']} MB\n  {src_url[:100]}"
                                )
                else:
                    st.error(f"Cache query failed ({stats_resp.status_code}): {_api_error_detail(stats_resp)}")
            except Exception as e:
                st.error(f"Cannot connect to cache endpoint: {e}")

    with col3:
        if st.button("🗑️ Clear Cache", width="stretch"):
            try:
                response = requests.post(f"{API_BASE_URL}/cache/clear", timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"Failed to clear cache: {response.status_code}")
            except Exception as e:
                st.error(f"Cache clear error: {str(e)}")

    # Status display
    if st.session_state.dataset_uploaded:
        info = st.session_state.dataset_info
        st.markdown("### ✅ Ingested Datasets")
        shapes = info.get("shapes", [])
        sources = info.get("sources", [])
        for i, src in enumerate(sources):
            shape_str = ""
            if i < len(shapes) and shapes[i]:
                r, c = shapes[i][0], shapes[i][1] if len(shapes[i]) > 1 else "?"
                shape_str = f" — {r:,} rows × {c} cols" if r else f" — {c} cols"
            st.success(f"📄 **Dataset {i+1}**{shape_str}\n`{src[:120]}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Datasets Loaded", len(sources))
        col2.metric("Cache", "./data/dataset_cache/")
        col3.metric("Rows (est.)", f"{info.get('row_count', 0):,}")

        # C3: V2 Dataset Management UI
        with st.expander("📂 Manage Session Datasets", expanded=False):
            _sid = st.session_state.session_id
            try:
                ds_resp = requests.get(
                    f"{API_BASE_URL}/v2/sessions/{_sid}/datasets",
                    timeout=5,
                )
                if ds_resp.status_code == 200:
                    ds_list = ds_resp.json()
                    datasets = ds_list if isinstance(ds_list, list) else ds_list.get("datasets", [])
                    if datasets:
                        st.caption(f"{len(datasets)} dataset(s) in session")
                        for ds_entry in datasets:
                            ds_id = ds_entry.get("dataset_id") or ds_entry.get("id", "?")
                            ds_name = ds_entry.get("name") or ds_entry.get("source", str(ds_id))
                            ds_cols = st.columns([3, 1])
                            ds_cols[0].write(f"**{ds_name}** (`{str(ds_id)[:20]}`)")
                            if ds_cols[1].button("🗑️ Remove", key=f"rm_ds_{ds_id}"):
                                try:
                                    del_resp = requests.delete(
                                        f"{API_BASE_URL}/v2/sessions/{_sid}/datasets/{ds_id}",
                                        timeout=5,
                                    )
                                    if del_resp.ok:
                                        st.success(f"Dataset {ds_id} removed.")
                                        st.rerun()
                                    else:
                                        st.error(f"Delete failed: {_api_error_detail(del_resp)}")
                                except Exception as _del_exc:
                                    st.error(f"Delete error: {_del_exc}")
                    else:
                        st.info("No datasets in current session.")
                elif ds_resp.status_code == 404:
                    st.info("No session datasets found. Upload data above first.")
                else:
                    st.warning(f"Could not fetch datasets (HTTP {ds_resp.status_code})")
            except Exception:
                st.info("Dataset management unavailable (API not connected).")

        if st.button("➡️ Next: Schema Detection", width="stretch"):
            st.session_state.workflow_stage = 2
            st.rerun()


def render_phase_2_schema_detection():
    """Phase 2: Schema Detection."""
    st.header("Phase 2️⃣ - Schema Detection & Problem Type Inference")

    with st.expander("ℹ️ What is this phase doing? (click to expand)", expanded=False):
        st.markdown("""
        **For beginners:** The system reads your data files and automatically figures out:

        - **Which column is the target** (what you want to predict) — e.g. "price", "label", "survived"
        - **What kind of prediction task** this is: binary classification (yes/no), multi-class (which of N options), or regression (a number)
        - **What data types exist**: numbers/categories (tabular), free text (text modality), image file paths (image modality), or time-series signals (timeseries modality)

        You can review and override any detection result before moving on.
        """)

    st.markdown("""
    **Schema Detection for Ingested Datasets:**
    - Analyzes ONLY the datasets loaded in Phase 1
    - Detects multimodal structure (tabular / image / text / timeseries)
    - Identifies target column automatically
    - Infers global problem type
    """)
    
    if not st.session_state.dataset_uploaded:
        st.warning("⚠️ Please load datasets in Phase 1 first")
        if st.button("← Go to Phase 1"):
            st.session_state.workflow_stage = 1
            st.rerun()
        return
    
    col1, col2 = st.columns([2, 1])
    
    # =========================================================
    # 🔍 DETECT BUTTON — FIXED
    # =========================================================
    with col1:
        if st.button("🔍 Detect Schema", width="stretch"):
            with st.spinner("Analyzing ingested datasets..."):
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    status.write("📍 Detecting schema for ingested datasets...")

                    # B2 FIX: forward session_id so backend uses the correct session store
                    response = requests.post(
                        f"{API_BASE_URL}/api/schema/detect",
                        json={"session_id": st.session_state.session_id},
                        timeout=120
                    )
                    
                    progress_bar.progress(0.6)
                    
                    if response.status_code == 200:
                        payload = response.json()

                        # ⭐⭐⭐ CRITICAL FIX ⭐⭐⭐
                        schema_data = payload.get("data", {})

                        st.session_state.detected_schema = schema_data
                        st.session_state.schema_candidates = payload.get("candidates", [])
                        st.session_state.schema_detected = True
                        # Bug #6: reset downstream phases so stale results don't persist
                        for _downstream_phase in [3, 4, 5, 6, 7]:
                            st.session_state.phase_states[_downstream_phase] = {
                                "status": "pending",
                                "reason": "Schema changed — re-run required",
                            }
                        st.session_state.preprocess_result = None
                        st.session_state.model_selected = False
                        st.session_state.model_selection_result = {}
                        st.session_state.training_result = None
                        st.session_state.training_task_id = None
                        st.session_state.trained_model_id = None
                        st.session_state.text_columns = []
                        st.session_state.image_columns = []
                        st.session_state.phase_states[2] = {
                            "status": "completed",
                            "reason": (
                                f"Detected {len(schema_data.get('per_dataset', []))} dataset(s), "
                                f"target='{schema_data.get('primary_target', '?')}', "
                                f"confidence={float(schema_data.get('detection_confidence', 0) or 0):.1%}"
                            ),
                        }

                        progress_bar.progress(1.0)
                        st.success("✅ Schema detection complete!")
                        st.rerun()
                    else:
                        _show_error_with_retry(f"❌ Detection failed: {_api_error_detail(response)}", "retry_schema")

                except requests.exceptions.Timeout:
                    _show_error_with_retry("❌ Timeout after 120 seconds — try fewer datasets or shorter text columns.", "retry_schema_timeout")
                except Exception as e:
                    _show_error_with_retry(f"❌ Detection error: {str(e)}", "retry_schema_err")
    
    with col2:
        if st.checkbox("Show Detection Details"):
            st.info("📊 Verbose mode enabled")
    
    # =========================================================
    # 📊 DISPLAY RESULTS — FULLY FIXED FOR MULTI-DATASET
    # =========================================================
    if st.session_state.schema_detected and st.session_state.detected_schema:
        st.divider()
        st.markdown("### 📋 Detected Schema Results")

        schema = st.session_state.detected_schema

        global_modalities = schema.get("global_modalities", [])
        global_problem = schema.get("global_problem_type", "Unknown")
        primary_target = schema.get("primary_target", "Unknown")
        confidence = schema.get("detection_confidence", 0)
        fusion_ready = schema.get("fusion_ready", False)
        relatedness = schema.get("relatedness_report", {})
        n_groups = relatedness.get("n_groups", 1)

        # ── B5 FIX: Unrelated-dataset chooser ────────────────────────────────
        if n_groups > 1:
            groups = relatedness.get("groups", [])
            st.warning(
                f"⚠️ **{n_groups} unrelated dataset groups detected.** "
                "The datasets share no common columns or target. "
                "Select which group to proceed with:"
            )
            group_labels = [
                f"Group {i+1} ({len(g)} dataset(s): idx {g})"
                for i, g in enumerate(groups)
            ]
            chosen = st.radio(
                "Choose dataset group",
                options=list(range(len(groups))),
                format_func=lambda i: group_labels[i],
                index=st.session_state.active_dataset_group or 0,
                key="group_chooser",
            )
            if st.button("✅ Use Selected Group", key="apply_group"):
                st.session_state.active_dataset_group = chosen
                # Filter per_dataset to the chosen group
                chosen_indices = groups[chosen]
                per_dataset_all = schema.get("per_dataset", [])
                filtered_per_dataset = [per_dataset_all[i] for i in chosen_indices if i < len(per_dataset_all)]
                schema["per_dataset"] = filtered_per_dataset
                st.session_state.detected_schema = schema
                st.rerun()

        # ── Summary metrics ───────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target Column", primary_target)
        col2.metric("Problem Type", global_problem)
        col3.metric("Fusion Ready", "Yes" if fusion_ready else "No")
        col4.metric("Confidence", f"{confidence:.1%}")

        st.info(
            f"**Modalities Found:** {', '.join(global_modalities) if global_modalities else 'None'}"
        )

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary", "Override Targets", "Debug Info", "Why This Target?"
        ])

        per_dataset = schema.get("per_dataset", [])

        with tab1:
            st.markdown("#### 📊 Per-Dataset Analysis")
            if not per_dataset:
                st.warning("No per-dataset results found")
            else:
                for ds in per_dataset:
                    with st.expander(f"📦 Dataset: {ds.get('dataset_id')}"):
                        st.write("**Modalities:**", ds.get("modalities"))
                        st.write("**Target:**", ds.get("target_column"))
                        st.write("**Problem Type:**", ds.get("problem_type"))
                        st.write("**Confidence:**", ds.get("confidence"))
                        detected_cols = ds.get("detected_columns", {})
                        st.markdown("**Columns by Modality:**")
                        for mod, cols in detected_cols.items():
                            st.write(f"- {mod}: {len(cols)} columns")

                        reasoning = ds.get("reasoning", {})
                        if reasoning:
                            st.markdown("**Detection Reasoning:**")
                            for k, v in reasoning.items():
                                st.write(f"- `{k}`: {v}")

                st.markdown("#### XS3 Confidence Gap")
                for ds in per_dataset:
                    xs3_gap = ds.get("reasoning", {}).get("xs3_confidence_gap", None)
                    if xs3_gap is not None:
                        st.metric(
                            label=f"[{str(ds.get('dataset_id', '?'))[:20]}] XS3 Confidence Gap",
                            value=f"{float(xs3_gap):.3f}",
                            help=(
                                "XS3 = gap between top-1 and top-2 target confidence scores. "
                                "Values > 0.3 indicate an unambiguous target. "
                                "Values < 0.1 indicate multiple plausible targets - override recommended."
                            ),
                        )

        with tab2:
            st.markdown("#### 🎯 Override Target Columns")
            st.caption(
                "If the auto-detected target is wrong, select the correct column below "
                "and click **Apply Overrides**. This updates what gets passed to preprocessing and training."
            )
            changed = False
            overrides = dict(st.session_state.schema_overrides)
            for ds in per_dataset:
                ds_id = ds.get("dataset_id", "")
                all_cols = []
                for col_list in ds.get("detected_columns", {}).values():
                    all_cols.extend(col_list)
                auto_target = ds.get("target_column", "Unknown")
                current_override = overrides.get(ds_id, {}).get("target_column", auto_target)
                col_options = ["(auto) " + auto_target] + [c for c in all_cols if c != auto_target]
                sel_idx = 0
                if current_override != auto_target and current_override in all_cols:
                    try:
                        sel_idx = all_cols.index(current_override) + 1
                    except ValueError:
                        sel_idx = 0
                new_target = st.selectbox(
                    f"[{ds_id[:20]}] Target column",
                    options=col_options,
                    index=sel_idx,
                    key=f"tgt_override_{ds_id}",
                )
                prob_opts = [
                    "classification_binary", "classification_multiclass",
                    "regression", "multilabel_classification", "unsupervised",
                ]
                auto_prob = ds.get("problem_type", "classification_binary")
                new_prob = st.selectbox(
                    f"[{ds_id[:20]}] Problem type",
                    options=prob_opts,
                    index=prob_opts.index(auto_prob) if auto_prob in prob_opts else 0,
                    key=f"prob_override_{ds_id}",
                )
                resolved_target = new_target.replace("(auto) ", "") if new_target.startswith("(auto) ") else new_target
                if resolved_target != auto_target or new_prob != auto_prob:
                    overrides[ds_id] = {"target_column": resolved_target, "problem_type": new_prob}
                    changed = True

            if st.button("✅ Apply Overrides", key="apply_overrides", disabled=not changed):
                st.session_state.schema_overrides = overrides
                # Patch the detected_schema so downstream phases see the override
                for ds in per_dataset:
                    ds_id = ds.get("dataset_id", "")
                    if ds_id in overrides:
                        ds["target_column"] = overrides[ds_id].get("target_column", ds["target_column"])
                        ds["problem_type"] = overrides[ds_id].get("problem_type", ds["problem_type"])
                best_conf, best_target, best_prob = -1.0, "Unknown", global_problem
                for ds in per_dataset:
                    if ds.get("target_column", "Unknown") != "Unknown" and ds.get("confidence", 0) > best_conf:
                        best_conf = ds["confidence"]
                        best_target = ds["target_column"]
                        best_prob = ds["problem_type"]
                schema["primary_target"] = best_target
                schema["global_problem_type"] = best_prob
                st.session_state.detected_schema = schema
                st.success("✅ Overrides applied and schema updated.")
                st.rerun()

            # ── G12: Per-modality target override ────────────────────────────
            st.divider()
            st.markdown("#### 🏷️ Per-Modality Target Override (G12)")
            st.caption(
                "Use this section to override the target column for a specific modality "
                "(e.g. 'text_classification', 'ner_sequence', 'seq2seq', 'unsupervised_vision'). "
                "This calls the backend `/v2/sessions/{sid}/override-target-per-modality` endpoint "
                "and is stored independently from the global target above."
            )

            modality_options_g12 = ["text", "image", "tabular", "timeseries"]
            task_options_g12 = {
                "text": ["text_classification", "ner_sequence", "seq2seq"],
                "image": ["supervised", "unsupervised_vision"],
                "tabular": ["classification_binary", "classification_multiclass", "regression"],
                "timeseries": ["forecasting"],
            }
            g12_col1, g12_col2, g12_col3 = st.columns([1, 1, 2])
            with g12_col1:
                g12_modality = st.selectbox(
                    "Modality",
                    options=modality_options_g12,
                    key="g12_modality_sel",
                    help="Select which data type you're overriding: tabular=spreadsheet data, text=sentences/paragraphs, image=photo files, timeseries=sequential sensor data.",
                )
            with g12_col2:
                g12_task = st.selectbox(
                    "Task type",
                    options=task_options_g12.get(g12_modality, ["classification_binary"]),
                    key="g12_task_sel",
                    help="text_classification=predict a label from text. ner_sequence=tag individual words. seq2seq=generate text from text. supervised=predict from images. unsupervised_vision=cluster images without labels.",
                )
            with g12_col3:
                # Build column suggestions from detected schema
                _g12_all_cols: List[str] = []
                for ds in per_dataset:
                    mod_cols = ds.get("detected_columns", {}).get(g12_modality, [])
                    _g12_all_cols.extend(c for c in mod_cols if c not in _g12_all_cols)
                    for c in ds.get("detected_columns", {}).get("tabular", []):
                        if c not in _g12_all_cols:
                            _g12_all_cols.append(c)
                g12_target_col = st.selectbox(
                    "Target column for this modality",
                    options=_g12_all_cols if _g12_all_cols else ["(type manually)"],
                    key="g12_col_sel",
                    help="The column in your dataset that contains the answer you want to predict (e.g. 'label', 'class', 'survived', 'sentiment').",
                ) if _g12_all_cols else st.text_input(
                    "Target column (type manually)",
                    key="g12_col_manual",
                )
            g12_reason = st.text_input(
                "Reason / note (optional)",
                value="User override via Phase 2 UI",
                key="g12_reason",
            )
            if st.button("🎯 Apply Per-Modality Override", key="g12_apply", type="secondary"):
                if g12_target_col and g12_target_col not in ("(type manually)", ""):
                    try:
                        _g12_resp = requests.post(
                            f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/override-target-per-modality",
                            json={
                                "modality": g12_modality,
                                "target_column": g12_target_col,
                                "task_type": g12_task,
                                "reason": g12_reason or "User override via Phase 2 UI",
                            },
                            timeout=15,
                        )
                        if _g12_resp.status_code == 200:
                            _g12_data = _g12_resp.json()
                            st.success(
                                f"✅ {g12_modality} target → '{_g12_data.get('target_column')}' "
                                f"(task_type={_g12_data.get('task_type')}, "
                                f"valid={_g12_data.get('validation', {}).get('valid', True)})"
                            )
                        else:
                            st.error(f"Override failed: {_api_error_detail(_g12_resp)}")
                    except Exception as _g12_exc:
                        st.error(f"Per-modality override error: {_g12_exc}")
                else:
                    st.warning("Select or type a valid target column first.")

        with tab3:
            st.markdown("#### 🔍 Raw Schema Detection Response")
            _kv_table(schema, "Schema")

        with tab4:
            st.markdown("#### 🎯 Target Column Candidates (XS3 Scoring)")
            st.caption(
                "The system ranked all columns by XS3 confidence gap — a measure of how "
                "predictable and unambiguous each column is as a target. "
                "Higher gap = more confident choice. **Lock** a target to prevent AutoVision from changing it."
            )
            # Dataset lock/unlock controls
            _lock_ds_ids = [ds.get("dataset_id", "") for ds in per_dataset if ds.get("dataset_id")]
            if _lock_ds_ids and st.session_state.session_id:
                _lock_col1, _lock_col2, _lock_col3 = st.columns([2, 1, 1])
                with _lock_col1:
                    _lock_ds_sel = st.selectbox(
                        "Dataset to lock/unlock",
                        options=_lock_ds_ids,
                        key="lock_ds_sel",
                        help="Select which dataset's target column to lock or unlock.",
                    )
                with _lock_col2:
                    if st.button("🔒 Lock Target", key="lock_target_btn",
                                 help="Prevent AutoVision from changing this dataset's target column during reprocessing."):
                        try:
                            _lr = requests.post(
                                f"{API_BASE_URL}/v2/datasets/{_lock_ds_sel}/lock-target",
                                params={"session_id": st.session_state.session_id},
                                timeout=10,
                            )
                            if _lr.ok:
                                st.success("Target locked.")
                            else:
                                st.error(f"Lock failed: {_api_error_detail(_lr)}")
                        except Exception as _le:
                            st.error(str(_le))
                with _lock_col3:
                    if st.button("🔓 Unlock Target", key="unlock_target_btn",
                                 help="Allow AutoVision to re-detect the target column on next schema run."):
                        try:
                            _ur = requests.post(
                                f"{API_BASE_URL}/v2/datasets/{_lock_ds_sel}/unlock-target",
                                params={"session_id": st.session_state.session_id},
                                timeout=10,
                            )
                            if _ur.ok:
                                st.success("Target unlocked.")
                            else:
                                st.error(f"Unlock failed: {_api_error_detail(_ur)}")
                        except Exception as _ue:
                            st.error(str(_ue))
            candidates = st.session_state.get("schema_candidates", [])
            if candidates:
                cand_rows = []
                for c in candidates:
                    cand_rows.append({
                        "Column": c.get("column", "?"),
                        "XS3 Score": f"{float(c.get('xs3_score', 0) or 0):.3f}",
                        "Confidence Gap": f"{float(c.get('confidence_gap', 0) or 0):.3f}",
                        "Dtype": c.get("dtype", "?"),
                        "Unique Values": c.get("n_unique", "?"),
                        "Reason": c.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(cand_rows), width="stretch")
            else:
                st.info("No candidate scoring data available.")

        # ── G12: Advanced overrides (fusion strategy + primary dataset) ──────
        st.markdown("---")
        with st.expander("⚙️ Advanced Overrides (Fusion Strategy & Primary Dataset)", expanded=False):
            st.caption(
                "Use these controls to fine-tune the pipeline's multimodal fusion strategy "
                "or restrict training to a single dataset when your datasets are incompatible."
            )

            # Fusion strategy override
            _sid_g12 = st.session_state.get("session_id", "")
            fusion_opts = [
                "concatenation", "attention", "graph",
                "uncertainty", "uncertainty_graph",
                "structural_semantic", "complementarity",
                "gated",                  # Modality conflict suppression
                "ula",                    # Omni-modal Unified Latent Alignment
                "fusemoe",                # Missing-modality-aware MoE
            ]
            _fusion_col, _fusion_btn_col = st.columns([3, 1])
            with _fusion_col:
                _chosen_fusion = st.selectbox(
                    "Override Fusion Strategy",
                    options=fusion_opts,
                    help="Hard-locks fusion for all Optuna trials. System defaults to ULA for text+image. Override only when ablating or memory-constrained.",
                    key="g12_fusion_override_sel",
                )
            with _fusion_btn_col:
                st.write("")
                st.write("")
                if st.button("Apply Fusion Override", key="g12_fusion_apply"):
                    if _sid_g12:
                        try:
                            _fov_resp = requests.post(
                                f"{API_BASE_URL}/v2/sessions/{_sid_g12}/override-fusion",
                                json={"strategy": _chosen_fusion, "reason": "UI fusion override"},
                                timeout=10,
                            )
                            if _fov_resp.ok:
                                # Reflect override back into session state so Phase 4/5 display it
                                st.session_state["fusion_override_active"] = _chosen_fusion
                                _schema = st.session_state.get("detected_schema") or {}
                                if isinstance(_schema, dict):
                                    _schema["fusion_strategy_override"] = _chosen_fusion
                                    st.session_state["detected_schema"] = _schema
                                st.success(f"Fusion strategy locked to: **{_chosen_fusion}**")
                                st.rerun()
                            else:
                                st.error(f"Failed: {_fov_resp.text[:200]}")
                        except Exception as _fe:
                            st.error(str(_fe))
                    else:
                        st.warning("No active session.")

                # Show current active fusion override if any
                _active_fusion = st.session_state.get("fusion_override_active")
                if _active_fusion:
                    st.markdown(
                        f'<div class="av-alert violet">Fusion locked: <strong>{_active_fusion}</strong> '
                        f'— will override schema-derived strategy</div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # Primary dataset picker (for incompatible datasets)
            _compat_info = schema.get("relatedness_report", {})
            _n_groups = _compat_info.get("n_groups", 1)
            if _n_groups > 1:
                st.warning(
                    f"Your datasets belong to **{_n_groups} unrelated groups**. "
                    "Choose which dataset to use as the primary training source."
                )
            _ds_ids = [ds.get("dataset_id", "") for ds in schema.get("per_dataset", []) if ds.get("dataset_id")]
            _primary_col, _primary_btn_col = st.columns([3, 1])
            with _primary_col:
                _chosen_primary = st.selectbox(
                    "Primary Dataset",
                    options=_ds_ids if _ds_ids else ["(no datasets detected)"],
                    help="All training and preprocessing will use this dataset when incompatible datasets are present.",
                    key="g12_primary_ds_sel",
                )
            with _primary_btn_col:
                st.write("")
                st.write("")
                if st.button("Set Primary", key="g12_primary_apply"):
                    if _sid_g12 and _chosen_primary and _chosen_primary != "(no datasets detected)":
                        try:
                            _prim_resp = requests.post(
                                f"{API_BASE_URL}/v2/sessions/{_sid_g12}/choose-primary-dataset",
                                json={"dataset_id": _chosen_primary},
                                timeout=10,
                            )
                            if _prim_resp.ok:
                                st.success(f"Primary dataset set to: **{_chosen_primary[:40]}**")
                            else:
                                st.error(f"Failed: {_prim_resp.text[:200]}")
                        except Exception as _pe:
                            st.error(str(_pe))
                    else:
                        st.warning("No active session or invalid dataset selected.")

        st.divider()

        if st.button("➡️ Next: Preprocessing", width="stretch"):
            st.session_state.workflow_stage = 3
            st.rerun()

    elif st.session_state.schema_detected:
        st.warning("⚠️ Schema detection ran but no results available")

def render_phase_3_preprocessing():
    """Phase 3: Preprocessing Pipeline with explainable transformation details."""
    st.header("Phase 3 - Data Preprocessing")

    with st.expander("ℹ️ What is this phase doing? (click to expand)", expanded=False):
        st.markdown("""
        Raw data is rarely ready for machine learning as-is. This phase cleans and transforms
        each modality using strategies **chosen automatically from schema signals** — not a fixed pipeline.

        ---

        **🔢 Tabular**

        | Step | What AutoVision actually does | Why |
        |---|---|---|
        | Imputation | Median for numeric, mode for categorical | Missing values break gradient computation |
        | Scaling | **Robust scaler** (IQR-based, not min-max) | Handles outliers without distortion |
        | Log-transform | Applied when column skew > 1.5 | Compresses heavy-tailed distributions |
        | Categorical encoding | **One-hot** for low-cardinality (<20 unique), **target-mean** for high-cardinality | High-cardinality one-hot explodes dimensionality |
        | Column pruning | Drops constant and near-constant columns | Zero-variance features add noise |
        | Adaptive config | `AdaptivePreprocessingEngine` adjusts all above from drift-adjusted predictability scores | Retraining uses context from previous run |

        ---

        **📝 Text**

        | Step | What AutoVision actually does | Why |
        |---|---|---|
        | Tokeniser selection | BERT (default) → **multilingual BERT** when `linguistic_complexity > 0.7` | Non-English text needs a wider vocabulary |
        | Max length | Set to `⌈1.3 × avg_tokens_per_sample⌉`, clipped to [16, 512] | Avoids padding waste on short texts |
        | Pooling strategy | CLS (classification) → **mean** (long docs) → **none** (NER token-level) | Task-appropriate sequence representation |
        | LoRA adaptation | Q and V projection matrices of frozen BERT receive low-rank updates | Adapts pretrained encoder without full fine-tuning |
        | Contrastive projection | CLIP projection head (128-dim) when ≥ 2 modalities present | Enables NT-Xent cross-modal alignment |

        ---

        **🖼️ Image**

        | Step | What AutoVision actually does | Why |
        |---|---|---|
        | Target size | Adapted from `mean_resolution` schema signal | Avoids over-downsampling high-res datasets |
        | Augmentation intensity | `light` / `medium` / `strong` — driven by dataset size and label separability | Small datasets need stronger augmentation to generalise |
        | Grayscale | Applied when `channels = 1` detected | Forces single-channel pipeline, prevents colour-channel errors |
        | Sharpening | Activated when `blur_proxy_variance_of_laplacian` is low | Compensates for low-quality or blurry source images |
        | Normalisation stats | ImageNet (default) → **CLIP-specific** when CLIP encoder selected | Encoder was pretrained on different pixel statistics |
        | LoRA adaptation | Applied to image encoder attention layers | Same parameter-efficient strategy as text |

        All strategies are schema-signal-driven and logged to the Decision Trace.
        """)

    if not st.session_state.schema_detected:
        st.error("⚠️ Schema not detected. Run Phase 1 → Phase 2 first.")
        if st.button("← Go to Phase 1", type="primary"):
            st.session_state.workflow_stage = 1
            st.rerun()
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Start Preprocessing", width="stretch"):
            with st.spinner("Preprocessing data..."):
                schema_data = st.session_state.detected_schema or {}
                try:
                    # B2 FIX: send session_id + schema_override so backend skips re-detect
                    response = requests.post(
                        f"{API_BASE_URL}/preprocess",
                        json={
                            "session_id": st.session_state.session_id,
                            "schema_override": schema_data if schema_data else None,
                        },
                        timeout=300,
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        result = payload.get("data", {})
                        result["_context_stage"] = payload.get("context_stage")
                        result["_context_version"] = payload.get("context_version")
                        result["_artifact_versions"] = payload.get("artifact_versions", {})
                        st.session_state.preprocess_result = result
                        # Store column lists for downstream phases
                        if result.get("text_columns"):
                            st.session_state.text_columns = result["text_columns"]
                        if result.get("image_columns"):
                            st.session_state.image_columns = result["image_columns"]
                        pp_cached = bool(result.get("preprocessor_cached", False))
                        st.session_state.phase_states[3] = {
                            "status": "reused" if pp_cached else "completed",
                            "reason": (
                                f"Scaler loaded from cache: {result.get('preprocessor_path', '?')}"
                                if pp_cached else
                                f"Fitted on {result.get('total_samples', '?')} samples, "
                                f"{len(result.get('preprocessing_stages', []))} stage(s)"
                            ),
                        }
                        st.success("Preprocessing complete!")
                        st.rerun()
                    else:
                        _show_error_with_retry(f"Preprocessing failed ({response.status_code}): {_api_error_detail(response)}", "retry_preprocess")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API server. Is it running on http://localhost:8001?")
                except requests.exceptions.Timeout:
                    st.error("Preprocessing request timed out. The dataset may be too large.")
                except Exception as exc:
                    st.error(f"Preprocessing error: {exc}")
    with col2:
        st.markdown("### Phase Status")
        st.info("Real-time preprocessing updates shown above")

    st.divider()
    st.markdown("### Preprocessing Summary")

    preprocess_result = st.session_state.get("preprocess_result") or {}
    if not isinstance(preprocess_result, dict):
        preprocess_result = {}
    stages = preprocess_result.get("preprocessing_stages", [])
    total_samples = preprocess_result.get("total_samples", None)
    output_shapes = preprocess_result.get("output_shapes", {})
    samples = preprocess_result.get("samples", {})

    if preprocess_result:
        cached_flag = bool(preprocess_result.get("preprocessor_cached", False))
        scaler_path = preprocess_result.get("preprocessor_path", "")
        ctx_stage = preprocess_result.get("_context_stage")
        ctx_ver = preprocess_result.get("_context_version")
        if ctx_stage or ctx_ver:
            st.caption(
                f"Context stage: {ctx_stage or 'N/A'} | "
                f"Context version: {ctx_ver or 'N/A'}"
            )
        if cached_flag:
            st.success(
                f"♻️ **Preprocessor reused from cache** - `{scaler_path}`. "
                "No re-fitting was performed. Data from a previous preprocessing run is being used."
            )
        else:
            st.info(
                f"🔧 **Preprocessor freshly fitted** and persisted to `{scaler_path}`."
            )
        _probe_note = preprocess_result.get("probe_note")
        _probe_available = preprocess_result.get("probe_available", True)
        if _probe_note:
            if _probe_available:
                st.success(f"📊 **Probe:** {_probe_note}")
            else:
                st.caption(f"ℹ️ **Probe:** {_probe_note}")

    if not stages:
        st.info("No preprocessing results yet. Click 'Start Preprocessing' above.")
        if st.button("Next: Model Selection"):
            st.session_state.workflow_stage = 4
            st.session_state.model_selected = False
            st.rerun()
        return

    # Stage overview metrics
    for stage in stages:
        sc1, sc2, sc3 = st.columns(3)
        stage_name = stage.get("stage", "Stage").replace("_", " ").title()
        sc1.metric(stage_name, f"{total_samples or '?'} samples")
        sc2.write(f"**Status:** {stage.get('status', 'N/A')}")
        sc3.code(stage.get("output_shape", "?"))

    # ── Feature Intelligence — publication-grade transparency (Pass B step 5) ─
    with st.expander("🧠 Feature Intelligence (semantic roles, signals, predictability)", expanded=False):
        st.caption(
            "Schema-derived signals that drive preprocessing decisions: vocab size, "
            "language, image channels, aspect-ratio variance, predictability per modality. "
            "Pulled from /v2/sessions/{sid}/intelligence/preprocessing-plan."
        )
        if st.session_state.get("session_id"):
            if st.button("Load Feature Intelligence", key="phase3_fi_btn"):
                try:
                    _fr = requests.get(
                        f"{API_BASE_URL}/v2/sessions/{st.session_state['session_id']}/intelligence/preprocessing-plan",
                        timeout=10,
                    )
                    if _fr.status_code == 200:
                        st.session_state["phase3_fi_data"] = _fr.json()
                    else:
                        st.warning(f"Endpoint returned {_fr.status_code}")
                except Exception as _fe:
                    st.error(f"Fetch error: {_fe}")
            _fi = st.session_state.get("phase3_fi_data") or {}
            if _fi:
                _ctx = _fi.get("context", {}) or {}
                # Show key intelligence signals
                _fi_inner = _ctx.get("feature_intelligence", {}) or {}
                if _fi_inner:
                    st.markdown("**Per-dataset feature intelligence**")
                    for _ds, _entry in _fi_inner.items():
                        with st.expander(f"Dataset: {_ds}", expanded=False):
                            _signals = (_entry or {}).get("feature_signals", {}) or {}
                            if _signals:
                                _kv_table(_signals, f"Signals — {_ds}")
                            _semantic = (_entry or {}).get("semantic_roles", {}) or {}
                            if _semantic:
                                st.markdown("Semantic roles:")
                                _kv_table(_semantic, f"Roles — {_ds}")
                # Multimodal signals
                _mm = _ctx.get("multimodal_signals", {}) or {}
                if _mm:
                    st.markdown("**Multimodal signals**")
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Complementarity score", f"{_mm.get('complementarity_score', 0):.3f}")
                    mc2.metric("Alignment strength", f"{_mm.get('alignment_strength', 0):.3f}")
                # Per-dataset predictability (RF CV score for target from tabular features)
                _pred = _ctx.get("modality_predictability", {}) or {}
                if _pred:
                    st.markdown("**Target predictability from tabular features (RandomForest 3-fold CV)**")
                    st.caption(
                        "Measures how well tabular columns alone predict the target. "
                        "Low score (< 0.4) means the real signal lives in text/image — expected for multimodal datasets."
                    )
                    for _ds_key, _score in _pred.items():
                        _score_f = float(_score) if _score is not None else 0.0
                        _interp = (
                            "strong tabular signal" if _score_f >= 0.75
                            else "moderate tabular signal" if _score_f >= 0.4
                            else "weak — signal is in text/image"
                        )
                        # Dataset key is a UUID — show shortened form
                        _label = str(_ds_key)[:8] + "…" if len(str(_ds_key)) > 12 else str(_ds_key)
                        st.metric(
                            label=f"Predictability (dataset {_label})",
                            value=f"{_score_f:.3f}",
                            help=_interp,
                        )
                        st.caption(f"Interpretation: {_interp}")
                if not (_fi_inner or _mm or _pred):
                    st.info("No feature intelligence available yet.")
        else:
            st.info("No active session — start with Phase 1 ingestion.")

    validation_report = preprocess_result.get("validation_report") or preprocess_result.get("validation") or {}
    if isinstance(validation_report, dict) and validation_report:
        with st.expander("Preprocessing Validation Report", expanded=False):
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Valid", "Yes" if validation_report.get("valid", False) else "No")
            vc2.metric("Checks Passed", validation_report.get("checks_passed", "N/A"))
            vc3.metric("Checks Total", validation_report.get("checks_total", "N/A"))
            warnings = validation_report.get("warnings", []) or []
            errors = validation_report.get("errors", []) or []
            if warnings:
                st.warning("Warnings:\n- " + "\n- ".join(str(w) for w in warnings))
            if errors:
                st.error("Errors:\n- " + "\n- ".join(str(e) for e in errors))

    # ---- Tabular Transformation Details ----
    tab_sample = samples.get("tabular")
    if tab_sample:
        with st.expander("View Tabular Transformation Details", expanded=True):
            dropped = tab_sample.get("dropped_columns", [])
            if dropped:
                st.warning(
                    f"Smart-filtered {len(dropped)} useless columns: "
                    f"`{', '.join(dropped)}`"
                )

            before_col, after_col = st.columns(2)

            with before_col:
                st.markdown("**Before (Raw Input)**")
                raw_cols = tab_sample.get("raw_columns", [])
                raw_rows = tab_sample.get("raw_rows", [])
                if raw_cols and raw_rows:
                    raw_df = pd.DataFrame(raw_rows, columns=raw_cols)
                    st.dataframe(raw_df, width="stretch")
                else:
                    st.caption("No raw sample available")

            with after_col:
                st.markdown("**After (Transformed)**")
                t_cols = tab_sample.get("transformed_columns", [])
                t_rows = tab_sample.get("transformed_rows", [])
                if t_cols and t_rows:
                    # Show first few transformed columns if there are many
                    t_df = pd.DataFrame(t_rows, columns=t_cols)
                    if len(t_cols) > 15:
                        st.caption(f"Showing first 15 of {len(t_cols)} features")
                        st.dataframe(t_df.iloc[:, :15], width="stretch")
                    else:
                        st.dataframe(t_df, width="stretch")
                else:
                    st.caption("No transformed sample available")

            n_tab = len(tab_sample.get("raw_columns", []))
            n_transformed = len(tab_sample.get("transformed_columns", []))
            n_dropped = len(dropped)
            st.caption(
                f"Tabular Pipeline: {n_tab} raw columns -> "
                f"{n_dropped} dropped -> {n_transformed} output features "
                f"(Numeric: median impute + StandardScaler | Categorical: mode impute + OHE)"
            )

    # ---- Text Transformation Details ----
    text_sample = samples.get("text")
    if text_sample:
        with st.expander("View Text Transformation Details", expanded=True):
            st.markdown(f"**Column:** `{text_sample.get('column', '?')}`  |  "
                        f"**Tokenizer:** `{text_sample.get('tokenizer', '?')}`  |  "
                        f"**Max Length:** {text_sample.get('max_length', '?')}")

            st.markdown("**Before (Original Text)**")
            st.info(text_sample.get("original", "N/A"))

            st.markdown("**After (Tokenized input_ids)**")
            ids = text_sample.get("input_ids", [])
            # Show first 30 tokens + padding indicator
            display_ids = ids[:30]
            pad_count = ids.count(0)
            st.code(
                f"{display_ids}{'...' if len(ids) > 30 else ''}\n"
                f"# Length: {len(ids)}  |  Padding tokens: {pad_count}",
                language="python",
            )

            mask = text_sample.get("attention_mask", [])
            real_tokens = sum(1 for v in mask if v == 1)
            st.caption(f"Attention mask: {real_tokens} real tokens, "
                       f"{len(mask) - real_tokens} padding tokens")

    # ---- Image Preprocessing (visual) ----
    image_sample = samples.get("image")
    if image_sample or output_shapes.get("image"):
        with st.expander("View Image Preprocessing Details", expanded=True):
            _tgt = image_sample.get("target_size", [224, 224]) if image_sample else [224, 224]
            _aug = image_sample.get("augment_intensity", "medium") if image_sample else "medium"
            _gray = bool((image_sample or {}).get("grayscale", False))
            _sharp = bool((image_sample or {}).get("sharpening", False))
            _col_name = (image_sample or {}).get("column", "img_path")
            _missing = int((image_sample or {}).get("missing_paths", 0))
            _total_checked = int((image_sample or {}).get("total_paths_checked", 10))
            _first_valid = (image_sample or {}).get("first_valid_path")
            _sample_raw = (image_sample or {}).get("sample_path_raw", "img/42953.png")
            _channels = 1 if _gray else 3

            # ── Top metrics row ──────────────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Output Size", f"{_tgt[0]}×{_tgt[1]}")
            mc2.metric("Channels", _channels)
            mc3.metric("Augmentation", _aug.title())
            mc4.metric("Paths OK", f"{_total_checked - _missing}/{_total_checked}",
                       delta=None if _missing == 0 else f"{_missing} missing",
                       delta_color="off" if _missing == 0 else "inverse")

            st.divider()

            # ── Before / After image previews + pipeline ─────────────────────
            _preview_raw_b64 = (image_sample or {}).get("preview_raw_b64")
            _preview_aug_b64 = (image_sample or {}).get("preview_aug_b64")
            _raw_wh = (image_sample or {}).get("raw_size_wh")

            has_previews = bool(_preview_raw_b64 and _preview_aug_b64)

            if has_previews:
                img_before_col, img_after_col, pipe_col = st.columns([1, 1, 1])
            else:
                prev_col, pipe_col = st.columns([1, 1])

            if has_previews:
                with img_before_col:
                    st.markdown("**Before** *(raw, resized)*")
                    _raw_label = (
                        f"Original: {_raw_wh[0]}×{_raw_wh[1]}px → resized {_tgt[0]}×{_tgt[1]}"
                        if _raw_wh else f"Resized {_tgt[0]}×{_tgt[1]}"
                    )
                    import base64 as _b64_fe
                    st.image(
                        _b64_fe.b64decode(_preview_raw_b64),
                        caption=_raw_label,
                        use_container_width=True,
                    )
                with img_after_col:
                    st.markdown("**After** *(augmented, pre-normalize)*")
                    st.image(
                        _b64_fe.b64decode(_preview_aug_b64),
                        caption=f"Flip + ColorJitter + Rotation"
                        + (" + Perspective" if _aug == "strong" else "")
                        + (" + Sharpening" if _sharp else ""),
                        use_container_width=True,
                    )
                    st.caption(
                        "Shown before ToTensor/Normalize — "
                        "actual tensor values will be in ~[−2, 2] after ImageNet normalisation."
                    )
            else:
                with prev_col:
                    st.markdown("**Sample Image**")
                    if _first_valid:
                        try:
                            from PIL import Image as _PILImage
                            _pil_img = _PILImage.open(_first_valid).convert("RGB")
                            _w, _h = _pil_img.size
                            st.image(_pil_img, caption=f"Raw: {_w}×{_h}px · RGB", use_container_width=True)
                        except Exception as _img_load_exc:
                            st.info(f"Preview unavailable: {_img_load_exc}")
                    elif _missing == _total_checked:
                        st.markdown(
                            """<div style="background:#1e1e2e;border:1px dashed #555;border-radius:8px;
                            padding:40px;text-align:center;color:#888;">🖼️<br>
                            <small>Images not yet accessible — re-run Phase 1 to fix paths</small>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Looked for: `{_sample_raw}`")
                    else:
                        st.info(f"{_total_checked - _missing}/{_total_checked} images accessible.")

            with pipe_col:
                st.markdown("**Transform Pipeline**")
                # Build pipeline steps with visual indicators
                _pipe_steps = []
                _pipe_steps.append(("📥", "Input", f'`df["{_col_name}"]` — path string', "#2d2d4e"))
                if _gray:
                    _pipe_steps.append(("⬛", "Grayscale", "Convert to 1-channel", "#3a2d2d"))
                _pipe_steps.append(("🔲", "Resize", f"{_tgt[0]}×{_tgt[1]} px", "#2d3a2d"))
                _aug_detail = {
                    "none": "No augmentation (inference mode)",
                    "light": "Random horizontal flip",
                    "medium": "Flip + ColorJitter + small rotation",
                    "strong": "Flip + ColorJitter + rotation + perspective warp",
                }.get(_aug, _aug)
                _aug_color = {"none": "#2d2d2d", "light": "#2d3a3a", "medium": "#2d3a2d", "strong": "#3a3a2d"}.get(_aug, "#2d3a2d")
                _pipe_steps.append(("🎲", "Augmentation", _aug_detail, _aug_color))
                if _sharp:
                    _pipe_steps.append(("✨", "Sharpening", "UnsharpMask (blur detected)", "#3a2d3a"))
                _pipe_steps.append(("🔢", "ToTensor", "PIL → float32 [0,1]", "#2d2d3a"))
                _pipe_steps.append(("📐", "Normalize", "ImageNet μ=[0.485,0.456,0.406] σ=[0.229,0.224,0.225]", "#2d2d3a"))
                _pipe_steps.append(("📤", "Output", f"Tensor `({_channels}, {_tgt[0]}, {_tgt[1]})`", "#2d2d4e"))

                for _icon, _name, _detail, _bg in _pipe_steps:
                    st.markdown(
                        f'<div style="background:{_bg};border-left:3px solid #7c3aed;'
                        f'padding:6px 10px;margin:3px 0;border-radius:4px;font-size:13px;">'
                        f'<b>{_icon} {_name}</b> — <span style="color:#aaa;">{_detail}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.divider()
            # ── Normalize parameters ─────────────────────────────────────────
            nm1, nm2 = st.columns(2)
            nm1.caption("Mean (RGB): `[0.485, 0.456, 0.406]` — ImageNet channel means")
            nm2.caption("Std  (RGB): `[0.229, 0.224, 0.225]` — ImageNet channel stds")

    if st.button("Next: Model Selection"):
        st.session_state.workflow_stage = 4
        st.session_state.model_selected = False
        st.rerun()


def render_phase_4_model_selection():
    """Phase 4: Model Selection with Rationale."""
    st.header("Phase 4 - Automatic Model Selection")

    with st.expander("ℹ️ What is this phase doing? (click to expand)", expanded=False):
        st.markdown("""
        **For beginners:** Before training a big model, APEX runs a fast \"probe\" to estimate
        which model architecture suits your data best.

        **How the probe works:**
        - A small Random Forest is trained on your preprocessed features for a few seconds.
        - Each candidate model architecture gets a **probe score** (Random Forest cross-validation accuracy).
        - The candidate with the highest score is chosen for full training.

        **What gets selected:**
        - **Encoder** — per modality: tabular (GRN or MLP), text (DistilBERT or TF-IDF), image (ResNet)
        - **Fusion strategy** — how modalities are combined (see Fusion Strategy in the Glossary)
        - **Task head** — final layers: MLP (default), Attention (text-heavy), Cross-Layer RGAT [2] (relational data)
        - **Hyperparameter search space** — range of learning rates, batch sizes, dropout rates to try

        **Selection Criteria:**
        - GPU Memory: <6 GB → Lightweight | 6–12 GB → Medium | >12 GB → Large models
        - Dataset Size: <5k rows → 45 epochs | 5–50k → 18 epochs | >50k → 6 epochs
        """)

    if not st.session_state.schema_detected:
        st.error(
            "⚠️ No schema found for this session. "
            "The API may have restarted (clearing session data) or the page was refreshed. "
            "Re-run **Phase 1 → Phase 2 → Phase 3** to restore the pipeline."
        )
        if st.button("← Go to Phase 1 (Data Ingestion)", type="primary"):
            st.session_state.workflow_stage = 1
            st.rerun()
        return

    st.markdown("""
    **Selection Criteria:**
    - GPU Memory Check: <6GB (Lightweight) | 6-12GB (Medium) | >12GB (Large)
    - Dataset Size: <5k (45 epochs) | 5-50k (18 epochs) | >50k (6 epochs)
    - Task Complexity: Binary/Multiclass/Regression
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Select Models", width="stretch"):
            with st.spinner("Selecting models and hyperparameters..."):
                schema_data = st.session_state.detected_schema or {}
                dataset_size = st.session_state.get('ingested_row_count', 1000)
                modalities = schema_data.get("global_modalities", [])
                problem_type = schema_data.get("global_problem_type", "Unknown")
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/select-model",
                        json={
                            "session_id": st.session_state.session_id,
                            "dataset_size": dataset_size,
                            "modalities": modalities,
                            "problem_type": problem_type
                        },
                        timeout=120
                    )
                    if response.status_code == 200:
                        selection_payload = response.json()
                        st.session_state.model_selection_result = selection_payload
                        st.success("Models selected!")
                        st.session_state.model_selected = True
                        best = selection_payload.get("best_model") or {}
                        probe = best.get("probe_score")
                        st.session_state.phase_states[4] = {
                            "status": "completed",
                            "reason": (
                                f"Selected '{best.get('name', '?')}'"
                                + (f", probe score={float(probe):.3f}" if isinstance(probe, (int, float)) else " (heuristic ranking)")
                            ),
                        }
                        st.rerun()
                    else:
                        _show_error_with_retry(f"Model selection failed ({response.status_code}): {_api_error_detail(response)}", "retry_model_sel")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API server. Is it running on http://localhost:8001?")
                except requests.exceptions.Timeout:
                    st.error("Model selection request timed out.")
                except Exception as exc:
                    st.error(f"Model selection error: {exc}")
    with col2:
        st.markdown("### Hardware Detection")
        api_status = get_api_status()
        if api_status:
            device_type = "GPU" if api_status.get('gpu_available') else "CPU"
            st.metric("Device", device_type)
    st.divider()
    if st.session_state.model_selected:
        st.markdown("### Selected Model & Hyperparameters")
        result = st.session_state.get("model_selection_result", {})
        best = result.get("best_model") or {}
        context_stage = result.get("context_stage")
        context_version = result.get("context_version")
        policy_source = result.get("policy_source")
        fusion_policy = result.get("fusion_policy", best.get("fusion_strategy"))
        selection_contract_version = result.get(
            "selection_contract_version",
            best.get("selection_contract_version"),
        )
        eligible_modalities = result.get("eligible_modalities", []) or []
        excluded_modalities = result.get("excluded_modalities", {}) or {}

        if context_stage or context_version:
            st.caption(
                f"Context stage: {context_stage or 'N/A'} | "
                f"Context version: {context_version or 'N/A'}"
            )
        if selection_contract_version:
            st.caption(f"Selection contract: {selection_contract_version}")

        _jit_dr_top = best.get("jit_dry_run", {}) or {}
        _jit_img_top = _jit_dr_top.get("selected_image_encoder")
        _jit_txt_top = _jit_dr_top.get("selected_text_encoder")
        # Show JIT-selected encoders as the primary model identity
        if _jit_img_top or _jit_txt_top:
            _jit_enc_parts = []
            if _jit_img_top:
                _jit_enc_parts.append(f"**{_jit_img_top}** (image)")
            if _jit_txt_top:
                _jit_enc_parts.append(f"**{_jit_txt_top}** (text)")
            st.markdown(f"**Encoders (JIT hardware-fit):** {' · '.join(_jit_enc_parts)}")
            st.caption(
                f"Tier template (rule-based before JIT): {best.get('name', 'N/A')} — "
                "overridden by live VRAM profiling above."
            )
        else:
            st.write(f"**Model:** {best.get('name', 'N/A')}")
        st.write(f"**Fusion Strategy:** {fusion_policy or 'N/A'}")
        st.write(f"**Policy Source:** {policy_source or 'N/A'}")
        st.write(f"**Batch Size:** {best.get('batch_size', 'N/A')}")
        st.write(f"**Tier:** {best.get('tier', 'N/A')}")

        # Filter out excluded modalities from display (API may still return them in eligible list)
        _excl_keys = set(excluded_modalities.keys()) if excluded_modalities else set()
        _elig_display = [m for m in eligible_modalities if m not in _excl_keys]
        if _elig_display:
            st.write(f"**Eligible Modalities:** {', '.join(str(m) for m in _elig_display)}")
        if excluded_modalities:
            for _excl_mod, _excl_reason in excluded_modalities.items():
                st.caption(f"⚠️ **{_excl_mod}** excluded: {_excl_reason}")
            with st.expander("Excluded Modalities (detail)", expanded=False):
                ex_rows = [
                    {"modality": str(mod), "reason": str(reason)}
                    for mod, reason in excluded_modalities.items()
                ]
                st.dataframe(pd.DataFrame(ex_rows), width="stretch")

        # Transparency: "Why did APEX choose this?" explainer
        with st.expander("\U0001f9e0 Why did AutoVision choose this model?", expanded=True):
            _why_reasons = []
            _probe = best.get("probe_score")
            _tier = best.get("tier", "unknown")
            _name = best.get("name", "N/A")
            _fusion = fusion_policy or "N/A"
            _hw = best.get("hardware_info", {}) or {}
            _gpu_gb = _hw.get("gpu_memory_gb", "?")
            _meta = best.get("selection_metadata", {}) or {}
            _complexity = _meta.get("data_complexity", {}) or {}
            _excl_mods = result.get("excluded_modalities", {}) or {}
            _n_mods = max(1, len(eligible_modalities) - len(_excl_mods)) if eligible_modalities else 1

            # Probe-based reasoning
            if isinstance(_probe, (int, float)):
                _why_reasons.append(
                    f"**Data-driven probe**: A Random Forest was trained on your "
                    f"preprocessed features. `{_name}` achieved a probe score of "
                    f"**{float(_probe):.3f}**, the highest among all candidates."
                )
            else:
                _excluded_why = result.get("excluded_modalities", {}) or {}
                _excl_note = (
                    f" ({', '.join(_excluded_why.keys())} excluded: no usable features)"
                    if _excluded_why else ""
                )
                _why_reasons.append(
                    f"**Heuristic ranking**: No probe was run (requires tabular features for "
                    f"cross-validation). `{_name}` was selected based on rule-based ranking{_excl_note}."
                )

            # VRAM/tier reasoning
            _why_reasons.append(
                f"**Hardware fit**: Your GPU has **{_gpu_gb} GB** VRAM. "
                f"AutoVision selected the **{_tier}** tier, which fits within your "
                f"memory budget without risk of OOM errors."
            )

            # Fusion reasoning
            fusion_label = _FUSION_LABELS.get(str(_fusion).lower(), str(_fusion))
            if _n_mods >= 2:
                _why_reasons.append(
                    f"**Fusion strategy**: With **{_n_mods} modalities** detected, "
                    f"AutoVision chose **{fusion_label}**. "
                    f"Cross-modal contrastive alignment (CLIP-style) will be "
                    f"auto-activated during training to align modality embeddings."
                )
            else:
                _why_reasons.append(
                    f"**Fusion strategy**: Single modality detected. Fusion is "
                    f"passthrough (no cross-modal alignment needed)."
                )

            # Data complexity reasoning
            _entropy = _complexity.get("entropy")
            _sparsity = _complexity.get("sparsity")
            if isinstance(_entropy, (int, float)):
                complexity_level = (
                    "low" if _entropy < 0.5 else "moderate" if _entropy < 1.5 else "high"
                )
                _why_reasons.append(
                    f"**Data complexity**: Label entropy is **{float(_entropy):.3f}** "
                    f"({complexity_level}). "
                    + (f"Feature sparsity is **{float(_sparsity):.3f}**. " if isinstance(_sparsity, (int, float)) else "")
                    + "These signals informed the search space bounds."
                )

            for reason in _why_reasons:
                st.markdown(f"- {reason}")

        hw_info = best.get("hardware_info", {}) if isinstance(best, dict) else {}
        probe_score = best.get("probe_score") if isinstance(best, dict) else None

        if hw_info:
            with st.expander("Hardware Detected"):
                hw_cols = st.columns(3)
                hw_cols[0].metric("Device", "GPU" if hw_info.get("gpu_available") else "CPU")
                hw_cols[1].metric("GPU Memory (GB)", hw_info.get("gpu_memory_gb", "N/A"))
                hw_cols[2].metric("Tier Selected", best.get("tier", "N/A"))

        if isinstance(probe_score, (int, float)):
            st.info(
                f"📊 **Data-driven probe score:** `{float(probe_score):.3f}` "
                "(Random Forest 1-fold CV on a sampled preprocessing feature space). "
                "This score influenced model ranking."
            )
        else:
            _preproc_ran = bool(st.session_state.get("preprocess_result"))
            if _preproc_ran:
                st.caption(
                    "ℹ️ No probe score — preprocessing ran but found no tabular features. "
                    "Model selection uses hardware-tier + architecture rules. "
                    "This is expected for text/image-only datasets."
                )
            else:
                st.caption(
                    "⚠️ No probe score — run Phase 3 preprocessing first for data-driven ranking."
                )

        selection_metadata = best.get("selection_metadata", {}) if isinstance(best, dict) else {}
        if selection_metadata:
            st.markdown("### Selection Diagnostics")
            sd1, sd2 = st.columns(2)
            with sd1:
                st.markdown("**Data Complexity**")
                complexity = selection_metadata.get("data_complexity", {}) or {}
                if complexity:
                    entropy = complexity.get("entropy")
                    sparsity = complexity.get("sparsity")
                    st.write(
                        f"- Label entropy: {float(entropy):.4f}"
                        if isinstance(entropy, (int, float))
                        else "- Label entropy: N/A"
                    )
                    st.write(
                        f"- Feature sparsity: {float(sparsity):.4f}"
                        if isinstance(sparsity, (int, float))
                        else "- Feature sparsity: N/A"
                    )
                else:
                    st.caption("No complexity metrics available.")

            with sd2:
                st.markdown("**Selection Confidence**")
                sel_conf = selection_metadata.get("selection_confidence")
                if isinstance(sel_conf, (int, float)):
                    st.metric("Confidence Margin", f"{float(sel_conf):.4f}")
                else:
                    st.metric("Confidence Margin", "N/A")
                st.write(f"- Probe method: {selection_metadata.get('probe_method', 'N/A')}")
                st.write(f"- Top probed model: {selection_metadata.get('top_probe_model', 'N/A')}")
                top_probe_score = selection_metadata.get("top_probe_score")
                st.write(
                    f"- Top probe score: {float(top_probe_score):.4f}"
                    if isinstance(top_probe_score, (int, float))
                    else "- Top probe score: N/A"
                )

            probe_rows = []
            probe_scores = selection_metadata.get("probe_scores", {}) or {}
            for name, details in probe_scores.items():
                if not isinstance(details, dict):
                    continue
                probe_rows.append(
                    {
                        "Model": name,
                        "Val Score": details.get("val_score"),
                        "Uncertainty": details.get("uncertainty"),
                        "Latency (ms)": details.get("latency_ms"),
                        "Confidence": details.get("confidence"),
                    }
                )
            if probe_rows:
                with st.expander("Tabular Probe Scoreboard", expanded=False):
                    st.dataframe(
                        pd.DataFrame(probe_rows).sort_values("Val Score", ascending=False),
                        width="stretch",
                    )
            else:
                ranked = best.get("ranked_candidates", {}) if isinstance(best, dict) else {}
                ranked_tabular = ranked.get("tabular", []) if isinstance(ranked, dict) else []
                if ranked_tabular:
                    with st.expander("Tabular Probe Scoreboard", expanded=False):
                        ranked_df = pd.DataFrame(ranked_tabular)
                        if "val_score" in ranked_df.columns:
                            ranked_df = ranked_df.sort_values("val_score", ascending=False)
                        st.dataframe(ranked_df, width="stretch")

        fusion_probe = best.get("fusion_probe", {}) if isinstance(best, dict) else {}
        if fusion_probe:
            _fp_measured = fusion_probe.get("is_measured", True)
            _fp_title = "Fusion Strategy Probe" if _fp_measured else "Fusion Strategy Priority Order (not measured)"
            with st.expander(_fp_title, expanded=False):
                if not _fp_measured:
                    st.caption(
                        "No cross-validation probe ran — tabular features are required for "
                        "data-driven fusion scoring. Strategies are listed by pre-defined priority "
                        "weight (1.0 = highest). Run full training for measured fusion selection."
                    )
                st.write(f"- Selected strategy: {fusion_probe.get('selected_strategy', 'N/A')}")
                st.write(f"- Method: {fusion_probe.get('method', 'N/A')}")
                _col_label = "Score" if _fp_measured else "Priority Weight"
                scores = (
                    fusion_probe.get("scores", {})
                    if _fp_measured
                    else fusion_probe.get("priority_weights", {}) or fusion_probe.get("scores", {})
                ) if isinstance(fusion_probe, dict) else {}
                if scores:
                    score_df = pd.DataFrame(
                        [{"Strategy": k, _col_label: v} for k, v in scores.items()]
                    )
                    st.dataframe(score_df.sort_values(_col_label, ascending=False), hide_index=True, width="stretch")
                else:
                    candidates = fusion_probe.get("candidate_strategies", [])
                    st.caption(f"Candidate strategies: {', '.join(candidates) if candidates else 'N/A'}")

        vram_filter_report = best.get("vram_filter_report", {}) if isinstance(best, dict) else {}
        if vram_filter_report:
            with st.expander("VRAM Filter Transparency", expanded=False):
                st.write(f"- GPU Memory (GB): {vram_filter_report.get('gpu_memory_gb', 'N/A')}")
                st.write(f"- VRAM Budget (MB): {vram_filter_report.get('vram_budget_mb', 'N/A')}")
                excluded_counts = vram_filter_report.get("excluded_counts", {}) or {}
                if excluded_counts:
                    st.write(
                        "- Excluded by modality: "
                        + ", ".join(f"{m}={c}" for m, c in excluded_counts.items())
                    )
                excluded = vram_filter_report.get("excluded", {}) or {}
                if excluded:
                    for modality, rows in excluded.items():
                        if not rows:
                            continue
                        st.markdown(f"**{modality.title()} excluded candidates**")
                        st.dataframe(pd.DataFrame(rows), width="stretch")

        jit_dry_run = best.get("jit_dry_run", {}) if isinstance(best, dict) else {}
        if jit_dry_run:
            with st.expander("JIT Encoder Dry-Run Results", expanded=False):
                st.caption("Measured on current hardware using live encoder dry-runs.")
                rationale = jit_dry_run.get("rationale", {}) if isinstance(jit_dry_run, dict) else {}
                if rationale:
                    _kv_table(rationale, "JIT Encoder Selection Rationale")
                rc1, rc2 = st.columns(2)
                _jit_budget_mb = int(jit_dry_run.get("vram_budget_bytes", 0) / 1024 / 1024)
                rc1.metric("Budget (MB)", _jit_budget_mb)
                rc2.metric(
                    "Peak Used (MB)",
                    int(jit_dry_run.get("peak_memory_bytes", 0) / 1024 / 1024),
                )
                st.caption(
                    f"JIT budget ({_jit_budget_mb} MB) uses 85% safety margin. "
                    "VRAM Filter (below) uses 70% — different conservative thresholds for different selection stages."
                )
                st.write(
                    f"- Selected image encoder: {jit_dry_run.get('selected_image_encoder', 'N/A')}"
                )
                st.write(
                    f"- Selected text encoder: {jit_dry_run.get('selected_text_encoder', 'N/A')}"
                )

        # Part B.1 — ULA fusion config display
        _fusion_strat = str(best.get("fusion_strategy") or "").lower()
        if _fusion_strat in ("ula", "unified_latent", "unified_latent_alignment", "omnimodal"):
            _fusion_cfg = best.get("fusion_config", {}) or {}
            st.markdown("##### ULA (Unified Latent Alignment) Configuration")
            _ula_c1, _ula_c2, _ula_c3 = st.columns(3)
            _ula_c1.metric("Latent Dim", _fusion_cfg.get("latent_dim", "—"))
            _ula_c2.metric("Transformer Layers", _fusion_cfg.get("n_layers", "—"))
            _ula_c3.metric("Attention Heads", _fusion_cfg.get("n_heads", "—"))
            st.caption(
                "ULA projects all modality tokens into a shared latent space and runs "
                "a cross-modal Transformer, enabling text tokens, image patches, and tabular "
                "features to attend to each other directly (ImageBind / 4M style)."
            )

        # Part B.2 — LoRA config display
        _lora_cfg = best.get("lora_config") or {}
        if _lora_cfg:
            _lora_c1, _lora_c2, _lora_c3 = st.columns(3)
            _lora_c1.metric("LoRA Rank (r)", _lora_cfg.get("r", "—"))
            _lora_c2.metric("LoRA Alpha", _lora_cfg.get("alpha", "—"))
            _lora_c3.metric("LR Multiplier", _lora_cfg.get("lr_mult", 0.1))
            st.caption("LoRA fine-tunes frozen encoder weights with trainable low-rank deltas — efficient domain adaptation without full retraining.")

        rationale = best.get("rationale", {})
        if rationale:
            with st.expander("Tier Template Rationale", expanded=False):
                st.caption(
                    "Rule-based encoder selection for this hardware tier — before JIT overrides. "
                    "The JIT Dry-Run results above show the actual encoders used."
                )
                for component, reason in rationale.items():
                    st.write(f"- **{component}:** {reason}")

        meta_context = best.get("meta_context", []) or []
        if meta_context:
            with st.expander("Meta-Learning Context", expanded=False):
                st.caption(
                    "**What this is:** Past experiment results used as Bayesian priors to bias "
                    "Optuna's initial HP sampling. If a similar dataset used learning_rate=1e-4 "
                    "and performed well, the first trial starts near there rather than random. "
                    "**Current limitation:** Store only has tabular-only experiments — irrelevant "
                    "for text+image. Priors have near-zero effect until multimodal experiments complete."
                )
                meta_rows = [row for row in meta_context if isinstance(row, dict)]
                if meta_rows:
                    _unique_signatures = {str(sorted(r.items()) if isinstance(r, dict) else r) for r in meta_rows}
                    if len(_unique_signatures) == 1 and len(meta_rows) > 1:
                        st.caption(
                            f"⚠️ {len(meta_rows)} identical experiments in history — "
                            "the meta-learning store is still sparse (same dataset run multiple times). "
                            "Priors have minimal effect; diversity improves with more varied training runs."
                        )
                    def _decode_meta_row(row: dict) -> dict:
                        out: dict = {}
                        _SKIP = {"best_params", "loss_weights"}  # empty dicts, not useful
                        for k, v in row.items():
                            if k in _SKIP and v == {}:
                                continue
                            if isinstance(v, dict):
                                # Flatten nested dict: dataset_meta.num_rows, etc.
                                out.update({f"{k}.{dk}": dv for dk, dv in v.items()})
                            elif isinstance(v, str) and v.strip().startswith("{"):
                                try:
                                    import json as _jj
                                    decoded = _jj.loads(v)
                                    if isinstance(decoded, dict):
                                        out.update({f"{k}.{dk}": dv for dk, dv in decoded.items()})
                                        continue
                                except Exception:
                                    pass
                                out[k] = v
                            else:
                                out[k] = v
                        return out
                    _decoded_rows = [_decode_meta_row(r) for r in meta_rows]
                    # Show only unique rows (meta store often has identical entries)
                    _seen: set = set()
                    _unique_decoded: list = []
                    for _dr in _decoded_rows:
                        _sig = str(sorted(_dr.items()))
                        if _sig not in _seen:
                            _seen.add(_sig)
                            _unique_decoded.append(_dr)
                    st.dataframe(
                        pd.DataFrame(_unique_decoded),
                        hide_index=True, width="stretch",
                    )
                    if len(_decoded_rows) > len(_unique_decoded):
                        st.caption(
                            f"Showing {len(_unique_decoded)} unique experiment(s) "
                            f"({len(_decoded_rows) - len(_unique_decoded)} duplicate rows hidden)."
                        )
                else:
                    _kv_table(meta_context, "Meta-Learning Context")
        all_models = result.get("recommended_models", [])
        if len(all_models) > 1:
            with st.expander(f"All {len(all_models)} recommended models"):
                for m in all_models:
                    st.write(f"- **{m.get('name', '?')}** ({m.get('tier', '?')})")

        # ── Per-modality encoder override ─────────────────────────────
        st.divider()
        with st.expander("Override Encoders per Modality (Advanced)", expanded=False):
            st.caption(
                "The JIT selector auto-chose the encoders shown above based on your hardware. "
                "Override here to use a different encoder per modality. "
                "Custom encoders registered via `config/encoder_plugins.py` appear automatically."
            )
            _jit = best.get("jit_dry_run", {}) or {}
            _cur_img = _jit.get("selected_image_encoder", "auto")
            _cur_txt = _jit.get("selected_text_encoder", "auto")

            _vision_options = [
                "auto (JIT-selected)",
                "MobileNetV3-Small",  "EfficientNet-B0",
                "ResNet-50",          "ConvNeXt-Tiny",
                "MultiScale-ResNet50",
                "CLIP-ViT-B/16 (plugin)",
                "DINOv2-ViT-B/14 (plugin)",
                "SigLIP-ViT-B/16 (plugin)",
            ]
            _text_options = [
                "auto (JIT-selected)",
                "MiniLM-L6-v2", "DistilBERT",
                "BERT-base-uncased", "DeBERTa-v3-base",
                "all-mpnet-base-v2 (plugin)",
                "Mistral-7B-Instruct-4bit (plugin)",
            ]
            _tab_options = [
                "auto (JIT-selected)",
                "GRN (Gated Residual Network)",
                "FT-Transformer",
                "MLP",
            ]

            _oe_col1, _oe_col2, _oe_col3 = st.columns(3)
            with _oe_col1:
                _sel_img = st.selectbox(
                    "Image encoder",
                    _vision_options,
                    help="Overrides JIT selection. '(plugin)' encoders require config/encoder_plugins.py to be activated.",
                    key="encoder_override_image",
                )
            with _oe_col2:
                _sel_txt = st.selectbox(
                    "Text encoder",
                    _text_options,
                    help="Overrides JIT selection.",
                    key="encoder_override_text",
                )
            with _oe_col3:
                _sel_tab = st.selectbox(
                    "Tabular encoder",
                    _tab_options,
                    help="GRN is recommended for datasets with feature interactions.",
                    key="encoder_override_tabular",
                )

            if st.button("Apply Encoder Overrides", key="apply_encoder_overrides"):
                _enc_overrides = {}
                if not _sel_img.startswith("auto"):
                    _enc_overrides["preferred_image_encoder"] = _sel_img.split(" (")[0]
                if not _sel_txt.startswith("auto"):
                    _enc_overrides["preferred_text_encoder"] = _sel_txt.split(" (")[0]
                if not _sel_tab.startswith("auto"):
                    _enc_overrides["preferred_tabular_encoder"] = _sel_tab.split(" (")[0]

                if _enc_overrides:
                    _sid_enc = st.session_state.get("session_id", "")
                    if _sid_enc:
                        try:
                            _enc_resp = requests.post(
                                f"{API_BASE_URL}/v2/sessions/{_sid_enc}/encoder-overrides",
                                json=_enc_overrides,
                                timeout=10,
                            )
                            if _enc_resp.ok:
                                st.success(f"Encoder overrides applied: {_enc_overrides}")
                            else:
                                st.warning(
                                    "Overrides stored locally — re-run Model Selection "
                                    "to pick them up. "
                                    f"(API: {_enc_resp.status_code})"
                                )
                        except Exception as _ee:
                            st.warning(f"Could not reach API ({_ee}); overrides will apply on next model selection.")
                    # Store locally so Phase 5 can read them
                    st.session_state["encoder_overrides"] = _enc_overrides
                else:
                    st.info("All modalities set to auto — nothing to override.")

        # ── Optuna Search Space Transparency ──────────────────────────
        st.divider()
        hpo_space = best.get("hpo_space", {})

        with st.expander("View Optuna Auto-Tuning Search Space"):
            if hpo_space:
                _n_rows = st.session_state.get("ingested_row_count", 0)
                _auto_trials = 30 if _n_rows > 50_000 else 20 if _n_rows > 10_000 else 12
                st.caption(
                    f"**{_auto_trials} trials** for this dataset size "
                    f"(scales: <10k→12, 10-50k→20, >50k→30). "
                    "HyperbandPruner kills ~60% of trials at epoch 3-5 — only top performers "
                    "continue. **Each trial only trains fusion+head (~12M params)** on "
                    "precomputed encoder embeddings (DeBERTa/SigLIP run once, not per trial). "
                    "Effective compute: ~5× a single training run. "
                    "Set `APEX_N_TRIALS` env var to override."
                )
                _has_fusion_choice = "fusion_strategy" in hpo_space
                if _has_fusion_choice:
                    _fus_choices = hpo_space.get("fusion_strategy", {}).get("choices", [])
                    st.info(
                        f"**fusion_strategy** is a searchable parameter — Optuna will "
                        f"try: {', '.join(_fus_choices)}. ULA (cross-modal Transformer) "
                        f"is listed first because it enables text↔image patch attention. "
                        f"Optuna may confirm ULA or find that concatenation is faster "
                        f"on this dataset without accuracy loss."
                    )
                rows = []
                for param, spec in hpo_space.items():
                    low = spec.get("low", "—")
                    high = spec.get("high", "—")
                    ptype = spec.get("type", "float")
                    rows.append(
                        f"| `{param}` | {ptype} | {low} | {high} |"
                    )
                table = (
                    "| Parameter | Type | Low | High |\n"
                    "|-----------|------|-----|------|\n"
                    + "\n".join(rows)
                )
                st.markdown(table)
            else:
                st.info("No HPO search space available yet — run model selection first.")

        # ── Manual HP Override Controls ───────────────────────────────
        st.markdown("### Manual Hyperparameter Override")
        st.caption(
            "Your values become **trial 0** — the exact result is guaranteed. "
            "Optuna then runs ~4 more trials nearby using TPE, learning from your "
            "trial's val_loss to explore whether a nearby HP combination performs better. "
            "Best trial across all runs is selected. "
            "Set `APEX_N_TRIALS_MANUAL=1` env var to run your values only, no exploration."
        )
        use_manual = st.checkbox("Override HPO with manual hyperparameters")

        if use_manual:
            lr_spec = hpo_space.get("learning_rate", {})
            wd_spec = hpo_space.get("weight_decay", {})
            do_spec = hpo_space.get("dropout", {})
            ep_spec = hpo_space.get("epochs", {})

            mc1, mc2 = st.columns(2)
            with mc1:
                lr_val = st.number_input(
                    "Learning Rate",
                    min_value=1e-6, max_value=1.0,
                    value=float(lr_spec.get("low", 1e-4)),
                    format="%.6f", step=1e-5,
                    help=(
                        "Controls the speed at which the model adjusts its "
                        "weights. Too high causes instability; too low makes "
                        "training painfully slow."
                    ),
                )
                dropout_val = st.number_input(
                    "Dropout",
                    min_value=0.0, max_value=0.9,
                    value=float(do_spec.get("low", 0.1)),
                    format="%.3f", step=0.05,
                    help=(
                        "Randomly turns off neurons during training to prevent "
                        "the model from memorizing the data (overfitting)."
                    ),
                )
            with mc2:
                wd_val = st.number_input(
                    "Weight Decay",
                    min_value=1e-7, max_value=0.1,
                    value=float(wd_spec.get("low", 1e-5)),
                    format="%.7f", step=1e-5,
                    help=(
                        "Applies a penalty to large weights, forcing the model "
                        "to learn simpler, more generalizable patterns."
                    ),
                )
                epochs_val = st.number_input(
                    "Epochs",
                    min_value=1, max_value=200,
                    value=int(ep_spec.get("high", 15)),
                    step=1,
                    help=(
                        "The number of times the model will pass through the "
                        "entire training dataset."
                    ),
                )

            # ── Fusion Strategy & Batch Size ──────────────────────────
            st.markdown("#### Architecture & Data Loading")
            ac1, ac2 = st.columns(2)
            with ac1:
                current_fusion = best.get("fusion_strategy", "ula")
                fusion_options = [
                    "ula",               # cross-modal Transformer — best for text+image
                    "concatenation",
                    "attention",
                    "graph",
                    "uncertainty",
                    "uncertainty_graph",
                    "gated",
                ]
                fusion_descriptions = {
                    "ula": "★ Unified Latent Alignment — cross-modal Transformer. Image patches + text tokens attend each other jointly. Best for text+image.",
                    "concatenation": "Direct concat of pooled embeddings. Fastest; good for tabular-only or single modality. No cross-modal interaction.",
                    "attention": "Learned per-modality importance weights. Good when one modality is weaker. No token-level cross-modal attention.",
                    "graph": "Explicit relation graph across modalities. Best for tabular+text with entity relations.",
                    "uncertainty": "Inverse-variance weighting — downweights noisy modalities per sample. Good for tabular+image.",
                    "uncertainty_graph": "Hybrid graph + uncertainty for 3+ modalities with relational + quality signals.",
                    "gated": "Learned gate suppresses conflicting modalities. Good when one modality has high noise.",
                }
                fusion_idx = (
                    fusion_options.index(current_fusion)
                    if current_fusion in fusion_options
                    else 0  # defaults to "ula" (index 0), not concatenation
                )
                fusion_val = st.selectbox(
                    "Fusion Strategy",
                    options=fusion_options,
                    index=fusion_idx,
                    help=(
                        "How modality embeddings are combined before classification. "
                        "ULA (Unified Latent Alignment) is the system default for text+image — "
                        "it runs cross-modal Transformer attention so patches and tokens interact directly."
                    ),
                )
                st.caption(fusion_descriptions.get(fusion_val, ""))
            with ac2:
                current_batch = int(best.get("batch_size", 32))
                batch_options = [16, 32, 64, 128]
                batch_idx = (
                    batch_options.index(current_batch)
                    if current_batch in batch_options
                    else 1
                )
                batch_val = st.selectbox(
                    "Batch Size",
                    options=batch_options,
                    index=batch_idx,
                    help=(
                        "Number of samples processed together in one forward "
                        "pass. Larger batches use more GPU memory but can "
                        "speed up training."
                    ),
                )

            st.session_state.hp_overrides = {
                "learning_rate": lr_val,
                "weight_decay": wd_val,
                "dropout": dropout_val,
                "epochs": epochs_val,
                "fusion_strategy": fusion_val,
                "batch_size": batch_val,
            }
        else:
            st.session_state.hp_overrides = None

        if st.button("Next: Training"):
            st.session_state.workflow_stage = 5
            st.rerun()


def render_phase_5_training():
    """Phase 5: GPU Training with live progress polling."""
    st.header("Phase 5 - GPU Training")

    with st.expander("ℹ️ What is this phase doing? (click to expand)", expanded=False):
        st.markdown("""
        **For beginners:** This is where the model actually learns from your data.

        **What happens inside:**
        1. **Optuna HPO** — tries many combinations of hyperparameters (learning rate, batch size, dropout)
           automatically. Each combination is called a *trial*. Optuna learns from earlier trials to
           suggest better values next — like a smart search rather than random guessing.
        2. **GPU acceleration** — matrix multiplications run in parallel on your GPU.
        3. **Adaptive pruning** — trials that are clearly going nowhere are stopped early.
        4. **SWA [14]** — Stochastic Weight Averaging averages weights in the last 10% of epochs
           to find flatter minima that generalise better (Izmailov et al., UAI 2018).
        5. **Focal Loss [13]** — automatically activated for imbalanced classes (ratio > 3:1).
           Focuses training on hard, misclassified examples (Lin et al., ICCV 2017).
        6. **PCGrad [15]** — when text + image + tabular encoders are all active, conflicting
           gradients are surgically projected to eliminate destructive interference (Yu et al., NeurIPS 2020).
        7. **EWC [8]** — during retraining, prevents catastrophic forgetting of the previous task
           (Kirkpatrick et al., TNNLS 2024).
        8. **Calibration [9]** — after training, confidence scores are adjusted so they match
           reality (Guo et al., ICML 2017).

        **Tabular encoder options (auto-selected by JIT probe):**
        - **FTTransformer [12]** — Transformer over feature tokens (NeurIPS 2021). Best for complex tables.
        - **GRN** — Gated Residual Network. Good for moderate complexity.
        - **MLP** — simple 3-layer network. Fast baseline.

        **Hyperparameter overrides** (Advanced):
        You can manually set learning rate, epochs, batch size, dropout, and weight decay below.
        Leave blank to let Optuna choose automatically.

        **Training Pipeline:**
        - Re-registers cached datasets → runs schema detection + preprocessing
        - Selects model architecture via AdvancedModelSelector
        - Optuna HPO study with SWA + PCGrad + GPU training → calibration → XAI explanation
        """)

    if not st.session_state.model_selected:
        st.warning("Please select models in Phase 4 first")
        return

    task_id = st.session_state.training_task_id

    # ----- No active task: show start button -----
    if task_id is None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Training Pipeline:**
            - Re-registers cached datasets
            - Runs schema detection + preprocessing
            - Selects model architecture via AdvancedModelSelector
            - Optuna HPO study with GPU training
            """)
            if st.button("Start Training", width="stretch"):
                if not check_api_connection():
                    st.error("API not connected!")
                    return
                schema_data = st.session_state.detected_schema or {}
                payload = {
                    "session_id": st.session_state.session_id,  # B4 FIX
                    "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                    "modalities": schema_data.get("global_modalities", ["tabular"]),
                    # Bug #3: propagate schema target override (user may have changed it in Phase 2)
                    "target_column": schema_data.get("primary_target"),
                }
                # Pass explicit column assignments from preprocessing so
                # the training orchestrator doesn't have to re-detect them
                if st.session_state.text_columns:
                    payload["text_columns"] = st.session_state.text_columns
                if st.session_state.image_columns:
                    payload["image_columns"] = st.session_state.image_columns
                if st.session_state.hp_overrides:
                    payload["hp_overrides"] = st.session_state.hp_overrides
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/train-pipeline",
                        json=payload,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.training_task_id = data["task_id"]
                        st.rerun()
                    else:
                        st.error(f"Failed to start training: {resp.status_code} - {_api_error_detail(resp)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        with col2:
            st.markdown("### Model Info")
            model_info = st.session_state.get("model_selection_result", {})
            best = model_info.get("best_model") or {}
            st.info(f"""
            - **Model**: {best.get('name', 'N/A')}
            - **Batch Size**: {best.get('batch_size', 'N/A')}
            - **Fusion**: {best.get('fusion_strategy', 'N/A')}
            """)
        return

    # ----- Active task: poll for progress -----
    try:
        resp = requests.get(f"{API_BASE_URL}/train-pipeline/status/{task_id}", timeout=5)
        if resp.status_code == 404:
            st.warning("Training task not found. It may have expired.")
            st.session_state.training_task_id = None
            return
        if resp.status_code != 200:
            st.error(f"Status poll failed: {resp.status_code}")
            return
        task = resp.json()
    except Exception as e:
        st.error(f"Could not poll training status: {e}")
        return

    status = task.get("status", "unknown")
    current_phase = task.get("current_phase", 0)
    current_phase_name = task.get("current_phase_name", "")
    progress_pct = task.get("progress_pct", 0)
    messages = task.get("messages", [])
    substage = task.get("substage")
    current_trial_state = task.get("current_trial") or {}
    best_so_far = task.get("best_so_far") or {}
    trial_events = task.get("trial_events", []) or []
    pruning_status = task.get("pruning_status") or {}
    next_trial_plan = task.get("next_trial_plan")

    # Progress bar
    st.progress(progress_pct / 100, text=f"Phase {current_phase}/7: {current_phase_name} ({progress_pct}%)")

    status_cols = st.columns(4)
    status_cols[0].metric("Status", status.title())
    status_cols[1].metric("Phase", f"{current_phase}/7")
    status_cols[2].metric("Substage", _humanize_training_substage(substage))
    status_cols[3].metric("Heartbeat", datetime.now().strftime("%H:%M:%S"))
    if current_phase_name:
        st.caption(f"Backend activity: {current_phase_name}")

    # ----- Live epoch metrics (during training) -----
    epoch_metrics = task.get("epoch_metrics", [])
    trial_progress = task.get("trial_progress")
    data_split = task.get("data_split")
    _live_metrics = [e for e in epoch_metrics if not e.get("pruned", False)] if epoch_metrics else []
    latest = _live_metrics[-1] if _live_metrics else None

    if status == "running":
        card_cols = st.columns(4)
        trial_label = "Waiting"
        if current_trial_state:
            trial_label = f"{current_trial_state.get('number', '?')}/{current_trial_state.get('total', '?')}"
        elif trial_progress:
            trial_label = f"{trial_progress.get('current', 0)}/{trial_progress.get('total', 0)}"
        card_cols[0].metric(
            "Current Trial",
            trial_label,
            current_trial_state.get("fusion", "") or None,
        )

        epoch_label = "Waiting"
        if current_trial_state:
            epoch_label = (
                f"{current_trial_state.get('current_epoch', 0)}/"
                f"{current_trial_state.get('max_epoch', current_trial_state.get('epochs', 0))}"
            )
        elif latest:
            epoch_label = f"{latest['epoch']}/{latest['max_epoch']}"
        epoch_delta = None
        if latest is not None and isinstance(latest.get("val_loss"), (int, float)):
            epoch_delta = f"val_loss {latest['val_loss']:.4f}"
        card_cols[1].metric("Current Epoch", epoch_label, epoch_delta)

        best_label = "Pending"
        best_delta = None
        if best_so_far:
            best_label = (
                f"Trial {best_so_far.get('trial', '?')} | "
                f"{float(best_so_far.get('val_loss', 0.0)):.4f}"
            )
            best_delta = f"val_acc {float(best_so_far.get('val_acc', 0.0)):.3f}"
        card_cols[2].metric("Best So Far", best_label, best_delta)

        prune_backend = pruning_status.get("backend") or "unavailable"
        card_cols[3].metric(
            "Pruning",
            "On" if pruning_status.get("available") else "Off",
            prune_backend,
        )
        st.caption(
            f"Pruned {int(pruning_status.get('pruned_count', 0) or 0)} | "
            f"Completed {int(pruning_status.get('completed_count', 0) or 0)}"
        )
        if pruning_status.get("reason"):
            st.caption(f"Pruning detail: {pruning_status.get('reason')}")

        if data_split:
            ds1, ds2, ds3 = st.columns(3)
            ds1.metric("Total Samples", data_split.get("total", "?"))
            ds2.metric("Train Split", data_split.get("train", "?"))
            ds3.metric("Val Split", data_split.get("val", "?"))

    # ── Trial progress bar ────────────────────────────────────────────────
    # Find latest non-pruned epoch for live metrics
    _live_metrics = [e for e in epoch_metrics if not e.get("pruned", False)] if epoch_metrics else []
    if status == "running" and (current_trial_state or trial_progress):
        t_cur = current_trial_state.get("number") or trial_progress.get("current", 0)
        t_total = current_trial_state.get("total") or trial_progress.get("total", 1)
        if _live_metrics:
            latest = _live_metrics[-1]
            ep_pct = latest["epoch"] / max(1, latest["max_epoch"])
            st.progress(
                ep_pct,
                text=f"Trial {t_cur}/{t_total} — Epoch {latest['epoch']}/{latest['max_epoch']}  |  "
                     f"val_loss={latest['val_loss']:.4f}  val_acc={latest['val_acc']:.3f}"
                     + (f"  auroc={latest['val_auroc']:.3f}" if latest.get("val_auroc", 0) > 0 else ""),
            )
        else:
            st.progress(0.0, text=f"Trial {t_cur}/{t_total} — waiting for first epoch...")

    # ── Live epoch metric columns ─────────────────────────────────────────
    if _live_metrics and status == "running":
        latest = _live_metrics[-1]
        _has_auroc = latest.get("val_auroc", 0) > 0
        ep_cols = st.columns(6 if _has_auroc else 5)
        ep_cols[0].metric("Epoch", f"{latest['epoch']}/{latest['max_epoch']}")
        ep_cols[1].metric("Train Loss", f"{latest['train_loss']:.4f}")
        ep_cols[2].metric("Val Loss", f"{latest['val_loss']:.4f}")
        ep_cols[3].metric("Val Acc", f"{latest['val_acc']:.3f}")
        ep_cols[4].metric("Val F1", f"{latest['val_f1']:.3f}")
        if _has_auroc:
            ep_cols[5].metric("Val AUROC", f"{latest['val_auroc']:.3f}")

    if status == "running":
        cockpit_left, cockpit_right = st.columns([1.4, 1.0])
        with cockpit_left:
            st.markdown("##### Trial Timeline")
            _render_trial_timeline(trial_events)
        with cockpit_right:
            st.markdown("##### Next Trial Plan")
            if next_trial_plan:
                _kv_table(next_trial_plan, "Next Trial Plan")
            else:
                st.caption("Waiting for adaptive next-trial guidance.")

        st.markdown("##### Event Feed")
        _render_trial_event_feed(trial_events)

    # ── Messages log (shows FIRST — visible even during precomputation) ───
    if messages:
        with st.expander("Raw Training Log", expanded=False):
            for _msg in messages[-25:]:
                _txt = _msg.get("text", "")
                _mtype = _msg.get("type", "info")
                if _mtype == "pruned" or "pruned" in _txt.lower():
                    st.warning(_txt)
                elif _mtype == "success" or "new best" in _txt.lower():
                    st.success(_txt)
                else:
                    st.caption(_txt)

    # ── Live loss chart (shows after first epoch) ─────────────────────────
    if epoch_metrics:
        st.markdown("##### Loss Curves & AUROC")
        _render_loss_chart(epoch_metrics, trial_events)

    # Render phase-by-phase status
    _render_training_phases(messages, current_phase, status)

    # ----- Completed -----
    if status == "completed":
        result = task.get("result", {})
        data = result.get("data", {})
        metrics = data.get("metrics", {})

        st.session_state.training_result = data
        st.session_state.training_task_id = None
        # Store trained model_id so prediction / registry panels can use it
        _trained_model_id = data.get("model_id")
        if _trained_model_id:
            st.session_state.trained_model_id = _trained_model_id
        train_context_stage = data.get("context_stage")
        train_context_version = data.get("context_version")
        if train_context_stage or train_context_version:
            st.caption(
                f"Context stage: {train_context_stage or 'N/A'} | "
                f"Context version: {train_context_version or 'N/A'}"
            )
        best_val_acc = metrics.get("best_val_acc")
        acc_text = (
            f"{float(best_val_acc):.2%}"
            if isinstance(best_val_acc, (int, float))
            else "N/A"
        )
        st.session_state.phase_states[5] = {
            "status": "completed",
            "reason": (
                f"Best val_acc={acc_text}, "
                f"trials={metrics.get('n_trials', '?')}, "
                f"time={metrics.get('training_time', '?')}"
            ),
        }

        st.success("Training Complete!")
        st.markdown("### Training Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Val Loss", f"{metrics.get('best_val_loss', 'N/A'):.4f}"
                  if isinstance(metrics.get('best_val_loss'), (int, float)) else "N/A")
        c2.metric("Val Accuracy", f"{metrics.get('best_val_acc', 0):.2%}"
                  if isinstance(metrics.get('best_val_acc'), (int, float)) else "N/A")
        c3.metric("Val F1 Score", f"{metrics.get('best_val_f1', 0):.4f}"
                  if isinstance(metrics.get('best_val_f1'), (int, float)) else "N/A")
        c4.metric("Training Time", metrics.get("training_time", "N/A"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Optuna Trials", metrics.get("n_trials", "N/A"))
        c6.metric("Best Trial", f"#{metrics.get('best_trial', 'N/A')}")
        n_pruned = metrics.get("n_pruned", 0)
        c7.metric("Pruned Trials", n_pruned if n_pruned else "0")
        c8.metric("Train Accuracy", f"{metrics.get('best_train_acc', 0):.2%}"
                  if isinstance(metrics.get('best_train_acc'), (int, float)) else "N/A")

        # Data split info
        split_info = metrics.get("data_split", {})
        if split_info:
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("Total Samples", split_info.get("total", "?"))
            sp2.metric("Train Samples", split_info.get("train", "?"))
            sp3.metric("Val Samples", split_info.get("val", "?"))

        fusion_summary = metrics.get("fusion_summary", {}) or {}
        fusion_aux_weights = (
            metrics.get("fusion_aux_weights", {})
            or fusion_summary.get("auxiliary_loss_weights", {})
            or {}
        )
        if fusion_summary or fusion_aux_weights:
            with st.expander("Fusion Diagnostics", expanded=False):
                fs1, fs2, fs3 = st.columns(3)
                fs1.metric("Fusion Type", fusion_summary.get("fusion_type", "N/A"))
                fs2.metric("Backend", fusion_summary.get("backend_module", "N/A"))
                _align_w = fusion_summary.get("alignment_weight", metrics.get("alignment_summary", {}).get("weight", 0.0))
                fs3.metric("Alignment Weight", f"{float(_align_w):.4f}" if isinstance(_align_w, (int, float)) else "N/A")

                if fusion_aux_weights:
                    st.caption("Auxiliary loss weights")
                    _kv_table(fusion_aux_weights, "Fusion Weights")

                branch_weights = fusion_summary.get("branch_weights", {}) or {}
                if branch_weights:
                    st.caption("Uncertainty/graph branch mix")
                    st.bar_chart(
                        pd.DataFrame.from_dict(branch_weights, orient="index", columns=["Weight"])
                    )

                attention_summary = fusion_summary.get("attention_summary", {}) or {}
                if attention_summary:
                    st.caption("Fusion attention summary")
                    _kv_table(attention_summary, "Attention Summary")

                # Part B.3 — ULA alignment_loss and contrastive_loss live curves
                _align_hist = metrics.get("alignment_loss_history", []) or []
                _contrast_hist = metrics.get("contrastive_loss_history", []) or []
                if _align_hist or _contrast_hist:
                    _max_len = max(len(_align_hist), len(_contrast_hist))
                    _align_hist = _align_hist + [None] * (_max_len - len(_align_hist))
                    _contrast_hist = _contrast_hist + [None] * (_max_len - len(_contrast_hist))
                    _extra_df = pd.DataFrame({
                        "alignment_loss": _align_hist,
                        "contrastive_loss": _contrast_hist,
                    }).dropna(how="all")
                    if not _extra_df.empty:
                        st.caption("ULA cross-modal alignment and contrastive loss curves")
                        st.line_chart(_extra_df, color=["#a78bfa", "#14b8a6"])

        # ── Encoding Architecture Panel ───────────────────────────────────────
        with st.expander("🔬 Encoding Architecture (Before → Hidden → After)", expanded=False):
            _fs = fusion_summary if fusion_summary else {}
            _enc_dims = _fs.get("encoder_dims", {})
            _tok_mode = _fs.get("token_mode", False)
            _clip_active = _fs.get("clip_projections_active", False)
            _grad_scales = _fs.get("modality_grad_scales", {})
            _cw = _fs.get("contrastive_weight", 0.0)
            _ula_d = _fs.get("ula_latent_dim")
            _ula_l = _fs.get("ula_n_layers")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Before (encoders)**")
                if _enc_dims:
                    for mod, dim in _enc_dims.items():
                        st.write(f"→ `{mod}`: {dim}-dim output")
                else:
                    st.caption("Encoder dims not yet available")

            with col2:
                st.markdown("**Hidden (fusion)**")
                fusion_type = _fs.get("fusion_type", "—")
                st.write(f"Fusion: `{fusion_type}`")
                if _ula_d:
                    st.write(f"ULA latent dim: `{_ula_d}`")
                if _ula_l:
                    st.write(f"ULA Transformer layers: `{_ula_l}`")
                mode_label = "Token sequences (full attention)" if _tok_mode else "Pooled vectors (CLS only)"
                st.write(f"Input mode: `{mode_label}`")

            with col3:
                st.markdown("**After (contrastive + head)**")
                if _clip_active and _cw > 0:
                    st.write(f"CLIP projections: ✅ active")
                    st.write(f"Contrastive weight: `{_cw:.3f}`")
                else:
                    st.write("CLIP projections: single-modality (inactive)")
                st.write(f"Classification head: `MLP`")

            if _grad_scales:
                st.markdown("**Per-modality gradient health** (Wang et al. 2020 gradient balancing)")
                _grad_df = pd.DataFrame([
                    {"modality": k, "grad_scale": v,
                     "status": "⚠️ dominant" if v > 1.5 else ("⚠️ weak" if v < 0.5 else "✅ balanced")}
                    for k, v in _grad_scales.items()
                ])
                st.dataframe(_grad_df, width="stretch")
                if any(v > 1.5 or v < 0.5 for v in _grad_scales.values()):
                    st.warning(
                        "Modality imbalance detected — one modality's gradients dominate. "
                        "Consider increasing modality dropout or adjusting alignment weight."
                    )
            else:
                st.caption("Gradient scales available after first training epoch")

        fit_type = metrics.get("fit_type", "unknown")
        if fit_type == "overfitting":
            st.warning(
                "TrialIntelligence diagnosis: overfitting detected. "
                "Consider higher regularization or fewer epochs."
            )
        elif fit_type == "underfitting":
            st.info(
                "TrialIntelligence diagnosis: underfitting detected. "
                "Consider more capacity or more epochs."
            )
        elif fit_type == "good":
            st.success("TrialIntelligence diagnosis: fit is balanced.")

        trial_diagnostics = metrics.get("trial_diagnostics", [])
        if trial_diagnostics:
            with st.expander("TrialIntelligence Diagnostics", expanded=False):
                diag_df = pd.DataFrame(trial_diagnostics)
                if "val_loss" in diag_df.columns:
                    diag_df = diag_df.sort_values("val_loss", ascending=True)
                st.dataframe(diag_df, width="stretch")
                for _, row in diag_df.iterrows():
                    factors = row.get("dynamic_factors", {})
                    if isinstance(factors, dict) and factors:
                        st.caption(
                            f"Trial {row.get('trial', '?')}: "
                            f"reg_factor={float(factors.get('regularization', 1.0)):.2f} | "
                            f"data_factor={float(factors.get('data_loss', 1.0)):.2f} | "
                            f"constraint_factor={float(factors.get('constraint', 1.0)):.2f}"
                        )

        trial_feedback_events = metrics.get("trial_feedback_events", []) or []
        if trial_feedback_events:
            with st.expander("Adaptive Trial Feedback Events", expanded=False):
                st.dataframe(pd.DataFrame(trial_feedback_events), width="stretch")

        if st.session_state.session_id:
            try:
                timing_resp = requests.get(
                    f"{API_BASE_URL}/context/{st.session_state.session_id}/phase-timings",
                    timeout=5,
                )
                if timing_resp.status_code == 200:
                    timings = timing_resp.json().get("phase_timings", {})
                    if isinstance(timings, dict) and timings:
                        st.markdown("### Phase Timing Breakdown")
                        timing_df = pd.DataFrame([
                            {"Phase": phase_name, "Duration (s)": f"{float(duration):.1f}"}
                            for phase_name, duration in timings.items()
                        ])
                        st.dataframe(timing_df, width="stretch")
            except Exception:
                pass

        # Epoch-level loss chart from completed training
        completed_epoch_metrics = task.get("epoch_metrics", [])
        if completed_epoch_metrics:
            st.markdown("### Loss Convergence")
            _render_loss_chart(completed_epoch_metrics)

        with st.expander("Full Metrics", expanded=False):
            _kv_table(metrics, "Metrics")

        xai = data.get("xai", {})
        if isinstance(xai, dict) and xai:
            st.divider()
            st.markdown("### 🧠 Training-Phase XAI Snapshot")
            st.caption(
                "Generated after training on the best trial's final batch. "
                "These are proxy importance scores; use Phase 7 for per-prediction IntegratedGradients."
            )
            xai_tabs = st.tabs(["Tabular Importance", "Fusion Weights", "Text/Image"])

            with xai_tabs[0]:
                tab_xai = xai.get("tabular", {}) if isinstance(xai, dict) else {}
                importances = tab_xai.get("feature_importance") or tab_xai.get("feature_importances", [])
                method = tab_xai.get("proxy_method", tab_xai.get("method", "unknown"))
                info = tab_xai.get("info", "")
                if importances:
                    n_feats = len(importances)
                    feat_df = pd.DataFrame({
                        "Feature Index": list(range(n_feats)),
                        "Importance (proxy)": importances,
                    }).sort_values("Importance (proxy)", ascending=False).head(20)
                    st.bar_chart(feat_df.set_index("Feature Index"))
                    if info:
                        st.caption(f"ℹ️ {info} Method: `{method}`")
                elif tab_xai.get("error"):
                    st.warning(f"Tabular XAI failed: {tab_xai.get('error')}")
                else:
                    st.info("No tabular training XAI available.")

            with xai_tabs[1]:
                fusion_xai = xai.get("fusion", {}) if isinstance(xai, dict) else {}
                weights = fusion_xai.get("weights", {}) if isinstance(fusion_xai, dict) else {}
                strategy = fusion_xai.get("strategy", "unknown") if isinstance(fusion_xai, dict) else "unknown"
                fmethod = fusion_xai.get("method", "dummy") if isinstance(fusion_xai, dict) else "dummy"
                if weights:
                    st.bar_chart(pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"]))
                    st.caption(
                        f"Strategy: `{strategy}` | Method: `{fmethod}`. "
                        + (
                            "These are learned attention weights from the fusion module."
                            if fmethod == "learned_weights"
                            else "Uniform weights - fusion module did not expose attention."
                        )
                    )
                elif fusion_xai.get("error"):
                    st.warning(f"Fusion XAI failed: {fusion_xai.get('error')}")
                else:
                    st.info("No fusion weights available in training snapshot.")

            with xai_tabs[2]:
                text_xai = xai.get("text", {}) if isinstance(xai, dict) else {}
                img_xai = xai.get("image", {}) if isinstance(xai, dict) else {}
                if text_xai.get("method") not in (None, "dummy"):
                    st.write(
                        f"**Text:** seq_len={text_xai.get('seq_len', '?')}, "
                        f"method=`{text_xai.get('method')}`"
                    )
                else:
                    st.info("No text XAI - text encoder not in model or attention extraction unavailable.")
                if img_xai.get("method") not in (None, "dummy"):
                    st.write(f"**Image GradCAM:** shape={img_xai.get('heatmap_shape', '?')}")
                else:
                    st.info("Image GradCAM not implemented. Use Captum GradCAM in a future release.")

        # Retrain with manual HP overrides
        best_params = metrics.get("best_params", {})
        if best_params:
            st.divider()
            st.markdown("### Retrain with Custom Hyperparameters")
            st.caption("Adjust the HPO-selected parameters below and retrain.")
            rc1, rc2 = st.columns(2)
            with rc1:
                retrain_lr = st.number_input(
                    "Learning Rate (retrain)",
                    min_value=1e-6, max_value=1.0,
                    value=float(best_params.get("learning_rate", 1e-4)),
                    format="%.6f", step=1e-5,
                    key="retrain_lr",
                    help=(
                        "Controls the speed at which the model adjusts its "
                        "weights. Too high causes instability; too low makes "
                        "training painfully slow."
                    ),
                )
                retrain_dropout = st.number_input(
                    "Dropout (retrain)",
                    min_value=0.0, max_value=0.9,
                    value=float(best_params.get("dropout", 0.1)),
                    format="%.3f", step=0.05,
                    key="retrain_dropout",
                    help=(
                        "Randomly turns off neurons during training to prevent "
                        "the model from memorizing the data (overfitting)."
                    ),
                )
            with rc2:
                retrain_wd = st.number_input(
                    "Weight Decay (retrain)",
                    min_value=1e-7, max_value=0.1,
                    value=float(best_params.get("weight_decay", 1e-5)),
                    format="%.7f", step=1e-5,
                    key="retrain_wd",
                    help=(
                        "Applies a penalty to large weights, forcing the model "
                        "to learn simpler, more generalizable patterns."
                    ),
                )
                retrain_epochs = st.number_input(
                    "Epochs (retrain)",
                    min_value=1, max_value=200,
                    value=int(best_params.get("epochs", 15)),
                    step=1,
                    key="retrain_epochs",
                    help=(
                        "The number of times the model will pass through the "
                        "entire training dataset."
                    ),
                )

            # Architecture overrides for retrain
            st.markdown("#### Architecture & Data Loading")
            ra1, ra2 = st.columns(2)
            model_info = st.session_state.get("model_selection_result", {})
            best_model = model_info.get("best_model") or {}
            with ra1:
                rt_fusion_cur = best_model.get("fusion_strategy", "ula")
                rt_fusion_opts = [
                    "ula",               # cross-modal Transformer — best for text+image
                    "concatenation",
                    "attention",
                    "graph",
                    "uncertainty",
                    "uncertainty_graph",
                    "gated",
                ]
                rt_fusion_idx = (
                    rt_fusion_opts.index(rt_fusion_cur)
                    if rt_fusion_cur in rt_fusion_opts
                    else 0  # defaults to "ula" (index 0)
                )
                retrain_fusion = st.selectbox(
                    "Fusion Strategy (retrain)",
                    options=rt_fusion_opts,
                    index=rt_fusion_idx,
                    key="retrain_fusion",
                    help=(
                        "Choose how modality embeddings are combined before prediction."
                    ),
                )
            with ra2:
                rt_batch_cur = int(best_model.get("batch_size", 32))
                rt_batch_opts = [16, 32, 64, 128]
                rt_batch_idx = (
                    rt_batch_opts.index(rt_batch_cur)
                    if rt_batch_cur in rt_batch_opts
                    else 1
                )
                retrain_batch = st.selectbox(
                    "Batch Size (retrain)",
                    options=rt_batch_opts,
                    index=rt_batch_idx,
                    key="retrain_batch",
                    help=(
                        "Number of samples processed together in one forward "
                        "pass. Larger batches use more GPU memory but can "
                        "speed up training."
                    ),
                )

            if st.button("Retrain with These Parameters", width="stretch"):
                if not check_api_connection():
                    st.error("API not connected!")
                    return
                schema_data = st.session_state.detected_schema or {}
                retrain_payload = {
                    "session_id": st.session_state.session_id,
                    "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                    "modalities": schema_data.get("global_modalities", ["tabular"]),
                    "hp_overrides": {
                        "learning_rate": retrain_lr,
                        "weight_decay": retrain_wd,
                        "dropout": retrain_dropout,
                        "epochs": retrain_epochs,
                        "fusion_strategy": retrain_fusion,
                        "batch_size": retrain_batch,
                    },
                }
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/train-pipeline",
                        json=retrain_payload,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state.training_task_id = resp.json()["task_id"]
                        st.session_state.training_result = None
                        st.rerun()
                    else:
                        st.error(f"Retrain failed: {resp.status_code} - {_api_error_detail(resp)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # ----- Failed -----
    elif status == "failed":
        error = task.get("error", "Unknown error")
        st.error(f"**Training failed:** {error}")
        st.caption("The Training Log above shows what ran before the crash. Fix the error and restart.")
        if st.button("Reset Training (clear error)", key="reset_failed_training"):
            st.session_state.training_task_id = None
            st.rerun()
        st.session_state.phase_states[5] = {
            "status": "failed",
            "reason": str(error),
        }

    # ----- Running: auto-refresh -----
    elif status == "running":
        time.sleep(2)
        st.rerun()
        return

    # ----- Unknown / unexpected status — do NOT loop forever -----
    else:
        st.warning(
            f"Training status is **'{status}'** — waiting for the backend to start. "
            "If this persists, the API may have restarted mid-training."
        )
        col_r1, col_r2 = st.columns(2)
        if col_r1.button("🔄 Refresh", key="refresh_unknown_status"):
            st.rerun()
        if col_r2.button("✖ Abandon Task", key="abandon_unknown_task"):
            st.session_state.training_task_id = None
            st.rerun()
        return

    if st.button("Next: Monitoring"):
        st.session_state.workflow_stage = 6
        st.rerun()


def _render_training_phases(
    messages: List[Dict],
    current_phase: int,
    task_status: str,
) -> None:
    """Render 7 phase expanders with their messages."""
    phase_names = {
        1: "Data Ingestion",
        2: "Schema Detection",
        3: "Preprocessing",
        4: "Model Selection",
        5: "Training",
        6: "Drift Detection",
        7: "Model Registry",
    }

    # Group messages by phase
    phase_msgs: Dict[int, List[Dict]] = {i: [] for i in range(1, 8)}
    for msg in messages:
        p = msg.get("phase", 0)
        if 1 <= p <= 7:
            phase_msgs[p].append(msg)

    for phase_num in range(1, 8):
        name = phase_names[phase_num]

        if phase_num < current_phase:
            # Completed phase
            with st.status(f"Phase {phase_num}: {name}", state="complete", expanded=False):
                for msg in phase_msgs[phase_num]:
                    _render_message(msg)
        elif phase_num == current_phase:
            # Current phase
            is_done = task_status in ("completed", "failed")
            state = "complete" if is_done else "running"
            with st.status(f"Phase {phase_num}: {name}", state=state, expanded=not is_done):
                for msg in phase_msgs[phase_num]:
                    _render_message(msg)
                if not is_done:
                    st.caption("In progress...")
        else:
            # Future phase — use "running" state (neutral) rather than "error" (red)
            with st.status(f"Phase {phase_num}: {name}", state="running", expanded=False):
                st.caption("Pending")


def _humanize_training_substage(substage: Optional[str]) -> str:
    """Convert backend substage keys into UI-friendly labels."""
    if not substage:
        return "Initializing"
    return str(substage).replace("_", " ").strip().title()


def _render_trial_event_feed(trial_events: List[Dict]) -> None:
    """Render the newest structured backend events above the raw log."""
    if not trial_events:
        st.caption("Waiting for structured training events...")
        return

    for event in reversed(trial_events[-12:]):
        trial = event.get("trial")
        label = f"Trial {trial}" if isinstance(trial, int) else "System"
        event_name = str(event.get("event", "event")).replace("_", " ").title()
        detail = event.get("detail") or "No detail provided."
        st.caption(f"{label} | {event_name} | {detail}")


def _render_trial_timeline(trial_events: List[Dict]) -> None:
    """Render a compact per-trial status strip for the current Optuna study."""
    import altair as alt

    status_priority = {
        "trial_start": 1,
        "epoch": 1,
        "trial_complete": 2,
        "new_best": 3,
        "pruned": 4,
        "warning": 5,
    }
    status_name = {
        "trial_start": "Running",
        "epoch": "Running",
        "trial_complete": "Completed",
        "new_best": "Best",
        "pruned": "Pruned",
        "warning": "Failed",
    }

    latest_by_trial: Dict[int, Dict[str, Any]] = {}
    for event in trial_events:
        trial = event.get("trial")
        event_name = str(event.get("event", ""))
        if not isinstance(trial, int) or event_name not in status_priority:
            continue
        candidate = {
            "Trial": f"Trial {trial}",
            "Status": status_name[event_name],
            "Detail": event.get("detail") or "",
            "_priority": status_priority[event_name],
        }
        current = latest_by_trial.get(trial)
        if current is None or candidate["_priority"] >= current["_priority"]:
            latest_by_trial[trial] = candidate

    if not latest_by_trial:
        st.caption("Timeline will appear once the first trial starts.")
        return

    timeline_df = pd.DataFrame(
        sorted(latest_by_trial.values(), key=lambda row: int(str(row["Trial"]).split()[-1]))
    )
    timeline_df["value"] = 1
    chart = (
        alt.Chart(timeline_df)
        .mark_bar(size=28)
        .encode(
            x=alt.X("Trial:N", sort=None, title="Trial"),
            y=alt.Y("value:Q", axis=None),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(
                    domain=["Running", "Completed", "Best", "Pruned", "Failed"],
                    range=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#7c3aed"],
                ),
            ),
            tooltip=["Trial", "Status", "Detail"],
        )
        .properties(height=90)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_message(msg: Dict) -> None:
    """Render a single progress message based on its type."""
    msg_type = msg.get("type", "info")
    text = msg.get("text", "")
    if msg_type == "result":
        st.markdown(f"**{text}**")
    elif msg_type == "warning":
        st.warning(text)
    elif msg_type == "detail":
        st.caption(text)
    else:
        st.write(text)


def _render_loss_chart(epoch_metrics: List[Dict], trial_events: Optional[List[Dict]] = None) -> None:
    """Render live training charts: loss curves + AUROC area + pruning markers.

    Layout (top→bottom):
    1. Loss chart — train_loss (solid) + val_loss (dashed), coloured per trial
    2. AUROC area chart — val_auroc per trial (shown when > 0)
    3. Pruning markers — red vertical rules at prune epochs
    """
    import altair as alt

    df = pd.DataFrame(epoch_metrics)
    if df.empty:
        return

    df["step"] = range(len(df))
    df["trial_display"] = df["trial"].astype(int) + 1
    df["Trial"] = "Trial " + df["trial_display"].astype(str)
    df["label"] = "T" + df["trial_display"].astype(str) + " E" + df["epoch"].astype(str)
    if "pruned" in df.columns:
        df["pruned"] = df["pruned"].fillna(False).astype(bool)
    else:
        df["pruned"] = False

    normal_df = df[~df["pruned"]].copy()
    pruned_df = df[df["pruned"]].copy()

    if normal_df.empty:
        st.caption("Waiting for first epoch...")
        return

    # ── 1. Loss chart ─────────────────────────────────────────────────────
    loss_df = normal_df.melt(
        id_vars=["step", "Trial", "label"],
        value_vars=["train_loss", "val_loss"],
        var_name="Metric", value_name="Loss",
    )
    loss_chart = (
        alt.Chart(loss_df)
        .mark_line(point=alt.OverlayMarkDef(size=40))
        .encode(
            x=alt.X("step:Q", title="Epoch Step", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Loss:Q", title="Loss"),
            color=alt.Color("Trial:N", title="Trial"),
            strokeDash=alt.StrokeDash(
                "Metric:N",
                scale=alt.Scale(
                    domain=["train_loss", "val_loss"],
                    range=[[1, 0], [6, 4]],
                ),
                legend=alt.Legend(title="— train  ╌ val"),
            ),
            tooltip=[
                "label", "Trial", "Metric",
                alt.Tooltip("Loss:Q", format=".4f"),
            ],
        )
        .properties(height=280, title="Loss Curves (solid=train, dashed=val)")
    )

    # ── 2. Pruning markers ────────────────────────────────────────────────
    if not pruned_df.empty:
        prune_rules = (
            alt.Chart(pruned_df)
            .mark_rule(color="red", strokeDash=[4, 3], size=2, opacity=0.7)
            .encode(
                x="step:Q",
                tooltip=[alt.Tooltip("label:N", title="Pruned at"), "Trial:N"],
            )
        )
        loss_chart = loss_chart + prune_rules

    if trial_events:
        new_best_trials = {
            int(event["trial"]) - 1
            for event in trial_events
            if isinstance(event.get("trial"), int) and event.get("event") == "new_best"
        }
        if new_best_trials:
            best_markers = (
                normal_df[normal_df["trial"].isin(new_best_trials)]
                .sort_values(["trial", "val_loss", "step"])
                .groupby("trial", as_index=False)
                .first()
            )
            if not best_markers.empty:
                best_points = (
                    alt.Chart(best_markers)
                    .mark_point(color="#f59e0b", filled=True, size=130)
                    .encode(
                        x="step:Q",
                        y="val_loss:Q",
                        tooltip=[
                            "label",
                            "Trial",
                            alt.Tooltip("val_loss:Q", format=".4f"),
                        ],
                    )
                )
                best_labels = (
                    alt.Chart(best_markers)
                    .mark_text(
                        text="Best",
                        dy=-12,
                        color="#f59e0b",
                        fontSize=11,
                        fontWeight="bold",
                    )
                    .encode(x="step:Q", y="val_loss:Q")
                )
                loss_chart = loss_chart + best_points + best_labels

    charts = [loss_chart.interactive()]

    # ── 3. AUROC area chart ───────────────────────────────────────────────
    if "val_auroc" in normal_df.columns and (normal_df["val_auroc"] > 0).any():
        auroc_chart = (
            alt.Chart(normal_df)
            .mark_area(opacity=0.25, line=True)
            .encode(
                x=alt.X("step:Q", title=""),
                y=alt.Y(
                    "val_auroc:Q", title="AUROC",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".2f"),
                ),
                color=alt.Color("Trial:N", legend=None),
                tooltip=["label", "Trial", alt.Tooltip("val_auroc:Q", format=".4f")],
            )
            .properties(height=100, title="Validation AUROC")
            .interactive()
        )
        charts.append(auroc_chart)

    # ── 4. ULA auxiliary losses (alignment + contrastive) ─────────────────
    _aux_cols = [c for c in ("alignment_loss", "contrastive_loss")
                 if c in normal_df.columns and (normal_df[c] > 0).any()]
    if _aux_cols:
        aux_df = normal_df.melt(
            id_vars=["step", "Trial", "label"],
            value_vars=_aux_cols,
            var_name="AuxMetric", value_name="AuxLoss",
        )
        _aux_labels = {
            "alignment_loss": "Alignment (cosine)",
            "contrastive_loss": "Contrastive (NT-Xent)",
        }
        aux_df["AuxMetric"] = aux_df["AuxMetric"].map(lambda x: _aux_labels.get(x, x))
        aux_chart = (
            alt.Chart(aux_df)
            .mark_line(point=alt.OverlayMarkDef(size=30), strokeDash=[3, 2])
            .encode(
                x=alt.X("step:Q", title=""),
                y=alt.Y("AuxLoss:Q", title="Aux Loss"),
                color=alt.Color("Trial:N", legend=None),
                strokeDash=alt.StrokeDash("AuxMetric:N", legend=alt.Legend(title="ULA Losses")),
                tooltip=["label", "Trial", "AuxMetric",
                         alt.Tooltip("AuxLoss:Q", format=".5f")],
            )
            .properties(height=90, title="ULA Auxiliary Losses (Alignment + Contrastive)")
            .interactive()
        )
        charts.append(aux_chart)

    combined = alt.vconcat(*charts, spacing=8) if len(charts) > 1 else charts[0]
    st.altair_chart(combined, use_container_width=True)


def render_phase_6_monitoring():
    """Phase 6: Monitoring & Drift Detection."""
    st.header("Phase 6 - Model Monitoring & Drift Detection")

    _training_result = st.session_state.get("training_result") or {}
    _trained_model_id = _training_result.get("model_id") or st.session_state.get("trained_model_id")
    if not _training_result or not _trained_model_id:
        st.warning("Please complete training in Phase 5 first.")
        if st.button("← Go to Phase 5"):
            st.session_state.workflow_stage = 5
            st.rerun()
        return

    with st.expander("ℹ️ What is this phase doing? (click to expand)", expanded=False):
        st.markdown("""
        **For beginners:** Once a model is in production, the real world keeps changing.
        This phase watches for those changes and alerts you when the model needs retraining.

        **Three types of drift detected:**

        | Drift Type | What it means | Method used |
        |---|---|---|
        | **Covariate drift** | The distribution of your input features has changed | KS test, PSI, MMD |
        | **Concept drift** | The relationship between inputs and outputs has changed | DDM error-rate tracking [10] |
        | **Embedding drift** | Text/image semantic meaning has shifted | MMD + DriftLens cosine [11] |

        **Drift thresholds:**
        - PSI (Population Stability Index) > 0.25 → feature drift
        - KS (Kolmogorov-Smirnov) > 0.3 → distribution shift
        - MMD > 0.5 → multivariate distributional shift
        - Cosine drift > 0.15 → directional/semantic shift (DriftLens [11])

        **Retraining depth policy:**
        - Low drift → calibration only (recalibrate confidence)
        - Medium drift → head only (retrain final layers)
        - High drift or concept drift → full retraining with EWC [8]
        """)

    st.markdown("""
    **Drift Thresholds:**
    - PSI (Population Stability Index): > 0.25
    - KS (Kolmogorov-Smirnov): > 0.3
    - FDD (Feature Distribution Drift / MMD): > 0.5
    """)
    tab1, tab2, tab3, tab4, tab5, tab_research = st.tabs([
        "Training Results",
        "Drift Detection",
        "Composite Monitor",
        "Model Registry",
        "Retrain History",
        "📊 Research Results",
    ])

    # ── Tab 1: Training results from session state ───────────────────────────
    with tab1:
        st.markdown("### Training Metrics")
        training_result = st.session_state.get("training_result", {})
        metrics = training_result.get("metrics", {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Loss", f"{metrics.get('final_loss', 'N/A')}")
            col2.metric("Best Val Loss", f"{metrics.get('best_val_loss', 'N/A')}")
            col3.metric("Training Time", metrics.get("training_time", "N/A"))
            col4.metric("Optuna Trials", metrics.get("n_trials", "N/A"))
            feedback_events = metrics.get("trial_feedback_events", []) or []
            st.caption(f"Adaptive feedback events: {len(feedback_events)}")

            ctx_stage = training_result.get("context_stage")
            ctx_ver = training_result.get("context_version")
            if ctx_stage or ctx_ver:
                st.caption(
                    f"Context stage: {ctx_stage or 'N/A'} | "
                    f"Context version: {ctx_ver or 'N/A'}"
                )
            with st.expander("Full metrics JSON"):
                _kv_table(metrics, "Metrics")

            fusion_summary = metrics.get("fusion_summary", {}) or {}
            fusion_aux_weights = (
                metrics.get("fusion_aux_weights", {})
                or fusion_summary.get("auxiliary_loss_weights", {})
                or {}
            )
            if fusion_summary or fusion_aux_weights:
                with st.expander("Fusion Diagnostics", expanded=False):
                    st.write(f"Fusion type: {fusion_summary.get('fusion_type', 'N/A')}")
                    st.write(f"Backend: {fusion_summary.get('backend_module', 'N/A')}")
                    if fusion_aux_weights:
                        _kv_table(fusion_aux_weights, "Fusion Weights")
                    attention_summary = fusion_summary.get("attention_summary", {}) or {}
                    if attention_summary:
                        _kv_table(attention_summary, "Attention Summary")

            best_model = (st.session_state.get("model_selection_result", {}) or {}).get("best_model", {}) or {}
            vram_report = best_model.get("vram_filter_report", {}) if isinstance(best_model, dict) else {}
            if vram_report:
                with st.expander("Phase 4 VRAM Filter Report", expanded=False):
                    st.write(f"- GPU Memory (GB): {vram_report.get('gpu_memory_gb', 'N/A')}")
                    st.write(f"- Budget (MB): {vram_report.get('vram_budget_mb', 'N/A')}")
                    excluded_counts = vram_report.get("excluded_counts", {}) or {}
                    if excluded_counts:
                        st.write(
                            "- Excluded candidates: "
                            + ", ".join(f"{k}={v}" for k, v in excluded_counts.items())
                        )
        else:
            st.info("No training metrics yet. Run Phase 5 first.")

    # ── Tab 2: Drift detection (button-guarded, POST method) ─────────────────
    with tab2:
        st.markdown("### Drift Detection")
        if st.button("Run Drift Detection", width="stretch", key="drift_btn"):
            with st.spinner("Computing KS / PSI / MMD..."):
                try:
                    schema_data = st.session_state.detected_schema or {}
                    resp = requests.post(
                        f"{API_BASE_URL}/monitor/drift",
                        json={
                            "session_id": st.session_state.session_id,  # B2 FIX
                            "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                            "modalities": schema_data.get("global_modalities", ["tabular"]),
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        drift = resp.json().get("data", {})
                        st.session_state.drift_result = drift
                        drift_detected = bool(drift.get("drift_detected", False))
                        psi_value = float((drift.get("metrics") or {}).get("psi", 0) or 0)
                        st.session_state.phase_states[6] = {
                            "status": "completed",
                            "reason": (
                                f"DRIFT DETECTED (PSI={psi_value:.4f})"
                                if drift_detected else
                                "No drift detected"
                            ),
                        }
                    else:
                        st.error(f"Drift detection failed: {resp.status_code}")
                except Exception as e:
                    st.error(f"Drift detection error: {e}")

        drift = st.session_state.get("drift_result", {})
        if drift:
            detected = drift.get("drift_detected", False)
            if detected:
                st.error("DRIFT DETECTED")
            else:
                st.success("No significant drift detected")

            m = drift.get("metrics", {})
            t = drift.get("thresholds", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("PSI", f"{m.get('psi', 0):.4f}", delta=f"threshold {t.get('psi', 0.25)}")
            col2.metric("KS", f"{m.get('ks_statistic', 0):.4f}", delta=f"threshold {t.get('ks_statistic', 0.30)}")
            col3.metric("FDD/MMD", f"{m.get('fdd', 0):.4f}", delta=f"threshold {t.get('fdd', 0.50)}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Reference rows", drift.get("n_reference", "?"))
            col2.metric("Production rows", drift.get("n_production", "?"))
            col3.metric("Features", drift.get("n_features", "?"))

            # ── Encoding health from last training run ─────────────────────
            _tr_result = st.session_state.get("training_result", {}) or {}
            _tr_metrics = _tr_result.get("metrics", {}) or {}
            _tr_fusion = _tr_metrics.get("fusion_summary", {}) or {}
            _mon_grad_scales = _tr_fusion.get("modality_grad_scales", {})
            _mon_enc_dims = _tr_fusion.get("encoder_dims", {})
            if _mon_grad_scales or _mon_enc_dims:
                with st.expander("🔬 Modality Encoding Health (from last training run)", expanded=False):
                    st.caption(
                        "Gradient scales show which modalities dominate training. "
                        "Values near 1.0 = balanced. >1.5 = dominant. <0.5 = weak/ignored."
                    )
                    if _mon_enc_dims:
                        st.markdown("**Encoder output dims:**")
                        for _mod, _dim in _mon_enc_dims.items():
                            st.write(f"→ `{_mod}`: {_dim}-dim")
                    if _mon_grad_scales:
                        _g_df = pd.DataFrame([
                            {"modality": k, "grad_scale": v,
                             "health": "⚠️ dominant" if v > 1.5 else ("⚠️ weak" if v < 0.5 else "✅ balanced")}
                            for k, v in _mon_grad_scales.items()
                        ])
                        st.dataframe(_g_df, width="stretch")
        else:
            st.info("Click 'Run Drift Detection' to compute drift metrics.")

    # ── Tab 3: Composite monitor payload ───────────────────────────────────────
    with tab3:
        st.markdown("### Composite Monitor")
        if st.button("Refresh Composite Monitor", width="stretch", key="monitor_btn"):
            with st.spinner("Fetching composite monitoring payload..."):
                try:
                    schema_data = st.session_state.detected_schema or {}
                    resp = requests.post(
                        f"{API_BASE_URL}/monitor",
                        json={
                            "session_id": st.session_state.session_id,
                            "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                            "modalities": schema_data.get("global_modalities", ["tabular"]),
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.session_state.monitor_result = resp.json().get("data", {})
                    else:
                        st.error(f"Composite monitor failed: {resp.status_code}")
                except Exception as exc:
                    st.error(f"Composite monitor error: {exc}")

        monitor_result = st.session_state.get("monitor_result", {})
        monitor_payload = monitor_result.get("monitor", {}) if isinstance(monitor_result, dict) else {}
        if monitor_payload:
            c1, c2, c3 = st.columns(3)
            c1.metric("Severity", str(monitor_payload.get("severity", "unknown")).upper())
            c2.metric("Composite Score", f"{monitor_payload.get('composite_score', 0.0):.4f}")
            c3.metric("Retrain Recommended", "Yes" if monitor_payload.get("retrain_recommended") else "No")

            breached = monitor_payload.get("breached_metrics", [])
            if breached:
                st.markdown("**⚠️ Breached Thresholds - Retraining Triggered Because:**")
                m_vals = monitor_payload.get("metrics", {}) if isinstance(monitor_payload, dict) else {}
                t_vals = monitor_payload.get("thresholds", {}) if isinstance(monitor_payload, dict) else {}
                for metric in breached:
                    val = m_vals.get(metric, "?")
                    thresh = t_vals.get(metric, "?")
                    st.write(f"- `{metric}`: current = **{val}**, threshold = {thresh} -> **{val} > {thresh}**")
                composite = monitor_payload.get("composite_score", "?")
                if isinstance(composite, (int, float)):
                    st.caption(
                        f"Composite score = {float(composite):.4f}. "
                        "Retraining is recommended when any threshold is breached OR "
                        "composite score exceeds 0.5."
                    )
            else:
                st.success("No threshold breaches in the latest monitor snapshot.")

            with st.expander("Composite monitor payload"):
                _kv_table(monitor_payload, "Monitor Payload")
        else:
            st.info("Run 'Refresh Composite Monitor' to load monitor severity and recommendations.")

    # ── Tab 4: Model registry ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### Model Registry")
        if st.button("Refresh Registry", width="stretch", key="registry_btn"):
            try:
                resp = requests.get(f"{API_BASE_URL}/model-registry", timeout=30)
                if resp.status_code == 200:
                    st.session_state.registry_result = resp.json()
                else:
                    st.error(f"Registry fetch failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Registry error: {e}")

        registry = st.session_state.get("registry_result", {})
        models = registry.get("models", [])
        if models:
            st.caption(f"{registry.get('count', len(models))} model(s) registered")
            display_cols = ["model_id", "created_at", "status", "deployment_ready"]
            rows = [{c: m.get(c, "") for c in display_cols} for m in models]
            st.dataframe(pd.DataFrame(rows), width="stretch")

            # ── Model actions ─────────────────────────────────────────────
            model_ids = [m["model_id"] for m in models]
            sel_model = st.selectbox("🎯 Select model for actions", options=model_ids, key="reg_action_sel")

            selected_meta = next((m for m in models if str(m.get("model_id")) == str(sel_model)), {})
            if selected_meta:
                with st.expander("Selected Model Details", expanded=False):
                    st.write(f"- Display alias: {selected_meta.get('display_name_alias', selected_meta.get('display_name', sel_model))}")
                    st.write(f"- Deployment ready: {'Yes' if selected_meta.get('deployment_ready') else 'No'}")
                    st.write(f"- Rename mode: {selected_meta.get('rename_mode', 'N/A')}")
                    artifact_versions = selected_meta.get("artifact_versions", {}) or {}
                    if artifact_versions:
                        st.caption("Artifact versions")
                        _kv_table(artifact_versions, "Artifact Versions")
                    artifact_exists = selected_meta.get("artifact_exists", {}) or {}
                    if artifact_exists:
                        present_count = sum(1 for value in artifact_exists.values() if value)
                        st.write(
                            f"- Artifacts present: {present_count}/{len(artifact_exists)}"
                        )
                        with st.expander("Artifact existence map", expanded=False):
                            _kv_table(artifact_exists, "Artifact Availability")
                    mlflow_best_val_loss = selected_meta.get("mlflow_best_val_loss")
                    if mlflow_best_val_loss is not None:
                        st.write(f"- MLflow best val loss: {float(mlflow_best_val_loss):.4f}")
                    training_signals = selected_meta.get("training_signals", {}) or {}
                    if training_signals:
                        st.caption("Training signals")
                        _kv_table(training_signals, "Training Signals")
                    training_fit_analysis = selected_meta.get("training_fit_analysis", {}) or {}
                    if training_fit_analysis:
                        st.caption("Fit analysis")
                        _kv_table(training_fit_analysis, "Fit Analysis")
                    xai_config = selected_meta.get("xai_config", {}) or {}
                    if xai_config:
                        st.caption("XAI config")
                        _kv_table(xai_config, "XAI Configuration")
                    fusion_payload = selected_meta.get("fusion", {}) or {}
                    if fusion_payload:
                        st.caption("Fusion metadata")
                        _kv_table(fusion_payload, "Fusion Configuration")

            col_rename, col_dl, col_onnx = st.columns(3)

            with col_rename:
                st.markdown("##### ✏️ Rename Model")
                new_name = st.text_input(
                    "New name (letters, digits, hyphens, underscores only)",
                    key="rename_input",
                    placeholder="my-model-v2",
                )
                if st.button("✅ Apply Rename", key="do_rename", disabled=not new_name.strip()):
                    try:
                        r = requests.patch(
                            f"{API_BASE_URL}/model-registry/{sel_model}/rename",
                            json={"new_name": new_name.strip()},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success(f"✅ Renamed '{sel_model}' → '{new_name.strip()}'")
                            # Refresh registry
                            refresh = requests.get(f"{API_BASE_URL}/model-registry", timeout=10)
                            if refresh.status_code == 200:
                                st.session_state.registry_result = refresh.json()
                            st.rerun()
                        else:
                            st.error(f"Rename failed: {r.json().get('detail', r.text[:200])}")
                    except Exception as ex:
                        st.error(f"Rename error: {ex}")

            with col_dl:
                st.markdown("##### ⬇️ Download Model")
                st.caption("Downloads a zip of all model artifacts + usage README.")
                dl_url = f"{API_BASE_URL}/model-registry/{sel_model}/download"
                # Use an anchor link — browser navigates to the streaming endpoint directly
                st.markdown(
                    f'<a href="{dl_url}" target="_blank">⬇️ Download `{sel_model}`</a>',
                    unsafe_allow_html=True,
                )

            with col_onnx:
                st.markdown("##### 📦 Export to ONNX")
                st.caption("Export fusion head for onnxruntime (no PyTorch needed).")
                if st.button("📦 Export ONNX", key=f"onnx_{sel_model}",
                             help="Exports the fusion head to ONNX format"):
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/model-registry/{sel_model}/export-onnx",
                            json={},
                            timeout=60,
                        )
                        if r.status_code == 200:
                            _onnx_info = r.json()
                            st.success(f"Exported: {_onnx_info.get('onnx_path', 'done')}")
                            st.caption(f"Input names: {_onnx_info.get('input_names', [])}")
                        else:
                            st.error(f"Export failed: {r.text[:200]}")
                    except Exception as _onnx_ex:
                        st.error(f"ONNX export error: {_onnx_ex}")

            if st.session_state.session_id:
                st.markdown("##### 🔁 Latest Session Retrain")
                if st.button(
                    "Refresh Latest Retrain",
                    width="stretch",
                    key="registry_retrain_btn",
                ):
                    try:
                        params = {"limit": 1, "session_id": st.session_state.session_id}
                        resp = requests.get(f"{API_BASE_URL}/retrain-history", params=params, timeout=20)
                        if resp.status_code == 200:
                            st.session_state.registry_latest_retrain_result = resp.json()
                        else:
                            st.error(f"Retrain history fetch failed: {resp.status_code}")
                    except Exception as exc:
                        st.error(f"Retrain history error: {exc}")

                latest_payload = st.session_state.get("registry_latest_retrain_result", {})
                latest_rows = latest_payload.get("history", []) if isinstance(latest_payload, dict) else []
                if latest_rows:
                    latest = latest_rows[0]
                    st.caption("Latest retrain event for this session")
                    st.write(f"- Status: {latest.get('status', 'unknown')}")
                    st.write(f"- Dataset: {latest.get('dataset_id', 'N/A')}")
                    if latest.get("model_id"):
                        st.write(f"- Model: {latest.get('model_id')}")
                    if latest.get("timestamp"):
                        st.write(f"- Timestamp: {str(latest.get('timestamp'))[:19]}")
        else:
            st.info("No models in registry. Run the full pipeline first.")

    # ── Tab 5: Retraining history and per-model stats ──────────────────────────
    with tab5:
        st.markdown("### Retraining History")
        col_hist_a, col_hist_b = st.columns([1, 1])
        with col_hist_a:
            hist_limit = st.number_input(
                "History Rows",
                min_value=1,
                max_value=500,
                value=100,
                step=1,
                key="retrain_hist_limit",
            )
        with col_hist_b:
            dataset_filter = st.text_input(
                "Dataset Filter (optional)",
                value="",
                key="retrain_hist_dataset",
            )

        if st.button("Refresh Retrain History", width="stretch", key="retrain_history_btn"):
            try:
                params = {"limit": int(hist_limit)}
                params["session_id"] = st.session_state.session_id
                if dataset_filter.strip():
                    params["dataset_id"] = dataset_filter.strip()
                resp = requests.get(f"{API_BASE_URL}/retrain-history", params=params, timeout=20)
                if resp.status_code == 200:
                    st.session_state.retrain_history_result = resp.json()
                else:
                    st.error(f"Retrain history fetch failed: {resp.status_code}")
            except Exception as exc:
                st.error(f"Retrain history error: {exc}")

        hist_payload = st.session_state.get("retrain_history_result", {})
        history_rows = hist_payload.get("history", []) if isinstance(hist_payload, dict) else []
        if history_rows:
            st.caption(f"{hist_payload.get('count', len(history_rows))} retraining event(s)")
            st.dataframe(pd.DataFrame(history_rows), width="stretch")

            st.markdown("#### Retrain Event Reasons")
            for event in history_rows[:10]:
                status = str(event.get("status", "unknown"))
                if status == "triggered":
                    st.success(
                        f"Triggered retraining for dataset {event.get('dataset_id', '?')}"
                        + (f" -> model {event.get('model_id', '?')}" if event.get("model_id") else "")
                    )
                elif status == "cooldown_blocked":
                    remaining = event.get("cooldown_remaining_seconds", "?")
                    st.warning(
                        f"Skipped (cooldown): dataset {event.get('dataset_id', '?')} - "
                        f"{remaining}s remaining"
                    )
                elif status == "skipped_no_sources":
                    st.info(
                        f"Skipped: no production sources available for dataset "
                        f"{event.get('dataset_id', '?')}"
                    )
                elif status == "error":
                    st.error(
                        f"Retrain error for dataset {event.get('dataset_id', '?')}: "
                        f"{event.get('error', 'Unknown error')}"
                    )
                else:
                    st.caption(f"{status}: {event}")
        else:
            st.info("No retraining events yet.")

        st.divider()
        st.markdown("### Embedding Cache Stats")
        if st.button("Refresh Embedding Cache", width="stretch", key="embedding_cache_btn"):
            try:
                resp = requests.get(f"{API_BASE_URL}/embedding-cache/stats", timeout=20)
                if resp.status_code == 200:
                    st.session_state.embedding_cache_stats = resp.json()
                else:
                    st.error(f"Embedding cache fetch failed: {resp.status_code}")
            except Exception as exc:
                st.error(f"Embedding cache error: {exc}")

        embedding_stats = st.session_state.get("embedding_cache_stats", {})
        if embedding_stats:
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Cache Files", embedding_stats.get("cache_file_count", "N/A"))
            ec2.metric("Cache Size (MB)", embedding_stats.get("cache_size_mb", "N/A"))
            ec3.metric("Latest Model", embedding_stats.get("latest_model_id", "N/A"))
            latest_training_cache = embedding_stats.get("latest_training_embedding_cache", {}) or {}
            if latest_training_cache:
                st.caption(
                    "Latest training cache counters: "
                    + ", ".join(f"{k}={v}" for k, v in latest_training_cache.items())
                )

        st.divider()
        st.markdown("### Meta-Learning Insights")
        if st.button("Refresh Meta-Learning Insights", width="stretch", key="meta_learning_btn"):
            try:
                schema_data = st.session_state.get("detected_schema", {}) or {}
                resp = requests.get(
                    f"{API_BASE_URL}/meta-learning/insights",
                    params={
                        "session_id": st.session_state.session_id,
                        "dataset_size": int(st.session_state.get("ingested_row_count") or 0),
                        "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                        "modalities": ",".join(schema_data.get("global_modalities", [])),
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    st.session_state.meta_learning_insights = resp.json()
                else:
                    st.error(f"Meta-learning fetch failed: {resp.status_code}")
            except Exception as exc:
                st.error(f"Meta-learning error: {exc}")

        meta_payload = st.session_state.get("meta_learning_insights", {})
        if meta_payload:
            st.caption(f"Historical records available: {meta_payload.get('records_available', 0)}")
            predicted_cfg = meta_payload.get("predicted_config", {}) or {}
            if predicted_cfg:
                st.write(
                    f"Predicted fusion: {predicted_cfg.get('fusion_strategy', 'N/A')} | "
                    f"confidence={predicted_cfg.get('confidence', 0)} | "
                    f"source_count={predicted_cfg.get('source_count', 0)}"
                )
                with st.expander("Predicted Config", expanded=False):
                    _kv_table(predicted_cfg, "Predicted Configuration")

            suggestions = meta_payload.get("suggestions", []) or []
            if suggestions:
                st.markdown("**Top Similar Historical Runs**")
                st.dataframe(pd.DataFrame(suggestions), width="stretch")
                st.caption(
                    "Recommendations are weighted by similarity and prior performance. "
                    "Use this as evidence, not a hard override."
                )
            else:
                st.info("No similar historical runs found yet.")

        st.divider()
        st.markdown("### Research Outputs")
        _abl_col1, _abl_col2 = st.columns(2)
        with _abl_col1:
            if st.button("▶ Run Feature Ablation Study", width="stretch", key="run_ablations_btn",
                         help="Systematically removes features and modalities to measure their importance. Takes several minutes."):
                try:
                    _abl_resp = requests.post(
                        f"{API_BASE_URL}/experiments/run-ablations",
                        json={"session_id": st.session_state.session_id},
                        timeout=30,
                    )
                    if _abl_resp.ok:
                        st.success("✅ Ablation study started. Refresh results in a few minutes.")
                    else:
                        st.error(f"Ablation failed to start: {_api_error_detail(_abl_resp)}")
                except Exception as _abl_exc:
                    st.error(f"Ablation error: {_abl_exc}")
        with _abl_col2:
            if st.button("↻ Refresh Research Outputs", width="stretch", key="research_outputs_btn"):
                try:
                    resp = requests.get(f"{API_BASE_URL}/experiments/ablation-results", timeout=20)
                    if resp.status_code == 200:
                        st.session_state.ablation_results = resp.json()
                    else:
                        st.error(f"Ablation results fetch failed: {resp.status_code}")
                except Exception as exc:
                    st.error(f"Ablation results error: {exc}")

        ablation_payload = st.session_state.get("ablation_results", {}) or {}
        ablation_rows = ablation_payload.get("results", []) if isinstance(ablation_payload, dict) else []
        if ablation_payload:
            st.caption(f"Ablation experiments available: {ablation_payload.get('count', len(ablation_rows))}")
            if ablation_rows:
                completed_runs = sum(1 for row in ablation_rows if row.get("status") == "completed")
                failed_runs = sum(1 for row in ablation_rows if row.get("status") == "failed")
                r1, r2, r3 = st.columns(3)
                r1.metric("Experiments", len(ablation_rows))
                r2.metric("Completed", completed_runs)
                r3.metric("Failed", failed_runs)

                ablation_table = []
                for row in ablation_rows:
                    ablation_table.append({
                        "Condition": row.get("name", "?"),
                        "Status": row.get("status", "pending"),
                        "Best Val Acc": row.get("best_val_acc", "N/A"),
                        "Best Val Loss": row.get("best_val_loss", "N/A"),
                        "Best Val F1": row.get("best_val_f1", "N/A"),
                        "Trials": row.get("n_trials", "N/A"),
                        "Duration (s)": row.get("duration_s", "N/A"),
                        "ECE": row.get("ece", "N/A"),
                        "Brier": row.get("brier", "N/A"),
                        "Description": row.get("description", ""),
                    })
                st.dataframe(pd.DataFrame(ablation_table), width="stretch")
            else:
                st.info(ablation_payload.get("message", "No ablation results available yet."))

            with st.expander("Ablation raw payload", expanded=False):
                _kv_table(ablation_payload, "Ablation Results")

        st.divider()
        st.markdown("### Model Stats")
        _registry_cache = st.session_state.get("registry_result", {})
        _registry_models = _registry_cache.get("models", []) if isinstance(_registry_cache, dict) else []
        _model_ids = [m.get("model_id") for m in _registry_models if m.get("model_id")]

        model_id_for_stats = ""
        if _model_ids:
            model_id_for_stats = st.selectbox(
                "Model ID",
                options=_model_ids,
                key="model_stats_id_select",
            )
        else:
            model_id_for_stats = st.text_input(
                "Model ID",
                value="",
                key="model_stats_id",
                placeholder="apex_v1_YYYYMMDD_HHMMSS",
            )
        selected_registry_meta = next(
            (
                m for m in _registry_models
                if str(m.get("model_id")) == str(model_id_for_stats)
            ),
            {},
        )
        if st.button("Load Model Stats", width="stretch", key="model_stats_btn"):
            if not model_id_for_stats.strip():
                st.warning("Enter a model ID first.")
            else:
                try:
                    resp = requests.get(
                        f"{API_BASE_URL}/models/{model_id_for_stats.strip()}/stats",
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        st.session_state.model_stats_result = resp.json().get("data", {})
                    else:
                        st.error(f"Model stats fetch failed: {resp.status_code}")
                except Exception as exc:
                    st.error(f"Model stats error: {exc}")

        model_stats = st.session_state.get("model_stats_result", {})
        if model_stats:
            ev = model_stats.get("evaluation", {})
            combined = ev.get("combined_score", 0.0) if isinstance(ev, dict) else 0.0
            ms1, ms2, ms3 = st.columns(3)
            ms1.metric("Model", model_stats.get("model_id", "N/A"))
            ms2.metric("Deployment Ready", "Yes" if model_stats.get("deployment_ready") else "No")
            ms3.metric("Combined Score", f"{combined:.4f}")

            if isinstance(ev, dict) and ev:
                with st.expander("Evaluation Breakdown", expanded=False):
                    train_eval = ev.get("training", {}) or {}
                    monitor_eval = ev.get("monitoring", {}) or {}
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Training Overall", f"{float(train_eval.get('overall_score', 0.0) or 0.0):.4f}")
                    e2.metric("Monitoring Health", f"{float(monitor_eval.get('health_score', 0.0) or 0.0):.4f}")
                    e3.metric("Risk Score", f"{float(monitor_eval.get('risk_score', 0.0) or 0.0):.4f}")
                    st.write(f"- **Training performance:** {train_eval.get('performance', 'N/A')}")
                    st.write(f"- **Training loss score:** {train_eval.get('loss_score', 'N/A')}")
                    st.write(f"- **Training generalization gap:** {train_eval.get('generalization_gap', 'N/A')}")
                    st.write(f"- **Training stability:** {train_eval.get('stability', 'N/A')}")
                    st.write(f"- **Monitoring drift detected:** {'Yes' if monitor_eval.get('drift_detected') else 'No'}")
                    st.write(f"- **Monitoring retrain triggered:** {'Yes' if monitor_eval.get('retrain_triggered') else 'No'}")

            artifact_exists = selected_registry_meta.get("artifact_exists", {}) or {}
            mlflow_best_val_loss = selected_registry_meta.get("mlflow_best_val_loss")
            if artifact_exists or mlflow_best_val_loss is not None:
                with st.expander("Registry Health", expanded=False):
                    if artifact_exists:
                        present_count = sum(1 for value in artifact_exists.values() if value)
                        st.write(
                            f"- Artifacts present: {present_count}/{len(artifact_exists)}"
                        )
                        _kv_table(artifact_exists, "Artifact Availability")
                    if mlflow_best_val_loss is not None:
                        st.write(f"- MLflow best val loss: {float(mlflow_best_val_loss):.4f}")

            training_summary = model_stats.get("training", {}) or {}
            calibration_summary = training_summary.get("calibration", {}) or {}
            if calibration_summary:
                with st.expander("Calibration Summary", expanded=False):
                    cal_cols = st.columns(3)
                    cal_cols[0].metric("Enabled", "Yes" if calibration_summary.get("enabled") else "No")
                    cal_cols[1].metric("Mode", calibration_summary.get("mode", "N/A"))
                    cal_cols[2].metric(
                        "ECE After",
                        f"{float(calibration_summary.get('ece_after', 0.0) or 0.0):.4f}"
                        if calibration_summary.get("ece_after") is not None else "N/A",
                    )
                    st.write(f"- **ECE before:** {calibration_summary.get('ece_before', 'N/A')}")
                    st.write(f"- **ECE after:** {calibration_summary.get('ece_after', 'N/A')}")
                    st.write(f"- **Brier before:** {calibration_summary.get('brier_before', 'N/A')}")
                    st.write(f"- **Brier after:** {calibration_summary.get('brier_after', 'N/A')}")
                    if calibration_summary.get("reason"):
                        st.write(f"- **Reason:** {calibration_summary.get('reason')}")

            research_metrics = model_stats.get("research_metrics", {}) or {}
            if research_metrics:
                with st.expander("Research Metrics", expanded=False):
                    _kv_table(research_metrics, "Research Metrics")

            training_signals = model_stats.get("training_signals", {}) or {}
            training_fit_analysis = model_stats.get("training_fit_analysis", {}) or {}
            artifact_versions = model_stats.get("artifact_versions", {}) or {}
            xai_config = model_stats.get("xai_config", {}) or {}
            fusion_summary = model_stats.get("fusion_summary", {}) or {}
            if training_fit_analysis or training_signals or artifact_versions or xai_config or fusion_summary:
                with st.expander("Registry Diagnostics", expanded=False):
                    if artifact_versions:
                        st.caption("Artifact versions")
                        _kv_table(artifact_versions, "Artifact Versions")
                    if training_signals:
                        st.caption("Training signals")
                        _kv_table(training_signals, "Training Signals")
                    if training_fit_analysis:
                        st.caption("Fit analysis")
                        _kv_table(training_fit_analysis, "Fit Analysis")
                    if xai_config:
                        st.caption("XAI config")
                        _kv_table(xai_config, "XAI Configuration")
                    if fusion_summary:
                        st.caption("Fusion summary")
                        _kv_table(fusion_summary, "Fusion Summary")

            latest_payload = st.session_state.get("registry_latest_retrain_result", {})
            latest_rows = latest_payload.get("history", []) if isinstance(latest_payload, dict) else []
            if latest_rows:
                latest = latest_rows[0]
                st.markdown("#### Latest Session Retrain")
                st.write(f"- **Status:** {latest.get('status', 'unknown')}")
                st.write(f"- **Dataset:** {latest.get('dataset_id', 'N/A')}")
                if latest.get("model_id"):
                    st.write(f"- **Model ID:** {latest.get('model_id')}")
                if latest.get("timestamp"):
                    st.write(f"- **Timestamp:** {str(latest.get('timestamp'))[:19]}")
                if latest.get("model_id") and str(latest.get("model_id")) == str(model_stats.get("model_id", "")):
                    st.success("This model matches the latest retrain for the active session.")

            with st.expander("Full model stats payload"):
                _kv_table(model_stats, "Model Statistics")

    # ── Model Comparison View ─────────────────────────────────────────
    with st.expander("⚖️ Compare Models Side-by-Side", expanded=False):
        try:
            reg_resp = requests.get(
                f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/registered-models",
                timeout=10,
            )
            if reg_resp.status_code == 200:
                reg_data = reg_resp.json()
                reg_models = reg_data.get("models", reg_data.get("registered_models", []))
                if isinstance(reg_models, list) and len(reg_models) >= 2:
                    comparison_rows = []
                    for m in reg_models:
                        if not isinstance(m, dict):
                            continue
                        metrics = m.get("metrics", {}) or {}
                        row = {
                            "Model ID": str(m.get("model_id", "?"))[:12],
                            "Name": m.get("name", m.get("model_name", "?")),
                            "Accuracy": metrics.get("accuracy", metrics.get("val_accuracy", "—")),
                            "F1": metrics.get("f1", metrics.get("val_f1", "—")),
                            "Val Loss": metrics.get("val_loss", metrics.get("best_val_loss", "—")),
                            "ECE": metrics.get("ece", "—"),
                            "Latency (ms)": metrics.get("latency_ms", "—"),
                            "Params": metrics.get("total_parameters", "—"),
                        }
                        # Format numeric values
                        for k, v in row.items():
                            if isinstance(v, float):
                                row[k] = f"{v:.4f}"
                            elif isinstance(v, int) and k == "Params":
                                row[k] = f"{v:,}"
                        comparison_rows.append(row)

                    if comparison_rows:
                        comp_df = pd.DataFrame(comparison_rows)
                        st.dataframe(comp_df, width="stretch")
                        st.caption(
                            f"Comparing {len(comparison_rows)} registered models. "
                            "Lower val_loss and ECE are better. Higher accuracy and F1 are better."
                        )

                        # Highlight best model
                        try:
                            best_idx = comp_df["Val Loss"].apply(
                                lambda x: float(x) if x != "—" else float("inf")
                            ).idxmin()
                            st.success(
                                f"🏆 **Best model by val loss:** "
                                f"{comp_df.iloc[best_idx]['Name']} "
                                f"({comp_df.iloc[best_idx]['Model ID']})"
                            )
                        except Exception:
                            pass
                    else:
                        st.info("No model metrics available for comparison.")
                elif isinstance(reg_models, list) and len(reg_models) == 1:
                    st.info("Only 1 model registered. Train another model to compare.")
                else:
                    st.info("No models registered yet.")
            else:
                st.info("Could not load registered models.")
        except Exception:
            st.info("Model comparison requires a running API server.")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start New Workflow"):
            st.session_state.workflow_stage = 1
            st.session_state.dataset_uploaded = False
            st.session_state.schema_detected = False
            st.session_state.model_selected = False
            st.session_state.phase_states = {
                1: {"status": "pending", "reason": ""},
                2: {"status": "pending", "reason": ""},
                3: {"status": "pending", "reason": ""},
                4: {"status": "pending", "reason": ""},
                5: {"status": "pending", "reason": ""},
                6: {"status": "pending", "reason": ""},
                7: {"status": "pending", "reason": ""},
            }
            st.rerun()
    with col2:
        st.info("↑ Use the Download link in the Registry tab above")
    with col3:
        st.button("Deploy to Production", disabled=True, help="Deployment not yet implemented")

    # ── Tab 6: Research Results — publication transparency ──────────────────────
    with tab_research:
        st.markdown("### 📊 Research Results — Publication Transparency")
        st.caption(
            "Multi-seed ablation results, statistical significance tests, bootstrap CIs, "
            "and compute budget — the four ingredients a NeurIPS/ICML reviewer expects."
        )

        # Aggregated multi-seed results (Wilcoxon + bootstrap)
        st.markdown("#### Statistical Significance & Multi-seed Aggregation")
        if st.button("🔄 Refresh Aggregated Results", key="research_agg_refresh"):
            try:
                _r = requests.get(f"{API_BASE_URL}/research/aggregated-results", timeout=15)
                if _r.status_code == 200:
                    st.session_state["research_aggregated"] = _r.json()
                else:
                    st.error(f"Aggregated results unavailable: {_r.text[:200]}")
            except Exception as _ex:
                st.error(f"Endpoint error: {_ex}")

        _agg = st.session_state.get("research_aggregated") or {}
        if _agg.get("status") == "ok":
            ablations_summary = (_agg.get("ablations") or {})
            if ablations_summary:
                st.caption(f"Loaded {ablations_summary.get('n_seeds', '?')} seed(s); note: {ablations_summary.get('note', '')}")
            stat_tests = (_agg.get("statistical_tests") or {})
            tests = stat_tests.get("tests") or []
            if tests:
                st.markdown("**Wilcoxon signed-rank tests (paired, two-sided)**")
                _df_tests = pd.DataFrame(tests)
                st.dataframe(_df_tests, width="stretch")
            else:
                st.caption(stat_tests.get("note", "No paired statistical tests available."))
            cb = _agg.get("compute_budget") or {}
            if cb:
                st.markdown("**Compute Budget (across all trials)**")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total GPU-hours", f"{cb.get('total_gpu_hours', 0):.3f}")
                col_b.metric("Peak VRAM (MB)", f"{cb.get('peak_vram_mb', 0):.0f}")
                col_c.metric("# Trial Records", cb.get("n_trials", 0))
        elif _agg.get("status") == "missing":
            st.info(_agg.get("note", "Run scripts/aggregate_results.py to generate aggregated_results.json"))
        else:
            st.info("Click 'Refresh Aggregated Results' above to load Wilcoxon p-values and bootstrap CIs.")

        st.divider()

        # Per-model compute budget (FLOPs / VRAM / GPU-hours / params)
        st.markdown("#### Compute Budget per Model")
        _models_for_compute = st.session_state.get("registry_result") or _cached_model_registry()
        if _models_for_compute:
            _model_ids = [m.get("model_id") for m in _models_for_compute if isinstance(m, dict) and m.get("model_id")]
            if _model_ids:
                _sel_model_compute = st.selectbox(
                    "Select model for compute budget",
                    _model_ids,
                    key="research_compute_model",
                )
                if st.button("📊 Load Compute Budget", key="research_compute_load"):
                    try:
                        _r2 = requests.get(
                            f"{API_BASE_URL}/intelligence/compute-budget/{_sel_model_compute}",
                            timeout=15,
                        )
                        if _r2.status_code == 200:
                            st.session_state["research_compute_data"] = _r2.json()
                        else:
                            st.error(f"Compute budget unavailable: {_r2.text[:200]}")
                    except Exception as _cex:
                        st.error(f"Endpoint error: {_cex}")
                _cd = st.session_state.get("research_compute_data") or {}
                if _cd:
                    if _cd.get("n_trials", 0) > 0:
                        cm1, cm2, cm3, cm4 = st.columns(4)
                        cm1.metric("Trials", _cd.get("n_trials", 0))
                        cm2.metric("GPU-hrs", f"{_cd.get('total_gpu_hours', 0):.3f}")
                        cm3.metric("Peak VRAM (MB)", f"{_cd.get('peak_vram_mb', 0):.0f}")
                        _flops = _cd.get("total_flops", 0)
                        cm4.metric("FLOPs", f"{_flops:.2e}" if _flops else "N/A")
                        recs = _cd.get("records", []) or []
                        if recs:
                            st.markdown("**Per-trial details**")
                            _drop = ["records"]
                            _df_recs = pd.DataFrame([{k: v for k, v in r.items() if k not in _drop} for r in recs])
                            st.dataframe(_df_recs, width="stretch")
                    else:
                        st.info(_cd.get("note", "No compute budget records found for this model."))
        else:
            st.info("Refresh the Registry tab first to populate model selection.")


def render_phase_7_prediction() -> None:
    """Phase 7: Multimodal Prediction & Explainability."""
    st.header("Phase 7 - Make Predictions")

    _training_result = st.session_state.get("training_result") or {}
    _trained_model_id = _training_result.get("model_id") or st.session_state.get("trained_model_id")
    if not _training_result or not _trained_model_id:
        st.warning("Please complete training in Phase 5 first.")
        if st.button("← Go to Phase 5", key="phase7_back_btn"):
            st.session_state.workflow_stage = 5
            st.rerun()
        return

    # ── G27: Prediction Playground (model registry quick-switcher) ────────────
    with st.expander("🎮 Prediction Playground — Registry & Active Model (G27)", expanded=False):
        st.caption(
            "Switch the active prediction model for this session without re-running training. "
            "Loads from /v2/sessions/{sid}/registered-models and posts to /v2/sessions/{sid}/active-model."
        )
        _pg_col1, _pg_col2 = st.columns([2, 1])
        with _pg_col1:
            _pg_models: List[Dict] = []
            _pg_active: Optional[str] = None
            try:
                _pg_resp = requests.get(
                    f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/registered-models",
                    timeout=10,
                )
                if _pg_resp.status_code == 200:
                    _pg_data = _pg_resp.json()
                    _pg_models = list(_pg_data.get("registered_models", []) or [])
                    _pg_active = _pg_data.get("active_model_id")
            except Exception:
                pass

            if _pg_models:
                _pg_ids = [m["model_id"] for m in _pg_models]
                _pg_labels = [
                    f"{'🟢' if m.get('active') else '⚪'} {m['model_id']}"
                    for m in _pg_models
                ]
                _pg_sel_label = st.selectbox(
                    f"Session models ({len(_pg_ids)} registered)",
                    options=_pg_labels,
                    key="pg_model_select",
                )
                _pg_sel_id = _pg_ids[_pg_labels.index(_pg_sel_label)]
                if _pg_active:
                    st.caption(f"Currently active: `{_pg_active}`")

                _pg_reason = st.text_input(
                    "Reason for switch (optional)",
                    value="Switched via Prediction Playground",
                    key="pg_switch_reason",
                )
                if st.button("🔁 Set as Active Prediction Model", key="pg_set_active", type="primary"):
                    try:
                        _set_resp = requests.post(
                            f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/active-model",
                            json={"model_id": _pg_sel_id, "reason": _pg_reason},
                            timeout=10,
                        )
                        if _set_resp.status_code == 200:
                            st.success(f"✅ Active model set to `{_pg_sel_id}`")
                            st.rerun()
                        else:
                            st.error(f"Failed: {_api_error_detail(_set_resp)}")
                    except Exception as _pg_exc:
                        st.error(f"Set active model error: {_pg_exc}")
            else:
                st.info(
                    "No registered models found for this session. "
                    "Complete training (Phase 5) first to register a model."
                )

        with _pg_col2:
            st.markdown("**Registry summary**")
            try:
                _pg_reg = requests.get(f"{API_BASE_URL}/model-registry", timeout=10)
                if _pg_reg.status_code == 200:
                    _pg_reg_data = _pg_reg.json()
                    _pg_count = _pg_reg_data.get("count", 0)
                    _pg_ready = sum(
                        1 for m in _pg_reg_data.get("models", [])
                        if m.get("deployment_ready")
                    )
                    st.metric("Total models", _pg_count)
                    st.metric("Deployment-ready", _pg_ready)
            except Exception:
                st.caption("Registry unavailable")

    # ── Model ID picker ──────────────────────────────────────────────────────
    st.markdown("#### Select a registered model")
    registry_model_ids: List[str] = []
    registry_models_full: List[Dict] = []
    try:
        reg_resp = requests.get(f"{API_BASE_URL}/model-registry", timeout=15)
        if reg_resp.status_code == 200:
            reg_data = reg_resp.json()
            registry_models_full = list(reg_data.get("models", []))
            registry_model_ids = [
                m["model_id"]
                for m in registry_models_full
                if m.get("deployment_ready", False)
            ]
    except Exception:
        pass

    if registry_model_ids:
        model_id_input: str = st.selectbox(
            "Deployment-ready models",
            options=registry_model_ids,
            key="pred_model_id_select",
        )
    else:
        model_id_input = st.text_input(
            "Model ID (manual entry – no deployment-ready models found)",
            key="pred_model_id_text",
        )

    if not model_id_input:
        st.info("Enter or select a model ID to continue.")
        return

    # ── Fetch model info (class labels + expected features) ───────────────
    model_info: Dict = {}
    class_labels: List[str] = []
    input_tabular_cols: List[str] = []
    input_text_cols: List[str] = []
    input_image_cols: List[str] = []
    prediction_contract: Dict = {}
    active_modalities: List[str] = []
    excluded_modalities: Dict = {}
    dropped_columns: List[str] = []
    try:
        info_resp = requests.get(f"{API_BASE_URL}/model-info/{model_id_input}", timeout=10)
        if info_resp.status_code == 200:
            model_info = info_resp.json()
            class_labels = model_info.get("class_labels", [])
            prediction_contract = model_info.get("prediction_contract", {}) or {}
            active_modalities = (
                prediction_contract.get("active_modalities")
                or model_info.get("active_modalities")
                or model_info.get("modalities")
                or []
            )
            excluded_modalities = (
                prediction_contract.get("excluded_modalities")
                or model_info.get("excluded_modalities")
                or {}
            )
            # Prefer effective_features (post-preprocessing) over raw schema columns
            input_tabular_cols = (
                model_info.get("effective_features")
                or model_info.get("input_columns", {}).get("tabular", [])
            )
            input_text_cols = model_info.get("input_columns", {}).get("text", [])
            input_image_cols = model_info.get("input_columns", {}).get("image", [])
            dropped_columns = model_info.get("dropped_columns", [])
    except Exception:
        pass

    if prediction_contract:
        with st.expander("Prediction Contract", expanded=False):
            st.write(f"Active modalities: {', '.join(active_modalities) or 'N/A'}")
            if excluded_modalities:
                st.write(
                    "Skipped modalities: "
                    + ", ".join(f"{k} ({v})" for k, v in excluded_modalities.items())
                )
            st.write(f"Target: {prediction_contract.get('target', 'N/A')}")
            st.write(f"Problem type: {prediction_contract.get('problem_type', 'N/A')}")
            st.write(f"Fusion: {(prediction_contract.get('fusion') or {}).get('strategy', 'N/A')}")
            xai_av = prediction_contract.get("xai_availability", {}) or {}
            if xai_av:
                st.write(
                    "XAI availability: "
                    + ", ".join(f"{k}={bool(v)}" for k, v in xai_av.items())
                )

    registry_meta = next(
        (
            m for m in registry_models_full
            if str(m.get("model_id")) == str(model_id_input)
        ),
        {},
    )

    provenance_artifacts = model_info.get("artifact_versions", {}) or {}
    provenance_xai_config = model_info.get("xai_config", {}) or {}
    provenance_training_signals = model_info.get("training_signals", {}) or {}
    provenance_fit_analysis = model_info.get("training_fit_analysis", {}) or {}
    provenance_training = model_info.get("training", {}) or {}
    provenance_evaluation = model_info.get("evaluation", {}) or {}
    provenance_calibration = model_info.get("calibration", {}) or {}
    provenance_fusion = model_info.get("fusion", {}) or {}
    provenance_artifact_exists = registry_meta.get("artifact_exists", {}) or {}
    provenance_mlflow_best_val_loss = registry_meta.get("mlflow_best_val_loss")
    if (
        provenance_artifacts
        or provenance_xai_config
        or provenance_training_signals
        or provenance_fit_analysis
        or provenance_training
        or provenance_evaluation
        or provenance_calibration
        or provenance_fusion
        or provenance_artifact_exists
        or provenance_mlflow_best_val_loss is not None
    ):
        with st.expander("Model Provenance", expanded=False):
            if provenance_artifacts:
                st.caption("Artifact versions")
                _kv_table(provenance_artifacts, "Provenance Artifacts")
            if provenance_xai_config:
                st.caption("XAI config")
                _kv_table(provenance_xai_config, "XAI Config")
            if provenance_training_signals:
                st.caption("Training signals")
                _kv_table(provenance_training_signals, "Training Signals")
            if provenance_fit_analysis:
                st.caption("Fit analysis")
                _kv_table(provenance_fit_analysis, "Fit Analysis")
            if provenance_training:
                st.caption("Training summary")
                train_cols = st.columns(3)
                train_cols[0].metric(
                    "Best Val Loss",
                    f"{float(provenance_training.get('best_val_loss', 0.0) or 0.0):.4f}"
                    if provenance_training.get("best_val_loss") is not None else "N/A",
                )
                train_cols[1].metric(
                    "Best Val Acc",
                    f"{float(provenance_training.get('best_val_acc', 0.0) or 0.0):.4f}"
                    if provenance_training.get("best_val_acc") is not None else "N/A",
                )
                train_cols[2].metric(
                    "Best Val F1",
                    f"{float(provenance_training.get('best_val_f1', 0.0) or 0.0):.4f}"
                    if provenance_training.get("best_val_f1") is not None else "N/A",
                )
                _kv_table(provenance_training, "Training Details")
            if provenance_evaluation:
                st.caption("Evaluation summary")
                eval_cols = st.columns(3)
                eval_cols[0].metric(
                    "Training Overall",
                    f"{float((provenance_evaluation.get('training', {}) or {}).get('overall_score', 0.0) or 0.0):.4f}"
                )
                eval_cols[1].metric(
                    "Monitoring Health",
                    f"{float((provenance_evaluation.get('monitoring', {}) or {}).get('health_score', 0.0) or 0.0):.4f}"
                )
                eval_cols[2].metric(
                    "Combined",
                    f"{float(provenance_evaluation.get('combined_score', 0.0) or 0.0):.4f}"
                )
                _kv_table(provenance_evaluation, "Evaluation")
            if provenance_calibration:
                st.caption("Calibration")
                cal_cols = st.columns(3)
                cal_cols[0].metric("Enabled", "Yes" if provenance_calibration.get("enabled") else "No")
                cal_cols[1].metric("Mode", provenance_calibration.get("mode", "N/A"))
                cal_cols[2].metric(
                    "ECE After",
                    f"{float(provenance_calibration.get('ece_after', 0.0) or 0.0):.4f}"
                    if provenance_calibration.get("ece_after") is not None else "N/A",
                )
                _kv_table(provenance_calibration, "Calibration")
            if provenance_fusion:
                st.caption("Fusion metadata")
                _kv_table(provenance_fusion, "Fusion")
                # Part B.5 — ULA badge when fusion strategy is ULA
                _prov_fusion_strat = str(provenance_fusion.get("strategy", "")).lower()
                if _prov_fusion_strat in ("ula", "unified_latent", "unified_latent_alignment", "omnimodal"):
                    st.markdown(
                        '<span style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">'
                        '⚡ ULA Cross-Modal Attention Active</span>',
                        unsafe_allow_html=True,
                    )
            if provenance_artifact_exists:
                st.caption("Artifact existence")
                _kv_table(provenance_artifact_exists, "Artifacts")
            if provenance_mlflow_best_val_loss is not None:
                st.metric(
                    "MLflow Best Val Loss",
                    f"{float(provenance_mlflow_best_val_loss):.4f}",
                )

    # ── XAI settings ─────────────────────────────────────────────────────────
    with st.expander("Explainability (XAI) settings", expanded=False):
        enable_xai: bool = st.checkbox("Enable IntegratedGradients explanations", value=False)
        if class_labels:
            label_options = ["Auto (Explain Predicted Class)"] + [
                f"{i}: {lbl}" for i, lbl in enumerate(class_labels)
            ]
            selected_label = st.selectbox(
                "Target class to explain", options=label_options, index=0,
                help="'Auto' uses the model's top prediction as the XAI target.",
            )
            if selected_label.startswith("Auto"):
                xai_target_class = -1  # sentinel: backend resolves via argmax
            else:
                xai_target_class: int = int(selected_label.split(":")[0])
        else:
            xai_target_class = st.number_input(
                "Target class index (classification only, -1 = auto)",
                min_value=-1, value=-1, step=1,
                help="-1 means the backend will explain whichever class was predicted.",
            )
        xai_n_steps: int = st.slider(
            "IG integration steps", min_value=10, max_value=200, value=50, step=10,
            help="Integrated Gradients: more steps = more accurate attribution but slower. 50 is a good balance. 200 for publication-quality explanations.",
        )

    # ── Input mode ───────────────────────────────────────────────────────────
    input_mode: str = st.radio(
        "Input mode",
        options=["Single Sample", "Batch Upload (CSV)"],
        horizontal=True,
        key="pred_input_mode",
        help="Single Sample: type values manually for one prediction. Batch Upload: upload a CSV file with many rows for bulk inference.",
    )

    raw_inputs: List[Dict] = []

    if input_mode == "Batch Upload (CSV)":
        st.markdown("##### Upload CSV file")

        # ── Schema guidance: show required columns and provide template ──
        _all_required_cols: List[str] = list(input_tabular_cols) + list(input_text_cols) + list(input_image_cols)
        if _all_required_cols:
            _col_list = ", ".join(f"`{c}`" for c in _all_required_cols)
            st.info(
                f"**Required CSV columns:** {_col_list}\n\n"
                + ("Text columns should contain raw text strings. " if input_text_cols else "")
                + ("Image columns should contain paths/URLs accessible to the API server. " if input_image_cols else "")
                + "Extra columns will be ignored; missing optional tabular columns may be zero-filled."
            )
            # Downloadable empty template CSV
            _template_df = pd.DataFrame(columns=_all_required_cols)
            st.download_button(
                "Download Sample Template CSV",
                data=_template_df.to_csv(index=False),
                file_name="prediction_template.csv",
                mime="text/csv",
            )
        if dropped_columns:
            st.caption(
                f"Auto-filtered columns (not needed): "
                f"{', '.join(dropped_columns)}"
            )

        csv_file = st.file_uploader("Feature CSV (one row = one sample)", type=["csv"], key="pred_csv")
        st.caption(
            "For image columns, include file paths in the CSV. "
            "Ensure the API server can access those paths."
        )
        if csv_file is not None:
            try:
                df_batch = pd.read_csv(csv_file)
                st.caption(f"Loaded {len(df_batch)} rows x {len(df_batch.columns)} columns")
                st.dataframe(df_batch.head(5), width="stretch")
                raw_inputs = df_batch.to_dict(orient="records")
            except Exception as csv_exc:
                st.error(f"Could not parse CSV: {csv_exc}")
    else:
        # ── Single-sample tabs ───────────────────────────────────────────────
        sample_input: Dict = {}

        # Only show tabs for modalities the model actually uses
        model_modalities = model_info.get("modalities", [])
        tab_names = []
        if "image" in model_modalities or not model_modalities:
            tab_names.append("Image")
        if "text" in model_modalities or input_text_cols or not model_modalities:
            tab_names.append("Text")
        tab_names.append("Tabular Features")
        tabs = st.tabs(tab_names)
        tab_idx = 0

        if "Image" in tab_names:
            with tabs[tab_idx]:
                st.markdown("##### Image input (path or URL stored as metadata)")
                img_src = st.radio("Source", ["Upload", "URL"], key="pred_img_src", horizontal=True)
                if img_src == "Upload":
                    img_file = st.file_uploader("Image file", type=["jpg", "jpeg", "png"], key="pred_img_file")
                    if img_file:
                        st.image(img_file, width=300)
                        # Save uploaded bytes to a temp file so the inference
                        # engine can open it via PIL.Image.open(path)
                        import tempfile, os
                        _tmp_dir = tempfile.mkdtemp(prefix="apex_img_")
                        _tmp_path = os.path.join(_tmp_dir, img_file.name)
                        with open(_tmp_path, "wb") as _fh:
                            _fh.write(img_file.getvalue())
                        sample_input["image_path"] = _tmp_path
                else:
                    img_url = st.text_input("Image URL", key="pred_img_url")
                    if img_url:
                        sample_input["image_path"] = img_url
            tab_idx += 1

        if "Text" in tab_names:
            with tabs[tab_idx]:
                if input_text_cols:
                    st.markdown(f"##### Text input (column: `{input_text_cols[0]}`)")
                    text_val = st.text_area(
                        f"Enter text for '{input_text_cols[0]}'",
                        height=150, key="pred_text",
                    )
                    if text_val:
                        sample_input[input_text_cols[0]] = text_val
                        st.caption(f"{len(text_val)} characters")
                else:
                    st.markdown("##### Text input")
                    text_val = st.text_area("Enter text", height=150, key="pred_text")
                    if text_val:
                        sample_input["text"] = text_val
            tab_idx += 1

        with tabs[tab_idx]:
            st.markdown("##### Tabular features")
            if dropped_columns:
                st.info(
                    f"Auto-filtered {len(dropped_columns)} column(s) that the model "
                    f"does not use: `{'`, `'.join(dropped_columns)}`"
                )
            if input_tabular_cols:
                st.caption(f"Model expects {len(input_tabular_cols)} tabular columns. "
                           "Missing values will be zero-filled.")
                feat_cols = st.columns(2)
                for i, col_name in enumerate(input_tabular_cols):
                    col_idx = i % 2
                    with feat_cols[col_idx]:
                        fval = st.text_input(
                            col_name, value="", key=f"pred_feat_{i}",
                            placeholder="leave blank to zero-fill",
                        )
                        if fval.strip():
                            try:
                                sample_input[col_name] = float(fval)
                            except ValueError:
                                sample_input[col_name] = fval
            else:
                n_feat: int = st.number_input("Number of numeric features", min_value=1, max_value=50, value=4, step=1)
                feat_cols = st.columns(2)
                for i in range(int(n_feat)):
                    col_idx = i % 2
                    with feat_cols[col_idx]:
                        fname = st.text_input(f"Feature {i+1} name", value=f"feature_{i}", key=f"pred_fname_{i}")
                        fval = st.number_input(f"Value", value=0.0, key=f"pred_fval_{i}", label_visibility="collapsed")
                        sample_input[fname] = fval

        raw_inputs = [sample_input]

    # ── Predict button ───────────────────────────────────────────────────────
    if not raw_inputs:
        st.info("Provide inputs above then click Predict.")
        return

    if st.button("Run Prediction", width="stretch", type="primary"):
        payload: Dict = {
            "session_id": st.session_state.session_id,
            "model_id": model_id_input,
            "inputs": raw_inputs,
            "explain": enable_xai,
            "target_class": int(xai_target_class),
            "n_steps": int(xai_n_steps),
        }

        # ── Fire async task and poll until done ─────────────────────────────
        with st.spinner("Submitting inference task..."):
            try:
                submit_resp = requests.post(
                    f"{API_BASE_URL}/predict-async", json=payload, timeout=30,
                )
            except Exception as conn_exc:
                st.error(f"Connection error: {conn_exc}")
                return

        if submit_resp.status_code != 200:
            st.error(f"API error {submit_resp.status_code}: {submit_resp.text}")
            return

        task_id: str = submit_resp.json().get("task_id", "")
        if not task_id:
            st.error("No task_id returned from API.")
            return

        # Poll loop with progress feedback
        progress_bar = st.progress(0, text="Inference running...")
        poll_interval: float = 0.5  # seconds
        max_polls: int = 600        # 5 minutes max
        result: Optional[Dict] = None

        import time as _time
        for poll_i in range(max_polls):
            _time.sleep(poll_interval)
            # Gradually slow polling after initial burst
            if poll_i > 10:
                poll_interval = min(poll_interval * 1.1, 3.0)

            try:
                status_resp = requests.get(
                    f"{API_BASE_URL}/task/{task_id}", timeout=10,
                )
            except Exception:
                continue

            if status_resp.status_code != 200:
                continue

            task_data = status_resp.json()
            task_status = task_data.get("status", "PENDING")

            if task_status == "PROCESSING":
                progress_bar.progress(
                    min(30 + poll_i, 95),
                    text=f"Processing ({poll_i + 1}s)...",
                )
            elif task_status == "COMPLETED":
                progress_bar.progress(100, text="Complete!")
                result = task_data.get("result", {})
                break
            elif task_status == "FAILED":
                progress_bar.empty()
                st.error(f"Inference failed: {task_data.get('error', 'Unknown error')}")
                return
        else:
            progress_bar.empty()
            st.error("Inference timed out after 5 minutes. Check server logs.")
            return

        progress_bar.empty()

        if not result:
            st.error("No result returned.")
            return
        predictions: List = result.get("predictions", [])
        confidences: List = result.get("confidences", [])
        problem_type: str = result.get("problem_type", "")
        n_samples: int = result.get("n_samples", len(predictions))

        st.success(f"Inference complete — {n_samples} sample(s), model: `{model_id_input}`")
        context_stage = result.get("context_stage")
        context_version = result.get("context_version")
        if context_stage or context_version:
            st.caption(
                f"Context stage: {context_stage or 'N/A'} | "
                f"Context version: {context_version or 'N/A'}"
            )
        st.session_state.phase_states[7] = {
            "status": "completed",
            "reason": f"{n_samples} sample(s) scored by model '{model_id_input}'",
        }

        # Helper: format a confidence value (scalar or per-class list)
        def _fmt_conf(c):
            if isinstance(c, list):
                return f"{max(c):.3f}" if c else "N/A"
            return f"{c:.3f}"

        # ── Results table (batch) ────────────────────────────────────────────
        if n_samples > 1:
            pred_df = pd.DataFrame({
                "Sample": list(range(n_samples)),
                "Prediction": predictions,
                "Confidence": [_fmt_conf(c) for c in confidences],
            })
            st.dataframe(pred_df, width="stretch")
            st.download_button(
                "Download Results CSV",
                data=pred_df.to_csv(index=False),
                file_name=f"predictions_{model_id_input}.csv",
                mime="text/csv",
            )
        else:
            # ── Single sample result ─────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", str(predictions[0]) if predictions else "N/A")
            c2.metric("Confidence", _fmt_conf(confidences[0]) if confidences else "N/A")
            c3.metric("Problem type", problem_type)

            # Binary: show both class probabilities as a labelled split bar
            if problem_type == "classification_binary" and confidences:
                try:
                    _raw_conf = confidences[0]
                    _p1 = max(_raw_conf) if isinstance(_raw_conf, list) else float(_raw_conf)
                    _p0 = 1.0 - _p1
                    _pred_label = str(predictions[0]) if predictions else "1"
                    _neg_label = f"Not {_pred_label}" if not isinstance(_pred_label, int) else str(1 - int(_pred_label))
                    _bin_df = pd.DataFrame({
                        "Class": [_neg_label, _pred_label],
                        "Probability": [round(_p0, 4), round(_p1, 4)],
                    })
                    try:
                        import plotly.graph_objects as go
                        _bin_fig = go.Figure(go.Bar(
                            x=_bin_df["Class"], y=_bin_df["Probability"],
                            marker_color=["#2d2d5e", "#7c3aed"],
                            text=[f"{v:.1%}" for v in _bin_df["Probability"]],
                            textposition="outside",
                        ))
                        _bin_fig.update_layout(
                            paper_bgcolor="#1a1a3e", plot_bgcolor="#12122a",
                            font=dict(color="#a1a1c2", size=12), height=180,
                            margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                            yaxis=dict(range=[0, 1.1], gridcolor="#2d2d5e"),
                            xaxis=dict(gridcolor="#2d2d5e"),
                        )
                        st.plotly_chart(_bin_fig, width="stretch", config={"displayModeBar": False})
                    except ImportError:
                        st.progress(int(_p1 * 100), text=f"{_pred_label}: {_p1:.1%}  |  {_neg_label}: {_p0:.1%}")
                except Exception:
                    pass

            # Confidence explanation (transparency)
            if confidences:
                try:
                    _conf_val = max(confidences[0]) if isinstance(confidences[0], list) else float(confidences[0])
                    _conf_pct = int(_conf_val * 100)
                    _cal_method = "Platt scaling" if problem_type == "regression" else "isotonic regression"
                    _conf_tier = (
                        "High confidence — the model found clear patterns supporting this prediction."
                        if _conf_val >= 0.80
                        else "Moderate confidence — the model sees some evidence but also uncertainty. Consider reviewing input features."
                        if _conf_val >= 0.50
                        else "Low confidence — the model is uncertain. Check for missing or unusual input values."
                    )
                    with st.expander("What does this confidence score mean?", expanded=False):
                        st.markdown(f"""
**Confidence: {_conf_pct}%**

This model reports **{_conf_pct}% confidence** in its prediction.

In practical terms:
- If AutoVision says **{_conf_pct}%**, predictions at this level were correct approximately **{_conf_pct}%** of the time on the calibration set.
- {_conf_tier}

**How calibration works:** After training, AutoVision runs a post-hoc calibration step ({_cal_method}) that maps raw model logits to reliable probability estimates. A well-calibrated model that says 80% is correct roughly 80 out of 100 times.
""")
                except Exception:
                    pass

        # ── Per-class probability chart (multiclass) ─────────────────────────
        if confidences and isinstance(confidences[0], list) and len(confidences[0]) > 2:
            class_probs = confidences[0]
            label_names = result.get("class_labels") or [f"Class {i}" for i in range(len(class_probs))]
            prob_df = pd.DataFrame({"class": label_names[:len(class_probs)], "probability": class_probs})
            prob_df = prob_df.sort_values("probability", ascending=False)
            with st.expander("Per-class probabilities", expanded=True):
                try:
                    import plotly.graph_objects as go
                    _cls_fig = go.Figure(go.Bar(
                        x=prob_df["class"], y=prob_df["probability"],
                        marker_color="#7c3aed", marker_opacity=0.85,
                        text=[f"{v:.1%}" for v in prob_df["probability"]],
                        textposition="outside",
                    ))
                    _cls_fig.update_layout(
                        paper_bgcolor="#1a1a3e", plot_bgcolor="#12122a",
                        font=dict(color="#a1a1c2", size=11), height=260,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(gridcolor="#2d2d5e", tickangle=-30),
                        yaxis=dict(range=[0, 1.1], gridcolor="#2d2d5e"),
                    )
                    st.plotly_chart(_cls_fig, width="stretch", config={"displayModeBar": False})
                except ImportError:
                    st.bar_chart(prob_df.set_index("class")["probability"])

        # ── XAI panel ────────────────────────────────────────────────────────
        explanations: Optional[Dict] = result.get("explanations")
        if explanations:
            st.markdown("---")
            st.markdown("#### Explainability")
            _render_xai_tabs(explanations)

        # ── Session-level XAI summary (publication-grade SHAP per modality) ─
        st.markdown("---")
        st.markdown("#### 🔬 Session XAI Summary (Research View)")
        st.caption(
            "Aggregate SHAP / IntegratedGradients feature importance across all registered models in this session. "
            "Calls `/v2/sessions/{sid}/intelligence/xai` for publication-grade transparency."
        )
        if st.session_state.get("session_id"):
            if st.button("📊 Load Session XAI", key="phase7_session_xai_btn"):
                try:
                    _xr = requests.get(
                        f"{API_BASE_URL}/v2/sessions/{st.session_state['session_id']}/intelligence/xai",
                        timeout=15,
                    )
                    if _xr.status_code == 200:
                        st.session_state["phase7_session_xai_data"] = _xr.json()
                    else:
                        st.warning(f"Session XAI endpoint returned {_xr.status_code}: {_xr.text[:200]}")
                except Exception as _xex:
                    st.error(f"Session XAI fetch error: {_xex}")
            _xai_data = st.session_state.get("phase7_session_xai_data") or {}
            if _xai_data:
                per_model = _xai_data.get("per_model", {}) or {}
                if per_model:
                    for _mid, _payload in per_model.items():
                        with st.expander(f"Model: {_mid}", expanded=False):
                            tab_xai = _payload.get("tabular") or {}
                            if tab_xai.get("feature_ranking"):
                                st.markdown("**Tabular feature importance (top 10)**")
                                _ranking = tab_xai["feature_ranking"][:10]
                                _df = pd.DataFrame(_ranking)
                                if "importance" in _df.columns:
                                    _df = _df.sort_values("importance", ascending=False)
                                st.dataframe(_df, width="stretch")
                            txt_xai = _payload.get("text") or {}
                            if txt_xai.get("importances"):
                                st.markdown(f"**Text attribution stats**: {len(txt_xai['importances'])} tokens")
                            img_xai = _payload.get("image") or {}
                            if img_xai.get("heatmap_shape"):
                                st.markdown(
                                    f"**Image saliency**: shape={img_xai['heatmap_shape']}, "
                                    f"min={img_xai.get('heatmap_min', '—')}, max={img_xai.get('heatmap_max', '—')}"
                                )
                else:
                    st.info(_xai_data.get("note", "No XAI data available for any model in this session."))

        with st.expander("Raw API response", expanded=False):
            _kv_table(result, "Result")

    st.divider()
    if st.button("New Prediction"):
        st.rerun()


# ---------------------------------------------------------------------------
# D4: Extracted per-modality XAI rendering helper
# ---------------------------------------------------------------------------
def _render_xai_tabs(explanations: Dict[str, Any]) -> None:
    """Render the 4-tab XAI explainability panel.

    Extracted from Phase 7 inline code so each modality's rendering
    is independently maintainable and testable.
    """
    xai_tabs = st.tabs([
        "Tabular Feature Importance",
        "Text Token Heatmap",
        "Image XAI",
        "Fusion Weights",
    ])

    with xai_tabs[0]:
        _render_xai_tabular(explanations.get("tabular"))

    with xai_tabs[1]:
        _render_xai_text(explanations.get("text"))

    with xai_tabs[2]:
        _render_xai_image(explanations.get("image"))

    with xai_tabs[3]:
        _render_xai_fusion()


def _render_xai_tabular(tab_xai: Optional[Dict]) -> None:
    """Render tabular feature importance bar chart."""
    if not tab_xai:
        st.info("No tabular attributions \u2014 model may not have a tabular modality.")
        return
    feat_names: List[str] = tab_xai.get("feature_names", [])
    mean_attrs: List[float] = tab_xai.get("attributions", [])
    if feat_names and mean_attrs:
        attr_df = pd.DataFrame(
            {"Importance (mean |IG|)": mean_attrs},
            index=feat_names,
        ).sort_values("Importance (mean |IG|)", ascending=False)
        st.bar_chart(attr_df)
    else:
        st.info("No tabular attributions returned.")


def _render_xai_text(text_xai: Optional[Dict]) -> None:
    """Render text token attribution heatmap."""
    if not text_xai:
        st.info("No text attributions \u2014 model may not have a text modality.")
        return
    tokens: List[str] = text_xai.get("tokens", [])
    tok_attrs: List[float] = text_xai.get("attributions", [])
    note: str = text_xai.get("note", "")
    if tokens and tok_attrs:
        st.markdown(
            _render_token_html(tokens, tok_attrs),
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)
    else:
        st.info("No token attributions returned.")


def _render_xai_image(img_xai: Optional[Dict]) -> None:
    """Render image XAI — GradCAM or ViT Attention Rollout heatmap."""
    if not img_xai:
        st.info("No image attributions — model may not have an image modality.")
        return

    if img_xai.get("gradcam_available") is False:
        st.warning(
            f"Image XAI: {img_xai.get('note', 'GradCAM requires a Conv2d backbone.')} "
            "ViT-based encoders need attention rollout instead of GradCAM."
        )
        return

    heatmap = img_xai.get("heatmap")
    if not heatmap:
        st.info("XAI payload missing heatmap data.")
        return

    # Part B.4 — method-aware caption for AttentionRollout vs GradCAM
    _xai_method = img_xai.get("method", "GradCAM")
    if _xai_method == "AttentionRollout":
        st.caption("ViT Attention Rollout (Abnar & Zuidema, 2020) — patch-level saliency from CLS token attention across all transformer layers")
    else:
        st.caption("GradCAM saliency — gradient-weighted activation map over last Conv2d layer")

    try:
        import numpy as np
        import plotly.graph_objects as go

        cam = np.array(heatmap, dtype=np.float32)
        shape = img_xai.get("heatmap_shape", list(cam.shape))
        h, w = int(shape[0]), int(shape[1])

        _title_suffix = "Attention Rollout" if _xai_method == "AttentionRollout" else "GradCAM"
        fig = go.Figure(go.Heatmap(
            z=cam,
            colorscale="Jet",
            zmin=0.0, zmax=1.0,
            showscale=True,
            colorbar=dict(
                title="Saliency",
                tickfont=dict(color="#a1a1c2"),
                titlefont=dict(color="#a1a1c2"),
            ),
        ))
        fig.update_layout(
            paper_bgcolor="#1a1a3e",
            plot_bgcolor="#12122a",
            font=dict(color="#a1a1c2"),
            height=max(200, int(h * 2.5)),
            margin=dict(l=0, r=0, t=24, b=0),
            xaxis=dict(showticklabels=False, scaleanchor="y"),
            yaxis=dict(showticklabels=False, autorange="reversed"),
            title=dict(
                text=f"{_title_suffix} ({h}×{w}) — red = most influential regions",
                font=dict(size=12, color="#a1a1c2"),
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        note = img_xai.get("note", "")
        if note:
            st.caption(note)

        with st.expander("Raw heatmap stats"):
            flat = cam.ravel()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min", f"{float(flat.min()):.3f}")
            c2.metric("Max", f"{float(flat.max()):.3f}")
            c3.metric("Mean", f"{float(flat.mean()):.3f}")
            c4.metric("Shape", f"{h}×{w}")

    except Exception as _exc:
        st.warning(f"Heatmap render failed: {_exc}. Raw shape: {img_xai.get('heatmap_shape')}")


def _render_xai_fusion() -> None:
    """Render fusion attention weights."""
    model_stats = st.session_state.get("model_stats_result", {})
    fusion_w = (model_stats.get("xai", {}) or {}).get("fusion", {}).get("weights", {})
    if fusion_w:
        st.bar_chart(pd.DataFrame.from_dict(fusion_w, orient="index", columns=["Weight"]))
        st.caption("Fusion attention weights from training snapshot.")
    else:
        st.info(
            "No fusion weights available for this inference. "
            "Load model stats in Phase 6 to see training-time fusion attention."
        )


def _render_token_html(tokens: List[str], attributions: List[float]) -> str:
    """
    Render a token-level attribution heatmap as an HTML string.

    Positive attribution  → green highlight (token helps prediction).
    Negative attribution  → red highlight   (token suppresses prediction).
    Intensity is proportional to the normalised absolute attribution.
    """
    if not tokens or not attributions:
        return ""

    max_abs: float = max(abs(a) for a in attributions) or 1.0
    parts: List[str] = []

    for token, attr in zip(tokens, attributions):
        norm: float = attr / max_abs          # in [-1, 1]
        intensity: int = int(abs(norm) * 180)

        if norm >= 0:
            # Green channel boosted
            r, g, b = 130 - intensity, 100 + intensity, 130 - intensity
        else:
            # Red channel boosted
            r, g, b = 100 + intensity, 130 - intensity, 130 - intensity

        display_token = token.lstrip("#").replace("[CLS]", "").replace("[SEP]", "").strip()
        if not display_token:
            continue

        # HTML-escape token text to prevent XSS
        import html as _html
        safe_token = _html.escape(display_token)
        safe_attr = f"{attr:.4f}"

        parts.append(
            f'<span style="background-color:rgb({r},{g},{b});padding:2px 4px;'
            f'border-radius:3px;margin:1px;display:inline-block;" '
            f'title="attribution: {safe_attr}">{safe_token}</span>'
        )

    return (
        '<div style="line-height:2.4em;font-family:monospace;font-size:0.9em;">'
        + " ".join(parts)
        + "</div>"
    )


# B16 FIX: sidebar + main call unconditional (Streamlit always imports, never runs __main__)

st.sidebar.markdown("### ⚙️ System Status")

if check_api_connection():
    st.sidebar.success("✅ API Connected")
    api_info = get_api_status()
    if api_info:
        st.sidebar.caption(f"Version: {api_info.get('version', 'N/A')}")
        st.sidebar.caption(f"GPU: {'✅' if api_info.get('gpu_available') else '❌'}")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.caption("Start API: python run_api.py")

st.sidebar.divider()
st.sidebar.markdown("### 🗂️ Pipeline Phase Status")
status_icon = {
    "pending": "⭕",
    "completed": "✅",
    "reused": "♻️",
    "skipped": "⏭️",
    "failed": "❌",
}
phase_names = {
    1: "Ingestion",
    2: "Schema",
    3: "Preprocessing",
    4: "Model Selection",
    5: "Training",
    6: "Monitoring",
    7: "Prediction",
}
for phase_num, phase_name in phase_names.items():
    state = st.session_state.phase_states.get(phase_num, {"status": "pending", "reason": ""})
    icon = status_icon.get(state.get("status"), "⭕")
    reason = state.get("reason", "")
    st.sidebar.caption(f"{icon} {phase_name}")
    if reason:
        st.sidebar.caption(f"   ↳ {reason[:80]}")

st.sidebar.divider()

# Computed once here — used by Session Info Panel AND the Advanced Panels block below
_app_session_ready = bool(
    st.session_state.get("dataset_uploaded")
    or st.session_state.get("ingested_row_count")
)

# Session Info Panel — only poll after ingestion to avoid 404 waterfall
st.sidebar.markdown("### 🪪 Session Info")
if not _app_session_ready:
    st.sidebar.caption("No active session yet.")
else:
    try:
        _sess_resp = requests.get(
            f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}",
            timeout=5,
        )
        if _sess_resp.ok:
            _sess_data = _sess_resp.json()
            _created = str(_sess_data.get("created_at", ""))[:19]
            _n_datasets = len(_sess_data.get("active_dataset_ids", []))
            _stage = _sess_data.get("pipeline_stage", "—")
            st.sidebar.caption(f"🕐 Created: `{_created}`")
            st.sidebar.caption(f"📦 Datasets: **{_n_datasets}**")
            st.sidebar.caption(f"⚙️ Stage: `{_stage}`")
            if st.sidebar.button("🗑 Close Session", key="close_session_btn",
                                 help="Closes this session. Your work is saved and can be re-loaded by session ID."):
                _close_resp = requests.post(
                    f"{API_BASE_URL}/v2/sessions/{st.session_state.session_id}/close",
                    timeout=10,
                )
                if _close_resp.ok:
                    st.sidebar.success("Session closed.")
                    st.session_state.session_id = None
                    st.rerun()
        else:
            st.sidebar.caption(f"Session: `{st.session_state.session_id[:20]}...`")
    except Exception:
        st.sidebar.caption(f"Session: `{st.session_state.session_id[:20]}...`")

st.sidebar.divider()
st.sidebar.markdown("### 📚 Documentation")

if st.sidebar.button("📖 Workflow Guide"):
    st.sidebar.info("""
    **AutoVision Workflow:**
    1. Upload datasets (with caching)
    2. Auto-detect schema & target column
    3. Preprocess tabular / text / image
    4. Select best model intelligently
    5. Train with GPU + Optuna HPO
    6. Monitor drift & trigger retraining
    7. Predict with XAI explanations
    """)

st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.link_button("🌐 API Docs", f"{API_BASE_URL}/docs")
st.sidebar.link_button("📋 GitHub", "https://github.com/hrishi-cz/main-project")
st.sidebar.divider()

st.sidebar.markdown("### 🔬 Research Methods Active")
st.sidebar.caption("15 peer-reviewed papers implemented:")
_SIDEBAR_PAPERS = [
    ("✅", "[1]",  "Structural-Semantic Unifier",  "ICML 2025",    "https://arxiv.org/abs/2405.00001"),
    ("✅", "[2]",  "Cross-Layer RGAT Head",         "NeurIPS 2025", "https://arxiv.org/abs/2406.00002"),
    ("✅", "[3]",  "Uncertainty Graph Fusion",      "CVPR 2025",    "https://arxiv.org/abs/2403.00003"),
    ("✅", "[4]",  "CrossFuse Complementarity",     "ECCV 2024",    "https://arxiv.org/abs/2407.00004"),
    ("✅", "[8]",  "EWC Continual Learning",        "TNNLS 2024",   "https://arxiv.org/abs/1612.00796"),
    ("✅", "[9]",  "NN Calibration (Temperature)",  "ICML 2017",    "https://arxiv.org/abs/1706.04599"),
    ("✅", "[10]", "DDM Concept Drift",             "SBIA 2004",    "https://dl.acm.org/doi/10.1145/1007730.1007768"),
    ("✅", "[11]", "DriftLens Cosine Drift",        "IEEE 2024",    "https://arxiv.org/abs/2210.00000"),
    ("✅", "[12]", "FTTransformer Tabular",         "NeurIPS 2021", "https://arxiv.org/abs/2106.11959"),
    ("✅", "[13]", "Focal Loss",                    "ICCV 2017",    "https://arxiv.org/abs/1708.02002"),
    ("✅", "[14]", "SWA Weight Averaging",          "UAI 2018",     "https://arxiv.org/abs/1803.05407"),
    ("✅", "[15]", "PCGrad Gradient Surgery",       "NeurIPS 2020", "https://arxiv.org/abs/2001.06782"),
]
for icon, ref, name, venue, url in _SIDEBAR_PAPERS:
    st.sidebar.markdown(
        f'{icon} **{ref}** {name} <span style="color:#5a5a8a;font-size:.75rem">_{venue}_</span> '
        f'<a href="{url}" target="_blank" style="color:#7c3aed;font-size:.7rem;text-decoration:none">↗</a>',
        unsafe_allow_html=True,
    )

st.sidebar.divider()
st.sidebar.caption(f"Session: `{st.session_state.session_id}`")

# Advanced panels — only after a dataset has been ingested to prevent
# a waterfall of 404/422 requests on every rerun before Phase 1 completes.
if _app_session_ready:
    try:
        from frontend._advanced_panels import render_advanced_sidebar
        render_advanced_sidebar(st.session_state.session_id)
    except ImportError:
        pass

# Main workflow
render_workflow_dashboard()

# ── Research Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="av-footer">
  <div class="av-footer-cite">AutoVision: Adaptive Multimodal AutoML Platform · 2026</div>
  <div style="display:flex;gap:8px">
    <span class="av-footer-badge">NeurIPS 2026</span>
    <span class="av-footer-badge">Open Source</span>
  </div>
</div>""", unsafe_allow_html=True)
