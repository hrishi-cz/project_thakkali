"""APEX AutoML Frontend - Comprehensive Multimodal ML Platform with Workflow Integration."""

import streamlit as st
import requests
import json
import time
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="APEX AutoML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .phase-header {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .phase-status {font-size: 14px; color: #666;}
    .success-badge {background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .info-badge {background-color: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .warning-badge {background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8001"

# API key auth (reads APEX_API_KEY — same env var as backend)
_API_KEY = os.environ.get("APEX_API_KEY", "")


def _api_headers() -> Dict[str, str]:
    """Return common headers for API requests (includes API key when set)."""
    headers: Dict[str, str] = {}
    if _API_KEY:
        headers["X-API-Key"] = _API_KEY
    return headers

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

# Session state initialization
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = 1
if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False
if 'schema_detected' not in st.session_state:
    st.session_state.schema_detected = False
if 'detected_schema' not in st.session_state:
    st.session_state.detected_schema = None
if 'model_selected' not in st.session_state:
    st.session_state.model_selected = False
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}
if 'ingested_row_count' not in st.session_state:
    st.session_state.ingested_row_count = None
if 'training_task_id' not in st.session_state:
    st.session_state.training_task_id = None
if 'hp_overrides' not in st.session_state:
    st.session_state.hp_overrides = None
if 'training_result' not in st.session_state:
    st.session_state.training_result = None
if 'training_epoch_metrics' not in st.session_state:
    st.session_state.training_epoch_metrics = []
if 'ingestion_task_id' not in st.session_state:
    st.session_state.ingestion_task_id = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def render_workflow_dashboard():
    """Render the main workflow dashboard."""
    st.title("🚀 APEX AutoML Framework")
    st.markdown("**Advanced Predictive Ensemble with eXtendable Modularity**\n---")
    
    # API Status Section
    col1, col2, col3 = st.columns(3)
    with col1:
        api_status = get_api_status()
        if api_status:
            st.success("✅ API Connected")
            st.caption(f"v{api_status.get('version', 'N/A')}")
        else:
            st.error("❌ API Disconnected")
            st.caption("Make sure API server is running on http://localhost:8001")
    
    with col2:
        st.info("📊 Workflow: Multi-Phase Training")
        st.caption(f"Current Stage: Phase {st.session_state.workflow_stage}")
    
    with col3:
        gpu_available = api_status.get('gpu_available', False) if api_status else False
        if gpu_available:
            device_name = api_status.get('device', 'GPU') if api_status else 'GPU'
            st.success(f"GPU: {device_name}")
        else:
            st.warning("CPU Mode")
    
    st.divider()
    
    # Workflow Progress
    st.subheader("📋 Workflow Stages")
    
    stages = [
        ("Phase 1", "Data Ingestion", "Upload & cache datasets"),
        ("Phase 2", "Schema Detection", "Detect columns & problem type"),
        ("Phase 3", "Preprocessing", "Prepare data for training"),
        ("Phase 4", "Model Selection", "Auto-select models & params"),
        ("Phase 5", "Training", "Train with GPU acceleration"),
        ("Phase 6", "Monitoring & Drift", "Track performance & drift"),
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
    _PHASE_LABELS = [
        "Phase 1: Data Ingestion",
        "Phase 2: Schema Detection",
        "Phase 3: Preprocessing",
        "Phase 4: Model Selection",
        "Phase 5: Training",
        "Phase 6: Monitoring & Drift",
        "Phase 7: Prediction",
    ]
    # Sync radio default from workflow_stage BEFORE the widget is instantiated.
    # This ensures navigation buttons (which set workflow_stage + rerun) are
    # reflected in the radio on the next render without violating Streamlit's
    # "no mutation after instantiation" rule.
    st.session_state["phase_radio"] = _PHASE_LABELS[
        st.session_state.workflow_stage - 1
    ]
    phase = st.radio(
        "Select Workflow Phase:",
        _PHASE_LABELS,
        index=st.session_state.workflow_stage - 1,
        horizontal=True,
        key="phase_radio",
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
            resp = requests.get(f"{API_BASE_URL}/ingest/status/{task_id}", headers=_api_headers(), timeout=5)
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
                from_cache = ds_info.get("from_cache", False)
                if ds_status == "success":
                    shape = ds_info.get("shape")
                    shape_str = f" — {shape[0]} rows x {shape[1]} cols" if shape else ""
                    cache_label = "Cached" if from_cache else "Loaded"
                    st.success(f"**Dataset {idx}** — {cache_label}{shape_str}\n{source[:80]}")
                else:
                    st.error(f"**Dataset {idx}** — {ds_status}\n{source[:80]}")

        # Still downloading
        if status == "running":
            st.info(f"📦 Processing {completed}/{total} dataset(s)...")
            time.sleep(2)
            st.rerun()
            return

        # Completed
        if status == "completed":
            result = task.get("result", {})
            ingestion = result.get("ingestion_progress", {})
            datasets_list = ingestion.get("datasets", datasets)

            st.session_state.ingestion_task_id = None

            # Update session state
            overall = ingestion.get("status", "failed")
            st.session_state.dataset_uploaded = overall in ("success", "partial")
            st.session_state.schema_detected = False
            st.session_state.detected_schema = None
            st.session_state.model_selected = False
            st.session_state.training_task_id = None
            st.session_state.hp_overrides = None
            st.session_state.training_result = None
            st.session_state.training_epoch_metrics = None
            if "model_selection_result" in st.session_state:
                del st.session_state.model_selection_result
            if "preprocess_result" in st.session_state:
                del st.session_state.preprocess_result
            if "drift_result" in st.session_state:
                del st.session_state.drift_result
            if "registry_result" in st.session_state:
                del st.session_state.registry_result

            # Extract dataset shapes
            dataset_shapes = []
            for ds_info in datasets_list:
                shape = ds_info.get("shape")
                if shape and ds_info.get("status") == "success":
                    dataset_shapes.append(shape)

            row_count = dataset_shapes[0][0] if dataset_shapes else 5000
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
            st.session_state.ingestion_task_id = None

        # Show navigation even after completion
        if st.session_state.dataset_uploaded:
            if st.button("➡️ Next: Schema Detection", use_container_width=True):
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

    # --- Per-link cache status check (before ingestion) ---
    if dataset_sources:
        try:
            cache_resp = requests.post(
                f"{API_BASE_URL}/ingest/check-cache",
                json={"urls": dataset_sources},
                headers=_api_headers(),
                timeout=5,
            )
            if cache_resp.status_code == 200:
                cache_results = cache_resp.json().get("results", [])
                for cr in cache_results:
                    url_display = cr["url"][:80]
                    if cr["cached"]:
                        size_str = f" ({cr['size_mb']} MB)" if cr.get("size_mb") else ""
                        st.success(f"Cached{size_str}: {url_display}")
                    else:
                        st.info(f"New — will download: {url_display}")
        except Exception:
            pass  # Non-critical — silently skip if endpoint unavailable

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
        if st.button("🔄 Load Datasets", use_container_width=True):
            if not dataset_sources:
                st.error("Please enter at least one dataset source (URL or local path) above.")
            elif not check_api_connection():
                st.error("API not connected!")
            else:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ingest/datasets",
                        json={
                            "dataset_urls": dataset_sources,
                            "session_id": datetime.now().isoformat(),
                        },
                        headers=_api_headers(),
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.ingestion_task_id = data["task_id"]
                        st.rerun()
                    else:
                        st.error(f"Failed to start ingestion: {resp.status_code} - {resp.text[:300]}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    with col2:
        if st.button("📊 View Cache", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/cache/stats", headers=_api_headers(), timeout=5)
                if response.status_code == 200:
                    cache_info = response.json()
                    st.success(f"✅ Cache Location: {cache_info['cache_location']}")

                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Cached Items", cache_info['total_items'])
                    col_c2.metric("Total Size (MB)", cache_info['total_size_mb'])
                    col_c3.metric("Status", "Ready")

                    if cache_info['items']:
                        with st.expander("📁 Cached Files"):
                            for item in cache_info['items']:
                                st.caption(f"• {item['name']} ({item['size_mb']} MB)")
                else:
                    st.error(f"Cache query failed ({response.status_code}): {response.text[:200]}")
            except Exception as e:
                st.error(f"Cannot connect to cache endpoint: {e}")

    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/cache/clear", headers=_api_headers(), timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"Failed to clear cache: {response.status_code}")
            except Exception as e:
                st.error(f"Cache clear error: {str(e)}")

    # Status display
    if st.session_state.dataset_uploaded:
        st.markdown("### ✅ Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Datasets Loaded", st.session_state.dataset_info.get("count", 0))
        col2.metric("Cache Location", "./data/dataset_cache/")
        col3.metric("Last Updated", "Just now")

        if st.button("➡️ Next: Schema Detection", use_container_width=True):
            st.session_state.workflow_stage = 2
            st.rerun()


def render_phase_2_schema_detection():
    """Phase 2: Schema Detection."""
    st.header("Phase 2️⃣ - Schema Detection & Problem Type Inference")
    
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
        if st.button("🔍 Detect Schema", use_container_width=True):
            with st.spinner("Analyzing ingested datasets..."):
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    status.write("📍 Detecting schema for ingested datasets...")
                    
                    response = requests.post(
                        f"{API_BASE_URL}/detect-schema",
                        json={},
                        headers=_api_headers(),
                        timeout=120
                    )
                    
                    progress_bar.progress(0.6)
                    
                    if response.status_code == 200:
                        payload = response.json()

                        # ⭐⭐⭐ CRITICAL FIX ⭐⭐⭐
                        schema_data = payload.get("data", {})

                        st.session_state.detected_schema = schema_data
                        st.session_state.schema_detected = True

                        progress_bar.progress(1.0)
                        st.success("✅ Schema detection complete!")
                        st.rerun()
                    else:
                        st.error(f"❌ Detection failed: {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout after 120 seconds")
                except Exception as e:
                    st.error(f"❌ Detection error: {str(e)}")
    
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

        # ⭐ GLOBAL FIELDS (NEW ENGINE)
        global_modalities = schema.get("global_modalities", [])
        global_problem = schema.get("global_problem_type", "Unknown")
        primary_target = schema.get("primary_target", "Unknown")
        confidence = schema.get("detection_confidence", 0)
        fusion_ready = schema.get("fusion_ready", False)

        # ---------------- SUMMARY ----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Target Column", primary_target)
        col2.metric("Problem Type", global_problem)
        col3.metric("Fusion Ready", fusion_ready)
        col4.metric("Confidence", f"{confidence:.1%}")

        st.info(
            f"**Modalities Found:** {', '.join(global_modalities) if global_modalities else 'None'}"
        )

        # ---------------- DATASET RELATEDNESS ----------------
        relatedness = schema.get("relatedness_report", {})
        n_groups = relatedness.get("n_groups", 1)
        n_datasets = relatedness.get("n_datasets", 0)

        if n_datasets > 1:
            st.markdown("#### Dataset Relatedness")
            groups = relatedness.get("groups", [])
            pairwise = relatedness.get("pairwise_scores", {})
            selected_group = relatedness.get("selected_group")
            selection_method = relatedness.get("selection_method", "auto")

            if n_groups == 1:
                st.success("All datasets are compatible (single related group)")
            else:
                st.warning(f"Datasets form **{n_groups} unrelated groups** — only one group is used for the global schema.")

                # Show pairwise scores
                if pairwise:
                    with st.expander("Pairwise Similarity Scores"):
                        score_rows = []
                        for pair_key, score in pairwise.items():
                            score_rows.append({"Pair": pair_key, "Similarity": f"{score:.2f}"})
                        if score_rows:
                            st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

                # Show groups and let user select
                per_dataset = schema.get("per_dataset", [])
                group_options = []
                for g_idx, group_indices in enumerate(groups):
                    ds_names = [per_dataset[i].get("dataset_id", f"Dataset {i}") if i < len(per_dataset) else f"Dataset {i}" for i in group_indices]
                    label = f"Group {g_idx + 1}: {', '.join(str(n)[:40] for n in ds_names)}"
                    group_options.append((label, list(group_indices)))

                if selected_group is not None:
                    st.info(f"Currently using: indices {selected_group} ({selection_method} selection)")

                selected_label = st.selectbox(
                    "Select dataset group to use:",
                    [opt[0] for opt in group_options],
                    key="schema_group_select",
                )
                selected_indices = next(opt[1] for opt in group_options if opt[0] == selected_label)

                if st.button("Re-detect with selected group"):
                    with st.spinner("Re-detecting schema with selected group..."):
                        try:
                            re_resp = requests.post(
                                f"{API_BASE_URL}/detect-schema",
                                json={"dataset_group": selected_indices},
                                headers=_api_headers(),
                                timeout=120,
                            )
                            if re_resp.status_code == 200:
                                re_payload = re_resp.json()
                                st.session_state.detected_schema = re_payload.get("data", {})
                                st.success("Schema re-detected with selected group!")
                                st.rerun()
                            else:
                                st.error(f"Re-detection failed: {re_resp.text}")
                        except Exception as e:
                            st.error(f"Re-detection error: {e}")

        # ---------------- TABS ----------------
        tab1, tab2 = st.tabs(["Summary", "Debug Info"])

        # ================= SUMMARY TAB =================
        with tab1:
            st.markdown("#### 📊 Per-Dataset Analysis")

            per_dataset = schema.get("per_dataset", [])

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

        # ================= DEBUG TAB =================
        with tab2:
            st.markdown("#### 🔍 Raw Schema Detection Response")
            st.json(schema)

        st.divider()

        if st.button("➡️ Next: Preprocessing", use_container_width=True):
            st.session_state.workflow_stage = 3
            st.rerun()

    elif st.session_state.schema_detected:
        st.warning("⚠️ Schema detection ran but no results available")

def render_phase_3_preprocessing():
    """Phase 3: Preprocessing Pipeline with explainable transformation details."""
    st.header("Phase 3 - Data Preprocessing")

    if not st.session_state.schema_detected:
        st.warning("Please detect schema in Phase 2 first")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Start Preprocessing", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                schema_data = st.session_state.detected_schema or {}
                try:
                    # The backend /preprocess endpoint uses session_ingested_hashes
                    # internally and ignores request body fields. Send empty body.
                    response = requests.post(
                        f"{API_BASE_URL}/preprocess",
                        json={},
                        headers=_api_headers(),
                        timeout=300,
                    )
                    if response.status_code == 200:
                        result = response.json().get("data", {})
                        st.session_state.preprocess_result = result
                        st.success("Preprocessing complete!")
                        st.rerun()
                    else:
                        st.error(f"Preprocessing failed ({response.status_code}): {response.text[:300]}")
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

    preprocess_result = st.session_state.get("preprocess_result", {})
    stages = preprocess_result.get("preprocessing_stages", [])
    total_samples = preprocess_result.get("total_samples", None)
    output_shapes = preprocess_result.get("output_shapes", {})
    samples = preprocess_result.get("samples", {})

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
                    import pandas as pd
                    raw_df = pd.DataFrame(raw_rows, columns=raw_cols)
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.caption("No raw sample available")

            with after_col:
                st.markdown("**After (Transformed)**")
                t_cols = tab_sample.get("transformed_columns", [])
                t_rows = tab_sample.get("transformed_rows", [])
                if t_cols and t_rows:
                    import pandas as pd
                    # Show first few transformed columns if there are many
                    t_df = pd.DataFrame(t_rows, columns=t_cols)
                    if len(t_cols) > 15:
                        st.caption(f"Showing first 15 of {len(t_cols)} features")
                        st.dataframe(t_df.iloc[:, :15], use_container_width=True)
                    else:
                        st.dataframe(t_df, use_container_width=True)
                else:
                    st.caption("No transformed sample available")

            st.caption(
                f"Pipeline: Numeric → Impute(median) → StandardScaler | "
                f"Categorical → Impute(mode) → OneHotEncoder"
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

    # ---- Image Preprocessing Note ----
    if output_shapes.get("image"):
        with st.expander("View Image Preprocessing Details"):
            st.markdown(
                f"**Output shape:** `{output_shapes['image']}`\n\n"
                f"Pipeline: Resize(224x224) → ToTensor → "
                f"Normalize(ImageNet mean/std)"
            )

    if st.button("Next: Model Selection"):
        st.session_state.workflow_stage = 4
        st.session_state.model_selected = False
        st.rerun()


def render_phase_4_model_selection():
    """Phase 4: Model Selection with Rationale."""
    st.header("Phase 4 - Automatic Model Selection")

    if not st.session_state.schema_detected:
        st.warning("Please detect schema in Phase 2 first")
        return

    st.markdown("""
    **Selection Criteria:**
    - **JIT Hardware Profiling:** Encoders are selected at training time via
      live VRAM measurement — the system maximizes model capacity while
      keeping peak memory within 85% of available GPU memory.
    - **Dataset-Aware Epochs:** Epoch bounds scale with dataset size and modality
      (e.g. small datasets get 30-50 epochs; large datasets 6-15).
    - **Task Complexity:** Binary/Multiclass/Regression drives encoder
      tier heuristics and Optuna search bounds.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Select Models", use_container_width=True):
            with st.spinner("Selecting models and hyperparameters..."):
                schema_data = st.session_state.detected_schema or {}
                dataset_size = st.session_state.get('ingested_row_count', 1000)
                modalities = schema_data.get("global_modalities", [])
                problem_type = schema_data.get("global_problem_type", "Unknown")
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/select-model",
                        json={
                            "dataset_size": dataset_size,
                            "modalities": modalities,
                            "problem_type": problem_type
                        },
                        headers=_api_headers(),
                        timeout=120
                    )
                    if response.status_code == 200:
                        st.session_state.model_selection_result = response.json()
                        st.success("Models selected!")
                        st.session_state.model_selected = True
                        st.rerun()
                    else:
                        err_body = response.text[:500]
                        st.error(f"Model selection failed ({response.status_code}): {err_body}")
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
        st.write(f"**Model:** {best.get('name', 'N/A')}")
        st.write(f"**Fusion Strategy:** {best.get('fusion_strategy', 'N/A')}")
        st.write(f"**Batch Size:** {best.get('batch_size', 'N/A')}")
        st.write(f"**Tier:** {best.get('tier', 'N/A')}")

        # ── Per-Modality Encoder Selection ───────────────────────────
        img_enc = best.get("image_encoder")
        txt_enc = best.get("text_encoder")
        tab_enc = best.get("tabular_encoder")
        if img_enc or txt_enc or tab_enc:
            enc_cols = st.columns(3)
            enc_cols[0].metric("Vision Encoder", img_enc or "N/A")
            enc_cols[1].metric("Text Encoder", txt_enc or "N/A")
            enc_cols[2].metric("Tabular Encoder", tab_enc or "N/A")

        hw_info = best.get("hardware_info", {})
        if hw_info:
            hw_cols = st.columns(3)
            hw_cols[0].metric("Device", hw_info.get("cuda_device", hw_info.get("device", "N/A")))
            hw_cols[1].metric("VRAM (GB)", hw_info.get("gpu_memory_gb", "N/A"))
            hw_cols[2].metric("GPU Available", "Yes" if hw_info.get("gpu_available") else "No")

        rationale = best.get("rationale", {})
        if rationale:
            with st.expander("Selection Rationale"):
                for component, reason in rationale.items():
                    st.write(f"- **{component}:** {reason}")
        hpo = best.get("hpo_space", {})
        if hpo:
            with st.expander("HPO Search Space"):
                st.json(hpo)
        all_models = result.get("recommended_models", [])
        if len(all_models) > 1:
            with st.expander(f"All {len(all_models)} recommended models"):
                for m in all_models:
                    st.write(f"- **{m.get('name', '?')}** ({m.get('tier', '?')})")

        # ── Manual Model / Encoder Override ──────────────────────────
        st.divider()
        st.markdown("### Model Override")
        st.caption(
            "The system has auto-selected the best encoders for your data. "
            "You may manually override individual encoder selections below."
        )
        use_model_override = st.checkbox(
            "Override auto-selected encoders",
            help="Manually choose vision, text, or tabular encoder tiers.",
        )
        if use_model_override:
            problem_type = result.get("problem_type", "")
            modalities = result.get("modalities", [])
            mo_c1, mo_c2, mo_c3 = st.columns(3)

            # Vision encoder options
            vision_options = ["(auto)", "EfficientNet-B0", "ResNet-50", "ConvNeXt-Tiny"]
            if problem_type == "segmentation":
                vision_options = ["ResNet-50 (U-Net)"]
            with mo_c1:
                vision_override = st.selectbox(
                    "Vision Encoder",
                    options=vision_options,
                    index=0,
                    help="Choose a specific vision backbone. (auto) uses the AI-recommended tier.",
                    disabled="image" not in modalities and problem_type != "segmentation",
                )

            # Text encoder options
            text_options = ["(auto)", "BERT-base", "DistilBERT", "TinyBERT"]
            with mo_c2:
                text_override = st.selectbox(
                    "Text Encoder",
                    options=text_options,
                    index=0,
                    help="Choose a specific text backbone.",
                    disabled="text" not in modalities,
                )

            # Tabular encoder options
            tabular_options = ["(auto)", "GRN", "MLP", "None"]
            with mo_c3:
                tabular_override = st.selectbox(
                    "Tabular Encoder",
                    options=tabular_options,
                    index=0,
                    help="Choose a specific tabular encoder.",
                    disabled="tabular" not in modalities,
                )

            # Store overrides in session state
            encoder_overrides = {}
            if vision_override != "(auto)":
                encoder_overrides["image_encoder"] = vision_override
            if text_override != "(auto)":
                encoder_overrides["text_encoder"] = text_override
            if tabular_override != "(auto)":
                encoder_overrides["tabular_encoder"] = tabular_override
            st.session_state.encoder_overrides = encoder_overrides if encoder_overrides else None
        else:
            st.session_state.encoder_overrides = None

        # ── Optuna Search Space Transparency ──────────────────────────
        st.divider()
        hpo_space = best.get("hpo_space", {})

        with st.expander("View Optuna Auto-Tuning Search Space"):
            if hpo_space:
                st.caption(
                    "If you leave manual override **unchecked**, the AI will "
                    "intelligently search within these boundaries using Optuna "
                    "Bayesian optimisation."
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
                st.json(hpo_space)
            else:
                st.info("No HPO search space available yet — run model selection first.")

        # ── Manual HP Override Controls ───────────────────────────────
        st.markdown("### Manual Hyperparameter Override")
        st.caption(
            "Leave unchecked to let Optuna auto-tune within the search space "
            "above. Check to lock in your own values for a single training run."
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
                current_fusion = best.get("fusion_strategy", "concatenation")
                fusion_options = ["concatenation", "attention"]
                fusion_idx = (
                    fusion_options.index(current_fusion)
                    if current_fusion in fusion_options
                    else 0
                )
                fusion_val = st.selectbox(
                    "Fusion Strategy",
                    options=fusion_options,
                    index=fusion_idx,
                    help=(
                        "Concatenation merges data directly. Attention allows "
                        "the model to dynamically weigh which modality (Image, "
                        "Text, or Tabular) is most important."
                    ),
                )
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


def _render_completed_training(data: dict, completed_epoch_metrics: list) -> None:
    """Render training completion view (reusable for both live and cached results)."""
    metrics = data.get("metrics", {})

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
    actual_ep = metrics.get("actual_epochs")
    max_ep = metrics.get("epochs")
    c8.metric("Epochs", f"{actual_ep}/{max_ep}" if actual_ep and max_ep else "N/A")

    # Stop reason display
    stop_reason = metrics.get("stop_reason", "")
    if stop_reason:
        reason_labels = {
            "converged": "Training converged — validation loss stabilized",
            "overfitting": "Early stopped — overfitting detected (train loss decreasing, val loss increasing)",
            "flatline": "Early stopped — loss flatlined (no further learning)",
            "underfitting_extended": "Underfitting detected — epochs were extended to allow more learning",
            "early_stopped": "Early stopped — validation loss stalled",
            "completed": "All scheduled epochs completed",
        }
        reason_text = reason_labels.get(stop_reason, f"Stop reason: {stop_reason}")
        if stop_reason in ("overfitting", "flatline", "early_stopped"):
            st.warning(reason_text)
        elif stop_reason == "converged":
            st.success(reason_text)
        elif stop_reason == "underfitting_extended":
            st.warning(reason_text)
        else:
            st.info(reason_text)

    # Data split info
    split_info = metrics.get("data_split", {})
    if split_info:
        sp1, sp2, sp3 = st.columns(3)
        sp1.metric("Total Samples", split_info.get("total", "?"))
        sp2.metric("Train Samples", split_info.get("train", "?"))
        sp3.metric("Val Samples", split_info.get("val", "?"))

    # Epoch-level loss chart
    if completed_epoch_metrics:
        st.markdown("### Loss Convergence")
        _render_loss_chart(completed_epoch_metrics)

    with st.expander("Full Metrics", expanded=False):
        st.json(metrics)

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
            rt_fusion_cur = best_model.get("fusion_strategy", "concatenation")
            rt_fusion_opts = ["concatenation", "attention"]
            rt_fusion_idx = (
                rt_fusion_opts.index(rt_fusion_cur)
                if rt_fusion_cur in rt_fusion_opts
                else 0
            )
            retrain_fusion = st.selectbox(
                "Fusion Strategy (retrain)",
                options=rt_fusion_opts,
                index=rt_fusion_idx,
                key="retrain_fusion",
                help=(
                    "Concatenation merges data directly. Attention allows "
                    "the model to dynamically weigh which modality (Image, "
                    "Text, or Tabular) is most important."
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

        if st.button("Retrain with These Parameters", use_container_width=True):
            if not check_api_connection():
                st.error("API not connected!")
                return
            schema_data = st.session_state.detected_schema or {}
            retrain_payload = {
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
                    headers=_api_headers(),
                    timeout=30,
                )
                if resp.status_code == 200:
                    st.session_state.training_task_id = resp.json()["task_id"]
                    st.session_state.training_result = None
                    st.rerun()
                else:
                    st.error(f"Retrain failed: {resp.status_code} - {resp.text[:300]}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    if st.button("Next: Monitoring"):
        st.session_state.workflow_stage = 6
        st.rerun()


def render_phase_5_training():
    """Phase 5: GPU Training with live progress polling."""
    st.header("Phase 5 - GPU Training")

    if not st.session_state.model_selected:
        st.warning("Please select models in Phase 4 first")
        return

    task_id = st.session_state.training_task_id

    # ----- No active task: show start button or completed view -----
    if task_id is None:
        # If training already completed, show results instead of start form
        if st.session_state.get("training_result"):
            _render_completed_training(st.session_state.training_result,
                                       st.session_state.get("training_epoch_metrics", []))
            return

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **Training Pipeline:**
            - Re-registers cached datasets
            - Runs schema detection + preprocessing
            - Selects model architecture via AdvancedModelSelector
            - Optuna HPO study with GPU training
            """)
            n_trials_val: int = st.slider(
                "Optuna HPO Trials",
                min_value=3, max_value=50, value=10, step=1,
                key="n_trials",
                help="Number of hyperparameter search trials. More trials = better "
                     "search but longer training. Ignored when manual HP overrides are set.",
            )
            if st.button("Start Training", use_container_width=True):
                if not check_api_connection():
                    st.error("API not connected!")
                    return
                schema_data = st.session_state.detected_schema or {}
                payload = {
                    "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                    "modalities": schema_data.get("global_modalities", ["tabular"]),
                    "n_trials": n_trials_val,
                }
                if st.session_state.hp_overrides:
                    payload["hp_overrides"] = st.session_state.hp_overrides
                if getattr(st.session_state, "encoder_overrides", None):
                    payload["encoder_overrides"] = st.session_state.encoder_overrides
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/train-pipeline",
                        json=payload,
                        headers=_api_headers(),
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.training_task_id = data["task_id"]
                        st.rerun()
                    else:
                        st.error(f"Failed to start training: {resp.status_code} - {resp.text[:300]}")
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
        resp = requests.get(f"{API_BASE_URL}/train-pipeline/status/{task_id}", headers=_api_headers(), timeout=5)
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

    # Progress bar
    st.progress(progress_pct / 100, text=f"Phase {current_phase}/7: {current_phase_name} ({progress_pct}%)")

    # ----- Live epoch metrics (during training) -----
    epoch_metrics = task.get("epoch_metrics", [])
    trial_progress = task.get("trial_progress")
    data_split = task.get("data_split")

    if data_split and status == "running":
        ds1, ds2, ds3 = st.columns(3)
        ds1.metric("Total Samples", data_split.get("total", "?"))
        ds2.metric("Train Split", data_split.get("train", "?"))
        ds3.metric("Val Split", data_split.get("val", "?"))

    if trial_progress and status == "running":
        t_cur = trial_progress.get("current", 0)
        t_total = trial_progress.get("total", 1)
        st.info(f"Optuna Trial {t_cur}/{t_total}")

    if epoch_metrics and status == "running":
        # Show latest epoch info
        latest = epoch_metrics[-1]
        ep_cols = st.columns(5)
        ep_cols[0].metric("Epoch", f"{latest['epoch']}/{latest['max_epoch']}")
        ep_cols[1].metric("Train Loss", f"{latest['train_loss']:.4f}")
        ep_cols[2].metric("Val Loss", f"{latest['val_loss']:.4f}")
        ep_cols[3].metric("Val Acc", f"{latest['val_acc']:.3f}")
        ep_cols[4].metric("Val F1", f"{latest['val_f1']:.3f}")

        # Loss chart across all epochs (Altair: disconnected per-trial lines)
        _render_loss_chart(epoch_metrics)

    # Render phase-by-phase status
    _render_training_phases(messages, current_phase, status)

    # ----- Completed -----
    if status == "completed":
        result = task.get("result", {})
        data = result.get("data", {})

        st.session_state.training_result = data
        st.session_state.training_epoch_metrics = task.get("epoch_metrics", [])
        st.session_state.training_task_id = None

        _render_completed_training(data, st.session_state.training_epoch_metrics)

    # ----- Failed -----
    elif status == "failed":
        error = task.get("error", "Unknown error")
        st.error(f"Training failed: {error}")
        st.session_state.training_task_id = None
        if st.button("Next: Monitoring"):
            st.session_state.workflow_stage = 6
            st.rerun()

    # ----- Running: auto-refresh -----
    else:
        time.sleep(2)
        st.rerun()
        return


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
        6: "Monitoring & Drift",
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


def _render_loss_chart(epoch_metrics: List[Dict]) -> None:
    """Render loss convergence using Altair with disconnected per-trial lines.

    Each Optuna trial is a separate colour; train/val loss are distinguished
    by stroke dash.  A hidden ``absolute_step`` column enforces chronological
    X-axis ordering (no lexicographic "T0 E10 before T0 E2" issues).
    Early-stopped trials are annotated with a vertical rule at the stop step.
    """
    import altair as alt

    chart_df = pd.DataFrame(epoch_metrics)
    chart_df["absolute_step"] = range(len(chart_df))
    chart_df["Trial_ID"] = "Trial " + chart_df["trial"].astype(str)
    chart_df["label"] = (
        "T" + chart_df["trial"].astype(str) + " E" + chart_df["epoch"].astype(str)
    )

    melted = chart_df.melt(
        id_vars=["absolute_step", "Trial_ID", "label"],
        value_vars=["train_loss", "val_loss"],
        var_name="Loss Type",
        value_name="Loss",
    )

    base_chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("absolute_step:Q", title="Step", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Loss:Q", title="Loss"),
            color=alt.Color("Trial_ID:N", title="Trial"),
            strokeDash=alt.StrokeDash("Loss Type:N"),
            tooltip=["label", "Trial_ID", "Loss Type", alt.Tooltip("Loss:Q", format=".4f")],
        )
        .properties(height=350)
        .interactive()
    )

    # Annotate early-stopped trials with vertical rules at their last step
    stop_marks = []
    if "max_epoch" in chart_df.columns:
        for trial_id in chart_df["trial"].unique():
            trial_rows = chart_df[chart_df["trial"] == trial_id]
            last_epoch = int(trial_rows["epoch"].max())
            max_epoch = int(trial_rows["max_epoch"].iloc[-1])
            if last_epoch < max_epoch:
                stop_step = int(trial_rows["absolute_step"].max())
                stop_marks.append({"absolute_step": stop_step, "label": f"T{trial_id} stopped"})

    if stop_marks:
        stop_df = pd.DataFrame(stop_marks)
        rule = (
            alt.Chart(stop_df)
            .mark_rule(strokeDash=[4, 4], color="red", opacity=0.6)
            .encode(x="absolute_step:Q", tooltip=["label:N"])
        )
        chart = base_chart + rule
    else:
        chart = base_chart

    st.altair_chart(chart, use_container_width=True)


def render_phase_6_monitoring():
    """Phase 6: Monitoring & Drift Detection."""
    st.header("Phase 6 - Model Monitoring & Drift Detection")
    st.markdown("""
    **Drift Thresholds:**
    - PSI (Population Stability Index): > 0.25
    - KS (Kolmogorov-Smirnov): > 0.3
    - FDD (Feature Distribution Drift / MMD): > 0.5
    """)
    tab1, tab2, tab3, tab4 = st.tabs(["Training Results", "Drift Detection", "Model Registry", "System Benchmarks"])

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
            with st.expander("Full metrics JSON"):
                st.json(metrics)
        else:
            st.info("No training metrics yet. Run Phase 5 first.")

    # ── Tab 2: Drift detection (button-guarded, POST method) ─────────────────
    with tab2:
        st.markdown("### Drift Detection")
        if st.button("Run Drift Detection", use_container_width=True, key="drift_btn"):
            with st.spinner("Computing KS / PSI / MMD..."):
                try:
                    schema_data = st.session_state.detected_schema or {}
                    resp = requests.post(
                        f"{API_BASE_URL}/monitor/drift",
                        json={
                            "problem_type": schema_data.get("global_problem_type", "classification_binary"),
                            "modalities": schema_data.get("global_modalities", ["tabular"]),
                        },
                        headers=_api_headers(),
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        drift = resp.json().get("data", {})
                        st.session_state.drift_result = drift
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

            # Actionable recommendations
            recommendations = drift.get("recommendations", [])
            if recommendations:
                st.markdown("#### Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")

            if detected:
                if st.button("Retrain Now", type="primary", key="retrain_drift_btn"):
                    st.session_state.workflow_stage = 5
                    st.session_state.training_task_id = None
                    st.rerun()
            else:
                st.success("Model stable — no retraining needed.")
        else:
            st.info("Click 'Run Drift Detection' to compute drift metrics.")

    # ── Tab 3: Model registry (button-guarded, correct response shape) ───────
    with tab3:
        st.markdown("### Model Registry")
        if st.button("Refresh Registry", use_container_width=True, key="registry_btn"):
            try:
                resp = requests.get(f"{API_BASE_URL}/model-registry", headers=_api_headers(), timeout=30)
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
            display_cols = ["model_id", "display_name", "created_at", "status", "deployment_ready"]
            rows = [{c: m.get(c, m.get("model_id", "") if c == "display_name" else "") for c in display_cols} for m in models]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ── Rename model ──────────────────────────────────────────
            with st.expander("Rename Model"):
                rename_model_id = st.selectbox(
                    "Model to rename",
                    [m.get("model_id", "") for m in models],
                    key="rename_model_select",
                )
                new_display_name = st.text_input("New display name", key="rename_display_name")
                if st.button("Rename", key="rename_btn"):
                    if not new_display_name.strip():
                        st.error("Display name cannot be empty.")
                    else:
                        try:
                            rename_resp = requests.patch(
                                f"{API_BASE_URL}/model-registry/{rename_model_id}/rename",
                                json={"display_name": new_display_name.strip()},
                                headers=_api_headers(),
                                timeout=15,
                            )
                            if rename_resp.status_code == 200:
                                st.success(f"Renamed to '{new_display_name.strip()}'")
                                st.session_state.pop("registry_result", None)
                                st.rerun()
                            else:
                                st.error(f"Rename failed: {rename_resp.status_code}")
                        except Exception as e:
                            st.error(f"Rename error: {e}")

            # ── Per-model detail expanders ────────────────────────────
            for mdl in models:
                mid = mdl.get("model_id", "unknown")
                dname = mdl.get("display_name", mid)
                with st.expander(f"Details: {dname} ({mid})"):
                    phases = mdl.get("phases_summary", {})
                    training = phases.get("TRAINING", {})
                    enc_sel = training.get("encoder_selection", {})

                    # Encoder selection
                    if enc_sel:
                        ec1, ec2, ec3 = st.columns(3)
                        ec1.metric("Vision Encoder", enc_sel.get("image_encoder") or "N/A")
                        ec2.metric("Text Encoder", enc_sel.get("text_encoder") or "N/A")
                        ec3.metric("Tabular Encoder", enc_sel.get("tabular_encoder") or "N/A")

                    # Training metrics
                    if training:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        bvl = training.get("best_val_loss")
                        mc1.metric("Best Val Loss", f"{bvl:.4f}" if isinstance(bvl, (int, float)) else "N/A")
                        bva = training.get("best_val_acc")
                        mc2.metric("Best Val Acc", f"{bva:.2%}" if isinstance(bva, (int, float)) else "N/A")
                        bvf = training.get("best_val_f1")
                        mc3.metric("Best Val F1", f"{bvf:.4f}" if isinstance(bvf, (int, float)) else "N/A")
                        dur = training.get("duration_seconds")
                        mc4.metric("Training Time", f"{dur:.1f}s" if isinstance(dur, (int, float)) else "N/A")

                        mc5, mc6, mc7, mc8 = st.columns(4)
                        mc5.metric("Trials", training.get("n_trials", "N/A"))
                        mc6.metric("Pruned", training.get("n_pruned", 0))
                        ae = training.get("actual_epochs")
                        me = training.get("epochs")
                        mc7.metric("Epochs", f"{ae}/{me}" if ae and me else "N/A")
                        mc8.metric("Problem Type", training.get("problem_type", "N/A"))

                        # Best hyperparameters
                        bp = training.get("best_params")
                        if bp:
                            st.markdown("**Best Hyperparameters:**")
                            st.json(bp)

                    # Use for Prediction button
                    if st.button("Use for Prediction", key=f"pred_{mid}"):
                        st.session_state.pred_model_preselect = mid
                        st.session_state.workflow_stage = 7
                        st.rerun()
        else:
            st.info("No models in registry. Run the full pipeline first.")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start New Workflow"):
            st.session_state.workflow_stage = 1
            st.session_state.dataset_uploaded = False
            st.session_state.schema_detected = False
            st.session_state.model_selected = False
            st.session_state.training_result = None
            st.session_state.training_epoch_metrics = None
            st.rerun()
    with col2:
        if models:
            dl_model_id = models[0].get("model_id", "")
            if len(models) > 1:
                dl_model_id = st.selectbox(
                    "Select model",
                    [m.get("model_id", "") for m in models],
                    key="dl_model_select",
                    label_visibility="collapsed",
                )
            if st.button("Download Model"):
                try:
                    dl_resp = requests.get(
                        f"{API_BASE_URL}/model-registry/{dl_model_id}/download",
                        headers=_api_headers(),
                        timeout=120,
                    )
                    if dl_resp.status_code == 200:
                        st.download_button(
                            label="Save ZIP",
                            data=dl_resp.content,
                            file_name=f"{dl_model_id}.zip",
                            mime="application/zip",
                            key="dl_zip_btn",
                        )
                    else:
                        st.error(f"Download failed: {dl_resp.status_code}")
                except Exception as e:
                    st.error(f"Download error: {e}")
        else:
            st.button("Download Model", disabled=True, help="No models available")
    with col3:
        if models:
            del_model_id = models[0].get("model_id", "")
            if len(models) > 1:
                del_model_id = st.selectbox(
                    "Select model to delete",
                    [m.get("model_id", "") for m in models],
                    key="del_model_select",
                    label_visibility="collapsed",
                )
            if st.button("Delete Model", type="secondary"):
                try:
                    del_resp = requests.delete(
                        f"{API_BASE_URL}/model-registry/{del_model_id}",
                        headers=_api_headers(),
                        timeout=30,
                    )
                    if del_resp.status_code == 200:
                        st.success(f"Model '{del_model_id}' deleted.")
                        st.session_state.pop("registry_result", None)
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {del_resp.status_code}")
                except Exception as e:
                    st.error(f"Delete error: {e}")
        else:
            st.button("Delete Model", disabled=True, help="No models available")

    # ── Tab 4: System Benchmarks ─────────────────────────────────────────────
    with tab4:
        st.markdown("### System Benchmarks")

        training_result = st.session_state.get("training_result", {})
        _bm_metrics = training_result.get("metrics", {}) if training_result else {}
        _enc_sel = _bm_metrics.get("encoder_selection", {})

        # ── 1. JIT Encoder Profiler ──────────────────────────────────────
        st.markdown("#### JIT Encoder Profiler")
        if _enc_sel:
            jit_c1, jit_c2, jit_c3, jit_c4 = st.columns(4)
            jit_c1.metric("Selection Method", _enc_sel.get("method", "N/A"))
            _vram_budget = _enc_sel.get("vram_budget_mb", 0)
            jit_c2.metric("VRAM Budget", f"{_vram_budget:.1f} MB")
            _peak_mem = _enc_sel.get("peak_memory_mb", 0)
            jit_c3.metric("Peak Memory Used", f"{_peak_mem:.1f} MB")
            _utilization = (_peak_mem / _vram_budget * 100) if _vram_budget > 0 else 0
            jit_c4.metric("VRAM Utilization", f"{_utilization:.0f}%")

            enc_c1, enc_c2, enc_c3 = st.columns(3)
            enc_c1.metric("Image Encoder", _enc_sel.get("image_encoder") or "N/A")
            enc_c2.metric("Text Encoder", _enc_sel.get("text_encoder") or "N/A")
            enc_c3.metric("Tabular Encoder", _enc_sel.get("tabular_encoder") or "N/A")
            st.caption(f"Total capacity: {_enc_sel.get('total_capacity', 0):,} parameters")

            # Per-encoder profiling table
            _profiles = _enc_sel.get("per_encoder_profiles", [])
            if _profiles:
                st.markdown("##### Per-Encoder VRAM Profiles")
                _prof_df = pd.DataFrame(_profiles)
                # Reorder columns for readability
                _col_order = [c for c in ["modality", "name", "capacity", "peak_memory_mb", "status"]
                              if c in _prof_df.columns]
                st.dataframe(_prof_df[_col_order], use_container_width=True)

            # Rationale
            _rationale = _enc_sel.get("rationale", {})
            if _rationale:
                with st.expander("Selection Rationale"):
                    for _comp, _reason in _rationale.items():
                        st.write(f"- **{_comp}:** {_reason}")
        else:
            st.info("No JIT profiler data available. Run the training pipeline first.")

        st.markdown("---")

        # ── 2. FinOps: Frozen Encoder Reuse Savings ──────────────────────
        st.markdown("#### FinOps: Frozen Encoder Reuse Savings")
        st.caption(
            "Frozen encoders are loaded once and shared across all Optuna "
            "trials, eliminating redundant VRAM allocations."
        )
        if _enc_sel and _bm_metrics.get("n_trials"):
            _n_trials = _bm_metrics.get("n_trials", 1)
            _peak_mb = _enc_sel.get("peak_memory_mb", 0)
            _naive_total = _peak_mb * _n_trials
            _shared_total = _peak_mb
            _savings_mb = _naive_total - _shared_total
            _savings_pct = (_savings_mb / _naive_total * 100) if _naive_total > 0 else 0

            finops_data = {
                "Metric": [
                    "Encoder VRAM per load",
                    "Optuna trials",
                    "Naive total (reload each trial)",
                    "Shared total (load once)",
                    "VRAM savings",
                    "Savings %",
                ],
                "Value": [
                    f"{_peak_mb:.1f} MB",
                    str(_n_trials),
                    f"{_naive_total:.1f} MB",
                    f"{_shared_total:.1f} MB",
                    f"{_savings_mb:.1f} MB",
                    f"{_savings_pct:.0f}%",
                ],
            }
            st.table(pd.DataFrame(finops_data))
        else:
            st.info("Complete a training run to see FinOps savings.")

        st.markdown("---")

        # ── 3. Multimodal Ablation Lift ──────────────────────────────────
        st.markdown("#### Multimodal Ablation Lift")
        st.caption(
            "Shows which modalities were included and the combined result. "
            "A full ablation study requires separate training runs with "
            "each modality removed."
        )
        if _bm_metrics:
            _ablation_rows = []
            for _mod in ["image", "text", "tabular"]:
                _enc_name = _enc_sel.get(f"{_mod}_encoder") if _enc_sel else None
                if _enc_name:
                    _ablation_rows.append({
                        "Modality": _mod.capitalize(),
                        "Encoder": _enc_name,
                        "Included": "Yes",
                    })
            if _ablation_rows:
                st.dataframe(pd.DataFrame(_ablation_rows), use_container_width=True)
                _bvl = _bm_metrics.get("best_val_loss", 0)
                _bva = _bm_metrics.get("best_val_acc", 0)
                _bvf = _bm_metrics.get("best_val_f1", 0)
                st.markdown(
                    f"**Combined Result:** Val Loss = {_bvl:.4f} | "
                    f"Val Acc = {_bva:.2%} | Val F1 = {_bvf:.4f}"
                )
            else:
                st.info("No modality data available.")
        else:
            st.info("Complete a training run to see ablation data.")

        st.markdown("---")

        # ── 4. API Inference Latency ─────────────────────────────────────
        st.markdown("#### API Inference Latency")
        _pred_history = st.session_state.get("prediction_history", [])
        _latencies = [
            p["latency_ms"] for p in _pred_history
            if isinstance(p, dict) and "latency_ms" in p
        ]
        if _latencies:
            _lat_arr = np.array(_latencies)
            lat_data = {
                "Metric": ["Predictions", "Mean Latency", "P50 Latency",
                           "P95 Latency", "Min", "Max"],
                "Value": [
                    str(len(_latencies)),
                    f"{np.mean(_lat_arr):.1f} ms",
                    f"{np.median(_lat_arr):.1f} ms",
                    f"{np.percentile(_lat_arr, 95):.1f} ms",
                    f"{np.min(_lat_arr):.1f} ms",
                    f"{np.max(_lat_arr):.1f} ms",
                ],
            }
            st.table(pd.DataFrame(lat_data))
        else:
            st.info("Run predictions in Phase 7 to see inference latency metrics.")

    if st.button("Next: Prediction"):
        st.session_state.workflow_stage = 7
        st.rerun()


def render_phase_7_prediction() -> None:
    """Phase 7: Multimodal Prediction & Explainability."""
    st.header("Phase 7 - Make Predictions")

    # ── Model ID picker ──────────────────────────────────────────────────────
    st.markdown("#### Select a registered model")
    registry_model_ids: List[str] = []
    _model_display_map: Dict[str, str] = {}  # model_id → display label
    try:
        reg_resp = requests.get(f"{API_BASE_URL}/model-registry", headers=_api_headers(), timeout=15)
        if reg_resp.status_code == 200:
            reg_data = reg_resp.json()
            for m in reg_data.get("models", []):
                if m.get("deployment_ready", False):
                    mid = m["model_id"]
                    registry_model_ids.append(mid)
                    dname = m.get("display_name") or mid
                    ptype = m.get("problem_type", "")
                    _model_display_map[mid] = f"{dname}  ({ptype})" if ptype else dname
    except Exception:
        pass

    if registry_model_ids:
        # Support preselection from Phase 6 "Use for Prediction" button
        preselect = st.session_state.pop("pred_model_preselect", None)
        default_idx = 0
        if preselect and preselect in registry_model_ids:
            default_idx = registry_model_ids.index(preselect)
        display_labels = [_model_display_map.get(mid, mid) for mid in registry_model_ids]
        selected_label = st.selectbox(
            "Deployment-ready models",
            options=display_labels,
            index=default_idx,
            key="pred_model_id_select",
        )
        # Reverse-map display label back to model_id
        _label_to_id = dict(zip(display_labels, registry_model_ids))
        model_id_input: str = _label_to_id.get(selected_label, registry_model_ids[default_idx])
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
    dropped_columns: List[str] = []
    try:
        info_resp = requests.get(f"{API_BASE_URL}/model-info/{model_id_input}", headers=_api_headers(), timeout=10)
        if info_resp.status_code == 200:
            model_info = info_resp.json()
            class_labels = model_info.get("class_labels", [])
            # Prefer effective_features (post-preprocessing) over raw schema columns
            input_tabular_cols = (
                model_info.get("effective_features")
                or model_info.get("input_columns", {}).get("tabular", [])
            )
            input_text_cols = model_info.get("input_columns", {}).get("text", [])
            dropped_columns = model_info.get("dropped_columns", [])
    except Exception:
        pass

    # ── Segmentation-specific prediction UI ─────────────────────────────
    _is_segmentation = model_info.get("problem_type") == "segmentation" if model_info else False
    if _is_segmentation:
        seg_cfg = model_info.get("seg_config", {})
        st.info(
            f"**Segmentation model** — {seg_cfg.get('decoder', 'U-Net')} decoder, "
            f"{seg_cfg.get('num_classes', 2)} classes, input {seg_cfg.get('input_size', 256)}px"
        )
        seg_source = st.radio("Image source", ["Upload", "URL"], horizontal=True, key="seg_source")
        seg_image_path = None
        if seg_source == "Upload":
            seg_file = st.file_uploader(
                "Upload image for segmentation",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                key="seg_image_upload",
            )
            if seg_file is not None:
                import tempfile, os
                _seg_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                _seg_tmp.write(seg_file.getvalue())
                _seg_tmp.close()
                seg_image_path = _seg_tmp.name
                st.image(seg_file, caption="Input image", width=300)
        else:
            seg_url = st.text_input("Image URL", key="seg_image_url")
            if seg_url:
                seg_image_path = seg_url

        if st.button("Run Segmentation", type="primary", use_container_width=True) and seg_image_path:
            with st.spinner("Running segmentation inference..."):
                try:
                    seg_payload = {
                        "model_id": model_id_input,
                        "inputs": [{"image_path": seg_image_path}],
                    }
                    seg_resp = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=seg_payload,
                        headers=_api_headers(),
                        timeout=120,
                    )
                    if seg_resp.status_code == 200:
                        seg_result = seg_resp.json()
                        st.success("Segmentation complete!")

                        import numpy as _np
                        mask_data = seg_result.get("predictions", [[]])[0]
                        mask_array = _np.array(mask_data, dtype=_np.uint8)
                        confidence = seg_result.get("confidences", [0])[0]

                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.metric("Mean Confidence", f"{confidence:.2%}")
                            n_cls = seg_cfg.get("num_classes", 2)
                            st.metric("Num Classes", n_cls)

                        # Visualise mask as colour-mapped image
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib.colors as mcolors
                            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                            if seg_image_path and not seg_image_path.startswith("http"):
                                from PIL import Image as _PILImage
                                orig = _PILImage.open(seg_image_path).convert("RGB")
                                axes[0].imshow(orig)
                                axes[0].set_title("Input Image")
                                axes[0].axis("off")
                            else:
                                axes[0].text(0.5, 0.5, "Original", ha="center")
                                axes[0].axis("off")

                            n_cls = seg_cfg.get("num_classes", 2)
                            cmap = plt.cm.get_cmap("tab20", n_cls)
                            axes[1].imshow(mask_array, cmap=cmap, vmin=0, vmax=n_cls - 1)
                            axes[1].set_title("Predicted Mask")
                            axes[1].axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        except ImportError:
                            st.write("Install matplotlib for mask visualisation.")
                            st.write(f"Mask shape: {mask_array.shape}")

                        with st.expander("Raw API Response"):
                            st.json(seg_result)
                    else:
                        st.error(f"Segmentation failed: {seg_resp.status_code}")
                except Exception as e:
                    st.error(f"Segmentation error: {e}")
        return  # Segmentation path complete; skip standard prediction UI

    # ── XAI settings ─────────────────────────────────────────────────────────
    _XAI_METHOD_MAP = {
        "IntegratedGradients": "ig",
        "GradientShap": "gradient_shap",
        "Saliency": "saliency",
        "Occlusion": "occlusion",
        "FeatureAblation": "feature_ablation",
        "SmoothGrad": "smoothgrad",
    }
    with st.expander("Explainability (XAI) settings", expanded=False):
        enable_xai: bool = st.checkbox("Enable XAI explanations", value=False)
        xai_method_display: str = st.selectbox(
            "Attribution method",
            options=list(_XAI_METHOD_MAP.keys()),
            index=0,
            help=(
                "Select the Captum attribution method. IntegratedGradients is the "
                "default and most interpretable. GradientShap is faster for large "
                "inputs. Saliency computes raw gradients. SmoothGrad adds noise "
                "sampling for smoother attribution maps."
            ),
        )
        xai_method: str = _XAI_METHOD_MAP.get(xai_method_display, "ig")
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
        xai_n_steps: int = st.slider("Integration steps", min_value=10, max_value=200, value=50, step=10)

    # ── Classification threshold ──────────────────────────────────────────────
    pred_threshold: float = st.slider(
        "Classification threshold (binary/multilabel)",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        key="pred_threshold",
        help="Decision boundary for binary and multilabel classification. "
             "Higher values produce fewer (more confident) positive predictions.",
    )

    # ── Input mode ───────────────────────────────────────────────────────────
    input_mode: str = st.radio(
        "Input mode",
        options=["Single Sample", "Batch Upload (CSV)"],
        horizontal=True,
        key="pred_input_mode",
    )

    raw_inputs: List[Dict] = []

    if input_mode == "Batch Upload (CSV)":
        st.markdown("##### Upload CSV file")

        # ── Schema guidance: show required columns and provide template ──
        _all_required_cols: List[str] = list(input_tabular_cols) + list(input_text_cols)
        if _all_required_cols:
            _col_list = ", ".join(f"`{c}`" for c in _all_required_cols)
            st.info(
                f"**Required CSV columns:** {_col_list}\n\n"
                + ("Text columns should contain raw text strings. " if input_text_cols else "")
                + "Extra columns will be ignored; missing columns will be zero-filled."
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
                st.dataframe(df_batch.head(5), use_container_width=True)
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
                        # Encode as base64 data URI so it works across machines
                        import base64 as _b64
                        _b64_str = _b64.b64encode(img_file.getvalue()).decode("utf-8")
                        _mime = img_file.type or "image/jpeg"
                        sample_input["image_path"] = f"data:{_mime};base64,{_b64_str}"
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

    if st.button("Run Prediction", use_container_width=True, type="primary"):
        payload: Dict = {
            "model_id": model_id_input,
            "inputs": raw_inputs,
            "explain": enable_xai,
            "target_class": int(xai_target_class),
            "n_steps": int(xai_n_steps),
            "method": xai_method,
            "threshold": pred_threshold,
        }

        # ── Fire async task and poll until done ─────────────────────────────
        _pred_start = time.time()
        with st.spinner("Submitting inference task..."):
            try:
                submit_resp = requests.post(
                    f"{API_BASE_URL}/predict-async", json=payload, headers=_api_headers(), timeout=30,
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
                    f"{API_BASE_URL}/task/{task_id}", headers=_api_headers(), timeout=10,
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

        # Surface schema mismatch warnings from the inference engine
        inference_warnings = result.get("warnings", [])
        for warn_msg in inference_warnings:
            st.warning(warn_msg)

        # Helper: format a confidence value (scalar or per-class list)
        def _fmt_conf(c):
            if isinstance(c, list):
                return f"{max(c):.3f}" if c else "N/A"
            return f"{c:.3f}"

        # Results table
        if n_samples > 1:
            pred_df = pd.DataFrame({
                "Sample": list(range(n_samples)),
                "Prediction": predictions,
                "Confidence": [_fmt_conf(c) for c in confidences],
            })
            st.dataframe(pred_df, use_container_width=True)

            # Batch confidence distribution
            _flat_confs = []
            for _c in confidences:
                if isinstance(_c, list):
                    _flat_confs.append(max(_c) if _c else 0.0)
                else:
                    _flat_confs.append(float(_c))
            if _flat_confs:
                st.markdown("##### Confidence Distribution")
                _hist_df = pd.DataFrame({"Confidence": _flat_confs})
                st.bar_chart(_hist_df.value_counts(bins=10).sort_index())
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", str(predictions[0]) if predictions else "N/A")
            c2.metric("Confidence", _fmt_conf(confidences[0]) if confidences else "N/A")
            c3.metric("Problem type", problem_type)

            # Single-sample per-class probability chart
            if confidences and isinstance(confidences[0], list) and len(confidences[0]) > 1:
                _conf_labels = class_labels[:len(confidences[0])] if class_labels else [
                    f"Class {i}" for i in range(len(confidences[0]))
                ]
                _conf_df = pd.DataFrame(
                    {"Probability": confidences[0]},
                    index=_conf_labels,
                )
                st.markdown("##### Class Probabilities")
                st.bar_chart(_conf_df)

        # ── XAI panel ────────────────────────────────────────────────────────
        explanations: Optional[Dict] = result.get("explanations")
        if explanations:
            st.markdown("---")
            st.markdown("#### Explainability")
            _xai_method_label = explanations.get("method", "IntegratedGradients")
            st.caption(f"Attribution method: **{_xai_method_label}**")

            # Convergence delta (for IG / GradientShap)
            _conv_delta = explanations.get("convergence_delta")
            if _conv_delta is not None:
                st.caption(f"Convergence delta: {_conv_delta:.6f} (lower = more accurate)")

            xai_tabs = st.tabs([
                "Tabular Feature Importance",
                "Text Token Heatmap",
                "Image Saliency",
                "Modality Attention",
            ])

            with xai_tabs[0]:
                tab_xai = explanations.get("tabular")
                if tab_xai:
                    feat_names: List[str] = tab_xai.get("feature_names", [])
                    mean_attrs: List[float] = tab_xai.get("attributions", [])
                    if feat_names and mean_attrs:
                        _col_label = f"Importance (mean |{_xai_method_label}|)"
                        attr_df = pd.DataFrame(
                            {_col_label: mean_attrs},
                            index=feat_names,
                        ).sort_values(_col_label, ascending=False)
                        st.bar_chart(attr_df)
                    else:
                        st.info("No tabular attributions returned.")
                else:
                    _diag = explanations.get("diagnostics", {})
                    st.info(_diag.get("tabular", "No tabular attributions — model may not have a tabular modality."))

            with xai_tabs[1]:
                text_xai = explanations.get("text")
                if text_xai:
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
                else:
                    _diag = explanations.get("diagnostics", {})
                    st.info(_diag.get("text", "No text attributions — model may not have a text modality."))

            with xai_tabs[2]:
                img_xai = explanations.get("image")
                if img_xai:
                    saliency_b64 = img_xai.get("saliency_base64")
                    if saliency_b64:
                        st.image(
                            f"data:image/png;base64,{saliency_b64}",
                            caption=f"Saliency map ({img_xai.get('method', 'GradCAM')})",
                            use_container_width=True,
                        )
                    _img_note = img_xai.get("note", "")
                    if _img_note:
                        st.caption(_img_note)
                else:
                    _diag = explanations.get("diagnostics", {})
                    st.info(_diag.get("image", "No image attributions — model may not have an image modality."))

            with xai_tabs[3]:
                attn_xai = explanations.get("attention")
                if attn_xai:
                    mod_names = attn_xai.get("modality_names", [])
                    weights = attn_xai.get("weights", [])
                    if mod_names and weights:
                        attn_df = pd.DataFrame(
                            {"Attention Weight": weights},
                            index=mod_names,
                        )
                        st.bar_chart(attn_df)
                    _attn_note = attn_xai.get("note", "")
                    if _attn_note:
                        st.caption(_attn_note)
                else:
                    _diag = explanations.get("diagnostics", {})
                    st.info(_diag.get("attention", "No attention weights — model may use concatenation fusion."))

        with st.expander("Raw API response", expanded=False):
            st.json(result)

        # ── Save to prediction history ────────────────────────────────────
        _pred_latency_ms = round((time.time() - _pred_start) * 1000, 1)
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "model_id": model_id_input,
            "n_samples": n_samples,
            "prediction": str(predictions[0]) if predictions else "N/A",
            "confidence": _fmt_conf(confidences[0]) if confidences else "N/A",
            "problem_type": problem_type,
            "latency_ms": _pred_latency_ms,
        })
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

        # ── Production Drift Monitor ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Production Drift Monitor")
        try:
            _buf_resp = requests.get(
                f"{API_BASE_URL}/predict/buffer-stats/{model_id_input}",
                headers=_api_headers(),
                timeout=5,
            )
            if _buf_resp.status_code == 200:
                _buf_data = _buf_resp.json()
                _n_buf = _buf_data.get("n_buffered", 0)
                _max_buf = _buf_data.get("max_buffer", 1000)
                _col_d1, _col_d2 = st.columns([3, 1])
                with _col_d1:
                    st.caption(f"Prediction buffer: {_n_buf}/{_max_buf} samples")
                with _col_d2:
                    _run_drift = st.button(
                        "Check Drift",
                        disabled=(_n_buf < 10),
                        help="Requires at least 10 buffered predictions",
                        key="pred_drift_btn",
                    )
                if _run_drift:
                    with st.spinner("Running drift detection on prediction buffer..."):
                        _drift_resp = requests.post(
                            f"{API_BASE_URL}/predict/drift-check",
                            json={"model_id": model_id_input},
                            headers=_api_headers(),
                            timeout=30,
                        )
                    if _drift_resp.status_code == 200:
                        _drift_data = _drift_resp.json()
                        if _drift_data.get("status") == "success":
                            _d = _drift_data["data"]
                            if _d["drift_detected"]:
                                st.error("DRIFT DETECTED in prediction inputs")
                            else:
                                st.success("No significant drift detected")
                            _dm = _d["metrics"]
                            _dc1, _dc2, _dc3 = st.columns(3)
                            _dc1.metric("PSI", f"{_dm['psi']:.4f}", delta=f"threshold 0.25")
                            _dc2.metric("KS", f"{_dm['ks_statistic']:.4f}", delta=f"threshold 0.30")
                            _dc3.metric("FDD/MMD", f"{_dm['fdd']:.4f}", delta=f"threshold 0.50")
                        elif _drift_data.get("status") == "insufficient_data":
                            st.info(_drift_data.get("message", "Not enough data."))
                        else:
                            st.info(_drift_data.get("message", "Drift check unavailable."))
                    else:
                        st.warning("Drift check request failed.")
        except Exception:
            st.caption("Drift monitor unavailable.")

    # ── Prediction History ────────────────────────────────────────────────
    if st.session_state.get("prediction_history"):
        with st.expander(
            f"Prediction History ({len(st.session_state.prediction_history)} entries)",
            expanded=False,
        ):
            _hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(_hist_df, use_container_width=True)

    st.divider()
    if st.button("New Prediction"):
        st.rerun()


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


# Main execution
if __name__ == "__main__":
    # Check API connection in sidebar
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
    
    # Sidebar navigation
    st.sidebar.divider()
    st.sidebar.markdown("### 📚 Documentation")
    
    if st.sidebar.button("📖 Workflow Guide"):
        st.sidebar.info("""
        **APEX AutoML Workflow:**
        1. Upload datasets (with caching)
        2. Auto-detect schema & columns
        3. Preprocess all modalities
        4. Select models intelligently
        5. Train with GPU acceleration
        6. Monitor performance & drift
        """)
    
    st.sidebar.markdown("### 🔗 Quick Links")
    st.sidebar.link_button("🌐 API Docs", "http://localhost:8001/docs")
    st.sidebar.link_button("📋 GitHub", "https://github.com/hrishi-cz/main-project")
    
    # Main workflow
    render_workflow_dashboard()

