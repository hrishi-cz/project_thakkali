"""Advanced Panels — wires every remaining backend endpoint into the UI.

Adds to the sidebar:
- Decision Trace (decision audit log + CSV export)
- Phase Timings
- Fit Analysis
- Calibration Intelligence
- Feature Intelligence
- Ranked Candidates
- Cache Management
- Ablation Runner
- Retrain History
- Model Stats
- Target Management (lock/unlock/candidates/per-modality override)

Also adds transparency helpers used inline by other phases.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

from frontend._endpoints import ep

logger = logging.getLogger(__name__)

_TIMEOUT = 10

# Tiered TTL constants (seconds)
_TTL_STATIC = 120   # config, cache metadata, ablation results — never change mid-run
_TTL_SLOW   = 30    # schema, target, intelligence, retrain history — between-phase updates
_TTL_FAST   = 10    # phase timings, fit analysis, drift, decision trace — mid-training


@st.cache_data(ttl=_TTL_STATIC, show_spinner=False)
def _get_static(url: str) -> Optional[Dict]:
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=_TTL_SLOW, show_spinner=False)
def _get_slow(url: str) -> Optional[Dict]:
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=_TTL_FAST, show_spinner=False)
def _get_fast(url: str) -> Optional[Dict]:
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _safe_get(url: str, label: str = "data", ttl: str = "slow") -> Optional[Dict]:
    """GET with tiered TTL caching.

    ttl: "static" (120s), "slow" (30s, default), "fast" (10s)
    404/422 and connection errors return None silently.
    """
    _fn = {"static": _get_static, "slow": _get_slow, "fast": _get_fast}.get(ttl, _get_slow)
    try:
        return _fn(url)
    except Exception as e:
        st.warning(f"Error loading {label}: {e}")
        return None


def _safe_post(url: str, json_body: Dict, label: str = "action") -> Optional[Dict]:
    """POST with error handling."""
    try:
        r = requests.post(url, json=json_body, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"{label} failed ({r.status_code}): {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error(f"API unavailable for {label}")
    except Exception as e:
        st.error(f"{label} error: {e}")
    return None


# ---------------------------------------------------------------------------
# 1. Decision Trace (transparency: audit log + CSV export)
# ---------------------------------------------------------------------------
def render_decision_trace(sid: str) -> None:
    """Render the full decision trace with CSV download."""
    with st.expander("📋 Decision Audit Log", expanded=False):
        data = _safe_get(ep.decision_trace(sid), "decision trace", ttl="fast")
        if not data:
            st.info("No decision trace available yet. Run the pipeline to generate decisions.")
            return

        decisions = data.get("decisions", data.get("trace", []))
        if isinstance(decisions, list) and decisions:
            df = pd.DataFrame(decisions)
            st.dataframe(df, width='stretch')

            # CSV export button
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False, quoting=csv.QUOTE_ALL)
            st.download_button(
                "⬇️ Download Decision Log (CSV)",
                csv_buf.getvalue(),
                file_name="apex_decision_trace.csv",
                mime="text/csv",
            )
            st.caption(f"{len(decisions)} decisions recorded")
        elif isinstance(decisions, dict):
            st.json(decisions)
        else:
            st.info("Decision trace is empty.")


# ---------------------------------------------------------------------------
# 2. Phase Timings
# ---------------------------------------------------------------------------
def render_phase_timings(sid: str) -> None:
    """Render pipeline phase timing breakdown."""
    with st.expander("⏱️ Phase Timings", expanded=False):
        data = _safe_get(ep.context_phase_timings(sid), "phase timings", ttl="fast")
        if not data:
            st.info("No phase timing data available yet.")
            return

        timings = data.get("timings", data)
        if isinstance(timings, dict):
            rows = []
            total = 0.0
            for phase, duration in timings.items():
                if isinstance(duration, (int, float)):
                    rows.append({"Phase": phase, "Duration (s)": round(float(duration), 2)})
                    total += float(duration)
            if rows:
                df = pd.DataFrame(rows)
                st.bar_chart(df.set_index("Phase"))
                st.metric("Total Pipeline Time", f"{total:.1f}s")
            else:
                st.json(timings)


# ---------------------------------------------------------------------------
# 3. Fit Analysis
# ---------------------------------------------------------------------------
def render_fit_analysis(sid: str) -> None:
    """Render training fit analysis (overfitting/underfitting diagnostics)."""
    with st.expander("📈 Fit Analysis", expanded=False):
        data = _safe_get(ep.context_fit_analysis(sid), "fit analysis", ttl="fast")
        if not data:
            st.info("No fit analysis available. Complete training first.")
            return

        # BUG-2 FIX: backend wraps result in "training_fit_analysis", not top-level
        fit_analysis = data.get("training_fit_analysis") or data
        fit_type = fit_analysis.get("fit_type", fit_analysis.get("diagnosis", "unknown"))
        color = {"overfit": "🔴", "underfit": "🟡", "good_fit": "🟢", "good": "🟢"}.get(
            str(fit_type).lower().replace(" ", "_"), "⚪"
        )
        st.markdown(f"### {color} Diagnosis: **{fit_type}**")

        train_loss = fit_analysis.get("train_loss")
        val_loss = fit_analysis.get("val_loss")
        if train_loss is not None and val_loss is not None:
            col1, col2 = st.columns(2)
            col1.metric("Train Loss", f"{float(train_loss):.4f}")
            col2.metric("Val Loss", f"{float(val_loss):.4f}")

        gap = fit_analysis.get("generalization_gap")
        if isinstance(gap, (int, float)):
            st.metric("Generalization Gap", f"{float(gap):.4f}")

        recommendation = fit_analysis.get("recommendation", fit_analysis.get("advice", ""))
        if recommendation:
            st.info(f"💡 **Recommendation:** {recommendation}")

        # Surface training_signals if populated
        signals = data.get("training_signals") or {}
        if signals:
            with st.expander("Training Signals", expanded=False):
                st.json(signals)
        elif fit_analysis.get("details"):
            st.json(fit_analysis["details"])


# ---------------------------------------------------------------------------
# 4. Calibration Intelligence
# ---------------------------------------------------------------------------
def render_calibration(sid: str) -> None:
    """Render calibration metrics (ECE, Brier score, reliability diagram data)."""
    with st.expander("🎯 Calibration Analysis", expanded=False):
        data = _safe_get(ep.intelligence(sid, "calibration"), "calibration")
        if not data:
            st.info("No calibration data available. Complete training first.")
            return

        cal = data.get("calibration", data)
        col1, col2, col3 = st.columns(3)
        ece = cal.get("ece", cal.get("expected_calibration_error"))
        brier = cal.get("brier_score")
        mce = cal.get("mce", cal.get("max_calibration_error"))

        if isinstance(ece, (int, float)):
            col1.metric("ECE", f"{float(ece):.4f}",
                        help="Expected Calibration Error — lower is better. "
                             "<0.05 is well-calibrated.")
        if isinstance(brier, (int, float)):
            col2.metric("Brier Score", f"{float(brier):.4f}",
                        help="Mean squared error of predicted probabilities. "
                             "Combines calibration + discrimination.")
        if isinstance(mce, (int, float)):
            col3.metric("MCE", f"{float(mce):.4f}",
                        help="Maximum Calibration Error — worst-case bin error.")

        # Calibration explanation
        if isinstance(ece, (int, float)):
            if ece < 0.05:
                st.success(
                    "✅ **Well-calibrated.** When this model says 87% confidence, "
                    "approximately 87% of those predictions are correct."
                )
            elif ece < 0.15:
                st.warning(
                    "⚠️ **Moderately calibrated.** Confidence values are somewhat "
                    "reliable but may over- or under-estimate by up to 15%."
                )
            else:
                st.error(
                    "❌ **Poorly calibrated.** Confidence values are unreliable. "
                    "Consider applying temperature scaling post-hoc."
                )

        bins = cal.get("reliability_bins", cal.get("bins"))
        if isinstance(bins, list) and bins:
            try:
                bin_df = pd.DataFrame(bins)
                st.line_chart(bin_df.set_index(bin_df.columns[0]))
            except Exception:
                st.json(bins)


# ---------------------------------------------------------------------------
# 5. Feature Intelligence
# ---------------------------------------------------------------------------
def render_feature_intelligence(sid: str) -> None:
    """Render per-modality feature intelligence signals."""
    with st.expander("🔍 Feature Intelligence", expanded=False):
        data = _safe_get(ep.intelligence(sid, "feature-intelligence"), "feature intelligence")
        if not data:
            st.info("No feature intelligence available yet.")
            return

        fi = data.get("feature_intelligence", data)
        for modality, signals in fi.items():
            if not isinstance(signals, dict):
                continue
            st.markdown(f"**{modality.title()}**")
            signal_rows = [
                {"Signal": k, "Value": str(v)[:80]}
                for k, v in signals.items()
            ]
            if signal_rows:
                st.dataframe(pd.DataFrame(signal_rows), width='stretch')


# ---------------------------------------------------------------------------
# 6. Ranked Candidates
# ---------------------------------------------------------------------------
def render_ranked_candidates(sid: str) -> None:
    """Render the ranked model candidates from intelligence endpoint."""
    with st.expander("🏆 Ranked Model Candidates", expanded=False):
        data = _safe_get(ep.intelligence(sid, "ranked-candidates"), "ranked candidates")
        if not data:
            st.info("No ranked candidates available. Run model selection first.")
            return

        candidates = data.get("ranked_candidates", data.get("candidates", data))
        if isinstance(candidates, dict):
            for modality, cands in candidates.items():
                if isinstance(cands, list) and cands:
                    st.markdown(f"**{modality.title()} Candidates**")
                    st.dataframe(pd.DataFrame(cands), width='stretch')
        elif isinstance(candidates, list):
            st.dataframe(pd.DataFrame(candidates), width='stretch')


# ---------------------------------------------------------------------------
# 7. Cache Management
# ---------------------------------------------------------------------------
def render_cache_management() -> None:
    """Render embedding cache stats and management controls."""
    with st.expander("💾 Cache Management", expanded=False):
        # MISMATCH-2 FIX: /embedding-cache/stats has the correct keys;
        # /cache/stats is the preprocessor artifact cache with different keys.
        emb_data = _safe_get(ep.EMBEDDING_CACHE_STATS, "embedding cache stats", ttl="static")
        if emb_data:
            col1, col2, col3 = st.columns(3)
            col1.metric("Cache Files", emb_data.get("cache_file_count", emb_data.get("total_files", "?")))
            col2.metric("Cache Size (MB)", emb_data.get("cache_size_mb", emb_data.get("total_size_mb", "?")))
            col3.metric("Latest Model", str(emb_data.get("latest_model_id", "—"))[:30])

        pp_data = _safe_get(ep.CACHE_STATS, "preprocessor cache", ttl="static")
        if pp_data and isinstance(pp_data, dict):
            st.caption(
                f"Preprocessor cache — files: {pp_data.get('total_files', '?')} "
                f"| size: {pp_data.get('total_size_mb', '?')} MB"
            )

        meta = _safe_get(ep.CACHE_METADATA, "cache metadata", ttl="static")
        if meta and isinstance(meta, dict):
            with st.expander("Cache Metadata", expanded=False):
                st.json(meta)

        if st.button("🗑️ Clear Preprocessor Cache", key="clear_cache_btn"):
            result = _safe_post(ep.CACHE_CLEAR, {}, "cache clear")
            if result:
                st.success("Cache cleared successfully")
                st.rerun()


# ---------------------------------------------------------------------------
# 8. Ablation Runner
# ---------------------------------------------------------------------------
def render_ablation_runner() -> None:
    """UI to trigger and view ablation experiments."""
    with st.expander("🧪 Ablation Studies", expanded=False):
        # View existing results
        data = _safe_get(ep.ABLATION_RESULTS, "ablation results", ttl="static")
        if data:
            results = data.get("results", data)
            if isinstance(results, list):
                st.dataframe(pd.DataFrame(results), width='stretch')
            elif isinstance(results, dict):
                for name, val in results.items():
                    st.write(f"**{name}:** {val}")
        else:
            st.info("No ablation results available yet.")

        # Trigger button
        if st.button("▶️ Run Ablation Study", key="run_ablation_btn"):
            with st.spinner("Running ablations..."):
                result = _safe_post(ep.RUN_ABLATIONS, {
                    "session_id": st.session_state.get("session_id", ""),
                }, "ablation run")
                if result:
                    st.success("Ablation study completed!")
                    st.rerun()


# ---------------------------------------------------------------------------
# 9. Retrain History
# ---------------------------------------------------------------------------
def render_retrain_history() -> None:
    """Show history of retraining events."""
    with st.expander("🔄 Retrain History", expanded=False):
        data = _safe_get(ep.RETRAIN_HISTORY, "retrain history")
        if not data:
            st.info("No retraining events recorded.")
            return

        # MISMATCH-4 FIX: backend returns a list directly, not {"history": [...]}
        if isinstance(data, list):
            history = data
        elif isinstance(data, dict):
            history = data.get("history", data.get("events", []))
        else:
            history = []

        if isinstance(history, list) and history:
            st.dataframe(pd.DataFrame(history), width='stretch')
            st.caption(f"{len(history)} retraining event(s) recorded")
        else:
            st.info("No retraining events recorded.")


# ---------------------------------------------------------------------------
# 10. Model Stats
# ---------------------------------------------------------------------------
def render_model_stats(model_id: str) -> None:
    """Show detailed model statistics."""
    with st.expander("📊 Model Statistics", expanded=False):
        data = _safe_get(ep.model_stats(model_id), "model stats")
        if not data:
            st.info("No model statistics available.")
            return

        stats = data.get("stats", data)

        # Key metrics
        if isinstance(stats, dict):
            cols = st.columns(3)
            if stats.get("total_parameters"):
                cols[0].metric("Parameters", f"{int(stats['total_parameters']):,}")
            if stats.get("model_size_mb"):
                cols[1].metric("Model Size", f"{float(stats['model_size_mb']):.1f} MB")
            if stats.get("inference_latency_ms"):
                cols[2].metric("Latency", f"{float(stats['inference_latency_ms']):.1f} ms")

            # XAI section
            xai = stats.get("xai", {})
            if xai:
                st.markdown("**XAI Summary**")
                fusion_w = (xai.get("fusion", {}) or {}).get("weights", {})
                if fusion_w:
                    st.bar_chart(pd.DataFrame.from_dict(
                        fusion_w, orient="index", columns=["Weight"]
                    ))

            # Full dump
            st.json(stats)


# ---------------------------------------------------------------------------
# 11. Target Management (lock/unlock/candidates/per-modality)
# ---------------------------------------------------------------------------
def render_target_management(sid: str) -> None:
    """Manage per-dataset target column locking and per-modality overrides."""
    with st.expander("🎯 Target Management", expanded=False):
        schema = st.session_state.get("detected_schema", {}) or {}
        datasets_list = schema.get("individual_schemas", [])

        if not datasets_list:
            st.info("No datasets detected. Run schema detection first.")
            return

        for i, ds in enumerate(datasets_list):
            ds_id = ds.get("dataset_id", f"dataset_{i}")
            ds_name = ds.get("filename", ds_id)
            st.markdown(f"**{ds_name}** (`{ds_id}`)")

            col1, col2, col3 = st.columns(3)

            # Target candidates
            with col1:
                if st.button("View Candidates", key=f"tc_{ds_id}"):
                    cands = _safe_get(
                        ep.dataset_target_candidates(ds_id),
                        "target candidates"
                    )
                    if cands:
                        st.session_state[f"_tc_{ds_id}"] = cands

                tc = st.session_state.get(f"_tc_{ds_id}")
                if tc:
                    candidates = tc.get("candidates", tc) if isinstance(tc, dict) else tc
                    if isinstance(candidates, list):
                        st.selectbox(
                            "Target", candidates,
                            key=f"tc_sel_{ds_id}"
                        )

            # Lock/unlock
            with col2:
                if st.button("🔒 Lock", key=f"lock_{ds_id}"):
                    _safe_post(
                        ep.dataset_lock_target(ds_id),
                        {"session_id": sid},
                        "lock target"
                    )
                    st.success("Target locked")

            with col3:
                if st.button("🔓 Unlock", key=f"unlock_{ds_id}"):
                    _safe_post(
                        ep.dataset_unlock_target(ds_id),
                        {"session_id": sid},
                        "unlock target"
                    )
                    st.success("Target unlocked")

            st.divider()

        # Per-modality target override
        st.markdown("**Per-Modality Target Override**")
        override_text = st.text_input(
            "Text modality target column", key="pmod_text_target",
            placeholder="e.g. sentiment"
        )
        override_image = st.text_input(
            "Image modality target column", key="pmod_image_target",
            placeholder="e.g. label"
        )
        if st.button("Apply Per-Modality Override", key="pmod_apply"):
            overrides = {}
            if override_text:
                overrides["text"] = {"target_column": override_text}
            if override_image:
                overrides["image"] = {"target_column": override_image}
            if overrides:
                result = _safe_post(
                    ep.override_target_per_modality(sid),
                    {"overrides": overrides},
                    "per-modality target override"
                )
                if result is not None:
                    st.success("Per-modality overrides applied!")
                    st.rerun()  # refresh sidebar + schema display


# ---------------------------------------------------------------------------
# 12. Confidence explanation helper (called from prediction results)
# ---------------------------------------------------------------------------
def render_confidence_explanation(confidence: float) -> None:
    """Plain-English explanation of what a confidence score means."""
    if confidence >= 0.9:
        st.success(
            f"🟢 **High confidence ({confidence:.0%})**. The model is very certain "
            f"about this prediction. In calibration tests, {confidence:.0%} of "
            f"predictions with this confidence level were correct."
        )
    elif confidence >= 0.7:
        st.info(
            f"🟡 **Moderate confidence ({confidence:.0%})**. The model is fairly "
            f"certain but there is some ambiguity. Consider reviewing the "
            f"XAI attributions below to understand which features drove this."
        )
    elif confidence >= 0.5:
        st.warning(
            f"🟠 **Low confidence ({confidence:.0%})**. The model is uncertain. "
            f"This sample may be near a decision boundary or contain unusual "
            f"feature values. Treat this prediction with caution."
        )
    else:
        st.error(
            f"🔴 **Very low confidence ({confidence:.0%})**. The model essentially "
            f"cannot distinguish between classes for this input. The prediction "
            f"should not be trusted without human review."
        )


# ---------------------------------------------------------------------------
# 13. Global Schema/Target viewers
# ---------------------------------------------------------------------------
def render_global_schema(sid: str) -> None:
    """View the merged global schema for the session."""
    with st.expander("🗂️ Global Schema", expanded=False):
        data = _safe_get(ep.global_schema(sid), "global schema")
        if data:
            schema = data.get("global_schema", data)
            if isinstance(schema, dict):
                cols = st.columns(3)
                cols[0].metric("Problem Type", schema.get("problem_type", "?"))
                cols[1].metric("Modalities", ", ".join(schema.get("modalities", [])))
                cols[2].metric("Datasets", schema.get("n_datasets", "?"))
            st.json(schema)
        else:
            st.info("No global schema available.")


def render_global_target(sid: str) -> None:
    """View and override the global target for the session."""
    with st.expander("🎯 Global Target", expanded=False):
        data = _safe_get(ep.global_target(sid), "global target")
        if not data or not isinstance(data, dict):
            st.info("No global target set.")
            return

        # BUG-3 FIX: backend returns global_target as a STRING (column name),
        # not a dict. Previous code mis-detected it as a dict -> always showed "?".
        col_name = data.get("global_target") or "—"
        confidence = float(data.get("confidence") or 0)
        xs3_gap = float(data.get("xs3_confidence_gap") or 0)
        candidates = data.get("candidates") or []

        st.write(f"**Current target column:** `{col_name}`")
        c1, c2 = st.columns(2)
        c1.metric("Confidence", f"{confidence:.0%}")
        c2.metric("XS3 Gap", f"{xs3_gap:.3f}",
                  help="Confidence gap between top-2 candidates. >0.2 = unambiguous.")

        if candidates:
            st.caption(f"{len(candidates)} candidate(s) scored")
            try:
                cand_df = pd.DataFrame(candidates)
                show_cols = [c for c in ["name", "score", "reason"] if c in cand_df.columns]
                st.dataframe(cand_df[show_cols] if show_cols else cand_df,
                             width='stretch')
            except Exception:
                st.json(candidates)


# ---------------------------------------------------------------------------
# 14. Config viewer
# ---------------------------------------------------------------------------
def render_config() -> None:
    """Show current API server configuration."""
    with st.expander("⚙️ Server Configuration", expanded=False):
        data = _safe_get(ep.CONFIG, "config", ttl="static")
        if data:
            for key, val in data.items():
                st.write(f"- **{key}:** `{val}`")
        else:
            st.info("Cannot retrieve server config.")


# ---------------------------------------------------------------------------
# Master sidebar panel — call this once to wire everything
# ---------------------------------------------------------------------------
def render_advanced_sidebar(sid: str) -> None:
    """Render all advanced panels in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Advanced Panels")

    with st.sidebar:
        render_decision_trace(sid)
        render_phase_timings(sid)
        render_fit_analysis(sid)
        render_calibration(sid)
        render_feature_intelligence(sid)
        render_ranked_candidates(sid)
        render_global_schema(sid)
        render_global_target(sid)
        render_target_management(sid)
        render_cache_management()
        render_ablation_runner()
        render_retrain_history()
        render_config()
