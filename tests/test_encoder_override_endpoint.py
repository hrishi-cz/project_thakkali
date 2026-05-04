"""Tests for encoder-overrides and fusion override API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.run_api import app

client = TestClient(app)


def _create_session():
    resp = client.post("/v2/sessions", json={"user_id": "test_user"})
    assert resp.status_code == 200, f"Session creation failed: {resp.text}"
    return resp.json()["session_id"]


class TestEncoderOverridesEndpoint:
    def test_endpoint_exists(self):
        """POST /v2/sessions/{sid}/encoder-overrides should exist and return 200."""
        sid = _create_session()
        resp = client.post(
            f"/v2/sessions/{sid}/encoder-overrides",
            json={"preferred_image_encoder": "CLIP-ViT-B/16", "reason": "test"},
        )
        assert resp.status_code == 200, f"Expected 200 got {resp.status_code}: {resp.text}"

    def test_response_has_overrides_field(self):
        sid = _create_session()
        resp = client.post(
            f"/v2/sessions/{sid}/encoder-overrides",
            json={"preferred_text_encoder": "DeBERTa-v3-base"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "encoder_overrides" in body
        assert body["encoder_overrides"].get("preferred_text_encoder") == "DeBERTa-v3-base"

    def test_multiple_overrides_accumulated(self):
        sid = _create_session()
        r1 = client.post(
            f"/v2/sessions/{sid}/encoder-overrides",
            json={"preferred_image_encoder": "CLIP-ViT-B/16"},
        )
        assert r1.status_code == 200
        r2 = client.post(
            f"/v2/sessions/{sid}/encoder-overrides",
            json={"preferred_text_encoder": "DeBERTa-v3-base"},
        )
        assert r2.status_code == 200
        body = r2.json()
        overrides = body["encoder_overrides"]
        # Both overrides should persist across calls
        assert "preferred_image_encoder" in overrides, f"Image encoder not in {overrides}"
        assert "preferred_text_encoder" in overrides, f"Text encoder not in {overrides}"

    def test_invalid_session_returns_error(self):
        resp = client.post(
            "/v2/sessions/nonexistent_session_xyz/encoder-overrides",
            json={"preferred_image_encoder": "ResNet-50"},
        )
        assert resp.status_code in (404, 400, 422, 500)

    def test_empty_payload_is_ok(self):
        """Empty payload (no overrides) should return 200 with empty overrides."""
        sid = _create_session()
        resp = client.post(
            f"/v2/sessions/{sid}/encoder-overrides",
            json={},
        )
        assert resp.status_code == 200


class TestFusionOverrideEndpoint:
    def test_fusion_override_accepted(self):
        sid = _create_session()
        for strategy in ("concatenation", "attention", "ula", "gated", "fusemoe"):
            resp = client.post(
                f"/v2/sessions/{sid}/override-fusion",
                json={"strategy": strategy, "reason": f"test {strategy}"},
            )
            assert resp.status_code == 200, (
                f"Strategy '{strategy}' rejected: {resp.status_code} {resp.text}"
            )

    def test_fusion_override_reflected_in_session(self):
        sid = _create_session()
        r = client.post(
            f"/v2/sessions/{sid}/override-fusion",
            json={"strategy": "ula", "reason": "testing ULA"},
        )
        assert r.status_code == 200
        body = r.json()
        # Response should confirm the strategy was set
        assert "ula" in str(body).lower() or body.get("status") == "ok"


class TestRenameEndpointMethodCompat:
    def test_post_rename_accepted(self):
        """Frontend uses POST for rename; API should accept POST (method compat fix)."""
        # Create a fake model dir to avoid 404
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from unittest.mock import patch
            fake_dir = Path(tmpdir) / "my_model"
            fake_dir.mkdir()
            meta = {"model_id": "my_model", "display_name": "My Model"}
            (fake_dir / "metadata.json").write_text(json.dumps(meta))

            with patch("api.run_api.MODEL_REGISTRY_DIR", Path(tmpdir)):
                resp = client.post(
                    "/model-registry/my_model/rename",
                    json={"alias": "New Display Name"},
                )
                assert resp.status_code in (200, 422), (
                    f"POST rename should be accepted, got {resp.status_code}"
                )

    def test_patch_rename_still_accepted(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from unittest.mock import patch
            fake_dir = Path(tmpdir) / "my_model2"
            fake_dir.mkdir()
            meta = {"model_id": "my_model2", "display_name": "Model 2"}
            (fake_dir / "metadata.json").write_text(json.dumps(meta))

            with patch("api.run_api.MODEL_REGISTRY_DIR", Path(tmpdir)):
                resp = client.patch(
                    "/model-registry/my_model2/rename",
                    json={"alias": "New Display Name"},
                )
                assert resp.status_code in (200, 422)


class TestSessionCleanup:
    def test_close_session_marks_closed(self):
        sid = _create_session()
        resp = client.post(f"/v2/sessions/{sid}/close")
        assert resp.status_code == 200

    def test_cleanup_stale_sessions_callable(self):
        """context_db.cleanup_stale_sessions should not raise."""
        from api.run_api import context_db as _cdb
        deleted = _cdb.cleanup_stale_sessions(max_age_hours=720)  # 30 days — won't delete test sessions
        assert isinstance(deleted, int)
        assert deleted >= 0
