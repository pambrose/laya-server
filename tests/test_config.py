import pytest

from laya_server.config import Settings


class TestDefaults:
    def test_auth_disabled_when_no_key_configured(self):
        s = Settings.from_env({})
        assert s.api_key is None
        assert s.auth_enabled is False

    def test_preloads_all_three_checkpoints(self):
        """max_loaded must cover every preloaded checkpoint so Router never
        evicts mid-request (router.py:188-195 is not concurrency-safe)."""
        s = Settings.from_env({})
        assert set(s.preload) == {"english", "multilingual", "typed-decisions"}

    def test_device_defaults_to_laya_autodetect(self):
        assert Settings.from_env({}).device is None


class TestEnvOverrides:
    def test_api_key_enables_auth(self):
        s = Settings.from_env({"LAYA_SERVER_API_KEY": "secret"})
        assert s.api_key == "secret"
        assert s.auth_enabled is True

    def test_device_override(self):
        assert Settings.from_env({"LAYA_SERVER_DEVICE": "cpu"}).device == "cpu"

    def test_preload_list_is_comma_separated(self):
        s = Settings.from_env({"LAYA_SERVER_PRELOAD": "english,multilingual"})
        assert s.preload == ("english", "multilingual")

    def test_preload_entries_are_trimmed(self):
        s = Settings.from_env({"LAYA_SERVER_PRELOAD": " english , multilingual "})
        assert s.preload == ("english", "multilingual")

    def test_max_queue_depth_override(self):
        assert Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "7"}).max_queue_depth == 7


class TestMalformedValues:
    """Issue 11: int() raised a bare ValueError naming neither the variable nor
    the value, so a typo produced an opaque startup traceback."""

    def test_non_numeric_queue_depth_names_the_variable(self):
        with pytest.raises(ValueError) as exc:
            Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "lots"})
        msg = str(exc.value)
        assert "LAYA_SERVER_MAX_QUEUE" in msg
        assert "lots" in msg

    def test_non_numeric_body_limit_names_the_variable(self):
        with pytest.raises(ValueError) as exc:
            Settings.from_env({"LAYA_SERVER_MAX_BODY_BYTES": "big"})
        assert "LAYA_SERVER_MAX_BODY_BYTES" in str(exc.value)

    def test_valid_values_still_parse(self):
        s = Settings.from_env(
            {"LAYA_SERVER_MAX_QUEUE": "7", "LAYA_SERVER_MAX_BODY_BYTES": "2048"}
        )
        assert (s.max_queue_depth, s.max_body_bytes) == (7, 2048)
