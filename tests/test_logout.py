import streamlit as st

from app import _logout


class DummyAuth:
    def __init__(self):
        self.called = False

    def sign_out(self):
        self.called = True


class DummySupabase:
    def __init__(self):
        self.auth = DummyAuth()


def test_logout_clears_session_state():
    # seed keys
    st.session_state.update({
        "authenticated": True,
        "username": "user@example.com",
        "quick_action_cache": {"hash": "h1"},
        "current_file_hash": "abc",
        "other": 123,
    })

    supabase = DummySupabase()

    _logout(supabase)

    # cleared keys
    for key in [
        "authenticated",
        "username",
        "login_error",
        "quick_action_output",
        "quick_action_type",
        "quick_action_title",
        "voice_audio_bytes",
        "voice_result",
        "voice_clear_pending",
        "embedding_notice_shown",
        "messages",
        "supabase_access_token",
        "supabase_refresh_token",
        "quick_action_cache",
        "current_file_hash",
    ]:
        assert key not in st.session_state

    # untouched unrelated keys
    assert "other" in st.session_state

    # sign_out called
    assert supabase.auth.called is True


def test_logout_handles_signout_errors_gracefully():
    class FailingAuth:
        def __init__(self):
            self.called = False

        def sign_out(self):
            self.called = True
            raise RuntimeError("boom")

    class FailingSupabase:
        def __init__(self):
            self.auth = FailingAuth()

    st.session_state.clear()
    st.session_state["quick_action_cache"] = {"hash": "h1"}

    supabase = FailingSupabase()
    # Should not raise
    _logout(supabase)

    assert "quick_action_cache" not in st.session_state
    assert supabase.auth.called is True
