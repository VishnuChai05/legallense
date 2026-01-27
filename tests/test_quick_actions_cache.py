import streamlit as st

from app import _precompute_quick_actions


def test_quick_action_cache_initializes_on_new_hash():
    _precompute_quick_actions("hash1", None)

    assert st.session_state["quick_action_cache"] == {"hash": "hash1", "data": {}}


def test_quick_action_cache_not_reset_for_same_hash():
    st.session_state["quick_action_cache"] = {"hash": "hash1", "data": {"a": 1}}

    _precompute_quick_actions("hash1", None)

    assert st.session_state["quick_action_cache"] == {"hash": "hash1", "data": {"a": 1}}


def test_quick_action_cache_resets_when_hash_changes():
    st.session_state["quick_action_cache"] = {"hash": "hash1", "data": {"a": 1}}

    _precompute_quick_actions("hash2", None)

    assert st.session_state["quick_action_cache"] == {"hash": "hash2", "data": {}}
