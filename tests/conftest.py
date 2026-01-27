import os
import pytest
import streamlit as st

# Ensure imports don't fail due to missing OpenAI key during tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def reset_session_state():
    """Clean Streamlit session state around each test."""
    st.session_state.clear()
    yield
    st.session_state.clear()
