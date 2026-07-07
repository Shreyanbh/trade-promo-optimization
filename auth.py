"""
Dashboard password gate.

Set DASHBOARD_PASSWORD in .env to enable authentication.
Leave it unset (or empty) to run without a password — suitable for local use only.

The password is compared with a timing-safe hash comparison so it can't be
brute-forced via timing side-channels.
"""

import os
import hashlib
import hmac

import streamlit as st

from src.security import audit


def _configured_password() -> str:
    # Priority: env var → streamlit secrets → empty (no auth)
    pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if not pw:
        try:
            pw = st.secrets.get("DASHBOARD_PASSWORD", "")
        except Exception:
            pw = ""
    return pw


def _hash(pw: str) -> bytes:
    return hashlib.sha256(pw.encode()).digest()


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(_hash(a), _hash(b))


def check_auth() -> None:
    """
    Call at the top of app.py before rendering any page content.
    Shows a login form and calls st.stop() if the user is not authenticated.
    Does nothing if DASHBOARD_PASSWORD is not set.
    """
    configured = _configured_password()
    if not configured:
        return

    if st.session_state.get("_auth_ok"):
        return

    st.markdown(
        """
        <style>
        #MainMenu, footer, header {visibility: hidden;}
        .auth-box {
            max-width: 380px; margin: 120px auto; padding: 40px;
            background: #0d1524; border: 1px solid #1e3a5f;
            border-radius: 12px; text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    st.markdown("## Sign in")
    st.caption("This dashboard is password-protected.")
    entered = st.text_input("Password", type="password", key="_login_pw",
                             label_visibility="collapsed",
                             placeholder="Enter dashboard password")
    if st.button("Sign in", type="primary", use_container_width=True):
        if _constant_time_compare(entered, configured):
            st.session_state["_auth_ok"] = True
            audit.log_auth(success=True)
            st.rerun()
        else:
            audit.log_auth(success=False)
            st.error("Incorrect password.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def logout():
    st.session_state.pop("_auth_ok", None)
