"""
Main application file for the DriverPulse AI web app.

This module defines the Streamlit interface for the driver portal.
Users can sign up or log in using their email and driver ID. After
authentication, drivers can navigate between different pages (My
Dashboard, My Safety, Earnings & Goals, My Driving Behavior).
Only the dashboard page is implemented here; other pages provide
placeholder messages to be completed by other team members.

Run this file with Streamlit:

```
streamlit run driver_pulse_ai/app.py
```
"""

import os
import sys
import importlib
import streamlit as st

from auth.auth_utils import (
    create_connection,
    create_user_table,
    add_user,
    get_user_by_email,
    verify_user,
)
from utils.data_loader import load_data


def load_script_page(module_name: str) -> None:
    """Import a script-style Streamlit page exactly once."""
    if module_name in sys.modules:
        del sys.modules[module_name]
    importlib.import_module(module_name)




def apply_global_style() -> None:
    """Apply a light editorial theme across the whole app."""
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');
      .stApp {
        background: #ffffff;
        color: #0a0a0a;
        font-family: 'DM Sans', sans-serif;
      }
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding-top: 1.5rem; max-width: 1440px; }
      h1, h2, h3 { color: #0a0a0a !important; }
      [data-testid="stSidebar"] { background: #f7f7f5; border-right: 1px solid #e5e7eb; }
      [data-testid="stSidebar"] * { color: #0a0a0a; }
      .stButton > button {
        background: #0a0a0a; color: #ffffff; border: none; border-radius: 10px;
        font-family: 'DM Sans', sans-serif; font-weight: 600; padding: 0.55rem 1rem;
      }
      .stButton > button:hover { background: #222; color: #fff; }
      .stTextInput > div > div > input, .stSelectbox > div > div {
        border-radius: 10px !important; border: 1px solid #d1d5db !important; background: #ffffff !important; color: #0a0a0a !important;
      }
      .stSelectbox svg { fill: #0a0a0a !important; }
      .stRadio label { font-family: 'DM Sans', sans-serif; }
      .app-hero-title { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; }
      .app-subtle { color: #6b7280; font-size: 0.95rem; }

      /* FIX FORM LABEL VISIBILITY */
      div[data-testid="stWidgetLabel"] label,
      div[data-testid="stWidgetLabel"] p,
      .stTextInput label,
      .stSelectbox label,
      .stRadio label {
        color: #111111 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
      }
                
      /* FIX TEXT CURSOR / CARET VISIBILITY */
     .stTextInput input,
     .stTextArea textarea,
     input[type="text"],
     input[type="password"],
     input[type="email"] {
     color: #111111 !important;
     caret-color: #111111 !important;
         }
     .stTextInput input::selection,
    .stTextArea textarea::selection,
    input[type="text"]::selection,
    input[type="password"]::selection,
    input[type="email"]::selection {
    background: #dbeafe !important;
    color: #111111 !important;
        
    </style>
    """, unsafe_allow_html=True)

def init_db() -> any:
    """Ensure the SQLite database and user table exist.

    Returns
    -------
    conn : sqlite3.Connection
        A connection object to the SQLite database.
    """
    db_path = os.path.join(os.path.dirname(__file__), "database", "users.db")
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = create_connection(db_path)
    create_user_table(conn)
    return conn


def show_login(conn) -> None:
    """Render the login and sign up interface when the user is not authenticated."""
    st.sidebar.title("DriverPulse AI")
    preferred_mode = st.session_state.pop("auth_mode", "Login")
    default_index = 1 if preferred_mode == "Sign Up" else 0
    page_type = st.sidebar.radio("Authentication", ["Login", "Sign Up"], index=default_index)

    flash = st.session_state.pop("auth_flash", None)
    if flash:
        st.success(flash)

    if page_type == "Login":
        st.title("Login")
        email = st.text_input("Email", value=st.session_state.pop("prefill_email", ""))
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = verify_user(conn, email, password)
            if user:
                # user tuple: (id, name, email, password_hash, salt, driver_id)
                st.session_state.logged_in = True
                st.session_state.driver_id = user[5]
                st.session_state.name = user[1]
                st.session_state.login_flash = "Logged in successfully!"
                st.rerun()
            else:
                st.error("Invalid email or password.")
    else:
        st.title("Sign Up")
        name = st.text_input("Name")
        email = st.text_input("Email", value=st.session_state.pop("prefill_email", ""))
        password = st.text_input("Password", type="password")
        driver_id = st.text_input("Driver ID")
        if st.button("Sign Up"):
            # Validate the driver_id exists in drivers.csv
            drivers_df = load_data("drivers.csv")
            if driver_id not in drivers_df["driver_id"].astype(str).values:
                st.error("Driver ID not found in drivers.csv. Please contact your administrator.")
            elif get_user_by_email(conn, email):
                st.error("An account with this email already exists.")
            else:
                add_user(conn, name, email, password, driver_id)
                st.session_state.auth_flash = "Account created successfully! Please login."
                st.session_state.auth_mode = "Login"
                st.session_state.prefill_email = email
                st.rerun()


def main() -> None:
    """Entry point for the DriverPulse Streamlit app."""
    st.set_page_config(
        page_title="DriverPulse AI",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_style()

    # Initialize or retrieve database connection
    conn = init_db()

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.driver_id = None
        st.session_state.name = None

    # If user is not logged in, show login/signup
    if not st.session_state.logged_in:
        show_login(conn)
        return

    if st.session_state.pop("login_flash", None):
        st.success("Logged in successfully!")

    # Display sidebar navigation for logged in users
    st.sidebar.write(f"👤 Logged in as **{st.session_state.name}** (ID: {st.session_state.driver_id})")
    page = st.sidebar.radio(
        "Navigation",
        [
            "My Dashboard",
            "My Safety",
            "Earnings & Goals",
            "My Driving Behavior",
            "Burnout Monitor",
            "Logout",
        ],
    )

    if page == "Logout":
        # Clear session and return to login screen
        st.session_state.logged_in = False
        st.session_state.driver_id = None
        st.session_state.name = None
        # Use st.rerun() on Streamlit ≥1.55 instead of experimental_rerun
        st.rerun()
        return

    if page == "My Dashboard":
        # Main dashboard summarising profile, trips, earnings, and safety
        from pages.page_dashboard import run as dashboard_run
        dashboard_run(st.session_state.driver_id)
    elif page == "My Safety":
        load_script_page("pages.page_safety")
    elif page == "Earnings & Goals":
        load_script_page("pages.page_earnings_goals")
    elif page == "My Driving Behavior":
        load_script_page("pages.page_driving_behaviour")
    elif page == "Burnout Monitor":
        load_script_page("pages.page_burnout_monitor")


if __name__ == "__main__":
    main()