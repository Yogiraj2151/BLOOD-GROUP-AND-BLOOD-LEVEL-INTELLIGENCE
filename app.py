import streamlit as st
import os
import re
import hashlib
import hmac
from datetime import date
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load local credentials from a .env file (kept out of Git with .gitignore).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Blood Group & Blood Level Intelligence",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLOOD_GROUPS = [
    "A+", "A-", "B+", "B-",
    "AB+", "AB-", "O+", "O-"
]

# =========================================================
# CSS
# =========================================================

def load_css(file_name):
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {file_name}")

load_css("style.css")

# =========================================================
# SUPABASE CONFIGURATION
# =========================================================

def get_setting(name, default=None):
    """Read a setting from Streamlit Secrets first, then environment variables."""
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        pass
    if not value:
        value = os.getenv(name)
    return value or default

SUPABASE_URL = get_setting("SUPABASE_URL")
# This app runs server-side in Streamlit. Prefer a service-role/secret key stored
# in Streamlit Secrets; never put it in frontend HTML or commit it to GitHub.
SUPABASE_KEY = (
    get_setting("SUPABASE_SERVICE_ROLE_KEY")
    or get_setting("SUPABASE_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    st.markdown('<div class="title">🩸 Blood Group & Blood Level Intelligence</div>', unsafe_allow_html=True)
    st.error("Supabase credentials are missing. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to your local .env file, environment variables, or Streamlit Secrets.")
    st.markdown("**For local development**, create a `.env` file beside `app.py` with:")
    st.code("SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co\nSUPABASE_SERVICE_ROLE_KEY=YOUR_SERVER_SIDE_SECRET_KEY\nADMIN_USERNAME=admin\nADMIN_PASSWORD=REPLACE_WITH_A_LONG_UNIQUE_PASSWORD", language="bash")
    st.stop()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as exc:
    st.error(f"Could not initialize Supabase: {exc}")
    st.stop()

# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):
    """Salted PBKDF2 hash for new passwords."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    """Verify current PBKDF2 hashes and legacy SHA-256 hashes from older versions."""
    if not stored_hash:
        return False
    try:
        parts = stored_hash.split("$")
        if len(parts) == 3 and parts[0] == "pbkdf2_sha256":
            salt = bytes.fromhex(parts[1])
            expected = bytes.fromhex(parts[2])
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
            return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False
    # Allow existing SHA-256 rows to continue working if data is migrated from
    # the old SQLite version. New account registrations use PBKDF2 above.
    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, str(stored_hash))

# =========================================================
# SMALL SQL-COMPATIBILITY LAYER
# The existing app's simple one-table queries are mapped to Supabase PostgREST.
# It supports SELECT with equality filters/order, INSERT, and UPDATE statements
# used in this app. It is not intended as a general-purpose SQL engine.
# =========================================================

TABLE_COLUMNS = {
    "admins": ["id", "username", "password"],
    "donors": ["id", "name", "age", "gender", "blood_group", "phone", "email", "location", "password", "available", "last_donation"],
    "hospitals": ["id", "hospital_name", "contact_person", "phone", "email", "location", "registration_id", "password"],
    "blood_stock": ["id", "blood_group", "units", "threshold"],
    "requests": ["id", "hospital_id", "hospital_name", "patient_name", "blood_group", "units", "priority", "status", "request_date"],
    "donations": ["id", "donor_id", "donor_name", "blood_group", "units", "donation_date"],
}


def _parse_literal_or_parameter(expression, params, param_index):
    expression = expression.strip()
    if expression == "?":
        if param_index >= len(params):
            raise ValueError("Not enough query parameters supplied.")
        return params[param_index], param_index + 1
    if len(expression) >= 2 and expression[0] == expression[-1] and expression[0] in ("'", '"'):
        return expression[1:-1], param_index
    if expression.lower() == "null":
        return None, param_index
    try:
        return int(expression), param_index
    except ValueError:
        try:
            return float(expression), param_index
        except ValueError:
            return expression, param_index


def _apply_where(builder, where_clause, params=(), start_index=0):
    if not where_clause:
        return builder, start_index
    idx = start_index
    conditions = re.split(r"\s+AND\s+", where_clause.strip(), flags=re.IGNORECASE)
    for condition in conditions:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\?|'.*?'|\".*?\"|[-+]?\d+(?:\.\d+)?)$", condition.strip())
        if not match:
            raise ValueError(f"Unsupported WHERE condition: {condition}")
        column, expression = match.groups()
        value, idx = _parse_literal_or_parameter(expression, params, idx)
        if value is None:
            builder = builder.is_(column, "null")
        else:
            builder = builder.eq(column, value)
    return builder, idx


def get_df(query, params=()):
    """Execute a simple SELECT via Supabase and return a pandas DataFrame."""
    match = re.match(r"^\s*SELECT\s+(.*?)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$", query.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Unsupported SELECT query: {query}")
    columns_text, table, tail = match.groups()
    table = table.lower()
    if table not in TABLE_COLUMNS:
        raise ValueError(f"Unsupported table: {table}")

    order_column = None
    order_desc = False
    order_match = re.search(r"\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(ASC|DESC))?\s*$", tail, flags=re.IGNORECASE)
    if order_match:
        order_column = order_match.group(1)
        order_desc = (order_match.group(2) or "ASC").upper() == "DESC"
        tail = tail[:order_match.start()].strip()

    where_clause = None
    where_match = re.search(r"\bWHERE\b(.*)$", tail, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1).strip()

    columns = "*" if columns_text.strip() == "*" else ",".join(c.strip() for c in columns_text.split(","))
    builder = supabase.table(table).select(columns)
    builder, _ = _apply_where(builder, where_clause, params, 0)
    if order_column:
        builder = builder.order(order_column, desc=order_desc)
    response = builder.limit(1000).execute()
    rows = response.data or []

    if columns == "*":
        result_columns = TABLE_COLUMNS[table]
    else:
        result_columns = [part.strip() for part in columns.split(",")]
    return pd.DataFrame(rows, columns=result_columns)


def run_query(query, params=()):
    """Execute the app's simple SELECT/INSERT/UPDATE operations with Supabase."""
    query_clean = query.strip().rstrip(";")
    if re.match(r"^SELECT\b", query_clean, re.IGNORECASE):
        df = get_df(query_clean, params)
        return list(df.itertuples(index=False, name=None))

    insert_match = re.match(
        r"^INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)\s*$",
        query_clean, flags=re.IGNORECASE | re.DOTALL
    )
    if insert_match:
        table, columns_text, _values_text = insert_match.groups()
        columns = [item.strip() for item in columns_text.split(",")]
        if len(columns) != len(params):
            raise ValueError(f"Expected {len(columns)} values for insert into {table}; got {len(params)}.")
        payload = dict(zip(columns, params))
        supabase.table(table.lower()).insert(payload).execute()
        return []

    update_match = re.match(
        r"^UPDATE\s+([A-Za-z_][A-Za-z0-9_]*)\s+SET\s+(.*?)\s+WHERE\s+(.*?)\s*$",
        query_clean, flags=re.IGNORECASE | re.DOTALL
    )
    if update_match:
        table, set_clause, where_clause = update_match.groups()
        table = table.lower()
        payload = {}
        set_idx = 0
        for assignment in set_clause.split(","):
            add_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*\?$", assignment.strip())
            direct_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\?$", assignment.strip())
            if add_match:
                target_col, source_col = add_match.groups()
                if set_idx >= len(params):
                    raise ValueError("Not enough values for UPDATE.")
                increment = params[set_idx]
                set_idx += 1
                read_builder = supabase.table(table).select("*")
                read_builder, _ = _apply_where(read_builder, where_clause, params, set_idx)
                existing = (read_builder.limit(1).execute().data or [])
                if existing:
                    payload[target_col] = (existing[0].get(source_col) or 0) + increment
            elif direct_match:
                column = direct_match.group(1)
                if set_idx >= len(params):
                    raise ValueError("Not enough values for UPDATE.")
                payload[column] = params[set_idx]
                set_idx += 1
            else:
                raise ValueError(f"Unsupported UPDATE assignment: {assignment}")
        builder = supabase.table(table).update(payload)
        builder, _ = _apply_where(builder, where_clause, params, set_idx)
        builder.execute()
        return []

    raise ValueError(f"Unsupported database operation in query: {query_clean[:100]}")


def create_database():
    """Seed the admin demo account and all eight blood groups if missing."""
    admin_username = get_setting("ADMIN_USERNAME", "admin")
    admin_password = get_setting("ADMIN_PASSWORD")
    if not admin_password:
        raise RuntimeError("Set ADMIN_PASSWORD in your .env file, environment variables, or Streamlit Secrets before first launch.")
    admin_rows = (supabase.table("admins").select("id").eq("username", admin_username).limit(1).execute().data or [])
    if not admin_rows:
        supabase.table("admins").insert({"username": admin_username, "password": hash_password(admin_password)}).execute()

    existing_rows = supabase.table("blood_stock").select("blood_group").limit(1000).execute().data or []
    existing_groups = {row.get("blood_group") for row in existing_rows}
    missing = [
        {"blood_group": group, "units": 0, "threshold": 5}
        for group in BLOOD_GROUPS if group not in existing_groups
    ]
    if missing:
        supabase.table("blood_stock").insert(missing).execute()


try:
    create_database()
except Exception as exc:
    st.markdown('<div class="title">🩸 Supabase Database Setup Required</div>', unsafe_allow_html=True)
    st.error("Could not read or initialize the Supabase tables. First run the SQL in `supabase_schema.sql` using Supabase Dashboard → SQL Editor.")
    st.code(str(exc))
    st.stop()


# =========================================================
# LOGIN FUNCTIONS
# =========================================================

def admin_login(username, password):
    result = run_query("SELECT id, password FROM admins WHERE username=?", (username,))
    return result[0][0] if result and verify_password(password, result[0][1]) else None


def donor_login(email, password):
    result = run_query("SELECT id, name, password FROM donors WHERE email=?", (email,))
    return (result[0][0], result[0][1]) if result and verify_password(password, result[0][2]) else None


def hospital_login(email, password):
    result = run_query("SELECT id, hospital_name, password FROM hospitals WHERE email=?", (email,))
    return (result[0][0], result[0][1]) if result and verify_password(password, result[0][2]) else None


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "logged_in": False,
    "role": "",
    "user_id": None,
    "user_name": "",
    "page": "Login"
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩸 BLOOD INTELLIGENCE")
    st.caption("Blood Group & Blood Level Intelligence")
    st.markdown("---")

    if not st.session_state.logged_in:

        st.subheader("🔐 Access")

        access = st.radio(
            "Select Portal",
            [
                "👨‍💼 ADMIN LOGIN",
                "🩸 DONOR LOGIN",
                "📝 DONOR SIGNUP",
                "🏥 HOSPITAL LOGIN",
                "🏥 HOSPITAL SIGNUP"
            ]
        )

        if access == "👨‍💼 ADMIN LOGIN":
            st.session_state.page = "Admin Login"

        elif access == "🩸 DONOR LOGIN":
            st.session_state.page = "Donor Login"

        elif access == "📝 DONOR SIGNUP":
            st.session_state.page = "Donor Signup"

        elif access == "🏥 HOSPITAL LOGIN":
            st.session_state.page = "Hospital Login"

        elif access == "🏥 HOSPITAL SIGNUP":
            st.session_state.page = "Hospital Signup"

    else:

        st.success(
            f"Logged in as {st.session_state.role.upper()}"
        )

        if st.session_state.role == "admin":

            menu = st.radio(
                "Admin Menu",
                [
                    "🏠 Dashboard",
                    "👥 Donors",
                    "🏥 Hospitals",
                    "🩸 Blood Stock",
                    "📋 Blood Requests",
                    "💉 Donations",
                    "🔎 Donor Matching",
                    "🤖 AI Prediction",
                    "📊 Analytics",
                    "📑 Reports",
                    "⚙️ Settings",
                    "🚪 Logout"
                ]
            )

            st.session_state.page = menu

        elif st.session_state.role == "donor":

            menu = st.radio(
                "Donor Menu",
                [
                    "👤 My Dashboard",
                    "📝 My Profile",
                    "❤️ My Donations",
                    "🟢 Availability",
                    "🔔 Blood Alerts",
                    "🚪 Logout"
                ]
            )

            st.session_state.page = menu

        elif st.session_state.role == "hospital":

            menu = st.radio(
                "Hospital Menu",
                [
                    "🏠 Hospital Dashboard",
                    "📝 Request Blood",
                    "🚨 Emergency Request",
                    "📋 My Requests",
                    "🩸 Check Blood Level",
                    "👤 Hospital Profile",
                    "🚪 Logout"
                ]
            )

            st.session_state.page = menu


# =========================================================
# ADMIN LOGIN
# =========================================================

if st.session_state.page == "Admin Login":

    st.markdown(
        '<div class="title">👨‍💼 ADMIN  LOGIN</div>',
        unsafe_allow_html=True
    )

    with st.form("admin_login_form"):

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        submit = st.form_submit_button(
            "🔐 Login",
            use_container_width=True
        )

        if submit:

            admin_id = admin_login(
                username,
                password
            )

            if admin_id:

                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.session_state.user_id = admin_id
                st.session_state.user_name = username
                st.session_state.page = "🏠 Dashboard"

                st.rerun()

            else:
                st.error("Invalid Admin Login")

   

# =========================================================
# DONOR SIGNUP
# =========================================================

elif st.session_state.page == "Donor Signup":

    st.markdown(
        '<div class="title">🩸 DONOR  SIGNUP</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Register yourself as a blood donor."
    )

    with st.form("donor_signup_form"):

        c1, c2 = st.columns(2)

        with c1:

            name = st.text_input("👤 Full Name")

            age = st.number_input(
                "🎂 Age",
                min_value=18,
                max_value=65,
                value=18
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

            blood_group = st.selectbox(
                "🩸 Blood Group",
                BLOOD_GROUPS
            )

        with c2:

            phone = st.text_input(
                "📱 Mobile Number"
            )

            email = st.text_input(
                "📧 Email"
            )

            location = st.text_input(
                "📍 Location"
            )

            password = st.text_input(
                "🔑 Password",
                type="password"
            )

            confirm = st.text_input(
                "🔐 Confirm Password",
                type="password"
            )

        submit = st.form_submit_button(
            "🩸 CREATE DONOR ACCOUNT",
            use_container_width=True
        )

        if submit:

            if not name or not email or not phone or not location:

                st.error(
                    "Please fill all required fields."
                )

            elif password != confirm:

                st.error(
                    "Passwords do not match."
                )

            elif len(password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

            else:

                try:

                    run_query("""
                        INSERT INTO donors
                        (
                            name,
                            age,
                            gender,
                            blood_group,
                            phone,
                            email,
                            location,
                            password,
                            available,
                            last_donation
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        age,
                        gender,
                        blood_group,
                        phone,
                        email,
                        location,
                        hash_password(password),
                        "Available",
                        ""
                    ))

                    st.success(
                        "🎉 Donor account created successfully!"
                    )

                    st.info(
                        "Go to 🩸 Donor Login from the sidebar."
                    )

                except Exception:

                    st.error(
                        "Email already registered."
                    )


# =========================================================
# DONOR LOGIN
# =========================================================

elif st.session_state.page == "Donor Login":

    st.markdown(
        '<div class="title">🩸 DONOR LOGIN</div>',
        unsafe_allow_html=True
    )

    with st.form("donor_login_form"):

        email = st.text_input(
            "📧 Email"
        )

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

        submit = st.form_submit_button(
            "🩸 LOGIN",
            use_container_width=True
        )

        if submit:

            donor = donor_login(
                email,
                password
            )

            if donor:

                st.session_state.logged_in = True
                st.session_state.role = "donor"
                st.session_state.user_id = donor[0]
                st.session_state.user_name = donor[1]
                st.session_state.page = "👤 My Dashboard"

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )


# =========================================================
# HOSPITAL SIGNUP
# =========================================================

elif st.session_state.page == "Hospital Signup":

    st.markdown(
        '<div class="title">🏥 HOSPITAL   SIGNUP</div>',
        unsafe_allow_html=True
    )

    with st.form("hospital_signup_form"):

        hospital_name = st.text_input(
            "🏥 Hospital Name"
        )

        contact = st.text_input(
            "👨‍⚕️ Contact Person"
        )

        phone = st.text_input(
            "📱 Mobile Number"
        )

        email = st.text_input(
            "📧 Email"
        )

        location = st.text_input(
            "📍 Location"
        )

        registration = st.text_input(
            "🪪 Registration ID"
        )

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

        confirm = st.text_input(
            "🔐 Confirm Password",
            type="password"
        )

        submit = st.form_submit_button(
            "🏥 CREATE HOSPITAL ACCOUNT",
            use_container_width=True
        )

        if submit:

            if not hospital_name or not email:

                st.error(
                    "Hospital name and email are required."
                )

            elif password != confirm:

                st.error(
                    "Passwords do not match."
                )

            else:

                try:

                    run_query("""
                        INSERT INTO hospitals
                        (
                            hospital_name,
                            contact_person,
                            phone,
                            email,
                            location,
                            registration_id,
                            password
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        hospital_name,
                        contact,
                        phone,
                        email,
                        location,
                        registration,
                        hash_password(password)
                    ))

                    st.success(
                        "🏥 Hospital account created!"
                    )

                except Exception:

                    st.error(
                        "Hospital email already registered."
                    )


# =========================================================
# HOSPITAL LOGIN
# =========================================================

elif st.session_state.page == "Hospital Login":

    st.markdown(
        '<div class="title">🏥 HOSPITAL     LOGIN</div>',
        unsafe_allow_html=True
    )

    with st.form("hospital_login_form"):

        email = st.text_input(
            "📧 Hospital Email"
        )

        password = st.text_input(
            "🔑 Password",
            type="password"
        )

        submit = st.form_submit_button(
            "🏥 LOGIN",
            use_container_width=True
        )

        if submit:

            hospital = hospital_login(
                email,
                password
            )

            if hospital:

                st.session_state.logged_in = True
                st.session_state.role = "hospital"
                st.session_state.user_id = hospital[0]
                st.session_state.user_name = hospital[1]
                st.session_state.page = "🏠 Hospital Dashboard"

                st.rerun()

            else:

                st.error(
                    "Invalid hospital login."
                )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "🏠 Dashboard"
):

    st.markdown(
        '<div class="title">🩸 Blood Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    donors = get_df(
        "SELECT * FROM donors"
    )

    hospitals = get_df(
        "SELECT * FROM hospitals"
    )

    stock = get_df(
        "SELECT * FROM blood_stock"
    )

    requests = get_df(
        "SELECT * FROM requests"
    )

    donations = get_df(
        "SELECT * FROM donations"
    )

    total_units = int(
        stock["units"].sum()
    )

    pending = len(
        requests[
            requests["status"] == "Pending"
        ]
    ) if not requests.empty else 0

    emergency = len(
        requests[
            requests["priority"] == "Emergency"
        ]
    ) if not requests.empty else 0

    critical = len(
        stock[
            stock["units"] <= stock["threshold"]
        ]
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "👥 Donors",
        len(donors)
    )

    c2.metric(
        "🏥 Hospitals",
        len(hospitals)
    )

    c3.metric(
        "🩸 Blood Units",
        total_units
    )

    c4.metric(
        "📋 Pending",
        pending
    )

    c5.metric(
        "🔴 Critical",
        critical
    )

    st.markdown("---")

    st.subheader("🩸 Blood Level Intelligence")

    cols = st.columns(4)

    for i, group in enumerate(BLOOD_GROUPS):

        row = stock[
            stock["blood_group"] == group
        ].iloc[0]

        units = int(row["units"])
        threshold = int(row["threshold"])

        if units <= threshold:
            status = "🔴 CRITICAL"
        elif units <= threshold * 2:
            status = "🟠 LOW"
        else:
            status = "🟢 NORMAL"

        with cols[i % 4]:

            st.markdown(
                f"""
                <div class="card">
                <h2>🩸 {group}</h2>
                <h3>{units} Units</h3>
                <p>{status}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    if not stock.empty:

        fig = px.bar(
            stock,
            x="blood_group",
            y="units",
            text="units",
            title="Current Blood Level"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if critical > 0:

        st.error(
            f"⚠️ {critical} blood group(s) are at critical level."
        )


# =========================================================
# ADMIN DONORS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "👥 Donors"
):

    st.title("👥 Donor Management")

    donors = get_df("""
        SELECT
            id,
            name,
            age,
            gender,
            blood_group,
            phone,
            email,
            location,
            available,
            last_donation
        FROM donors
        ORDER BY id DESC
    """)

    search = st.text_input(
        "🔎 Search donor"
    )

    if search:

        donors = donors[
            donors["name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        donors,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ADMIN HOSPITALS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "🏥 Hospitals"
):

    st.title("🏥 Hospital Management")

    hospitals = get_df("""
        SELECT
            id,
            hospital_name,
            contact_person,
            phone,
            email,
            location,
            registration_id
        FROM hospitals
        ORDER BY id DESC
    """)

    st.dataframe(
        hospitals,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# BLOOD STOCK
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "🩸 Blood Stock"
):

    st.title("🩸 Blood Stock Management")

    stock = get_df(
        "SELECT * FROM blood_stock"
    )

    st.dataframe(
        stock,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Update Blood Level")

    group = st.selectbox(
        "Blood Group",
        BLOOD_GROUPS
    )

    current = int(
        stock[
            stock["blood_group"] == group
        ]["units"].iloc[0]
    )

    units = st.number_input(
        "Available Units",
        min_value=0,
        value=current
    )

    threshold = st.number_input(
        "Critical Threshold",
        min_value=0,
        value=5
    )

    if st.button(
        "💾 Update Blood Level",
        use_container_width=True
    ):

        run_query("""
            UPDATE blood_stock
            SET units=?, threshold=?
            WHERE blood_group=?
        """, (
            units,
            threshold,
            group
        ))

        st.success(
            "Blood level updated!"
        )

        st.rerun()


# =========================================================
# ADMIN REQUESTS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "📋 Blood Requests"
):

    st.title("📋 Blood Requests")

    requests = get_df("""
        SELECT *
        FROM requests
        ORDER BY id DESC
    """)

    if requests.empty:

        st.info(
            "No blood requests."
        )

    else:

        st.dataframe(
            requests,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Update Request Status")

        request_id = st.selectbox(
            "Request ID",
            requests["id"].tolist()
        )

        status = st.selectbox(
            "Status",
            [
                "Pending",
                "Approved",
                "Rejected",
                "Completed"
            ]
        )

        if st.button(
            "Update Status"
        ):

            run_query("""
                UPDATE requests
                SET status=?
                WHERE id=?
            """, (
                status,
                request_id
            ))

            st.success(
                "Request status updated."
            )

            st.rerun()


# =========================================================
# DONATIONS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "💉 Donations"
):

    st.title("💉 Record Blood Donation")

    donors = get_df(
        "SELECT * FROM donors"
    )

    if donors.empty:

        st.info(
            "No donors registered."
        )

    else:

        donor_map = {
            f"{r['name']} - {r['blood_group']}":
            r["id"]
            for _, r in donors.iterrows()
        }

        selected = st.selectbox(
            "Select Donor",
            list(donor_map.keys())
        )

        donor_id = donor_map[selected]

        donor = donors[
            donors["id"] == donor_id
        ].iloc[0]

        units = st.number_input(
            "Donation Units",
            min_value=1,
            max_value=10,
            value=1
        )

        donation_date = st.date_input(
            "Donation Date",
            date.today()
        )

        if st.button(
            "🩸 Record Donation",
            use_container_width=True
        ):

            run_query("""
                INSERT INTO donations
                (
                    donor_id,
                    donor_name,
                    blood_group,
                    units,
                    donation_date
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                donor_id,
                donor["name"],
                donor["blood_group"],
                units,
                str(donation_date)
            ))

            run_query("""
                UPDATE blood_stock
                SET units = units + ?
                WHERE blood_group=?
            """, (
                units,
                donor["blood_group"]
            ))

            run_query("""
                UPDATE donors
                SET last_donation=?
                WHERE id=?
            """, (
                str(donation_date),
                donor_id
            ))

            st.success(
                "Donation recorded and blood level increased!"
            )

            st.rerun()


# =========================================================
# DONOR MATCHING
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "🔎 Donor Matching"
):

    st.title("🔎 Smart Donor Matching")

    group = st.selectbox(
        "Required Blood Group",
        BLOOD_GROUPS
    )

    location = st.text_input(
        "📍 Preferred Location"
    )

    if st.button(
        "🔎 Find Donors",
        use_container_width=True
    ):

        donors = get_df("""
            SELECT
                name,
                age,
                gender,
                blood_group,
                phone,
                email,
                location,
                available,
                last_donation
            FROM donors
            WHERE blood_group=?
            AND available='Available'
        """, (
            group,
        ))

        if location and not donors.empty:

            donors["location_match"] = donors[
                "location"
            ].str.contains(
                location,
                case=False,
                na=False
            )

            donors = donors.sort_values(
                "location_match",
                ascending=False
            )

        if donors.empty:

            st.warning(
                "No available donor found."
            )

        else:

            st.success(
                f"{len(donors)} available donor(s) found."
            )

            st.dataframe(
                donors,
                use_container_width=True,
                hide_index=True
            )

    st.info(
        "For academic/demo use. Actual transfusion compatibility "
        "must be verified by qualified medical professionals."
    )


# =========================================================
# AI PREDICTION
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "🤖 AI Prediction"
):

    st.title("🤖 AI Blood Demand Intelligence")

    group = st.selectbox(
        "Select Blood Group",
        BLOOD_GROUPS
    )

    requests = get_df("""
        SELECT request_date, units
        FROM requests
        WHERE blood_group=?
        ORDER BY request_date
    """, (
        group,
    ))

    if len(requests) < 3:

        st.info(
            "At least 3 historical requests are needed for AI prediction."
        )

    else:

        requests["request_date"] = pd.to_datetime(
            requests["request_date"]
        )

        daily = requests.groupby(
            "request_date"
        )["units"].sum().reset_index()

        daily["day"] = (
            daily["request_date"]
            - daily["request_date"].min()
        ).dt.days

        model = LinearRegression()

        model.fit(
            daily[["day"]],
            daily["units"]
        )

        future_day = daily["day"].max() + 7

        prediction = model.predict(
            [[future_day]]
        )[0]

        prediction = max(
            0,
            round(prediction, 2)
        )

        current_stock = int(
            get_df("""
                SELECT units
                FROM blood_stock
                WHERE blood_group=?
            """, (
                group,
            )).iloc[0]["units"]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Current Stock",
            f"{current_stock} Units"
        )

        c2.metric(
            "Predicted Demand",
            f"{prediction} Units"
        )

        if current_stock < prediction:

            c3.error(
                "⚠️ STOCK MAY BE LOW"
            )

        else:

            c3.success(
                "🟢 STOCK LOOKS OK"
            )

        fig = px.line(
            daily,
            x="request_date",
            y="units",
            markers=True,
            title=f"{group} Blood Demand Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# ANALYTICS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "📊 Analytics"
):

    st.title("📊 Blood Intelligence Analytics")

    stock = get_df(
        "SELECT * FROM blood_stock"
    )

    donations = get_df(
        "SELECT * FROM donations"
    )

    requests = get_df(
        "SELECT * FROM requests"
    )

    if not stock.empty:

        fig = px.bar(
            stock,
            x="blood_group",
            y="units",
            text="units",
            title="🩸 Blood Stock Level"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if not donations.empty:

        data = donations.groupby(
            "blood_group"
        )["units"].sum().reset_index()

        fig = px.bar(
            data,
            x="blood_group",
            y="units",
            text="units",
            title="💉 Donations by Blood Group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if not requests.empty:

        data = requests.groupby(
            "blood_group"
        ).size().reset_index(
            name="requests"
        )

        fig = px.pie(
            data,
            names="blood_group",
            values="requests",
            title="📋 Blood Request Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# REPORTS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "📑 Reports"
):

    st.title("📑 Reports")

    report = st.selectbox(
        "Select Report",
        [
            "Donors",
            "Hospitals",
            "Blood Stock",
            "Requests",
            "Donations"
        ]
    )

    if report == "Donors":

        df = get_df(
            "SELECT * FROM donors"
        )

    elif report == "Hospitals":

        df = get_df(
            "SELECT * FROM hospitals"
        )

    elif report == "Blood Stock":

        df = get_df(
            "SELECT * FROM blood_stock"
        )

    elif report == "Requests":

        df = get_df(
            "SELECT * FROM requests"
        )

    else:

        df = get_df(
            "SELECT * FROM donations"
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        csv,
        f"{report}.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# SETTINGS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "admin"
    and st.session_state.page == "⚙️ Settings"
):

    st.title("⚙️ System Settings")

    st.subheader("🩸 Blood Level")

    st.write(
        "Default critical threshold = 5 units"
    )

    st.subheader("🔔 Alerts")

    st.checkbox(
        "Critical Blood Level Alert",
        True
    )

    st.checkbox(
        "Emergency Request Alert",
        True
    )

    st.checkbox(
        "New Donation Alert",
        True
    )


# =========================================================
# DONOR DASHBOARD
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "donor"
    and st.session_state.page == "👤 My Dashboard"
):

    donor = get_df("""
        SELECT *
        FROM donors
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    donations = get_df("""
        SELECT *
        FROM donations
        WHERE donor_id=?
    """, (
        st.session_state.user_id,
    ))

    if not donor.empty:

        d = donor.iloc[0]

        st.markdown(
            f'<div class="title">🩸 Welcome, {d["name"]}</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🩸 Blood Group",
            d["blood_group"]
        )

        c2.metric(
            "❤️ Donations",
            len(donations)
        )

        c3.metric(
            "🟢 Status",
            d["available"]
        )

        st.markdown("---")

        st.subheader("👤 My Information")

        st.write(
            f"**Name:** {d['name']}"
        )

        st.write(
            f"**Age:** {d['age']}"
        )

        st.write(
            f"**Gender:** {d['gender']}"
        )

        st.write(
            f"**Blood Group:** {d['blood_group']}"
        )

        st.write(
            f"**Location:** {d['location']}"
        )

        st.write(
            f"**Last Donation:** "
            f"{d['last_donation'] or 'Not yet'}"
        )


# =========================================================
# DONOR PROFILE
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "donor"
    and st.session_state.page == "📝 My Profile"
):

    st.title("📝 My Profile")

    df = get_df("""
        SELECT
            name,
            age,
            gender,
            blood_group,
            phone,
            email,
            location,
            available,
            last_donation
        FROM donors
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DONOR DONATIONS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "donor"
    and st.session_state.page == "❤️ My Donations"
):

    st.title("❤️ My Donation History")

    df = get_df("""
        SELECT
            donor_name,
            blood_group,
            units,
            donation_date
        FROM donations
        WHERE donor_id=?
        ORDER BY donation_date DESC
    """, (
        st.session_state.user_id,
    ))

    if df.empty:

        st.info(
            "No donation history."
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DONOR AVAILABILITY
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "donor"
    and st.session_state.page == "🟢 Availability"
):

    st.title("🟢 Donor Availability")

    current = get_df("""
        SELECT available
        FROM donors
        WHERE id=?
    """, (
        st.session_state.user_id,
    )).iloc[0]["available"]

    status = st.selectbox(
        "Your Availability",
        [
            "Available",
            "Maybe Available",
            "Not Available"
        ],
        index=[
            "Available",
            "Maybe Available",
            "Not Available"
        ].index(current)
        if current in [
            "Available",
            "Maybe Available",
            "Not Available"
        ]
        else 0
    )

    if st.button(
        "💾 Update Availability"
    ):

        run_query("""
            UPDATE donors
            SET available=?
            WHERE id=?
        """, (
            status,
            st.session_state.user_id
        ))

        st.success(
            "Availability updated!"
        )

        st.rerun()


# =========================================================
# DONOR BLOOD ALERTS
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "donor"
    and st.session_state.page == "🔔 Blood Alerts"
):

    st.title("🔔 Blood Alerts")

    donor = get_df("""
        SELECT blood_group
        FROM donors
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    group = donor.iloc[0]["blood_group"]

    requests = get_df("""
        SELECT
            hospital_name,
            patient_name,
            blood_group,
            units,
            priority,
            status,
            request_date
        FROM requests
        WHERE blood_group=?
        AND status='Pending'
        ORDER BY id DESC
    """, (
        group,
    ))

    if requests.empty:

        st.success(
            "No pending blood request for your group."
        )

    else:

        st.warning(
            f"🚨 {len(requests)} request(s) need {group} blood."
        )

        st.dataframe(
            requests,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# HOSPITAL DASHBOARD
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "🏠 Hospital Dashboard"
):

    st.title(
        f"🏥 {st.session_state.user_name}"
    )

    requests = get_df("""
        SELECT *
        FROM requests
        WHERE hospital_id=?
    """, (
        st.session_state.user_id,
    ))

    stock = get_df(
        "SELECT * FROM blood_stock"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📋 Total Requests",
        len(requests)
    )

    c2.metric(
        "⏳ Pending",
        len(
            requests[
                requests["status"] == "Pending"
            ]
        ) if not requests.empty else 0
    )

    c3.metric(
        "🩸 Total Blood Units",
        int(stock["units"].sum())
    )

    st.subheader("🩸 Current Blood Levels")

    st.dataframe(
        stock[
            ["blood_group", "units"]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# HOSPITAL REQUEST BLOOD
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "📝 Request Blood"
):

    st.title("📝 Request Blood")

    hospital = get_df("""
        SELECT hospital_name
        FROM hospitals
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    hospital_name = hospital.iloc[0]["hospital_name"]

    with st.form("hospital_request"):

        patient = st.text_input(
            "👤 Patient Name"
        )

        group = st.selectbox(
            "🩸 Blood Group",
            BLOOD_GROUPS
        )

        units = st.number_input(
            "Units Required",
            min_value=1,
            value=1
        )

        priority = st.selectbox(
            "Priority",
            [
                "Normal",
                "Urgent"
            ]
        )

        submit = st.form_submit_button(
            "🩸 SEND REQUEST",
            use_container_width=True
        )

        if submit:

            run_query("""
                INSERT INTO requests
                (
                    hospital_id,
                    hospital_name,
                    patient_name,
                    blood_group,
                    units,
                    priority,
                    status,
                    request_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                st.session_state.user_id,
                hospital_name,
                patient,
                group,
                units,
                priority,
                "Pending",
                str(date.today())
            ))

            st.success(
                "Blood request sent successfully!"
            )


# =========================================================
# HOSPITAL EMERGENCY
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "🚨 Emergency Request"
):

    st.title("🚨 Emergency Blood Request")

    hospital = get_df("""
        SELECT hospital_name
        FROM hospitals
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    hospital_name = hospital.iloc[0]["hospital_name"]

    with st.form("emergency_form"):

        patient = st.text_input(
            "Patient Name"
        )

        group = st.selectbox(
            "Required Blood Group",
            BLOOD_GROUPS
        )

        units = st.number_input(
            "Units Required",
            min_value=1,
            value=1
        )

        submit = st.form_submit_button(
            "🚨 SEND EMERGENCY REQUEST",
            use_container_width=True
        )

        if submit:

            run_query("""
                INSERT INTO requests
                (
                    hospital_id,
                    hospital_name,
                    patient_name,
                    blood_group,
                    units,
                    priority,
                    status,
                    request_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                st.session_state.user_id,
                hospital_name,
                patient,
                group,
                units,
                "Emergency",
                "Pending",
                str(date.today())
            ))

            st.success(
                "🚨 Emergency request sent!"
            )


# =========================================================
# HOSPITAL REQUEST HISTORY
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "📋 My Requests"
):

    st.title("📋 My Blood Requests")

    requests = get_df("""
        SELECT
            id,
            patient_name,
            blood_group,
            units,
            priority,
            status,
            request_date
        FROM requests
        WHERE hospital_id=?
        ORDER BY id DESC
    """, (
        st.session_state.user_id,
    ))

    if requests.empty:

        st.info(
            "No requests found."
        )

    else:

        st.dataframe(
            requests,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# HOSPITAL CHECK BLOOD
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "🩸 Check Blood Level"
):

    st.title("🩸 Live Blood Level")

    stock = get_df(
        "SELECT * FROM blood_stock"
    )

    for _, row in stock.iterrows():

        units = int(row["units"])
        threshold = int(row["threshold"])

        if units <= threshold:
            status = "🔴 CRITICAL"
        elif units <= threshold * 2:
            status = "🟠 LOW"
        else:
            status = "🟢 AVAILABLE"

        st.write(
            f"### 🩸 {row['blood_group']} — "
            f"{units} Units — {status}"
        )


# =========================================================
# HOSPITAL PROFILE
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.role == "hospital"
    and st.session_state.page == "👤 Hospital Profile"
):

    st.title("👤 Hospital Profile")

    profile = get_df("""
        SELECT
            hospital_name,
            contact_person,
            phone,
            email,
            location,
            registration_id
        FROM hospitals
        WHERE id=?
    """, (
        st.session_state.user_id,
    ))

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# LOGOUT
# =========================================================

elif (
    st.session_state.logged_in
    and st.session_state.page == "🚪 Logout"
):

    st.session_state.logged_in = False
    st.session_state.role = ""
    st.session_state.user_id = None
    st.session_state.user_name = ""
    st.session_state.page = "Admin Login"

    st.rerun()
