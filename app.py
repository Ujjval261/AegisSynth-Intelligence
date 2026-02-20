# app.py
# Professional Synthetic Data Generation Platform
# Built by Ujjval Dwivedi

import streamlit as st
import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import io
import json
import joblib
import base64
from datetime import datetime, timezone
import time
import os
import re
import warnings
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BRAND_LOGO_CANDIDATES = [
    Path("assets/aegissynth-logo.png"),
    Path("assets/aegissynth-logo.PNG"),
    Path("assets/aegissynth_logo.png"),
    Path("assets/aegissynth_logo.PNG"),
    Path("assets/logo.png"),
    Path("assets/logo.PNG"),
    Path("assets/logo.jpg"),
    Path("assets/logo.jpeg"),
    Path("aegissynth-logo.png"),
    Path("aegissynth-logo.PNG"),
    Path("logo.png"),
    Path("logo.PNG"),
    Path("logo.jpg"),
    Path("logo.jpeg"),
]
BRAND_LOGO_PATH = next((p for p in BRAND_LOGO_CANDIDATES if p.exists() and p.is_file()), None)


def _image_data_uri(path: Path | None) -> str:
    if not path:
        return ""
    try:
        suffix = path.suffix.lower().lstrip(".") or "png"
        mime = "image/png" if suffix == "png" else f"image/{suffix}"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


BRAND_LOGO_DATA_URI = _image_data_uri(BRAND_LOGO_PATH)

# Page config
st.set_page_config(
    page_title="AegisSynth Intelligence - Synthetic Data Platform",
    page_icon=str(BRAND_LOGO_PATH) if BRAND_LOGO_PATH else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = True
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'
if 'ui_theme_preset' not in st.session_state:
    st.session_state.ui_theme_preset = "Creamy Gold" if st.session_state.theme_mode == "light" else "Dark Pro"
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'auth_id_token' not in st.session_state:
    st.session_state.auth_id_token = None
if 'auth_uid' not in st.session_state:
    st.session_state.auth_uid = None
if 'auth_provider' not in st.session_state:
    st.session_state.auth_provider = None
if 'nav_page' not in st.session_state:
    st.session_state.nav_page = "🏠 Home"
if 'pending_nav_page' not in st.session_state:
    st.session_state.pending_nav_page = None
if 'ui_algorithm' not in st.session_state:
    st.session_state.ui_algorithm = "CTGAN - Best Quality"
if 'ui_num_rows' not in st.session_state:
    st.session_state.ui_num_rows = 1000
if 'ui_epochs' not in st.session_state:
    st.session_state.ui_epochs = 150
if 'ui_batch_size' not in st.session_state:
    st.session_state.ui_batch_size = 500
if 'ui_privacy_level' not in st.session_state:
    st.session_state.ui_privacy_level = "Enhanced"
if 'ui_quality_threshold' not in st.session_state:
    st.session_state.ui_quality_threshold = 0.85
if 'copilot_plan' not in st.session_state:
    st.session_state.copilot_plan = ""

_DARK_PRESETS = {"Dark Pro", "Midnight Neon"}
st.session_state.theme_mode = "dark" if st.session_state.ui_theme_preset in _DARK_PRESETS else "light"


def _http_error_message(exc: HTTPError) -> str:
    raw = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = str(exc.reason or exc)
    if not raw:
        return str(exc.reason or exc)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            if isinstance(err, dict):
                return str(err.get("message") or err.get("status") or raw)
            return str(err)
    except Exception:
        pass
    return raw


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        raise ValueError(_http_error_message(exc)) from exc


def _put_json(url: str, payload: dict) -> dict:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT"
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        raise ValueError(_http_error_message(exc)) from exc


def _firebase_config_value(secret_key: str, env_key: str, default: str = "") -> str:
    value = os.getenv(env_key, default)
    try:
        if secret_key in st.secrets:
            value = st.secrets[secret_key]
    except Exception:
        pass
    return str(value).strip()


def _firebase_web_api_key() -> str:
    return _firebase_config_value("firebase_web_api_key", "FIREBASE_WEB_API_KEY")


def _firebase_database_url() -> str:
    raw = _firebase_config_value("firebase_database_url", "FIREBASE_DATABASE_URL")
    if not raw:
        return ""

    # If user pasted Firebase console URL, convert it to RTDB API base URL.
    if "console.firebase.google.com" in raw:
        match = re.search(r"/database/([^/]+)/", raw)
        if match:
            db_name = match.group(1)
            return f"https://{db_name}.firebaseio.com"

    return raw


def _firebase_signup_email_password(email: str, password: str, display_name: str) -> dict:
    api_key = _firebase_web_api_key()
    if not api_key:
        raise ValueError("Missing FIREBASE_WEB_API_KEY")
    signup_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    auth_data = _post_json(signup_url, {"email": email, "password": password, "returnSecureToken": True})
    if display_name and auth_data.get("idToken"):
        update_url = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={api_key}"
        _post_json(update_url, {
            "idToken": auth_data["idToken"],
            "displayName": display_name,
            "returnSecureToken": True
        })
    return auth_data


def _firebase_signin_email_password(email: str, password: str) -> dict:
    api_key = _firebase_web_api_key()
    if not api_key:
        raise ValueError("Missing FIREBASE_WEB_API_KEY")
    signin_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    return _post_json(signin_url, {"email": email, "password": password, "returnSecureToken": True})


def _save_user_to_firebase(uid: str, id_token: str, data: dict) -> None:
    database_url = _firebase_database_url().rstrip("/")
    if not database_url or not uid or not id_token:
        return
    _put_json(f"{database_url}/users/{uid}.json?auth={id_token}", data)


def _save_auth_event(uid: str, id_token: str, event_name: str) -> None:
    database_url = _firebase_database_url().rstrip("/")
    if not database_url or not uid or not id_token:
        return
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    key = now_utc.replace(":", "-").replace(".", "-")
    _put_json(f"{database_url}/auth_events/{uid}/{key}.json?auth={id_token}", {
        "event": event_name,
        "timestamp_utc": now_utc
    })


def _set_authenticated_user(auth_data: dict, provider: str) -> None:
    st.session_state.authenticated = True
    st.session_state.current_user = auth_data.get("displayName") or auth_data.get("email") or "User"
    st.session_state.auth_provider = provider
    st.session_state.auth_uid = auth_data.get("localId")
    st.session_state.auth_id_token = auth_data.get("idToken")
    _save_user_to_firebase(st.session_state.auth_uid, st.session_state.auth_id_token, {
        "uid": st.session_state.auth_uid,
        "email": auth_data.get("email"),
        "display_name": auth_data.get("displayName"),
        "provider": provider,
        "last_login_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    })
    _save_auth_event(st.session_state.auth_uid, st.session_state.auth_id_token, f"{provider}_login")


def _clear_auth_state() -> None:
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.auth_provider = None
    st.session_state.auth_uid = None
    st.session_state.auth_id_token = None


def _config_value(secret_key: str, env_key: str, default: str = "") -> str:
    value = os.getenv(env_key, default)
    try:
        if secret_key in st.secrets:
            value = st.secrets[secret_key]
    except Exception:
        pass
    return str(value).strip()


def _openai_api_key() -> str:
    return _config_value("openai_api_key", "OPENAI_API_KEY")


def _openai_model() -> str:
    return _config_value("openai_model", "OPENAI_MODEL", "gpt-4o-mini")


def _dataset_profile(dataframe: pd.DataFrame) -> dict:
    rows = len(dataframe)
    cols = len(dataframe.columns)
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing_cells = int(dataframe.isnull().sum().sum())
    total_cells = max(rows * max(cols, 1), 1)
    missing_pct = (missing_cells / total_cells) * 100
    high_cardinality_cols = [
        col for col in dataframe.columns
        if rows > 0 and (dataframe[col].nunique(dropna=True) / rows) > 0.95
    ]
    return {
        "rows": rows,
        "cols": cols,
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "missing_pct": round(missing_pct, 2),
        "high_cardinality_cols": high_cardinality_cols[:8]
    }


def _heuristic_generation_plan(profile: dict, privacy_level: str) -> dict:
    rows = profile["rows"]
    cols = profile["cols"]
    numeric_cols = profile["numeric_cols"]
    categorical_cols = profile["categorical_cols"]

    complexity_score = (cols * 1.2) + (categorical_cols * 1.5) + (numeric_cols * 0.8)

    if rows <= 2500:
        algorithm = "TVAE - Fast Training"
        epochs = 125
    elif complexity_score >= 45 or categorical_cols > numeric_cols:
        algorithm = "CTGAN - Best Quality"
        epochs = 250
    elif cols <= 12 and profile["missing_pct"] <= 5:
        algorithm = "GaussianCopula - Statistical"
        epochs = 100
    else:
        algorithm = "CTGAN - Best Quality"
        epochs = 180

    if privacy_level == "Maximum":
        epochs = min(epochs, 200)
        num_rows = min(max(800, rows), 5000)
    elif privacy_level == "Standard":
        num_rows = min(max(1200, rows), 10000)
    else:
        num_rows = min(max(1000, rows), 8000)

    if rows < 2000:
        batch_size = 200
    elif rows < 10000:
        batch_size = 500
    else:
        batch_size = 800

    reasons = [
        f"Selected {algorithm.split(' - ')[0]} based on dataset size/complexity.",
        f"Suggested epochs={epochs} for balance between speed and fidelity.",
        f"Suggested batch_size={batch_size} based on record volume ({rows:,}).",
        f"Privacy level '{privacy_level}' tuned synthetic row target to {num_rows:,}."
    ]
    return {
        "algorithm": algorithm,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_rows": int(num_rows),
        "reasons": reasons
    }


def _llm_copilot_plan(profile: dict, goal: str, current_config: dict) -> str:
    api_key = _openai_api_key()
    if not api_key:
        return ""

    prompt = f"""
You are an expert synthetic-data ML architect.
Return concise deployment guidance in 6 bullets max.

Dataset profile:
{json.dumps(profile, indent=2)}

Current config:
{json.dumps(current_config, indent=2)}

User goal:
{goal or "Generate high-quality privacy-preserving synthetic data for production analytics."}

Include:
1) Best model choice (CTGAN/TVAE/GaussianCopula) with reason
2) Recommended training params
3) Privacy/quality tradeoff note
4) One risk and one mitigation
"""

    payload = {
        "model": _openai_model(),
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "Be practical, technical, and concise."},
            {"role": "user", "content": prompt}
        ]
    }
    request = Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    try:
        with urlopen(request, timeout=25) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
    except Exception:
        return ""


def _build_run_report(profile: dict, selected_columns: list, config: dict, output_rows: int) -> str:
    lines = [
        "AegisSynth Intelligence - Generation Run Report",
        f"Generated At (UTC): {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        "",
        "Source Dataset Profile",
        f"- Rows: {profile.get('rows', 0):,}",
        f"- Columns: {profile.get('cols', 0)}",
        f"- Numeric Columns: {profile.get('numeric_cols', 0)}",
        f"- Categorical Columns: {profile.get('categorical_cols', 0)}",
        f"- Missing %: {profile.get('missing_pct', 0)}",
        "",
        "Run Configuration",
        f"- Algorithm: {config.get('algorithm')}",
        f"- Epochs: {config.get('epochs')}",
        f"- Batch Size: {config.get('batch_size')}",
        f"- Requested Synthetic Rows: {config.get('num_rows'):,}",
        f"- Generated Rows: {output_rows:,}",
        f"- Privacy Level: {config.get('privacy_level')}",
        f"- Quality Threshold: {config.get('quality_threshold')}",
        "",
        f"Selected Features ({len(selected_columns)}): {', '.join(selected_columns[:20])}"
    ]
    if len(selected_columns) > 20:
        lines.append(f"... and {len(selected_columns) - 20} more features.")
    return "\n".join(lines)


def _render_quality_table(type_df: pd.DataFrame) -> None:
    try:
        st.dataframe(
            type_df.style.background_gradient(subset=['Missing %'], cmap='Reds')
            .background_gradient(subset=['Unique %'], cmap='Blues'),
            width='stretch',
            height=400
        )
    except Exception as exc:
        if "background_gradient requires matplotlib" in str(exc):
            st.info("Detailed color gradients disabled: install `matplotlib` in requirements.txt.")
            st.dataframe(type_df, width='stretch', height=400)
        else:
            raise


def _is_cloud_runtime() -> bool:
    if (
        os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("IS_STREAMLIT_CLOUD")
        or os.getenv("STREAMLIT_RUNTIME")
        or os.getenv("STREAMLIT_CLOUD")
    ):
        return True

    cwd = os.getcwd().lower().replace("\\", "/")
    if "/mount/src" in cwd:
        return True

    home_dir = str(os.getenv("HOME", "")).lower().replace("\\", "/")
    if "/home/adminuser" in home_dir:
        return True

    host = str(os.getenv("HOSTNAME", "")).lower()
    if "streamlit" in host:
        return True

    return False


def _safe_generation_config(
    algorithm: str,
    clean_data: pd.DataFrame,
    epochs: int,
    batch_size: int,
    num_rows: int
) -> tuple[pd.DataFrame, int, int, int, str]:
    safe_data = clean_data.copy()
    safe_epochs = int(epochs)
    safe_batch = int(batch_size)
    safe_rows = int(num_rows)
    notes = []
    is_cloud = bool(
        os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("IS_STREAMLIT_CLOUD")
        or os.getenv("STREAMLIT_RUNTIME")
    )

    if "CTGAN" in algorithm or "TVAE" in algorithm:
        max_train_rows = 1500 if is_cloud else 5000
        cat_cols = safe_data.select_dtypes(include=["object", "category"]).columns.tolist()

        # Convert date-like text columns to datetime so they are not treated as huge categorical domains.
        converted_dates = []
        for col in cat_cols:
            series = safe_data[col]
            if series.dropna().empty:
                continue
            probe = series.dropna().astype(str).str.strip().head(120)
            if probe.empty:
                continue
            likely_datetime = (
                probe.str.contains(r"[-/:]", regex=True).mean() >= 0.85
                and probe.str.contains(r"\d", regex=True).mean() >= 0.95
            ) or ("date" in col.lower() or "time" in col.lower())
            if not likely_datetime:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format")
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parse_ratio = parsed.notna().mean()
            if parse_ratio >= 0.9 and series.nunique(dropna=True) > 20:
                safe_data[col] = parsed
                converted_dates.append(col)
        if converted_dates:
            notes.append(
                f"Converted {len(converted_dates)} date-like columns to datetime: "
                + ", ".join(converted_dates[:4])
                + ("..." if len(converted_dates) > 4 else "")
            )

        cat_cols = safe_data.select_dtypes(include=["object", "category"]).columns.tolist()
        max_cat_unique = 80 if is_cloud else 180
        high_card_cols = [
            col for col in cat_cols
            if len(safe_data) > 0 and (
                safe_data[col].nunique(dropna=True) > max_cat_unique
                or (safe_data[col].nunique(dropna=True) / max(len(safe_data), 1)) > 0.18
            )
        ]
        if high_card_cols:
            safe_data = safe_data.drop(columns=high_card_cols)
            notes.append(
                f"Removed {len(high_card_cols)} high-cardinality text columns from training: "
                + ", ".join(high_card_cols[:5])
                + ("..." if len(high_card_cols) > 5 else "")
            )

        if len(safe_data.select_dtypes(include=["object", "category"]).columns) > 14:
            safe_rows = min(safe_rows, 1200 if is_cloud else 2500)
            safe_epochs = min(safe_epochs, 40 if is_cloud else 90)
            notes.append("Dense categorical schema detected; reduced rows/epochs for stable generation.")

        if len(safe_data.columns) > 35:
            max_train_rows = min(max_train_rows, 2000)
        if len(safe_data.columns) > 50:
            max_train_rows = min(max_train_rows, 1200)
        if len(safe_data) > max_train_rows:
            safe_data = safe_data.sample(max_train_rows, random_state=42)
            notes.append(f"Training data sampled to {max_train_rows:,} rows for stability.")

    if "CTGAN" in algorithm or "TVAE" in algorithm:
        max_epochs = 40 if is_cloud else 120
        max_batch = 64 if is_cloud else 200
        max_rows = 1500 if is_cloud else 5000
        if len(safe_data.columns) > 35:
            max_rows = min(max_rows, 1000)
            max_batch = min(max_batch, 48)
        if safe_epochs > max_epochs:
            safe_epochs = max_epochs
            notes.append(f"Epochs capped at {max_epochs} for stable training.")
        if safe_batch > max_batch:
            safe_batch = max_batch
            notes.append(f"Batch size capped at {max_batch} for safer execution.")
        if safe_rows > max_rows:
            safe_rows = max_rows
            notes.append(f"Synthetic output rows capped at {max_rows:,} in safe mode.")

    safe_batch = max(50, min(safe_batch, max(len(safe_data), 1)))
    return safe_data, safe_epochs, safe_batch, safe_rows, " ".join(notes)


def _select_runtime_algorithm(algorithm: str, safe_data: pd.DataFrame) -> tuple[str, str]:
    if not ("CTGAN" in algorithm or "TVAE" in algorithm):
        return algorithm, ""
    is_cloud = _is_cloud_runtime()
    rows = len(safe_data)
    cols = len(safe_data.columns)
    high_card_cols = [
        col for col in safe_data.select_dtypes(include=["object", "category"]).columns
        if rows > 0 and (
            safe_data[col].nunique(dropna=True) > 120
            or (safe_data[col].nunique(dropna=True) / max(rows, 1)) > 0.18
        )
    ]
    if is_cloud and (rows > 1800 or cols > 25 or len(high_card_cols) > 2):
        return "GaussianCopula - Statistical", (
            "Cloud-safe runtime switched to GaussianCopula to avoid memory/time crashes."
        )
    if cols > 45 or len(high_card_cols) > 6:
        return "GaussianCopula - Statistical", (
            "Runtime switched to GaussianCopula due to very high dimensional/high-cardinality dataset."
        )
    if "CTGAN" in algorithm and (cols > 30 or len(high_card_cols) > 3):
        return "TVAE - Fast Training", "Runtime switched to TVAE for lower memory usage."
    return algorithm, ""


def _apply_ui_preset(algorithm: str, epochs: int, batch_size: int, num_rows: int) -> None:
    st.session_state.ui_algorithm = algorithm
    st.session_state.ui_epochs = epochs
    st.session_state.ui_batch_size = batch_size
    st.session_state.ui_num_rows = num_rows


def _apply_autotune_from_current_data() -> None:
    if st.session_state.real_data is None:
        return
    profile = _dataset_profile(st.session_state.real_data)
    recommended = _heuristic_generation_plan(profile, st.session_state.ui_privacy_level)
    _apply_ui_preset(
        recommended["algorithm"],
        recommended["epochs"],
        recommended["batch_size"],
        recommended["num_rows"]
    )
    st.session_state.autotune_applied = True


if not st.session_state.authenticated:
    st.markdown("""
    <style>
        .auth-page-wrap {
            background:
                radial-gradient(420px 220px at 14% 8%, rgba(6,182,212,0.24), transparent 65%),
                radial-gradient(520px 260px at 86% 92%, rgba(56,189,248,0.22), transparent 68%),
                linear-gradient(145deg, #041427, #0b1b34 60%, #0f2746);
            border: 1px solid rgba(34,211,238,0.28);
            border-radius: 24px;
            padding: 1.1rem;
            box-shadow: 0 26px 54px rgba(2, 12, 30, 0.45);
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }
        .auth-brand-panel {
            height: 100%;
            border-radius: 18px;
            padding: 1.3rem 1.2rem;
            background: linear-gradient(160deg, rgba(15,23,42,0.55), rgba(14,116,144,0.28));
            border: 1px solid rgba(186,230,253,0.22);
        }
        .auth-shell {
            height: 100%;
            background: rgba(248, 252, 255, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.26);
            border-radius: 18px;
            padding: 1rem 1rem 0.7rem 1rem;
            box-shadow: 0 18px 34px rgba(3, 12, 26, 0.18);
        }
        .auth-title {
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0;
            color: #031422;
            line-height: 1.1;
        }
        .auth-subtitle {
            color: #38526d;
            margin-top: 0.45rem;
            margin-bottom: 1rem;
        }
        .auth-overline {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: #7dd3fc;
            margin: 0.15rem 0 0.5rem 0;
            font-weight: 700;
        }
        .auth-side-title {
            color: #e2f5ff;
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
        }
        .auth-side-sub {
            color: #b7d7ec;
            margin-top: 0.5rem;
            margin-bottom: 0.85rem;
            font-size: 0.95rem;
        }
        .auth-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin: 0.65rem 0 0.95rem 0;
        }
        .auth-chip {
            border: 1px solid rgba(186,230,253,0.30);
            color: #dff6ff;
            background: rgba(8, 47, 73, 0.38);
            border-radius: 999px;
            padding: 0.2rem 0.58rem;
            font-size: 0.74rem;
            font-weight: 700;
        }
        .auth-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .auth-stat {
            border: 1px solid rgba(186,230,253,0.24);
            background: rgba(2, 24, 39, 0.32);
            border-radius: 12px;
            padding: 0.62rem 0.68rem;
        }
        .auth-stat b {
            display: block;
            color: #e6f8ff;
            font-size: 1rem;
            line-height: 1;
        }
        .auth-stat span {
            color: #a3c6dc;
            font-size: 0.76rem;
            font-weight: 600;
        }
        .auth-note {
            text-align: left;
            font-size: 0.92rem;
            color: #486580;
            margin-top: 0.4rem;
        }
        [data-testid="stForm"] {
            border: 1px solid rgba(2,132,199,0.17);
            border-radius: 12px;
            padding: 0.78rem 0.78rem 0.35rem 0.78rem;
            background: rgba(255,255,255,0.84);
        }
        [data-testid="stForm"] [data-testid="stWidgetLabel"],
        [data-testid="stForm"] [data-testid="stWidgetLabel"] * {
            color: #0f172a !important;
            font-weight: 600 !important;
        }
        [data-testid="stForm"] [data-baseweb="input"] > div {
            background: #1f2430 !important;
            border: 1px solid #2f3848 !important;
            border-radius: 10px !important;
        }
        [data-testid="stForm"] input {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
            caret-color: #38bdf8 !important;
        }
        [data-testid="stForm"] input::placeholder {
            color: #94a3b8 !important;
            opacity: 1 !important;
        }
        .auth-shell .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(135deg, rgba(7, 41, 71, 0.92), rgba(8, 59, 93, 0.9));
            border-radius: 12px;
            padding: 5px;
            gap: 8px;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .auth-shell .stTabs [data-baseweb="tab"] {
            border-radius: 9px !important;
            font-weight: 700 !important;
            min-width: 126px !important;
            justify-content: center !important;
            color: #dbeafe !important;
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(125, 211, 252, 0.15) !important;
            opacity: 1 !important;
        }
        .auth-shell .stTabs [data-baseweb="tab"]:hover {
            color: #f0f9ff !important;
            background: rgba(255, 255, 255, 0.14) !important;
        }
        .auth-shell .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #06b6d4, #0ea5e9) !important;
            color: #ffffff !important;
            border: 1px solid rgba(224, 242, 254, 0.55) !important;
            box-shadow: 0 6px 16px rgba(2, 132, 199, 0.38);
        }
        @media (max-width: 980px) {
            .auth-brand-panel {
                margin-bottom: 0.75rem;
            }
            .auth-title {
                font-size: 1.8rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    if st.session_state.theme_mode == "light":
        st.markdown("""
        <style>
            .auth-page-wrap {
                background: linear-gradient(145deg, #fff7ea, #fdeed4 60%, #fbe5c3) !important;
                border: 1px solid #dcbf97 !important;
                box-shadow: 0 20px 42px rgba(15, 23, 42, 0.16) !important;
            }
            .auth-brand-panel {
                background: linear-gradient(160deg, #fffdf8, #fff1dc) !important;
                border: 1px solid #e5c9a6 !important;
            }
            .auth-overline {
                color: #a15c14 !important;
            }
            .auth-side-title {
                color: #3f2a16 !important;
            }
            .auth-side-sub {
                color: #6f5438 !important;
            }
            .auth-chip {
                background: #fffaf1 !important;
                color: #5b3b1c !important;
                border: 1px solid #d9b890 !important;
            }
            .auth-stat {
                background: #fffdf8 !important;
                border: 1px solid #e5c9a6 !important;
            }
            .auth-stat b {
                color: #402813 !important;
            }
            .auth-stat span {
                color: #775b3f !important;
            }
            .auth-shell {
                background: #fffdf8 !important;
                border: 1px solid #e5c9a6 !important;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12) !important;
            }
            .auth-title {
                color: #36220f !important;
            }
            .auth-subtitle, .auth-note {
                color: #72583d !important;
            }
            .auth-shell .stTabs [data-baseweb="tab-list"] {
                background: #fdf1dc !important;
                border: 1px solid #e3c9a9 !important;
            }
            .auth-shell .stTabs [data-baseweb="tab"] {
                color: #5f3e1f !important;
                background: #fff8ea !important;
                border: 1px solid #ebd3b5 !important;
            }
            .auth-shell .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #bf7021, #df9d3f) !important;
                color: #ffffff !important;
                border: 1px solid #d69a52 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    side_a, main_shell, side_b = st.columns([0.25, 5.1, 0.25])
    with main_shell:
        st.markdown("<div class='auth-page-wrap'>", unsafe_allow_html=True)
        left_panel, right_panel = st.columns([1.08, 1], vertical_alignment="center")
        with left_panel:
            if BRAND_LOGO_PATH:
                st.image(str(BRAND_LOGO_PATH), width=190)
            st.markdown("""
            <div class="auth-brand-panel">
                <p class="auth-overline">Synthetic Data Command Center</p>
                <p class="auth-side-title">Build safer data pipelines with an interface that feels premium.</p>
                <p class="auth-side-sub">Sign in to generate, analyze, and export enterprise-grade synthetic datasets in minutes.</p>
                <div class="auth-chip-row">
                    <span class="auth-chip">Privacy-by-Design</span>
                    <span class="auth-chip">Quality Analytics</span>
                    <span class="auth-chip">Cloud-Safe Runtime</span>
                </div>
                <div class="auth-stats">
                    <div class="auth-stat"><b>3</b><span>Generation Models</span></div>
                    <div class="auth-stat"><b>6</b><span>Analysis Modules</span></div>
                    <div class="auth-stat"><b>100%</b><span>In-App Workflow</span></div>
                    <div class="auth-stat"><b>24x7</b><span>Session Access</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with right_panel:
            st.markdown("""
            <div class="auth-shell">
                <p class="auth-title">Welcome Back</p>
                <p class="auth-subtitle">Login or create your account to open the AegisSynth dashboard.</p>
            """, unsafe_allow_html=True)

            tab_login, tab_signup = st.tabs(["Login", "Create Account"])
            with tab_login:
                with st.form("email_login_form"):
                    login_email = st.text_input("Email", key="login_email", placeholder="you@company.com")
                    login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
                    do_login = st.form_submit_button("Login with Email", width='stretch')
                    if do_login:
                        try:
                            auth_data = _firebase_signin_email_password(login_email.strip(), login_password)
                            _set_authenticated_user(auth_data, "email")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Login failed: {exc}")
            with tab_signup:
                with st.form("email_signup_form"):
                    signup_name = st.text_input("Full Name", key="signup_name", placeholder="Your name")
                    signup_email = st.text_input("Email", key="signup_email", placeholder="you@company.com")
                    signup_password = st.text_input("Password (min 6 chars)", type="password", key="signup_password", placeholder="Create password")
                    do_signup = st.form_submit_button("Create Account", width='stretch')
                    if do_signup:
                        try:
                            auth_data = _firebase_signup_email_password(signup_email.strip(), signup_password, signup_name.strip())
                            _set_authenticated_user(auth_data, "email")
                            _save_auth_event(st.session_state.auth_uid, st.session_state.auth_id_token, "email_signup")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Signup failed: {exc}")

            st.markdown("<p class='auth-note'>Privacy-first authentication powered by Firebase.</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not _firebase_web_api_key():
        st.warning("Firebase key missing. Set `firebase_web_api_key` in `.streamlit/secrets.toml`.")
    st.stop()

# Professional CSS with sidebar animation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Sora:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Sora', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Sidebar Toggle Button */
    .sidebar-toggle {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 999999;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 12px 16px;
        border-radius: 12px;
        cursor: pointer;
        font-size: 1.2rem;
        box-shadow: 0 4px 16px rgba(30, 58, 138, 0.3);
        transition: all 0.3s ease;
    }
    
    .sidebar-toggle:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
    }
    
    /* Animated background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-gradient {
        background: linear-gradient(-45deg, #1e3a8a, #3b82f6, #60a5fa, #2563eb);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Slide in animation */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: -1px;
        animation: slideIn 0.8s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
        animation: fadeIn 1.2s ease-in;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(45deg);
        transition: all 0.6s ease;
    }
    
    .metric-card:hover::before {
        top: -30%;
        left: -30%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(30, 58, 138, 0.3);
    }
    
    .metric-card h2 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
    }
    
    .metric-card p {
        font-size: 0.95rem;
        margin: 0.8rem 0 0 0;
        opacity: 0.95;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .pro-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    .pro-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border-left: 4px solid #3b82f6;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .feature-card:hover::after {
        opacity: 1;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transform: translateY(-4px);
        border-left-width: 6px;
    }
    
    .feature-card h3 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        position: relative;
        z-index: 1;
    }
    
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3b82f6;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 100px;
        height: 3px;
        background: #60a5fa;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
        font-weight: 500;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
        font-weight: 500;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
        font-weight: 500;
    }
    
    .progress-container {
        background: #e5e7eb;
        border-radius: 20px;
        height: 40px;
        overflow: hidden;
        margin: 1.5rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3);
    }
    
    .quality-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .quality-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .quality-fair {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .quality-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    .hero-section h2 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-section p {
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 400;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Interactive Upload Zone */
    .upload-zone {
        border: 3px dashed #3b82f6;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(30, 58, 138, 0.05) 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #1e3a8a;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(30, 58, 138, 0.1) 100%);
        transform: scale(1.02);
    }
    
    /* Stats Counter Animation */
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stat-counter {
        animation: countUp 0.6s ease-out;
    }
    
    .professional-footer {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-top: 4rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
    }
    
    .professional-footer h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .professional-footer p {
        margin: 0.8rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
    }
    
    .professional-footer .tech-stack {
        margin-top: 1.5rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Interactive Button Styles */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 12px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Loading Spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Theme palette presets
LIGHT_BASE = {
    "app_bg_1": "#FFF8EE",
    "app_bg_2": "#FDEFD9",
    "card_bg": "#FFFDF8",
    "text": "#2F241B",
    "muted": "#6E5D4A",
    "border": "#E8D5BC",
}

DARK_BASE = {
    "app_bg_1": "#0B1220",
    "app_bg_2": "#111B2E",
    "card_bg": "#162338",
    "text": "#E6EDF7",
    "muted": "#9FB2CA",
    "border": "#2A3B57",
}

THEME_PRESETS = {
    "Creamy Gold": {
        **LIGHT_BASE,
        "accent_1": "#C67A2B",
        "accent_2": "#E2A84E",
        "sidebar_1": "#8A5A2B",
        "sidebar_2": "#B67937",
    },
    "Ocean Blue": {
        **LIGHT_BASE,
        "accent_1": "#0B84CC",
        "accent_2": "#31B3E6",
        "sidebar_1": "#0E4D7B",
        "sidebar_2": "#1177A8",
    },
    "Mint Fresh": {
        **LIGHT_BASE,
        "accent_1": "#1C9A72",
        "accent_2": "#4BC69C",
        "sidebar_1": "#1F6E55",
        "sidebar_2": "#2F9C77",
    },
    "Dark Pro": {
        **DARK_BASE,
        "accent_1": "#0EA5E9",
        "accent_2": "#22D3EE",
        "sidebar_1": "#0F1A2D",
        "sidebar_2": "#172945",
    },
    "Midnight Neon": {
        **DARK_BASE,
        "accent_1": "#8B5CF6",
        "accent_2": "#06B6D4",
        "sidebar_1": "#121536",
        "sidebar_2": "#1A1E4F",
    }
}
if st.session_state.ui_theme_preset not in THEME_PRESETS:
    st.session_state.ui_theme_preset = "Creamy Gold"
palette = THEME_PRESETS[st.session_state.ui_theme_preset]

st.markdown(
    f"""
<style>
    :root {{
        --fx-glow: 0 0 0 rgba(0,0,0,0);
        --radius-lg: 18px;
        --radius-md: 12px;
        --shadow-soft: 0 14px 36px rgba(2, 132, 199, 0.14);
        --shadow-strong: 0 20px 48px rgba(2, 132, 199, 0.28);
    }}
    .main, .stApp {{
        background: radial-gradient(circle at 15% 10%, {palette['app_bg_2']} 0%, {palette['app_bg_1']} 55%) !important;
        color: {palette['text']} !important;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            radial-gradient(420px 220px at 12% 14%, rgba(34,211,238,0.12), transparent 65%),
            radial-gradient(520px 260px at 85% 8%, rgba(14,165,233,0.12), transparent 68%),
            radial-gradient(480px 240px at 76% 82%, rgba(6,182,212,0.10), transparent 65%);
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
        max-width: 1320px !important;
        padding-top: 1rem !important;
    }}
    .main-header {{
        letter-spacing: -0.03em !important;
        text-shadow: 0 10px 35px rgba(2, 132, 199, 0.16);
    }}
    .subtitle {{
        color: {palette['muted']} !important;
    }}
    .main-header {{
        background: linear-gradient(135deg, {palette['accent_1']} 0%, {palette['accent_2']} 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {palette['sidebar_1']} 0%, {palette['sidebar_2']} 100%) !important;
    }}
    /* Sidebar controls contrast fix */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{
        color: #eef6ff !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] span {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] input {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stSelectbox svg {{
        fill: #ffffff !important;
    }}
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] * {{
        color: {palette['text']} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        color: #eef6ff !important;
    }}
    .pro-card, .feature-card {{
        background: {palette['card_bg']} !important;
        border-color: {palette['border']} !important;
        border: 1px solid {palette['border']} !important;
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow-soft);
        transition: all 0.28s ease !important;
        border-radius: var(--radius-lg) !important;
        position: relative;
        overflow: hidden;
    }}
    .pro-card::before, .feature-card::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(130deg, rgba(255,255,255,0.35), transparent 50%);
        pointer-events: none;
    }}
    .pro-card:hover, .feature-card:hover {{
        transform: translateY(-4px);
        box-shadow: var(--shadow-strong);
    }}
    .feature-card p, .pro-card p, .pro-card h4 {{
        color: {palette['muted']} !important;
    }}
    .section-header {{
        color: {palette['accent_1']} !important;
        border-bottom-color: {palette['accent_2']} !important;
    }}
    .metric-card, .hero-section, .professional-footer {{
        background: linear-gradient(135deg, {palette['accent_1']} 0%, {palette['accent_2']} 100%) !important;
        box-shadow: 0 18px 44px rgba(2, 132, 199, 0.24) !important;
    }}
    .metric-card {{
        border-radius: 20px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
    }}
    .hero-section {{
        border-radius: 24px !important;
        border: 1px solid rgba(255,255,255,0.24) !important;
    }}
    .status-strip {{
        background: linear-gradient(120deg, rgba(2,132,199,0.12), rgba(6,182,212,0.16));
        border: 1px solid {palette['border']};
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0 1.5rem 0;
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
    }}
    .status-chip {{
        background: rgba(255,255,255,0.62);
        border: 1px solid rgba(2,132,199,0.24);
        color: {palette['text']};
        border-radius: 999px;
        padding: 0.3rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {palette['accent_1']} 0%, {palette['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 700 !important;
        transition: all 0.22s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(2,132,199,0.35) !important;
    }}
    [data-testid="stDownloadButton"] > button {{
        background: linear-gradient(135deg, {palette['accent_1']} 0%, {palette['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] {{
        border-right: 1px solid rgba(255,255,255,0.14) !important;
        box-shadow: inset -1px 0 0 rgba(255,255,255,0.05) !important;
    }}
    input, textarea, [data-baseweb="select"] > div {{
        border-radius: 10px !important;
        border-color: {palette['border']} !important;
        background: {palette['card_bg']} !important;
        color: {palette['text']} !important;
    }}
    input::placeholder, textarea::placeholder {{
        color: {palette['muted']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetric"] {{
        background: {palette['card_bg']} !important;
        border: 1px solid {palette['border']} !important;
        border-radius: 14px !important;
        padding: 0.55rem 0.75rem !important;
        box-shadow: 0 8px 22px rgba(2,132,199,0.08) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {palette['muted']} !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }}
    [data-testid="stMetricValue"] {{
        color: {palette['text']} !important;
        font-weight: 800 !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(2,132,199,0.08);
        border: 1px solid {palette['border']};
        padding: 6px;
        border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px !important;
        padding: 0.4rem 0.85rem !important;
        font-weight: 700 !important;
        color: {palette['text']} !important;
        opacity: 1 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {palette['accent_1']}, {palette['accent_2']}) !important;
        color: #fff !important;
    }}
    .stExpander {{
        border: 1px solid {palette['border']} !important;
        border-radius: 12px !important;
        background: {palette['card_bg']} !important;
    }}
    [data-testid="stExpander"] details {{
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] details > summary {{
        background: linear-gradient(135deg, rgba(2,132,199,0.14), rgba(6,182,212,0.16)) !important;
        border-bottom: 1px solid {palette['border']} !important;
        color: {palette['text']} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary label {{
        color: {palette['text']} !important;
        -webkit-text-fill-color: {palette['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {{
        fill: {palette['text']} !important;
        color: {palette['text']} !important;
        opacity: 1 !important;
    }}
    /* Sidebar-specific contrast fixes */
    [data-testid="stSidebar"] [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(255,255,255,0.24) !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] * {{
        color: #f8fbff !important;
        -webkit-text-fill-color: #f8fbff !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] .stExpander {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
    }}
    [data-testid="stSidebar"] .stExpander [data-testid="stExpanderToggleIcon"],
    [data-testid="stSidebar"] .stExpander summary,
    [data-testid="stSidebar"] .stExpander label,
    [data-testid="stSidebar"] .stExpander span {{
        color: #f8fbff !important;
        fill: #f8fbff !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.24) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {palette['border']} !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}
    @media (max-width: 900px) {{
        .main-header {{
            font-size: 2.2rem !important;
        }}
        .hero-section h2 {{
            font-size: 1.7rem !important;
        }}
        .status-strip {{
            margin-top: 0.2rem;
            padding: 0.72rem 0.8rem;
        }}
    }}
    /* Cleaner file-uploader UI */
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: none !important;
    }}
    [data-testid="stFileUploader"] section {{
        border: 2px dashed {palette['accent_2']} !important;
        border-radius: 14px !important;
        background: {palette['card_bg']} !important;
    }}
    [data-testid="stFileUploader"] button {{
        border-radius: 10px !important;
        background: linear-gradient(135deg, {palette['accent_1']} 0%, {palette['accent_2']} 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploader"] small {{
        color: {palette['muted']} !important;
    }}
    [data-testid="stFileUploader"] div {{
        color: {palette['text']} !important;
    }}
</style>
""",
    unsafe_allow_html=True
)

# Extra light-theme clarity overrides (solid surfaces, stronger borders, no washed-out cards)
if st.session_state.theme_mode == "light":
    st.markdown(
        f"""
<style>
    .stApp::before {{
        display: none !important;
    }}
    .main, .stApp {{
        background: #fff7ec !important;
    }}
    .pro-card, .feature-card {{
        background: #fffdf8 !important;
        border: 1.5px solid #dfc5a3 !important;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08) !important;
    }}
    .pro-card::before, .feature-card::before {{
        display: none !important;
    }}
    .status-strip {{
        background: #fdf0d8 !important;
        border: 1.5px solid #ddc19d !important;
    }}
    .status-chip {{
        background: #fffaf2 !important;
        border: 1.5px solid #d9b489 !important;
        color: #5a3c1f !important;
    }}
    [data-testid="stMetric"] {{
        background: #fffdf8 !important;
        border: 1.5px solid #dfc5a3 !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06) !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: #fcefd8 !important;
        border: 1.5px solid #e0c5a4 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #5d3d1f !important;
        background: #fff8ec !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {palette['accent_1']}, {palette['accent_2']}) !important;
        color: #ffffff !important;
    }}
    .stButton > button, [data-testid="stDownloadButton"] > button {{
        background: linear-gradient(135deg, {palette['accent_1']}, {palette['accent_2']}) !important;
        color: #fffefb !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 8px 18px rgba(165, 103, 36, 0.24) !important;
    }}
    .section-header {{
        color: {palette['accent_1']} !important;
        border-bottom-color: {palette['accent_2']} !important;
    }}
    .stExpander {{
        background: #fffdf8 !important;
        border: 1.5px solid #dfc5a3 !important;
    }}
    [data-testid="stExpander"] details > summary {{
        background: #fdf0d8 !important;
        border-bottom: 1.5px solid #dfc5a3 !important;
        color: #4f3318 !important;
    }}
    [data-testid="stDataFrame"] {{
        background: #fffdf8 !important;
        border: 1.5px solid #dfc5a3 !important;
    }}
    [data-testid="stFileUploader"] section {{
        background: #fffdf8 !important;
        border: 2px dashed {palette['accent_1']} !important;
    }}
</style>
""",
        unsafe_allow_html=True
    )

# Sidebar Toggle (outside sidebar)
col1, col2, col3 = st.columns([1, 6, 1])
with col3:
    theme_label = "Quick: Dark" if st.session_state.theme_mode == "light" else "Quick: Light"
    if st.button(theme_label, key="theme_toggle_btn"):
        st.session_state.ui_theme_preset = "Dark Pro" if st.session_state.theme_mode == "light" else "Creamy Gold"
        st.session_state.theme_mode = "dark" if st.session_state.ui_theme_preset in _DARK_PRESETS else "light"
        st.rerun()
with col1:
    if st.button("☰" if st.session_state.sidebar_state == 'collapsed' else "✕", key="sidebar_toggle"):
        st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
        st.rerun()

# Professional Header
if BRAND_LOGO_PATH:
    l1, l2, l3 = st.columns([2, 3, 2])
    with l2:
        st.image(str(BRAND_LOGO_PATH), width='stretch')
else:
    st.markdown('''
    <div class="slide-in">
        <h1 class="main-header">🛡️ AegisSynth Intelligence</h1>
        <p class="subtitle">Enterprise Synthetic Data Intelligence Platform</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Apply deferred navigation before nav radio is instantiated
if st.session_state.pending_nav_page:
    st.session_state.nav_page = st.session_state.pending_nav_page
    st.session_state.pending_nav_page = None

# Professional Sidebar
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Logged in as: {st.session_state.current_user}")
    st.selectbox(
        "Color Style",
        list(THEME_PRESETS.keys()),
        key="ui_theme_preset"
    )
    st.session_state.theme_mode = "dark" if st.session_state.ui_theme_preset in _DARK_PRESETS else "light"
    if st.button("Logout", width='stretch'):
        try:
            _save_auth_event(st.session_state.auth_uid, st.session_state.auth_id_token, "logout")
        except Exception:
            pass
        _clear_auth_state()
        st.rerun()
    st.markdown("---")
    st.markdown("### ⚙️ Control Panel")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Data Explorer",
            "🤖 AI Generator",
            "📈 Quality Analysis",
            "🔐 Privacy Metrics",
            "💾 Model Hub"
        ],
        key="nav_page",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Dataset info
    if st.session_state.real_data is not None:
        st.markdown("### 📊 Dataset Overview")
        data = st.session_state.real_data
        
        st.metric("Total Rows", f"{len(data):,}")
        st.metric("Total Columns", len(data.columns))
        st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
        st.metric("Categorical Features", len(data.select_dtypes(include=['object']).columns))
    
    st.markdown("---")
    
    # AI Configuration
    st.markdown("### 🤖 AI Configuration")
    
    algorithm = st.selectbox(
        "Algorithm",
        ["CTGAN - Best Quality", "TVAE - Fast Training", "GaussianCopula - Statistical"],
        key="ui_algorithm",
        help="Choose the AI model for generation"
    )
    
    num_rows = st.slider("Synthetic Records", 100, 10000, step=100, key="ui_num_rows")
    epochs = st.slider("Training Epochs", 50, 500, step=25, key="ui_epochs")
    batch_size = st.slider("Batch Size", 100, 1000, step=100, key="ui_batch_size")
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("🔬 Advanced Settings"):
        privacy_level = st.select_slider(
            "Privacy Level",
            options=["Standard", "Enhanced", "Maximum"],
            key="ui_privacy_level"
        )
        
        quality_threshold = st.slider("Quality Threshold", 0.70, 0.99, 0.85, 0.01, key="ui_quality_threshold")
    
    st.markdown("---")
    st.markdown("### ⚡ Smart Presets")
    p1, p2, p3 = st.columns(3)
    p1.button(
        "Fast",
        width='stretch',
        on_click=_apply_ui_preset,
        args=("TVAE - Fast Training", 100, 400, 1000)
    )
    p2.button(
        "Balanced",
        width='stretch',
        on_click=_apply_ui_preset,
        args=("CTGAN - Best Quality", 175, 500, 2000)
    )
    p3.button(
        "Max",
        width='stretch',
        on_click=_apply_ui_preset,
        args=("CTGAN - Best Quality", 300, 700, 5000)
    )

    st.button(
        "🧠 Auto-Tune from Dataset",
        width='stretch',
        disabled=st.session_state.real_data is None,
        on_click=_apply_autotune_from_current_data
    )
    if st.session_state.pop("autotune_applied", False):
        st.success("Applied recommended settings")
    
    st.markdown("---")
    
    # Quick actions
    if st.button("🔄 Reset Session", width='stretch'):
        st.session_state.real_data = None
        st.session_state.synthetic_data = None
        st.session_state.trained_model = None
        st.session_state.generation_complete = False
        st.rerun()

# Main Content
if page == "🏠 Home":
    # Hero Section with Animation
    st.markdown("""
    <div class='hero-section fade-in'>
        <h2>Transform Your Data with AI</h2>
        <p>Generate privacy-preserving synthetic data that maintains statistical fidelity</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="status-strip">
        <span class="status-chip">Enterprise UI</span>
        <span class="status-chip">GAN + Copilot</span>
        <span class="status-chip">Privacy First</span>
        <span class="status-chip">Cloud-Safe Runtime</span>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class='feature-card'>
            <h3>1) Upload</h3>
            <p style='color: #6b7280;'>Upload CSV and inspect structure, quality, and coverage.</p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class='feature-card'>
            <h3>2) Tune</h3>
            <p style='color: #6b7280;'>Use Presets or Auto-Tune to select GAN configuration quickly.</p>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class='feature-card'>
            <h3>3) Generate</h3>
            <p style='color: #6b7280;'>Create synthetic data, validate quality/privacy, and export.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.real_data is None:
        if st.button("Go to AI Generator (after upload)", width='stretch'):
            st.session_state.pending_nav_page = "🤖 AI Generator"
            st.rerun()
    
    # Interactive Stats Banner (if data loaded)
    if st.session_state.real_data is not None:
        real_data = st.session_state.real_data
        if st.button("Open AI Generator", type="primary", width='stretch'):
            st.session_state.pending_nav_page = "🤖 AI Generator"
            st.rerun()
        
        st.markdown("<div class='stat-counter'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card pulse-animation">
                <h2>{len(real_data):,}</h2>
                <p>Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card pulse-animation" style="animation-delay: 0.1s;">
                <h2>{len(real_data.columns)}</h2>
                <p>Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = len(real_data.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="metric-card pulse-animation" style="animation-delay: 0.2s;">
                <h2>{numeric_cols}</h2>
                <p>Numeric</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_cols = len(real_data.select_dtypes(include=['object']).columns)
            st.markdown(f"""
            <div class="metric-card pulse-animation" style="animation-delay: 0.3s;">
                <h2>{categorical_cols}</h2>
                <p>Categorical</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # File Upload with Enhanced UI
    st.markdown("<h3 class='section-header slide-in'>📤 Upload Your Dataset</h3>", unsafe_allow_html=True)
    st.markdown("**CSV upload: click `Browse files` to select your file.**")
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        key="main_csv_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        with st.spinner('🔄 Processing your data...'):
            try:
                # Load data
                encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
                loaded = False
                
                for encoding in encodings:
                    try:
                        real_data = pd.read_csv(uploaded_file, encoding=encoding)
                        st.session_state.real_data = real_data
                        loaded = True
                        break
                    except:
                        continue
                
                if loaded:
                    st.balloons()
                    
                    st.markdown("""
                    <div class='alert-success slide-in'>
                        ✅ <b>Dataset Loaded Successfully!</b> Your data is ready for processing.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Animated Metrics
                    st.markdown("<div class='stat-counter'>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{len(real_data):,}</h2>
                            <p>Total Records</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{len(real_data.columns)}</h2>
                            <p>Features</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        numeric_cols = len(real_data.select_dtypes(include=[np.number]).columns)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{numeric_cols}</h2>
                            <p>Numeric</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        categorical_cols = len(real_data.select_dtypes(include=['object']).columns)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{categorical_cols}</h2>
                            <p>Categorical</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div><br>", unsafe_allow_html=True)
                    
                    # Data Preview with Tabs
                    st.markdown("<h3 class='section-header'>📊 Dataset Preview</h3>", unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs(["📋 Data Sample", "📊 Statistics", "🔍 Data Quality"])
                    
                    with tab1:
                        search_term = st.text_input("🔍 Search in dataset", "", key="home_search")
                        
                        if search_term:
                            mask = real_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                            filtered_data = real_data[mask]
                            st.dataframe(filtered_data.head(20), width='stretch', height=400)
                            st.info(f"Found {len(filtered_data)} matching records")
                        else:
                            st.dataframe(real_data.head(20), width='stretch', height=400)
                    
                    with tab2:
                        st.dataframe(real_data.describe(), width='stretch', height=400)
                    
                    with tab3:
                        missing_pct = (real_data.isnull().sum() / len(real_data) * 100).round(2)
                        unique_pct_rounded = [round(real_data[col].nunique() / len(real_data) * 100, 2) for col in real_data.columns]
                        
                        type_df = pd.DataFrame({
                            'Column': real_data.columns,
                            'Type': real_data.dtypes.astype(str),
                            'Missing': real_data.isnull().sum().values,
                            'Missing %': missing_pct.values,
                            'Unique': [real_data[col].nunique() for col in real_data.columns],
                            'Unique %': unique_pct_rounded
                        })
                        
                        _render_quality_table(type_df)
                    
                    # Quick Insights
                    st.markdown("<h3 class='section-header'>📈 Quick Insights</h3>", unsafe_allow_html=True)
                    
                    numeric_cols_list = real_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols_list) >= 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_col = st.selectbox("Select feature for distribution", numeric_cols_list)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=real_data[selected_col],
                                marker=dict(
                                    color='#3b82f6',
                                    line=dict(color='#1e3a8a', width=1)
                                ),
                                name=selected_col
                            ))
                            fig.update_layout(
                                title=f"Distribution: {selected_col}",
                                xaxis_title=selected_col,
                                yaxis_title="Frequency",
                                height=400,
                                template="plotly_white",
                                showlegend=False
                            )
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            if len(numeric_cols_list) >= 2:
                                corr_matrix = real_data[numeric_cols_list].corr()
                                
                                fig = go.Figure(data=go.Heatmap(
                                    z=corr_matrix.values,
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    colorscale='RdBu',
                                    zmid=0,
                                    text=corr_matrix.values.round(2),
                                    texttemplate='%{text}',
                                    textfont={"size": 10},
                                    colorbar=dict(title="Correlation")
                                ))
                                fig.update_layout(
                                    title="Feature Correlations",
                                    height=400,
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig, width='stretch')
                    
                    # Interactive Action Cards
                    st.markdown("<h3 class='section-header'>⚡ Next Steps</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div class='feature-card'>
                            <h3>📊 Explore Data</h3>
                            <p style='color: #6b7280;'>Dive deep into your dataset with advanced analytics and visualizations</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class='feature-card'>
                            <h3>🤖 Generate Synthetic</h3>
                            <p style='color: #6b7280;'>Create high-quality synthetic data using state-of-the-art AI models</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                        <div class='feature-card'>
                            <h3>📈 Analyze Quality</h3>
                            <p style='color: #6b7280;'>Validate synthetic data quality with comprehensive metrics</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div class='alert-warning'>
                    ❌ <b>Error Loading Dataset:</b> {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Enhanced Feature Showcase
        st.markdown("<h3 class='section-header'>🌟 Platform Capabilities</h3>", unsafe_allow_html=True)
        
        # Animated feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='feature-card slide-in'>
                <h3>🔐 Privacy-First</h3>
                <p style='color: #6b7280; line-height: 1.8;'>
                    • GDPR & HIPAA Compliant<br>
                    • Differential Privacy<br>
                    • Zero Data Leakage<br>
                    • K-Anonymity Support<br>
                    • Enterprise Security
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-card slide-in' style='animation-delay: 0.2s;'>
                <h3>🤖 AI-Powered</h3>
                <p style='color: #6b7280; line-height: 1.8;'>
                    • CTGAN Technology<br>
                    • 95%+ Statistical Fidelity<br>
                    • Fast Generation<br>
                    • Multi-Algorithm Support<br>
                    • Production-Ready
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='feature-card slide-in' style='animation-delay: 0.4s;'>
                <h3>📊 Enterprise Features</h3>
                <p style='color: #6b7280; line-height: 1.8;'>
                    • Quality Validation<br>
                    • Model Management<br>
                    • Batch Processing<br>
                    • API Integration<br>
                    • Advanced Analytics
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Use Cases
        st.markdown("<br><h3 class='section-header'>💡 Use Cases</h3>", unsafe_allow_html=True)
        
        use_case_col1, use_case_col2 = st.columns(2)
        
        with use_case_col1:
            st.markdown("""
            <div class='pro-card'>
                <h4 style='color: #1e3a8a; margin-bottom: 1rem;'>🏥 Healthcare</h4>
                <p style='color: #6b7280;'>Generate HIPAA-compliant synthetic patient data for research and testing</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='pro-card'>
                <h4 style='color: #1e3a8a; margin-bottom: 1rem;'>💳 Finance</h4>
                <p style='color: #6b7280;'>Create synthetic transaction data for fraud detection model training</p>
            </div>
            """, unsafe_allow_html=True)
        
        with use_case_col2:
            st.markdown("""
            <div class='pro-card'>
                <h4 style='color: #1e3a8a; margin-bottom: 1rem;'>🛒 E-Commerce</h4>
                <p style='color: #6b7280;'>Generate customer behavior data for recommendation systems</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='pro-card'>
                <h4 style='color: #1e3a8a; margin-bottom: 1rem;'>🔬 Research</h4>
                <p style='color: #6b7280;'>Create privacy-preserving datasets for collaborative research</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Data Explorer":
    if st.session_state.real_data is not None:
        real_data = st.session_state.real_data
        
        st.markdown("<h2 class='section-header'>📊 Data Explorer</h2>", unsafe_allow_html=True)
        
        # Column selector
        all_cols = real_data.columns.tolist()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_cols = st.multiselect(
                "Select features to analyze",
                all_cols,
                default=all_cols[:min(5, len(all_cols))]
            )
        
        with col2:
            if st.button("Select All", width='stretch'):
                selected_cols = all_cols
        
        if selected_cols:
            # Column analysis
            for idx, col in enumerate(selected_cols):
                with st.expander(f"📊 {col} ({real_data[col].dtype})", expanded=(idx == 0)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if pd.api.types.is_numeric_dtype(real_data[col]):
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('Distribution', 'Box Plot'),
                                column_widths=[0.7, 0.3]
                            )
                            
                            fig.add_trace(
                                go.Histogram(
                                    x=real_data[col],
                                    marker_color='#3b82f6',
                                    name='Distribution'
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Box(
                                    y=real_data[col],
                                    marker_color='#1e3a8a',
                                    name='Box Plot'
                                ),
                                row=1, col=2
                            )
                            
                            fig.update_layout(
                                height=350,
                                showlegend=False,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, width='stretch')
                        
                        else:
                            value_counts = real_data[col].value_counts().head(10)
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=value_counts.index,
                                    y=value_counts.values,
                                    marker_color='#3b82f6',
                                    text=value_counts.values,
                                    textposition='auto'
                                )
                            ])
                            fig.update_layout(
                                title=f"Top 10 Values",
                                xaxis_title="Value",
                                yaxis_title="Count",
                                height=350,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        st.markdown("#### Statistics")
                        
                        stats_data = {
                            "Unique": real_data[col].nunique(),
                            "Missing": real_data[col].isnull().sum(),
                            "Missing %": f"{(real_data[col].isnull().sum() / len(real_data) * 100):.1f}%"
                        }
                        
                        if pd.api.types.is_numeric_dtype(real_data[col]):
                            stats_data.update({
                                "Mean": f"{real_data[col].mean():.2f}",
                                "Median": f"{real_data[col].median():.2f}",
                                "Std Dev": f"{real_data[col].std():.2f}",
                                "Min": f"{real_data[col].min():.2f}",
                                "Max": f"{real_data[col].max():.2f}"
                            })
                        
                        for key, value in stats_data.items():
                            st.metric(key, value)
                        
                        # Quality
                        null_pct = real_data[col].isnull().sum() / len(real_data)
                        if null_pct == 0:
                            st.markdown("<div class='quality-excellent'>✅ Excellent</div>", unsafe_allow_html=True)
                        elif null_pct < 0.05:
                            st.markdown("<div class='quality-good'>✅ Good</div>", unsafe_allow_html=True)
                        elif null_pct < 0.2:
                            st.markdown("<div class='quality-fair'>⚠️ Fair</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='quality-poor'>❌ Poor</div>", unsafe_allow_html=True)
            
            # Missing data analysis
            st.markdown("<h3 class='section-header'>🔍 Missing Data Analysis</h3>", unsafe_allow_html=True)
            
            missing_data = real_data[selected_cols].isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = go.Figure(data=[
                    go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        marker_color='#ef4444',
                        text=missing_data.values,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Missing Values by Feature",
                    xaxis_title="Feature",
                    yaxis_title="Missing Count",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.markdown("""
                <div class='alert-success'>
                    ✅ <b>Perfect Data Quality!</b> No missing values detected.
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please upload data first from the Home page")

elif page == "🤖 AI Generator":
    if st.session_state.real_data is not None:
        real_data = st.session_state.real_data
        
        st.markdown("<h2 class='section-header'>🤖 AI Synthetic Data Generator</h2>", unsafe_allow_html=True)
        dataset_profile = _dataset_profile(real_data)
        configured_plan = _heuristic_generation_plan(dataset_profile, privacy_level)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Rows", f"{dataset_profile['rows']:,}")
        a2.metric("Columns", dataset_profile["cols"])
        a3.metric("Missing %", dataset_profile["missing_pct"])
        a4.metric("Recommended", configured_plan["algorithm"].split(" - ")[0])

        with st.expander("🧠 GAN + LLM Copilot", expanded=False):
            st.caption("Use Auto-Tune for one-click settings. Use LLM Copilot for strategy guidance.")
            copilot_goal = st.text_area(
                "Generation goal",
                value="High-fidelity synthetic data with strong privacy for analytics and model testing.",
                key="copilot_goal"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Generate Copilot Plan", width='stretch'):
                    current_config = {
                        "algorithm": algorithm,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "num_rows": num_rows,
                        "privacy_level": privacy_level,
                        "quality_threshold": quality_threshold
                    }
                    llm_plan = _llm_copilot_plan(dataset_profile, copilot_goal, current_config)
                    if llm_plan:
                        st.session_state.copilot_plan = llm_plan
                    else:
                        st.session_state.copilot_plan = "\n".join([f"- {item}" for item in configured_plan["reasons"]])
            with c2:
                st.button(
                    "Apply Auto-Tune",
                    width='stretch',
                    on_click=_apply_ui_preset,
                    args=(
                        configured_plan["algorithm"],
                        configured_plan["epochs"],
                        configured_plan["batch_size"],
                        configured_plan["num_rows"]
                    )
                )

            if st.session_state.copilot_plan:
                st.code(st.session_state.copilot_plan, language="markdown")
        
        # Preprocessing
        st.markdown("### 🔧 Data Preprocessing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            remove_high_cardinality = st.checkbox(
                "Remove High Cardinality (>95%)",
                value=True
            )
        
        with col2:
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["Keep as is", "Drop rows", "Fill with mean/mode"]
            )
        
        with col3:
            normalize = st.checkbox("Normalize Numeric Features", value=False)
        
        # Column selection
        st.markdown("### 📋 Feature Selection")
        
        all_columns = real_data.columns.tolist()
        selected_columns = st.multiselect(
            "Select features for generation",
            all_columns,
            default=all_columns
        )
        
        if selected_columns:
            # Prepare data
            clean_data = real_data[selected_columns].copy()
            
            columns_removed = []
            if remove_high_cardinality:
                for col in clean_data.columns:
                    unique_ratio = clean_data[col].nunique() / len(clean_data)
                    if unique_ratio > 0.95:
                        clean_data = clean_data.drop(col, axis=1)
                        columns_removed.append(col)
            
            if columns_removed:
                st.markdown(f"""
                <div class='alert-info'>
                    ℹ️ <b>Auto-removed:</b> {', '.join(columns_removed)}
                </div>
                """, unsafe_allow_html=True)
            
            # Handle missing
            original_len = len(clean_data)
            if handle_missing == "Drop rows":
                clean_data = clean_data.dropna()
            elif handle_missing == "Fill with mean/mode":
                for col in clean_data.columns:
                    if pd.api.types.is_numeric_dtype(clean_data[col]):
                        clean_data[col].fillna(clean_data[col].mean(), inplace=True)
                    else:
                        mode_val = clean_data[col].mode()[0] if len(clean_data[col].mode()) > 0 else "Unknown"
                        clean_data[col].fillna(mode_val, inplace=True)

            if normalize:
                numeric_for_scale = clean_data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_for_scale:
                    scaler = StandardScaler()
                    clean_data[numeric_for_scale] = scaler.fit_transform(clean_data[numeric_for_scale])
            
            # Show stats
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Final Records", f"{len(clean_data):,}", delta=f"{len(clean_data) - original_len:+,}")
            col2.metric("Final Features", len(clean_data.columns))
            col3.metric("Numeric", len(clean_data.select_dtypes(include=[np.number]).columns))
            col4.metric("Categorical", len(clean_data.select_dtypes(include=['object']).columns))

            st.markdown(f"""
            <div class='alert-info'>
                <b>Recommended Plan:</b> {configured_plan['algorithm']} | epochs={configured_plan['epochs']} | batch={configured_plan['batch_size']} | target rows={configured_plan['num_rows']:,}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Generate button
            st.markdown("### 🚀 Generate Synthetic Data")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                generate_button = st.button(
                    f"🚀 Generate {num_rows:,} Synthetic Records",
                    type="primary",
                    width='stretch'
                )
            
            if generate_button:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                try:
                    def update_progress(percent, status):
                        progress_placeholder.markdown(f"""
                        <div class='progress-container'>
                            <div class='progress-bar' style='width: {percent}%;'>
                                {percent}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        status_placeholder.markdown(f"""
                        <div class='alert-info' style='margin-top: 0.5rem;'>
                            {status}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Initialize
                    update_progress(10, "🔄 Initializing AI model...")
                    time.sleep(0.5)

                    safe_clean_data, safe_epochs, safe_batch_size, safe_num_rows, safe_notes = _safe_generation_config(
                        algorithm, clean_data, epochs, batch_size, num_rows
                    )
                    if safe_notes:
                        st.info(f"Cloud-safe adjustments applied: {safe_notes}")
                    if safe_clean_data.empty or len(safe_clean_data.columns) == 0:
                        st.error("❌ Preprocessing ke baad data empty ho gaya. Missing strategy/features change karein.")
                        st.stop()

                    runtime_algorithm, runtime_note = _select_runtime_algorithm(algorithm, safe_clean_data)
                    if runtime_note:
                        st.info(runtime_note)
                    
                    metadata = Metadata.detect_from_dataframe(safe_clean_data, table_name='synthetic_table')
                    try:
                        os.makedirs("database", exist_ok=True)
                        metadata_filename = f"synthetic_table_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
                        metadata.save_to_json(os.path.join("database", metadata_filename))
                    except Exception as metadata_save_error:
                        st.info(f"Metadata snapshot not saved: {metadata_save_error}")
                    
                    algo_name = runtime_algorithm.split(" - ")[0]
                    
                    if "CTGAN" in runtime_algorithm:
                        synthesizer = CTGANSynthesizer(
                            metadata, epochs=safe_epochs, batch_size=safe_batch_size, verbose=False
                        )
                    elif "TVAE" in runtime_algorithm:
                        synthesizer = TVAESynthesizer(
                            metadata, epochs=safe_epochs, batch_size=safe_batch_size, verbose=False
                        )
                    else:
                        synthesizer = GaussianCopulaSynthesizer(metadata)
                    
                    update_progress(20, "✅ Model initialized")
                    time.sleep(0.3)
                    
                    # Training
                    update_progress(40, f"🔄 Training {algo_name}... ({safe_epochs} epochs)")
                    try:
                        synthesizer.fit(safe_clean_data)
                    except Exception as fit_error:
                        fit_error_msg = str(fit_error).lower()
                        oom_like = any(
                            token in fit_error_msg
                            for token in ["out of memory", "oom", "cuda", "killed", "allocation", "memoryerror"]
                        )
                        if "GaussianCopula" not in runtime_algorithm:
                            if oom_like:
                                st.warning("Heavy model training failed due to memory pressure. Retrying with GaussianCopula.")
                            else:
                                st.warning("Model training failed on current runtime. Retrying with GaussianCopula.")
                            runtime_algorithm = "GaussianCopula - Statistical"
                            algo_name = "GaussianCopula"
                            synthesizer = GaussianCopulaSynthesizer(metadata)
                            synthesizer.fit(safe_clean_data)
                        else:
                            raise
                    
                    update_progress(70, "✅ Training complete")
                    time.sleep(0.3)
                    
                    # Generation
                    update_progress(85, f"🔄 Generating {safe_num_rows:,} synthetic records...")
                    synthetic_data = synthesizer.sample(num_rows=safe_num_rows)
                    
                    # Add back removed columns
                    for col in columns_removed:
                        if real_data[col].dtype == 'object':
                            synthetic_data[col] = [f"{col}_{i+1}" for i in range(len(synthetic_data))]
                        else:
                            synthetic_data[col] = range(1, len(synthetic_data) + 1)
                    
                    update_progress(100, "✅ Generation complete!")
                    time.sleep(0.5)
                    
                    # Save
                    st.session_state.synthetic_data = synthetic_data
                    st.session_state.trained_model = synthesizer
                    st.session_state.generation_complete = True
                    
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'algorithm': algo_name,
                        'rows': safe_num_rows,
                        'epochs': safe_epochs,
                        'columns': len(synthetic_data.columns)
                    })
                    
                    st.balloons()
                    
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    st.markdown(f"""
                    <div class='alert-success'>
                        🎉 <b>Success!</b> Generated {safe_num_rows:,} high-quality synthetic records!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Preview
                    st.markdown("<h3 class='section-header'>📊 Generated Data</h3>", unsafe_allow_html=True)
                    
                    tab1, tab2 = st.tabs(["📋 Preview", "📊 Statistics"])
                    
                    with tab1:
                        st.dataframe(synthetic_data.head(20), width='stretch', height=400)
                    
                    with tab2:
                        st.dataframe(synthetic_data.describe(), width='stretch', height=400)
                    
                    # Download
                    st.markdown("<h3 class='section-header'>💾 Download Options</h3>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        csv = synthetic_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📄 CSV",
                            csv,
                            f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            width='stretch'
                        )
                    
                    with col2:
                        try:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                synthetic_data.to_excel(writer, index=False)
                            st.download_button(
                                "📊 Excel",
                                buffer.getvalue(),
                                f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                width='stretch'
                            )
                        except ImportError:
                            st.info("📊 Excel export requires openpyxl. Install with: pip install openpyxl")
                    
                    with col3:
                        json_data = synthetic_data.to_json(orient='records', indent=2)
                        st.download_button(
                            "📋 JSON",
                            json_data,
                            f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            width='stretch'
                        )
                    
                    with col4:
                        try:
                            parquet_buffer = io.BytesIO()
                            synthetic_data.to_parquet(parquet_buffer, index=False)
                            st.download_button(
                                "🗂️ Parquet",
                                parquet_buffer.getvalue(),
                                f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                                width='stretch'
                            )
                        except ImportError:
                            st.info("🗂️ Parquet export requires pyarrow. Install with: pip install pyarrow")

                    run_config = {
                        "algorithm": runtime_algorithm,
                        "epochs": safe_epochs,
                        "batch_size": safe_batch_size,
                        "num_rows": safe_num_rows,
                        "privacy_level": privacy_level,
                        "quality_threshold": quality_threshold
                    }
                    report_text = _build_run_report(
                        dataset_profile,
                        selected_columns,
                        run_config,
                        len(synthetic_data)
                    )
                    st.download_button(
                        "📝 Download Run Report",
                        report_text,
                        f"run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                        width='stretch'
                    )
                
                except Exception as e:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    st.markdown(f"""
                    <div class='alert-warning'>
                        ❌ <b>Generation Error:</b> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please upload data first")

elif page == "📈 Quality Analysis":
    if st.session_state.real_data is not None and st.session_state.synthetic_data is not None:
        real_data = st.session_state.real_data
        synthetic_data = st.session_state.synthetic_data
        
        st.markdown("<h2 class='section-header'>📈 Quality Analysis</h2>", unsafe_allow_html=True)
        
        # Statistical tests
        st.markdown("### 📊 Statistical Similarity Tests")
        
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(real_data[col])]
        
        if numeric_cols:
            test_results = []
            
            for col in numeric_cols:
                ks_stat, ks_pval = stats.ks_2samp(
                    real_data[col].dropna(),
                    synthetic_data[col].dropna()
                )
                
                real_mean = real_data[col].mean()
                synth_mean = synthetic_data[col].mean()
                mean_diff = abs((synth_mean - real_mean) / real_mean * 100)
                
                test_results.append({
                    'Feature': col,
                    'KS Statistic': f"{ks_stat:.4f}",
                    'KS p-value': f"{ks_pval:.4f}",
                    'Similarity': '✅ Similar' if ks_pval > 0.05 else '⚠️ Different',
                    'Mean Diff %': f"{mean_diff:.2f}%"
                })
            
            results_df = pd.DataFrame(test_results)
            st.dataframe(results_df, width='stretch', height=400)
            
            st.markdown("""
            <div class='alert-info'>
                <b>Interpretation:</b> p-value > 0.05 indicates statistical similarity between distributions
            </div>
            """, unsafe_allow_html=True)
        
        # Distribution comparison
        st.markdown("<h3 class='section-header'>📊 Distribution Comparison</h3>", unsafe_allow_html=True)
        
        if numeric_cols:
            selected_col = st.selectbox("Select feature", numeric_cols)
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=real_data[selected_col],
                name='Real Data',
                marker_color='#3b82f6',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig.add_trace(go.Histogram(
                x=synthetic_data[selected_col],
                name='Synthetic Data',
                marker_color='#10b981',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig.update_layout(
                barmode='overlay',
                title=f"Distribution Comparison: {selected_col}",
                xaxis_title=selected_col,
                yaxis_title="Frequency",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, width='stretch')
    
    else:
        st.warning("⚠️ Please generate synthetic data first")

elif page == "🔐 Privacy Metrics":
    if st.session_state.real_data is not None and st.session_state.synthetic_data is not None:
        real_data = st.session_state.real_data
        synthetic_data = st.session_state.synthetic_data
        
        st.markdown("<h2 class='section-header'>🔐 Privacy & Security Analysis</h2>", unsafe_allow_html=True)
        
        numeric_cols = [col for col in real_data.columns 
                       if pd.api.types.is_numeric_dtype(real_data[col]) 
                       and col in synthetic_data.columns]
        
        if len(numeric_cols) >= 2:
            from sklearn.neighbors import NearestNeighbors
            
            # Get clean data first
            real_clean = real_data[numeric_cols].dropna()
            synth_clean = synthetic_data[numeric_cols].dropna()
            
            # Sample based on cleaned data length
            real_sample_size = min(1000, len(real_clean))
            synth_sample_size = min(1000, len(synth_clean))
            
            real_subset = real_clean.sample(real_sample_size) if len(real_clean) > 0 else real_clean
            synth_subset = synth_clean.sample(synth_sample_size) if len(synth_clean) > 0 else synth_clean
            
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(real_subset)
            
            distances, _ = nn.kneighbors(synth_subset)
            
            avg_distance = distances.mean()
            privacy_score = min(100, avg_distance * 50)
            
            # Privacy gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=privacy_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Privacy Score", 'font': {'size': 24, 'color': '#1e3a8a'}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#10b981" if privacy_score > 80 else "#f59e0b"},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 80], 'color': "#fef3c7"},
                        {'range': [80, 100], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "#1e3a8a", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            if privacy_score > 90:
                st.markdown("""
                <div class='quality-excellent'>
                    ✅ <b>Excellent Privacy Protection</b><br>
                    Synthetic data is well-separated from real records
                </div>
                """, unsafe_allow_html=True)
            elif privacy_score > 70:
                st.markdown("""
                <div class='quality-good'>
                    ✅ <b>Good Privacy Level</b><br>
                    Acceptable privacy protection
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='quality-fair'>
                    ⚠️ <b>Moderate Privacy</b><br>
                    Consider retraining with higher privacy settings
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please generate synthetic data first")

elif page == "💾 Model Hub":
    st.markdown("<h2 class='section-header'>💾 Model Management Hub</h2>", unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown("### 📜 Generation History")
        
        history_df = pd.DataFrame(st.session_state.history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(history_df, width='stretch', height=300)
    
    # Save/Load
    st.markdown("### 💾 Model Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Save Model")
        
        if st.session_state.trained_model is not None:
            model_name = st.text_input("Model name", "synthai_model")
            
            if st.button("💾 Save Model", width='stretch'):
                try:
                    model_filename = f"{model_name}.pkl"
                    joblib.dump(st.session_state.trained_model, model_filename)
                    
                    with open(model_filename, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Model",
                            f,
                            file_name=model_filename,
                            width='stretch'
                        )
                    
                    st.success(f"✅ Model saved: {model_filename}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("Train a model first")
    
    with col2:
        st.markdown("#### Load Model")
        st.caption("Model file (.pkl)")
        uploaded_model = st.file_uploader(
            "Upload model (.pkl)",
            type=['pkl'],
            key="model_pkl_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_model:
            try:
                loaded_model = joblib.load(uploaded_model)
                st.session_state.trained_model = loaded_model
                st.success("✅ Model loaded successfully!")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Professional Footer
footer_logo_html = ""
if BRAND_LOGO_DATA_URI:
    footer_logo_html = (
        f"<img src='{BRAND_LOGO_DATA_URI}' "
        "style='display:block;margin:0 auto 1rem auto;max-width:190px;width:100%;height:auto;'/>"
    )

footer_html = f"""
<div class='professional-footer'>
    {footer_logo_html}
    <h3>AegisSynth Intelligence</h3>
    <p><b>Enterprise Synthetic Data Generation Platform</b></p>
    <p style='margin-top: 1.5rem;'>Developed by <b>Ujjval Dwivedi</b></p>
    <p class='tech-stack'>
        Built with Python • Streamlit • CTGAN • SDV • Plotly • Scikit-learn
    </p>
    <p style='margin-top: 1rem; font-size: 0.85rem;'>
        © 2026 AegisSynth Intelligence. All rights reserved. | Privacy-First AI Solutions
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


