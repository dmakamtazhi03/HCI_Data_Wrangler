import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import json
import io
import re
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI-Assisted Data Wrangler & Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# COLOR PALETTE
# =========================
FOREST = "#04202C"
DEEP_TEXT = "#304040"
PINE = "#5B7065"
MIST = "#C9D1C8"
WHITE_BAR = "#F4F4F1"

PASTEL_PALETTE = [
    "#5B7065",
    "#7A8D82",
    "#93A69B",
    "#A9BAB0",
    "#C9D1C8",
    "#B8C5BD",
    "#DCE3DD",
]

# =========================
# CSS
# =========================
def inject_custom_css():
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-main: {MIST};
            --card-bg: rgba(255,255,255,0.18);
            --title-color: {FOREST};
            --text-main: {DEEP_TEXT};
            --accent: {PINE};
            --button-text: {MIST};
            --soft-border: rgba(4, 32, 44, 0.18);
            --nav-bg: {WHITE_BAR};
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-main: {FOREST};
                --card-bg: rgba(255,255,255,0.06);
                --title-color: {MIST};
                --text-main: {MIST};
                --accent: {PINE};
                --button-text: {MIST};
                --soft-border: rgba(201, 209, 200, 0.18);
                --nav-bg: #13303A;
            }}
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: var(--bg-main) !important;
        }}

        [data-testid="stSidebar"] {{
            display: none !important;
        }}

        .block-container {{
            padding-top: 0.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px !important;
        }}

        /* navigation white strip */
        .top-nav-wrap {{
            background: var(--nav-bg);
            border-radius: 0 0 24px 24px;
            padding: 0.95rem 1.4rem 1.1rem 1.4rem;
            margin-top: -0.2rem;
            margin-bottom: 1.8rem;
            box-shadow: none;
        }}

        /* title */
        .custom-main-title {{
            text-align: center;
            font-size: 3.2rem;
            line-height: 1.12;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 2.0rem;
            color: var(--title-color);
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
            color: var(--title-color) !important;
            letter-spacing: 0.2px;
        }}

        p, li, small {{
            font-family: "Trebuchet MS", Arial, sans-serif !important;
            color: var(--text-main) !important;
        }}

        label, input, textarea, .stSelectbox, .stMultiSelect, .stNumberInput, .stTextInput {{
            font-family: "Trebuchet MS", Arial, sans-serif !important;
            color: var(--text-main) !important;
        }}

        /* radio nav */
        div[role="radiogroup"] {{
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 1.0rem !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }}

        div[role="radiogroup"] > label {{
            background: transparent !important;
            border: none !important;
            padding: 0.15rem 0.15rem !important;
            margin: 0 !important;
            border-radius: 0 !important;
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
            font-size: 1.12rem !important;
            color: var(--title-color) !important;
        }}

        /* buttons */
        .stButton > button,
        .stDownloadButton > button {{
            background-color: var(--title-color) !important;
            color: var(--button-text) !important;
            border: 1px solid var(--soft-border) !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-family: "Trebuchet MS", Arial, sans-serif !important;
            font-weight: 600 !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background-color: var(--accent) !important;
            color: var(--button-text) !important;
        }}

        .stButton > button p,
        .stDownloadButton > button p {{
            color: var(--button-text) !important;
            font-family: "Trebuchet MS", Arial, sans-serif !important;
        }}

        /* metrics */
        [data-testid="metric-container"] {{
            background-color: var(--card-bg) !important;
            border: 1px solid var(--soft-border) !important;
            border-radius: 18px !important;
            padding: 0.9rem 1rem !important;
        }}

        [data-testid="metric-container"] label {{
            color: var(--title-color) !important;
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
            font-size: 1rem !important;
        }}

        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: var(--text-main) !important;
            font-family: "Trebuchet MS", Arial, sans-serif !important;
        }}

        /* tables */
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--soft-border) !important;
            border-radius: 14px !important;
            overflow: hidden !important;
        }}

        /* expanders - fix the overlap issue */
        [data-testid="stExpander"] {{
            border-radius: 16px !important;
            border: 1px solid var(--soft-border) !important;
            background: rgba(255,255,255,0.10) !important;
            overflow: hidden !important;
        }}

        [data-testid="stExpander"] summary {{
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
            color: var(--title-color) !important;
            font-size: 1.12rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }}

        [data-testid="stExpander"] summary p {{
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif !important;
            color: var(--title-color) !important;
            font-size: 1.12rem !important;
            margin: 0 !important;
        }}

        /* form text */
        .small-note {{
            font-family: "Trebuchet MS", Arial, sans-serif !important;
            color: var(--text-main);
            font-size: 0.95rem;
            margin-top: -0.3rem;
            margin-bottom: 0.55rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# =========================
# CHART STYLE
# =========================
def set_chart_style():
    mpl.rcParams.update({
        "font.family": "Trebuchet MS",
        "axes.facecolor": MIST,
        "figure.facecolor": MIST,
        "axes.edgecolor": DEEP_TEXT,
        "axes.labelcolor": DEEP_TEXT,
        "xtick.color": DEEP_TEXT,
        "ytick.color": DEEP_TEXT,
        "text.color": DEEP_TEXT,
        "axes.titlecolor": FOREST,
    })

set_chart_style()

def get_chart_colors():
    return PASTEL_PALETTE

def get_pastel_cmap():
    return LinearSegmentedColormap.from_list(
        "custom_pastel_green",
        [FOREST, PINE, MIST]
    )

# =========================
# SESSION STATE
# =========================
def init_session():
    defaults = {
        "original_df": None,
        "working_df": None,
        "history": [],
        "transformation_log": [],
        "validation_violations": pd.DataFrame(),
        "loaded_file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def reset_session():
    keys = [
        "original_df", "working_df", "history", "transformation_log",
        "validation_violations", "loaded_file_name"
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    init_session()

def push_history():
    if st.session_state["working_df"] is not None:
        st.session_state["history"].append(st.session_state["working_df"].copy())

def log_step(operation_name, parameters, affected_columns):
    st.session_state["transformation_log"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation_name,
        "parameters": parameters,
        "affected_columns": affected_columns
    })

def undo_last_step():
    if st.session_state["history"]:
        st.session_state["working_df"] = st.session_state["history"].pop()
        if st.session_state["transformation_log"]:
            st.session_state["transformation_log"].pop()

# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    suffix = file_name.lower().split(".")[-1]
    file_obj = io.BytesIO(file_bytes)

    if suffix == "csv":
        return pd.read_csv(file_obj)
    elif suffix == "xlsx":
        return pd.read_excel(file_obj)
    elif suffix == "json":
        return pd.read_json(file_obj)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, or JSON.")

@st.cache_data(show_spinner=False)
def profile_dataframe(df):
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values
    }).sort_values("missing_count", ascending=False)

    dtypes = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })

    duplicate_count = int(df.duplicated().sum())

    numeric_summary = df.select_dtypes(include=np.number).describe().T.reset_index() \
        if not df.select_dtypes(include=np.number).empty else pd.DataFrame()

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    cat_summary_rows = []
    for c in cat_cols:
        try:
            top_val = df[c].mode(dropna=True)
            top_val = top_val.iloc[0] if len(top_val) else None
        except Exception:
            top_val = None

        cat_summary_rows.append({
            "column": c,
            "unique_values": int(df[c].nunique(dropna=True)),
            "top_value": str(top_val),
            "missing_count": int(df[c].isna().sum())
        })

    categorical_summary = pd.DataFrame(cat_summary_rows)

    return {
        "missing": missing,
        "dtypes": dtypes,
        "duplicate_count": duplicate_count,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }

def get_numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def clean_dirty_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,\$\€\£\₸\₹\¥\s]", "", regex=True),
        errors="coerce"
    )

def safe_to_datetime(series, fmt=None):
    if fmt and fmt.strip():
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def compute_outlier_mask_iqr(df, cols):
    mask = pd.Series(False, index=df.index)
    summary_rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_mask = (s < lower) | (s > upper)
        mask = mask | col_mask.fillna(False)
        summary_rows.append({
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(col_mask.fillna(False).sum())
        })
    return mask, pd.DataFrame(summary_rows)

def cap_outliers_iqr(df, cols):
    df2 = df.copy()
    changed = {}
    for col in cols:
        s = pd.to_numeric(df2[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = s.copy()
        df2[col] = s.clip(lower, upper)
        changed[col] = int((before != df2[col]).fillna(False).sum())
    return df2, changed

def min_max_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return s
    return (s - smin) / (smax - smin)

def z_score_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std()
    if pd.isna(std) or std == 0:
        return s
    return (s - mean) / std

def formula_to_python(expr):
    python_expr = expr
    python_expr = re.sub(
        r"mean\(\[([^\]]+)\]\)",
        lambda m: f"df[{repr(m.group(1))}].mean()",
        python_expr
    )
    python_expr = re.sub(
        r"median\(\[([^\]]+)\]\)",
        lambda m: f"df[{repr(m.group(1))}].median()",
        python_expr
    )
    python_expr = re.sub(
        r"std\(\[([^\]]+)\]\)",
        lambda m: f"df[{repr(m.group(1))}].std()",
        python_expr
    )
    python_expr = re.sub(
        r"log\(\[([^\]]+)\]\)",
        lambda m: f"np.log(df[{repr(m.group(1))}])",
        python_expr
    )
    python_expr = re.sub(
        r"\[([^\]]+)\]",
        lambda m: f"df[{repr(m.group(1))}]",
        python_expr
    )
    return python_expr

def apply_formula(df, new_col_name, expr):
    python_expr = formula_to_python(expr)
    allowed_globals = {"df": df, "np": np, "__builtins__": {}}
    result = eval(python_expr, allowed_globals, {})
    df2 = df.copy()
    df2[new_col_name] = result
    return df2

def df_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()

def json_bytes(data):
    return json.dumps(data, indent=2, default=str).encode("utf-8")

# =========================
# TOP NAV
# =========================
st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["Upload & Overview", "Cleaning & Preparation Studio", "Visualization Builder", "Export & Report"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN TITLE
# =========================
st.markdown(
    '<div class="custom-main-title">AI-Assisted Data Wrangler & Visualizer</div>',
    unsafe_allow_html=True
)

# =========================
# PAGE A - UPLOAD & OVERVIEW
# =========================
if page == "Upload & Overview":
    st.header("Upload & Overview")

    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or JSON",
        type=["csv", "xlsx", "json"]
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        load_clicked = st.button("Load dataset")
    with c2:
        if st.button("Reset session"):
            reset_session()
            st.rerun()

    if uploaded_file is not None and load_clicked:
        try:
            file_bytes = uploaded_file.getvalue()
            df = load_data(file_bytes, uploaded_file.name)
            st.session_state["original_df"] = df.copy()
            st.session_state["working_df"] = df.copy()
            st.session_state["history"] = []
            st.session_state["transformation_log"] = []
            st.session_state["validation_violations"] = pd.DataFrame()
            st.session_state["loaded_file_name"] = uploaded_file.name
            st.success("Dataset loaded successfully.")
        except Exception as e:
            st.error(f"Could not load file: {e}")

    df = st.session_state["working_df"]

    if df is not None:
        profile = profile_dataframe(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{df.shape[0]}")
        m2.metric("Columns", f"{df.shape[1]}")
        m3.metric("Duplicates", f"{profile['duplicate_count']}")

        st.subheader("Column names and data types")
        st.dataframe(profile["dtypes"], use_container_width=True)

        st.subheader("Missing values by column")
        st.dataframe(profile["missing"], use_container_width=True)

        csum1, csum2 = st.columns(2)

        with csum1:
            st.subheader("Numeric summary")
            if not profile["numeric_summary"].empty:
                st.dataframe(profile["numeric_summary"], use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with csum2:
            st.subheader("Categorical summary")
            if not profile["categorical_summary"].empty:
                st.dataframe(profile["categorical_summary"], use_container_width=True)
            else:
                st.info("No categorical columns found.")

        st.subheader("Data preview")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Upload a dataset and click Load dataset.")

# =========================
# PAGE B - CLEANING
# =========================
elif page == "Cleaning & Preparation Studio":
    st.header("Cleaning & Preparation Studio")

    df = st.session_state["working_df"]
    if df is None:
        st.warning("Please load a dataset first on the Upload & Overview page.")
        st.stop()

    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("Undo last step"):
            undo_last_step()
            st.success("Last step undone.")
            st.rerun()
    with top2:
        if st.button("Reset to original dataset"):
            if st.session_state["original_df"] is not None:
                st.session_state["working_df"] = st.session_state["original_df"].copy()
                st.session_state["history"] = []
                st.session_state["transformation_log"] = []
                st.session_state["validation_violations"] = pd.DataFrame()
                st.success("Dataset reset to original.")
                st.rerun()

    profile = profile_dataframe(df)

    with st.expander("4.1 Missing values (null handling)", expanded=True):
        st.write("Missing value summary (count + % per column)")
        st.dataframe(profile["missing"], use_container_width=True)

        st.markdown("**Drop rows with missing values in selected columns**")
        mv_cols = st.multiselect("Select columns for row-drop null check", df.columns.tolist(), key="mv_drop_rows_cols")
        if st.button("Drop rows with missing values"):
            if mv_cols:
                push_history()
                before_rows = len(df)
                new_df = df.dropna(subset=mv_cols)
                st.session_state["working_df"] = new_df
                log_step(
                    "Drop rows with missing values",
                    {"selected_columns": mv_cols, "rows_before": before_rows, "rows_after": len(new_df)},
                    mv_cols
                )
                st.success(f"Dropped rows. Before: {before_rows}, After: {len(new_df)}")
                st.rerun()
            else:
                st.warning("Please select at least one column.")

        st.markdown("**Drop columns with missing values above threshold (%)**")
        threshold = st.slider("Missingness threshold (%)", 0, 100, 50, key="mv_threshold")
        if st.button("Drop columns above threshold"):
            push_history()
            missing_pct = df.isna().mean() * 100
            drop_cols = missing_pct[missing_pct > threshold].index.tolist()
            new_df = df.drop(columns=drop_cols) if drop_cols else df.copy()
            st.session_state["working_df"] = new_df
            log_step(
                "Drop columns above missing threshold",
                {"threshold_percent": threshold, "dropped_columns": drop_cols},
                drop_cols
            )
            st.success(f"Dropped columns: {drop_cols if drop_cols else 'None'}")
            st.rerun()

        st.markdown("**Fill missing values**")
        fill_col = st.selectbox("Column to fill", df.columns.tolist(), key="fill_col")
        fill_method = st.selectbox(
            "Fill method",
            ["constant", "mean", "median", "mode", "most_frequent", "forward_fill", "backward_fill"],
            key="fill_method"
        )
        constant_value = st.text_input("Constant value (used only if method = constant)", key="constant_fill_value")

        if st.button("Apply missing value fill"):
            push_history()
            new_df = df.copy()
            before_missing = int(new_df[fill_col].isna().sum())

            try:
                if fill_method == "constant":
                    new_df[fill_col] = new_df[fill_col].fillna(constant_value)
                elif fill_method == "mean":
                    new_df[fill_col] = pd.to_numeric(new_df[fill_col], errors="coerce").fillna(
                        pd.to_numeric(new_df[fill_col], errors="coerce").mean()
                    )
                elif fill_method == "median":
                    new_df[fill_col] = pd.to_numeric(new_df[fill_col], errors="coerce").fillna(
                        pd.to_numeric(new_df[fill_col], errors="coerce").median()
                    )
                elif fill_method in ["mode", "most_frequent"]:
                    mode_series = new_df[fill_col].mode(dropna=True)
                    fill_val = mode_series.iloc[0] if not mode_series.empty else None
                    new_df[fill_col] = new_df[fill_col].fillna(fill_val)
                elif fill_method == "forward_fill":
                    new_df[fill_col] = new_df[fill_col].ffill()
                elif fill_method == "backward_fill":
                    new_df[fill_col] = new_df[fill_col].bfill()

                after_missing = int(new_df[fill_col].isna().sum())
                st.session_state["working_df"] = new_df
                log_step(
                    "Fill missing values",
                    {
                        "column": fill_col,
                        "method": fill_method,
                        "constant_value": constant_value if fill_method == "constant" else None,
                        "missing_before": before_missing,
                        "missing_after": after_missing
                    },
                    [fill_col]
                )
                st.success(f"Filled missing values in '{fill_col}'. Before: {before_missing}, After: {after_missing}")
                preview = pd.DataFrame({
                    "before": df[fill_col].head(10).astype(str),
                    "after": new_df[fill_col].head(10).astype(str)
                })
                st.dataframe(preview, use_container_width=True)
                st.rerun()
            except Exception as e:
                st.error(f"Could not fill missing values: {e}")

    with st.expander("4.2 Duplicates", expanded=False):
        st.write(f"Full-row duplicates found: {int(df.duplicated().sum())}")

        dup_subset_cols = st.multiselect("Select key columns for duplicate check", df.columns.tolist(), key="dup_subset_cols")
        if dup_subset_cols:
            dup_subset_df = df[df.duplicated(subset=dup_subset_cols, keep=False)].sort_values(by=dup_subset_cols)
            st.write("Duplicate groups by selected keys")
            st.dataframe(dup_subset_df.head(200), use_container_width=True)

        keep_option = st.selectbox("When removing duplicates, keep", ["first", "last"], key="dup_keep_option")
        remove_mode = st.selectbox("Duplicate removal mode", ["full_row", "subset_of_columns"], key="dup_remove_mode")

        if st.button("Remove duplicates"):
            push_history()
            before_rows = len(df)
            if remove_mode == "full_row":
                new_df = df.drop_duplicates(keep=keep_option)
                affected_cols = list(df.columns)
            else:
                if not dup_subset_cols:
                    st.warning("Select subset columns first.")
                    st.stop()
                new_df = df.drop_duplicates(subset=dup_subset_cols, keep=keep_option)
                affected_cols = dup_subset_cols

            st.session_state["working_df"] = new_df
            log_step(
                "Remove duplicates",
                {
                    "mode": remove_mode,
                    "subset_columns": dup_subset_cols if remove_mode == "subset_of_columns" else None,
                    "keep": keep_option,
                    "rows_before": before_rows,
                    "rows_after": len(new_df)
                },
                affected_cols
            )
            st.success(f"Duplicates removed. Before: {before_rows}, After: {len(new_df)}")
            st.rerun()

    with st.expander("4.3 Data types & parsing", expanded=False):
        dtype_col = st.selectbox("Select column to convert", df.columns.tolist(), key="dtype_col")
        target_type = st.selectbox("Convert to", ["numeric", "categorical", "datetime"], key="target_type")
        dt_format = st.text_input("Datetime format (optional, e.g. %Y-%m-%d)", key="dt_format")
        clean_numeric_strings = st.checkbox("Clean dirty numeric strings (commas, currency signs)", value=True, key="clean_numeric_strings")

        if st.button("Apply type conversion"):
            push_history()
            new_df = df.copy()
            try:
                if target_type == "numeric":
                    if clean_numeric_strings:
                        new_df[dtype_col] = clean_dirty_numeric(new_df[dtype_col])
                    else:
                        new_df[dtype_col] = pd.to_numeric(new_df[dtype_col], errors="coerce")
                elif target_type == "categorical":
                    new_df[dtype_col] = new_df[dtype_col].astype("category")
                elif target_type == "datetime":
                    new_df[dtype_col] = safe_to_datetime(new_df[dtype_col], dt_format)

                st.session_state["working_df"] = new_df
                log_step(
                    "Convert column type",
                    {
                        "column": dtype_col,
                        "target_type": target_type,
                        "datetime_format": dt_format if target_type == "datetime" else None,
                        "clean_dirty_numeric_strings": clean_numeric_strings if target_type == "numeric" else None
                    },
                    [dtype_col]
                )
                st.success(f"Converted '{dtype_col}' to {target_type}.")
                st.rerun()
            except Exception as e:
                st.error(f"Type conversion failed: {e}")

    with st.expander("4.4 Categorical data tools", expanded=False):
        cat_cols = df.columns.tolist()
        cat_col = st.selectbox("Categorical column", cat_cols, key="cat_col_tools")
        standardize_action = st.selectbox("Value standardization", ["trim_whitespace", "lower_case", "title_case"], key="cat_standardize_action")

        if st.button("Apply standardization"):
            push_history()
            new_df = df.copy()
            s = new_df[cat_col].astype(str)
            if standardize_action == "trim_whitespace":
                new_df[cat_col] = s.str.strip()
            elif standardize_action == "lower_case":
                new_df[cat_col] = s.str.strip().str.lower()
            elif standardize_action == "title_case":
                new_df[cat_col] = s.str.strip().str.title()

            st.session_state["working_df"] = new_df
            log_step(
                "Standardize categorical values",
                {"column": cat_col, "action": standardize_action},
                [cat_col]
            )
            st.success(f"Applied {standardize_action} to '{cat_col}'.")
            st.rerun()

        st.markdown("**Mapping / replacement**")
        mapping_text = st.text_area(
            'Enter mapping as JSON, example: {{"ny": "New York", "la": "Los Angeles"}}',
            height=120,
            key="mapping_text"
        )
        unmatched_other = st.checkbox("Set unmatched values to 'Other' (otherwise keep unchanged)", key="unmatched_other")

        if st.button("Apply mapping"):
            try:
                mapping = json.loads(mapping_text) if mapping_text.strip() else {}
                if not isinstance(mapping, dict):
                    raise ValueError("Mapping must be a JSON object/dictionary.")
                push_history()
                new_df = df.copy()
                if unmatched_other:
                    new_df[cat_col] = new_df[cat_col].map(mapping).fillna("Other")
                else:
                    new_df[cat_col] = new_df[cat_col].replace(mapping)

                st.session_state["working_df"] = new_df
                log_step(
                    "Apply categorical mapping",
                    {"column": cat_col, "mapping": mapping, "unmatched_to_other": unmatched_other},
                    [cat_col]
                )
                st.success(f"Mapping applied to '{cat_col}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Mapping failed: {e}")

        st.markdown("**Rare category grouping**")
        rare_threshold = st.number_input("Minimum frequency count to keep category", min_value=1, value=5, step=1, key="rare_threshold")
        if st.button("Group rare categories into Other"):
            push_history()
            new_df = df.copy()
            counts = new_df[cat_col].value_counts(dropna=False)
            keepers = counts[counts >= rare_threshold].index
            new_df[cat_col] = new_df[cat_col].where(new_df[cat_col].isin(keepers), "Other")
            st.session_state["working_df"] = new_df
            log_step(
                "Group rare categories",
                {"column": cat_col, "min_frequency_threshold": int(rare_threshold), "replacement_value": "Other"},
                [cat_col]
            )
            st.success(f"Rare categories grouped into 'Other' in '{cat_col}'.")
            st.rerun()

        st.markdown("**One-hot encoding**")
        one_hot_cols = st.multiselect("Select columns to one-hot encode", df.columns.tolist(), key="one_hot_cols")
        if st.button("Apply one-hot encoding"):
            if one_hot_cols:
                push_history()
                before_cols = df.shape[1]
                new_df = pd.get_dummies(df, columns=one_hot_cols, dummy_na=False)
                st.session_state["working_df"] = new_df
                log_step(
                    "One-hot encode columns",
                    {"columns": one_hot_cols, "columns_before": before_cols, "columns_after": new_df.shape[1]},
                    one_hot_cols
                )
                st.success(f"One-hot encoding complete. Columns before: {before_cols}, after: {new_df.shape[1]}")
                st.rerun()
            else:
                st.warning("Select at least one column.")

    with st.expander("4.5 Numeric cleaning", expanded=False):
        numeric_cols = get_numeric_cols(df)
        if numeric_cols:
            outlier_cols = st.multiselect("Select numeric columns for outlier handling", numeric_cols, default=numeric_cols[:1], key="outlier_cols")
            if outlier_cols:
                mask, outlier_summary = compute_outlier_mask_iqr(df, outlier_cols)
                st.write("Outlier detection summary (IQR)")
                st.dataframe(outlier_summary, use_container_width=True)

                outlier_action = st.selectbox("Choose action", ["do_nothing", "cap_winsorize", "remove_outlier_rows"], key="outlier_action")
                if st.button("Apply outlier action"):
                    if outlier_action == "do_nothing":
                        st.info("No changes applied.")
                    elif outlier_action == "cap_winsorize":
                        push_history()
                        new_df, changed = cap_outliers_iqr(df, outlier_cols)
                        st.session_state["working_df"] = new_df
                        log_step(
                            "Cap outliers",
                            {"columns": outlier_cols, "changed_value_counts": changed, "method": "IQR capping"},
                            outlier_cols
                        )
                        st.success(f"Outliers capped. Value changes: {changed}")
                        st.rerun()
                    elif outlier_action == "remove_outlier_rows":
                        push_history()
                        before_rows = len(df)
                        new_df = df.loc[~mask].copy()
                        st.session_state["working_df"] = new_df
                        log_step(
                            "Remove outlier rows",
                            {"columns": outlier_cols, "rows_before": before_rows, "rows_after": len(new_df), "method": "IQR"},
                            outlier_cols
                        )
                        st.success(f"Outlier rows removed. Before: {before_rows}, After: {len(new_df)}")
                        st.rerun()
        else:
            st.info("No numeric columns available.")

    with st.expander("4.6 Normalization / scaling", expanded=False):
        numeric_cols = get_numeric_cols(df)
        if numeric_cols:
            scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols, key="scale_cols")
            scale_method = st.selectbox("Scaling method", ["min_max", "z_score"], key="scale_method")

            if st.button("Apply scaling"):
                if scale_cols:
                    push_history()
                    new_df = df.copy()
                    before_stats = new_df[scale_cols].describe().T[["mean", "std", "min", "max"]].reset_index()

                    for col in scale_cols:
                        if scale_method == "min_max":
                            new_df[col] = min_max_scale(new_df[col])
                        else:
                            new_df[col] = z_score_scale(new_df[col])

                    after_stats = new_df[scale_cols].describe().T[["mean", "std", "min", "max"]].reset_index()

                    st.session_state["working_df"] = new_df
                    log_step(
                        "Scale numeric columns",
                        {"columns": scale_cols, "method": scale_method},
                        scale_cols
                    )
                    st.success(f"Applied {scale_method} scaling.")
                    st.write("Before stats")
                    st.dataframe(before_stats, use_container_width=True)
                    st.write("After stats")
                    st.dataframe(after_stats, use_container_width=True)
                    st.rerun()
                else:
                    st.warning("Select at least one numeric column.")
        else:
            st.info("No numeric columns available.")

    with st.expander("4.7 Column operations", expanded=False):
        st.markdown("**Rename columns**")
        rename_old = st.selectbox("Current column name", df.columns.tolist(), key="rename_old")
        rename_new = st.text_input("New column name", key="rename_new")
        if st.button("Rename column"):
            if rename_new.strip():
                push_history()
                new_df = df.copy().rename(columns={rename_old: rename_new.strip()})
                st.session_state["working_df"] = new_df
                log_step(
                    "Rename column",
                    {"old_name": rename_old, "new_name": rename_new.strip()},
                    [rename_old]
                )
                st.success(f"Renamed '{rename_old}' to '{rename_new.strip()}'.")
                st.rerun()
            else:
                st.warning("Enter a new column name.")

        st.markdown("**Drop columns**")
        drop_cols = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols")
        if st.button("Drop selected columns"):
            if drop_cols:
                push_history()
                new_df = df.drop(columns=drop_cols)
                st.session_state["working_df"] = new_df
                log_step(
                    "Drop columns",
                    {"dropped_columns": drop_cols},
                    drop_cols
                )
                st.success(f"Dropped columns: {drop_cols}")
                st.rerun()
            else:
                st.warning("Select at least one column.")

        st.markdown("**Create new column using formula**")
        st.markdown(
            """
            <div class="small-note">
            Use square brackets around column names.<br>
            Examples:<br>
            [Sales] / [Quantity]<br>
            log([Price])<br>
            [Amount] - mean([Amount])
            </div>
            """,
            unsafe_allow_html=True
        )
        new_formula_col = st.text_input("New column name", key="new_formula_col")
        formula_expr = st.text_input("Formula", key="formula_expr")
        if st.button("Create formula column"):
            if new_formula_col.strip() and formula_expr.strip():
                try:
                    push_history()
                    new_df = apply_formula(df, new_formula_col.strip(), formula_expr.strip())
                    st.session_state["working_df"] = new_df
                    log_step(
                        "Create column with formula",
                        {"new_column": new_formula_col.strip(), "formula": formula_expr.strip()},
                        [new_formula_col.strip()]
                    )
                    st.success(f"Created new column '{new_formula_col.strip()}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not create formula column: {e}")
            else:
                st.warning("Enter both a new column name and a formula.")

        st.markdown("**Binning numeric columns into categories**")
        numeric_cols = get_numeric_cols(df)
        if numeric_cols:
            bin_col = st.selectbox("Numeric column to bin", numeric_cols, key="bin_col")
            bin_method = st.selectbox("Binning method", ["equal_width", "quantile"], key="bin_method")
            bin_count = st.number_input("Number of bins", min_value=2, value=4, step=1, key="bin_count")
            bin_new_col = st.text_input("New binned column name", value="binned_category", key="bin_new_col")
            if st.button("Create binned column"):
                try:
                    push_history()
                    new_df = df.copy()
                    if bin_method == "equal_width":
                        new_df[bin_new_col] = pd.cut(new_df[bin_col], bins=int(bin_count))
                    else:
                        new_df[bin_new_col] = pd.qcut(new_df[bin_col], q=int(bin_count), duplicates="drop")
                    st.session_state["working_df"] = new_df
                    log_step(
                        "Bin numeric column",
                        {
                            "source_column": bin_col,
                            "new_column": bin_new_col,
                            "method": bin_method,
                            "bin_count": int(bin_count)
                        },
                        [bin_col, bin_new_col]
                    )
                    st.success(f"Created binned column '{bin_new_col}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Binning failed: {e}")
        else:
            st.info("No numeric columns available for binning.")

    with st.expander("4.8 Data validation rules", expanded=False):
        vtabs = st.tabs(["Numeric range check", "Allowed categories", "Non-null constraint"])

        with vtabs[0]:
            numeric_cols = get_numeric_cols(df)
            if numeric_cols:
                v_num_col = st.selectbox("Numeric column", numeric_cols, key="v_num_col")
                v_min = st.number_input("Minimum allowed value", value=0.0, key="v_min")
                v_max = st.number_input("Maximum allowed value", value=100.0, key="v_max")
                if st.button("Run numeric range check"):
                    bad = df[(pd.to_numeric(df[v_num_col], errors="coerce") < v_min) |
                             (pd.to_numeric(df[v_num_col], errors="coerce") > v_max)].copy()
                    bad["violation_rule"] = f"{v_num_col} outside [{v_min}, {v_max}]"
                    st.session_state["validation_violations"] = bad
                    st.write("Violations table")
                    st.dataframe(bad, use_container_width=True)
            else:
                st.info("No numeric columns available.")

        with vtabs[1]:
            v_cat_col = st.selectbox("Categorical column for allowed list", df.columns.tolist(), key="v_cat_col")
            allowed_vals_text = st.text_input("Allowed values (comma separated)", key="allowed_vals_text")
            if st.button("Run allowed categories check"):
                allowed_vals = [x.strip() for x in allowed_vals_text.split(",") if x.strip()]
                bad = df[~df[v_cat_col].astype(str).isin(allowed_vals)].copy()
                bad["violation_rule"] = f"{v_cat_col} not in allowed list"
                st.session_state["validation_violations"] = bad
                st.write("Violations table")
                st.dataframe(bad, use_container_width=True)

        with vtabs[2]:
            non_null_cols = st.multiselect("Columns that must be non-null", df.columns.tolist(), key="non_null_cols")
            if st.button("Run non-null check"):
                if non_null_cols:
                    bad = df[df[non_null_cols].isna().any(axis=1)].copy()
                    bad["violation_rule"] = f"Null found in required columns: {', '.join(non_null_cols)}"
                    st.session_state["validation_violations"] = bad
                    st.write("Violations table")
                    st.dataframe(bad, use_container_width=True)
                else:
                    st.warning("Select at least one column.")

        if isinstance(st.session_state["validation_violations"], pd.DataFrame) and not st.session_state["validation_violations"].empty:
            csv_viol = st.session_state["validation_violations"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export violations csv",
                data=csv_viol,
                file_name="validation_violations.csv",
                mime="text/csv"
            )

    st.subheader("Current working dataset preview")
    st.dataframe(st.session_state["working_df"].head(20), use_container_width=True)

    st.subheader("Transformation log")
    if st.session_state["transformation_log"]:
        st.dataframe(pd.DataFrame(st.session_state["transformation_log"]), use_container_width=True)
    else:
        st.info("No transformations applied yet.")

# =========================
# PAGE C - VISUALIZATION
# =========================
elif page == "Visualization Builder":
    st.header("Visualization Builder")

    df = st.session_state["working_df"]
    if df is None:
        st.warning("Please load a dataset first on the Upload & Overview page.")
        st.stop()

    working = df.copy()

    st.subheader("Filters")
    filter_cols1, filter_cols2 = st.columns(2)

    with filter_cols1:
        cat_filter_col = st.selectbox("Categorical filter column (optional)", ["None"] + working.columns.tolist(), key="viz_cat_filter_col")
        if cat_filter_col != "None":
            options = working[cat_filter_col].dropna().astype(str).unique().tolist()
            selected_cat_values = st.multiselect("Select values", options, default=options[:min(5, len(options))], key="viz_cat_values")
            if selected_cat_values:
                working = working[working[cat_filter_col].astype(str).isin(selected_cat_values)]

    with filter_cols2:
        numeric_cols = get_numeric_cols(working)
        num_filter_col = st.selectbox("Numeric range filter column (optional)", ["None"] + numeric_cols, key="viz_num_filter_col")
        if num_filter_col != "None":
            num_min = float(pd.to_numeric(working[num_filter_col], errors="coerce").min())
            num_max = float(pd.to_numeric(working[num_filter_col], errors="coerce").max())
            selected_range = st.slider("Select numeric range", min_value=num_min, max_value=num_max, value=(num_min, num_max), key="viz_num_slider")
            working = working[(pd.to_numeric(working[num_filter_col], errors="coerce") >= selected_range[0]) &
                              (pd.to_numeric(working[num_filter_col], errors="coerce") <= selected_range[1])]

    st.subheader("Choose your chart")
    chart_type = st.selectbox(
        "Plot type",
        ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Heatmap / Correlation Matrix"],
        key="chart_type"
    )

    x_col = st.selectbox("X column", ["None"] + working.columns.tolist(), key="viz_x_col")
    y_col = st.selectbox("Y column", ["None"] + working.columns.tolist(), key="viz_y_col")
    group_col = st.selectbox("Optional color / group column", ["None"] + working.columns.tolist(), key="viz_group_col")
    agg_func = st.selectbox("Optional aggregation", ["none", "sum", "mean", "count", "median"], key="viz_agg")
    top_n = st.number_input("Top N categories for bar chart", min_value=1, value=10, step=1, key="viz_top_n")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = get_chart_colors()

    try:
        if chart_type == "Histogram":
            if x_col == "None":
                st.warning("Choose a numeric X column.")
            else:
                s = pd.to_numeric(working[x_col], errors="coerce").dropna()
                ax.hist(s, color=colors[0], edgecolor=DEEP_TEXT)
                ax.set_title(f"Histogram of {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequency")

        elif chart_type == "Box Plot":
            if x_col == "None":
                st.warning("Choose a numeric X column.")
            else:
                s = pd.to_numeric(working[x_col], errors="coerce").dropna()
                ax.boxplot(
                    s,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[2], color=DEEP_TEXT),
                    medianprops=dict(color=FOREST, linewidth=2),
                    whiskerprops=dict(color=DEEP_TEXT),
                    capprops=dict(color=DEEP_TEXT)
                )
                ax.set_title(f"Box Plot of {x_col}")
                ax.set_ylabel(x_col)

        elif chart_type == "Scatter Plot":
            if x_col == "None" or y_col == "None":
                st.warning("Choose both X and Y columns.")
            else:
                x = pd.to_numeric(working[x_col], errors="coerce")
                y = pd.to_numeric(working[y_col], errors="coerce")
                if group_col != "None":
                    groups = working[group_col].astype(str).fillna("Missing").unique().tolist()
                    for i, g in enumerate(groups):
                        subset = working[working[group_col].astype(str).fillna("Missing") == g]
                        ax.scatter(
                            pd.to_numeric(subset[x_col], errors="coerce"),
                            pd.to_numeric(subset[y_col], errors="coerce"),
                            color=colors[i % len(colors)],
                            edgecolor=DEEP_TEXT,
                            alpha=0.8,
                            label=g
                        )
                    ax.legend()
                else:
                    ax.scatter(x, y, color=colors[1], edgecolor=DEEP_TEXT, alpha=0.8)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)

        elif chart_type == "Line Chart":
            if x_col == "None" or y_col == "None":
                st.warning("Choose both X and Y columns.")
            else:
                plot_df = working.copy()
                plot_df = plot_df.sort_values(by=x_col)
                if group_col != "None":
                    groups = plot_df[group_col].astype(str).fillna("Missing").unique().tolist()
                    for i, g in enumerate(groups):
                        subset = plot_df[plot_df[group_col].astype(str).fillna("Missing") == g]
                        ax.plot(
                            subset[x_col],
                            pd.to_numeric(subset[y_col], errors="coerce"),
                            color=colors[i % len(colors)],
                            linewidth=2.2,
                            label=g
                        )
                    ax.legend()
                else:
                    ax.plot(plot_df[x_col], pd.to_numeric(plot_df[y_col], errors="coerce"), color=colors[0], linewidth=2.5)
                ax.set_title(f"Line Chart: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)

        elif chart_type == "Bar Chart":
            if x_col == "None":
                st.warning("Choose at least an X column.")
            else:
                plot_df = working.copy()
                if agg_func == "none":
                    counts = plot_df[x_col].astype(str).value_counts().head(int(top_n))
                    x_vals = counts.index.tolist()
                    y_vals = counts.values.tolist()
                else:
                    if y_col == "None":
                        st.warning("Choose a Y column for aggregation.")
                        st.stop()
                    grouped = plot_df.groupby(x_col)[y_col]
                    if agg_func == "sum":
                        res = grouped.sum()
                    elif agg_func == "mean":
                        res = grouped.mean()
                    elif agg_func == "count":
                        res = grouped.count()
                    elif agg_func == "median":
                        res = grouped.median()
                    else:
                        res = grouped.sum()

                    res = res.sort_values(ascending=False).head(int(top_n))
                    x_vals = res.index.astype(str).tolist()
                    y_vals = res.values.tolist()

                bar_colors = [colors[i % len(colors)] for i in range(len(x_vals))]
                ax.bar(x_vals, y_vals, color=bar_colors, edgecolor=DEEP_TEXT)
                ax.set_title(f"Bar Chart of {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Value")
                ax.tick_params(axis='x', rotation=45)

        elif chart_type == "Heatmap / Correlation Matrix":
            numeric_df = working.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                st.warning("Need at least 2 numeric columns for a correlation matrix.")
            else:
                corr = numeric_df.corr(numeric_only=True)
                cmap = get_pastel_cmap()
                im = ax.imshow(corr, cmap=cmap)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.columns)
                ax.set_title("Correlation Matrix")
                fig.colorbar(im, ax=ax)

        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart could not be created: {e}")

    st.subheader("Filtered data preview")
    st.dataframe(working.head(20), use_container_width=True)

# =========================
# PAGE D - EXPORT
# =========================
elif page == "Export & Report":
    st.header("Export & Report")

    df = st.session_state["working_df"]
    if df is None:
        st.warning("Please load a dataset first on the Upload & Overview page.")
        st.stop()

    st.subheader("Transformation log")
    if st.session_state["transformation_log"]:
        log_df = pd.DataFrame(st.session_state["transformation_log"])
        st.dataframe(log_df, use_container_width=True)
    else:
        log_df = pd.DataFrame()
        st.info("No transformations logged yet.")

    st.subheader("Export cleaned dataset")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned csv",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    try:
        excel_data = df_to_excel_bytes(df)
        st.download_button(
            "Download cleaned excel",
            data=excel_data,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.info("Excel export is unavailable. If needed, install openpyxl.")

    st.subheader("Export transformation report / recipe")
    recipe = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": st.session_state.get("loaded_file_name"),
        "steps": st.session_state["transformation_log"],
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist()
    }

    st.download_button(
        "Download json recipe / report",
        data=json_bytes(recipe),
        file_name="transformation_recipe.json",
        mime="application/json"
    )

    if not log_df.empty:
        report_csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download transformation log csv",
            data=report_csv,
            file_name="transformation_report.csv",
            mime="text/csv"
        )

    if isinstance(st.session_state["validation_violations"], pd.DataFrame) and not st.session_state["validation_violations"].empty:
        viol_csv = st.session_state["validation_violations"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download validation violations csv",
            data=viol_csv,
            file_name="validation_violations.csv",
            mime="text/csv"
        )

    st.subheader("Current dataset preview")
    st.dataframe(df.head(30), use_container_width=True)
