import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import tempfile
import os
import re
import requests

st.set_page_config(layout="wide", page_title="Data Cleaning  App")

def push_undo():
    # Save current state before modifying
    st.session_state.undo_stack.append(
        st.session_state.working_df.copy()
    )
    # Once a new action happens, redo history is invalid



def undo():
    if st.session_state.undo_stack:
        st.session_state.redo_stack.append(
            st.session_state.working_df.copy()
        )
        st.session_state.working_df = st.session_state.undo_stack.pop()
        st.rerun()


def redo():
    if st.session_state.redo_stack:
        st.session_state.undo_stack.append(
            st.session_state.working_df.copy()
        )
        st.session_state.working_df = st.session_state.redo_stack.pop()
        st.rerun()

# =========================
# SESSION STATE
# =========================
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []
if "working_df" not in st.session_state:
    st.session_state.working_df = None
if "kpis" not in st.session_state:
    st.session_state.kpis = []
if "dup_key" not in st.session_state:
    st.session_state.dup_key = None
if "groupby_result" not in st.session_state:
    st.session_state.groupby_result = None
if "run_kpi" not in st.session_state:
    st.session_state.run_kpi = False
# =========================
# POSSIBLE UNIQUE ID COLUMNS
# =========================
POSSIBLE_KEYS = [
    "order_id", "invoice_id", "transaction_id",
    "bill_id", "customer_id", "id"
]

# =========================
# DERIVED COLUMNS
# =========================
DERIVED_COLUMNS = {
    "total_amount": {
        "requires": ["quantity", "price"],
        "formula": lambda df: df["quantity"] * df["price"]
    },
    "profit": {
        "requires": ["total_sales", "cost"],
        "formula": lambda df: df["total_sales"] - df["cost"]
    },
    "revenue": {
        "requires": ["total_sales"],
        "formula": lambda df: df["total_sales"]
    }
}

# =========================
# GOOGLE DRIVE LINK CONVERTER
# =========================
def convert_gdrive_link(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        return None
    file_id = match.group(1)
    return f"https://drive.google.com/uc?id={file_id}&export=download"

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙ Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("↩ Undo"):
        undo()
        st.rerun()

with col2:
    if st.button("↪ Redo"):
        redo()
        st.rerun()

if st.sidebar.button("🔄 Reset cleaning"):
    st.session_state.original_df = None
    st.session_state.working_df = None
    st.session_state.kpis = []
    st.session_state.dup_key = None
    st.session_state.groupby_result = None
    st.rerun()

# =========================
# TITLE
# =========================
st.title("🧹 Data Cleaning App")

st.info(
    "✔ CSV / Excel / TXT supported\n"
    "✔ ZIP folder upload supported\n"
    "✔ Google Drive file links supported"
)

# =========================
# FILE / ZIP / LINK INPUT
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV / Excel / TXT / ZIP (folder)",
    type=["csv", "xlsx", "txt", "zip"]
)

gdrive_link = st.text_input(
    "Or paste Google Drive file link (optional)"
)

def load_dataframe_from_file(file, name):
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file, delimiter=None, engine="python")

if uploaded_file or gdrive_link:
    try:
        if gdrive_link:
            download_url = convert_gdrive_link(gdrive_link)
            response = requests.get(download_url)
            buffer = io.BytesIO(response.content)
            df = pd.read_csv(buffer)

        elif uploaded_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)

                files = []
                for root, _, filenames in os.walk(tmpdir):
                    for name in filenames:
                        if name.endswith((".csv", ".xlsx", ".txt")):
                            files.append(os.path.join(root, name))

                selected_file = st.selectbox("Select file from uploaded folder", files)
                df = load_dataframe_from_file(selected_file, selected_file)

        else:
            df = load_dataframe_from_file(uploaded_file, uploaded_file.name)

        st.session_state.original_df = df.copy()
        st.session_state.working_df = df.copy()
        st.success("Dataset loaded successfully")

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if st.session_state.working_df is None:
    st.stop()

df = st.session_state.working_df

# =========================
# AUTO-DETECT UNIQUE KEY
# =========================
st.session_state.dup_key = None
for key in POSSIBLE_KEYS:
    if key in df.columns:
        st.session_state.dup_key = key
        break

# =========================
# DUPLICATE ROWS (UNCHANGED)
# =========================
st.subheader("🔁 Duplicate Rows (Automatic Check)")

if st.session_state.dup_key:
    key = st.session_state.dup_key

    rows_before = len(df)
    dup_before = df.duplicated(subset=[key]).sum()

    push_undo()
    df = df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    st.session_state.working_df = df

    rows_after = len(df)
    dup_after = df.duplicated(subset=[key]).sum()

    left, right = st.columns(2)

    with left:
        st.metric(f"Duplicate rows BEFORE removal (based on `{key}`)", dup_before)
        st.metric(f"Duplicate rows AFTER removal (based on `{key}`)", dup_after)

    with right:
        st.metric("Total Rows (Before)", rows_before)
        st.metric("Total Rows (After)", rows_after)

else:
    st.warning("No unique ID column found. Duplicate removal skipped.")

# =========================
# MANUAL DUPLICATE REMOVAL
# =========================
st.subheader("🧍 Manual Duplicate Removal")

df = st.session_state.working_df

# Select column for manual duplicate check
manual_col = st.selectbox(
    "Select column to check duplicates",
    df.columns,
    key="manual_dup_col"
)

# Count duplicates for selected column
manual_dup_count = df.duplicated(subset=[manual_col]).sum()

st.info(f"Duplicate rows found based on `{manual_col}`: {manual_dup_count}")

# Choose which row to keep
keep_option = st.radio(
    "Which duplicate row to keep?",
    ["First", "Last"],
    horizontal=True
)

# Remove duplicates ONLY on button click
if st.button("🗑 Remove Duplicates Manually"):
    push_undo()
    df = df.drop_duplicates(
        subset=[manual_col],
        keep="first" if keep_option == "First" else "last"
    ).reset_index(drop=True)

    st.session_state.working_df = df
    st.success("Manual duplicate removal completed successfully ✅")

# =========================
# DERIVED COLUMNS
# =========================
# 1️⃣ Remove old derived columns
for col in DERIVED_COLUMNS.keys():
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 2️⃣ Recreate derived columns safely
for col, meta in DERIVED_COLUMNS.items():
    required_cols = meta["requires"]
    if all(c in df.columns for c in required_cols):
        try:
            df[col] = meta["formula"](df)
        except Exception:
            pass

st.session_state.working_df = df

# =========================
# DATA PREVIEW
# =========================
st.subheader("📄 Data Preview")
st.dataframe(df.head(10))

# =========================
# MISSING VALUES BEFORE
# =========================
st.subheader("📉 Missing Values (Before Cleaning)")
missing_before = df.isnull().sum()
st.dataframe(missing_before.to_frame("Missing Count"))

# =========================
# FIX MISSING VALUES (UNCHANGED)
# =========================
st.subheader("🩹 Fix Missing Values")

for col in df.columns:
    if col in DERIVED_COLUMNS:
        continue

    miss = df[col].isnull().sum()
    if miss == 0:
        continue

    st.markdown(f"**{col} (missing: {miss})**")

    if pd.api.types.is_numeric_dtype(df[col]):
        opt = st.selectbox(
            f"Fix {col}",
            ["None", "Mean", "Median", "Mode", "Drop rows"],
            key=f"fix_{col}"
        )

        if opt == "Mean":
            push_undo()
            df[col] = df[col].fillna(df[col].mean())
        elif opt == "Median":
            push_undo()
            df[col] = df[col].fillna(df[col].median())
        elif opt == "Mode":
            push_undo()
            df[col] = df[col].fillna(df[col].mode()[0])
        elif opt == "Drop rows":
            push_undo()
            df = df.dropna(subset=[col])
    else:
        opt = st.selectbox(
            f"Fix {col}",
            ["None", "Mode", "Drop rows"],
            key=f"fix_{col}"
        )

        if opt == "Mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif opt == "Drop rows":
            df = df.dropna(subset=[col])

st.session_state.working_df = df

# =========================
# MISSING VALUES AFTER
# =========================
st.subheader("✅ Missing Values (After Cleaning)")
missing_after = df.isnull().sum()
st.dataframe(missing_after.to_frame("Missing Count"))

# =========================
# FILTER PANEL (UNCHANGED)
# =========================
st.sidebar.header("🎛 Filters")

filtered_df = df.copy()
filter_cols = st.sidebar.multiselect(
    "Select columns to filter",
    df.columns.tolist()
)

for col in filter_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        mn, mx = float(df[col].min()), float(df[col].max())
        r = st.sidebar.slider(col, mn, mx, (mn, mx))
        filtered_df = filtered_df[
            (filtered_df[col] >= r[0]) & (filtered_df[col] <= r[1])
        ]
    else:
        vals = st.sidebar.multiselect(col, df[col].dropna().unique())
        if vals:
            filtered_df = filtered_df[filtered_df[col].isin(vals)]

# =========================
# KPI BUILDER (FORM – FIXED)
# =========================
st.sidebar.header("📌 KPI Builder")

with st.sidebar.form("kpi_form", clear_on_submit=False):

    kpi_col = st.selectbox(
        "Select KPI column",
        filtered_df.columns
    )

    if pd.api.types.is_numeric_dtype(filtered_df[kpi_col]):
        kpi_opts = ["Sum", "Average", "Min", "Max", "Count"]
    else:
        kpi_opts = ["Count", "Unique Count", "Most Frequent"]

    kpi_metric = st.selectbox(
        "Select KPI",
        kpi_opts
    )

    add_kpi = st.form_submit_button("➕ Add KPI")
    clear_last = st.form_submit_button("↩ Clear Last KPI")
    clear_all = st.form_submit_button("❌ Clear All KPIs")


if add_kpi:
    st.session_state.kpis.append((kpi_col, kpi_metric))

if clear_last:
    if st.session_state.kpis:
        st.session_state.kpis.pop()

if clear_all:
    st.session_state.kpis = []


# =========================
# GROUP BY (ONLY ADDITION)
# =========================
st.sidebar.header("📊 Group By")

with st.sidebar.form("groupby_form", clear_on_submit=False):

    gb_col = st.selectbox(
        "Group by column",
        filtered_df.columns,
        key="gb_col"
    )

    val_col = st.selectbox(
        "Value column",
        filtered_df.columns,
        key="gb_val_col"
    )

    agg = st.selectbox(
        "Aggregation",
        ["Sum", "Average", "Min", "Max", "Count"],
        key="gb_agg"
    )

    run_groupby = st.form_submit_button("▶ Run Group By")
    clear_groupby = st.form_submit_button("❌ Clear Group By")



if run_groupby:
    if agg == "Sum":
        result = filtered_df.groupby(gb_col)[val_col].sum()
    elif agg == "Average":
        result = filtered_df.groupby(gb_col)[val_col].mean()
    elif agg == "Min":
        result = filtered_df.groupby(gb_col)[val_col].min()
    elif agg == "Max":
        result = filtered_df.groupby(gb_col)[val_col].max()
    else:
        result = filtered_df.groupby(gb_col)[val_col].count()

    st.session_state.groupby_result = result.reset_index()
    
if clear_groupby:
    st.session_state.groupby_result = None


# =========================
# KPI RESULTS
# =========================
st.subheader("📊 KPI Results")

if st.session_state.kpis:
    cols = st.columns(4)

    for i, (col, metric) in enumerate(st.session_state.kpis):
        c = cols[i % 4]

        if metric == "Sum":
            val = filtered_df[col].sum()
        elif metric == "Average":
            val = filtered_df[col].mean()
        elif metric == "Min":
            val = filtered_df[col].min()
        elif metric == "Max":
            val = filtered_df[col].max()
        elif metric == "Count":
            val = filtered_df[col].count()
        elif metric == "Unique Count":
            val = filtered_df[col].nunique()
        else:
            val = filtered_df[col].mode()[0]

        c.metric(f"{metric} of {col}", val)
else:
    st.info("Add KPIs from the sidebar")


# =========================
# GROUP BY RESULT (RIGHT SIDE)
# =========================
if st.session_state.groupby_result is not None:
    st.subheader("📊 Group By Result")
    st.dataframe(st.session_state.groupby_result)

    # 🔒 STOP auto rerun
    st.session_state.run_groupby = False
# =========================
# FINAL DATA
# =========================
st.subheader("✅ Cleaned & Filtered Data")
st.dataframe(filtered_df)

# =========================
# DOWNLOAD
# =========================
st.subheader("⬇ Download Cleaned Data")

download_type = st.selectbox(
    "Select file format",
    ["CSV", "Excel", "Text"]
)

if download_type == "CSV":
    st.download_button(
        "Download CSV",
        filtered_df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv"
    )
elif download_type == "Excel":
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False)
    st.download_button(
        "Download Excel",
        buffer.getvalue(),
        "cleaned_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.download_button(
        "Download Text",
        filtered_df.to_string(index=False),
        "cleaned_data.txt",
        "text/plain"
    )