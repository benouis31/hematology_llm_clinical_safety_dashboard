from datetime import datetime
import re
import csv

import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# ============================================================
# CONFIGURATION
# ============================================================

SAFETY_LABELS = [
    "Insignificant",
    "Unsafe, minor harm",
    "Unsafe, major harm if omitted",
    "Unsafe, major harm if implemented",
    "Output correct / partially correct",
]

ANNOTATION_COLUMNS = [
    "reviewer_id",
    "timestamp",
    "QID",
    "model",
    "ground_truth",
    "prediction",
    "error_type",
    "safety_rating",
    "ground_truth_outdated",
    "needs_consensus_review",
    "clinician_comment",
]

REQUIRED_COLUMNS = {
    "QID",
    "model",
    "ground_truth",
    "prediction",
    "error_type",
}

SPREADSHEET_ID = "1e0MGEdJAXGe3TwPz_JTu5rGR65Af4Q4tX_bMAFQqhVY"
WORKSHEET_GID = 0


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Clinical Safety Review Dashboard",
    layout="wide",
)

st.title("Clinical Safety Review Dashboard")
st.caption("Blinded clinician review of LLM failures in hematology MCQs")

st.info(
    "Annotations are not autosaved. After reviewing a question, click "
    "'Save all annotations for this question'."
)

if "last_save_message" in st.session_state:
    st.success(st.session_state.pop("last_save_message"))


# ============================================================
# GOOGLE SHEETS CONNECTION
# ============================================================

@st.cache_resource
def get_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    service_account_info = dict(st.secrets["gcp_service_account"])
    service_account_info["private_key"] = service_account_info[
        "private_key"
    ].replace("\\n", "\n")

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes,
    )

    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    try:
        worksheet = spreadsheet.get_worksheet_by_id(WORKSHEET_GID)
    except Exception:
        worksheet = spreadsheet.get_worksheet(0)

    if worksheet is None:
        raise ValueError(
            f"Could not find worksheet with gid={WORKSHEET_GID}."
        )

    return worksheet


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Question_ID": "QID",
        "question_id": "QID",
        "qid": "QID",
        "Model": "model",
        "Ground_Truth": "ground_truth",
        "groundtruth": "ground_truth",
        "Prediction": "prediction",
        "Failure_Type": "error_type",
        "failure_type": "error_type",
        "Error_Type": "error_type",
        "Clinical_Question": "clinical_question",
        "Question": "clinical_question",
        "Options": "options_sorted",
        "Options_Sorted": "options_sorted",
        "Vignette": "vignette",
        "Full_Prompt": "full_prompt",
    }

    return df.rename(columns=rename_map)


def validate_data(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        st.error(
            "Missing required columns:\n\n"
            f"{', '.join(sorted(missing))}"
        )
        st.stop()


def normalize_key(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_qid(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value).strip()

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def clean_answer(answer) -> str:
    """
    Normalize MCQ answers while preserving leading zeros and original order.

    Examples:
        3.0          -> "3"
        "3.0"        -> "3"
        "0 1 2 4"    -> "0124"
        "[0,1,2]"    -> "012"
        "01234"      -> "01234"
        NaN          -> ""
    """
    try:
        if pd.isna(answer):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(answer, float) and answer.is_integer():
        return str(int(answer))

    text = str(answer).strip()

    if text == "":
        return ""

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    digits = re.findall(r"\d", text)

    if not digits:
        return text

    seen = set()
    ordered_digits = []

    for d in digits:
        if d not in seen:
            ordered_digits.append(d)
            seen.add(d)

    return "".join(ordered_digits)


def format_answer(answer) -> str:
    cleaned = clean_answer(answer)

    if cleaned == "":
        return "No answer"

    if re.fullmatch(r"\d+", cleaned):
        return " ".join(list(cleaned))

    return cleaned


def safe_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def to_sheet_bool(value) -> str:
    return "TRUE" if bool(value) else "FALSE"


def parse_sheet_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def empty_annotation_table() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def is_quota_error(error) -> bool:
    """
    Detect Google Sheets API quota / rate-limit errors.
    """
    text = str(error).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def show_sheet_read_warning_once(error):
    """
    Show a non-blocking warning only once per Streamlit session.
    This prevents the whole app from crashing when Google temporarily
    blocks read requests.
    """
    if not st.session_state.get("sheet_read_warning_shown", False):
        st.warning(
            "Google Sheets read quota was temporarily exceeded. "
            "Existing annotations/progress may not refresh immediately, "
            "but saving can still work. Please wait a few minutes if this persists."
        )
        st.session_state["sheet_read_warning_shown"] = True


def latest_annotation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest annotation for each reviewer_id + QID + model.

    The app now saves in append-only mode to avoid expensive read-before-write
    operations. Historical duplicate rows are kept in Google Sheets as an audit
    trail, but only the latest row is used in the dashboard and downloads.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_row_order"] = range(len(df))
    df["_timestamp_dt"] = pd.to_datetime(
        df.get("timestamp", ""),
        errors="coerce",
    )

    df = df.sort_values(["_timestamp_dt", "_row_order"], na_position="first")

    df = df.drop_duplicates(
        subset=["reviewer_id", "QID", "model"],
        keep="last",
    )

    df = df.drop(columns=["_row_order", "_timestamp_dt"], errors="ignore")
    return df.reset_index(drop=True)


# ============================================================
# GOOGLE SHEET SAFE READ FUNCTIONS
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_annotation_values(_sheet):
    """
    Read only annotation columns A:K.

    Cached for 60 seconds to reduce Google Sheets API quota errors.
    The leading underscore in _sheet tells Streamlit not to hash
    the gspread worksheet object.

    Important:
    Do NOT use get_all_records().
    Do NOT use get_all_values() on the full worksheet.
    """
    values = _sheet.get(
        "A:K",
        value_render_option="FORMATTED_VALUE",
    )

    return values or []


def ensure_annotation_header(sheet):
    """
    No repeated header check.

    The annotation sheet already has the correct header. Re-reading A1:K1
    on every rerun can trigger Google Sheets API quota errors, especially
    when several reviewers use the app at the same time.
    """
    return


def sheet_values_to_dataframe(sheet) -> pd.DataFrame:
    """
    Read Google Sheet values as text from A:K only.

    This preserves leading zeros and avoids automatic conversion:
        01234 -> 1234
    """
    ensure_annotation_header(sheet)
    values = get_annotation_values(sheet)

    if not values or len(values) < 2:
        return empty_annotation_table()

    headers = [str(h).strip() for h in values[0]]
    rows = values[1:]

    cleaned_rows = []

    for row in rows:
        if not any(str(cell).strip() for cell in row):
            continue

        padded = row + [""] * max(0, len(headers) - len(row))
        cleaned_rows.append(padded[:len(headers)])

    if not cleaned_rows:
        return empty_annotation_table()

    df = pd.DataFrame(cleaned_rows, columns=headers)
    df.columns = df.columns.str.strip()

    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in ["reviewer_id", "QID", "model", "ground_truth", "prediction"]:
        df[col] = df[col].astype(str).str.strip()

    return df[ANNOTATION_COLUMNS].copy()


def sheet_records_with_row_numbers(sheet) -> list:
    """
    Return records as text, including the real Google Sheet row number.

    Used for update/delete while preserving leading zeros.
    """
    ensure_annotation_header(sheet)
    values = get_annotation_values(sheet)

    if not values or len(values) < 2:
        return []

    headers = [str(h).strip() for h in values[0]]
    records = []

    for sheet_row_number, row in enumerate(values[1:], start=2):
        if not any(str(cell).strip() for cell in row):
            continue

        padded = row + [""] * max(0, len(headers) - len(row))

        record = {
            headers[i]: str(padded[i]).strip()
            for i in range(len(headers))
        }

        record["_sheet_row_number"] = sheet_row_number
        records.append(record)

    return records


def load_annotations(reviewer_id: str) -> pd.DataFrame:
    sheet = get_sheet()

    try:
        df = sheet_values_to_dataframe(sheet)
    except Exception as e:
        if is_quota_error(e):
            show_sheet_read_warning_once(e)
            return empty_annotation_table()
        raise

    if df.empty or "reviewer_id" not in df.columns:
        return empty_annotation_table()

    reviewer_df = df[
        df["reviewer_id"].apply(normalize_key) == normalize_key(reviewer_id)
    ].copy()

    return latest_annotation_rows(reviewer_df)


def get_active_reviewers() -> list:
    sheet = get_sheet()

    try:
        df = sheet_values_to_dataframe(sheet)
    except Exception as e:
        if is_quota_error(e):
            show_sheet_read_warning_once(e)
            return []
        raise

    if df.empty or "reviewer_id" not in df.columns:
        return []

    return sorted(
        set(
            normalize_key(x)
            for x in df["reviewer_id"]
            if normalize_key(x)
        )
    )



# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_annotation(
    row,
    reviewer_id,
    safety_rating,
    outdated,
    consensus,
    comment,
):
    """
    Save one annotation in append-only mode.

    This is the best option for multi-reviewer Streamlit use because it avoids
    read-before-write operations, which were causing Google Sheets API 429
    quota errors. If a reviewer saves the same QID/model more than once, the
    latest timestamped row is used by the dashboard via latest_annotation_rows().
    """
    sheet = get_sheet()

    new_row = [
        normalize_key(reviewer_id),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        normalize_qid(row["QID"]),
        normalize_key(row["model"]),
        clean_answer(row["ground_truth"]),
        clean_answer(row["prediction"]),
        normalize_key(row["error_type"]),
        safety_rating,
        to_sheet_bool(outdated),
        to_sheet_bool(consensus),
        normalize_key(comment),
    ]

    sheet.append_row(
        new_row,
        value_input_option="RAW",
    )


def save_all_current_question(current_df: pd.DataFrame, reviewer_id: str) -> int:
    saved_count = 0

    for _, row in current_df.sort_values("model").iterrows():
        row_data = row.to_dict()

        safety_key = f"safety_{row['QID']}_{row['model']}"
        outdated_key = f"outdated_{row['QID']}_{row['model']}"
        consensus_key = f"consensus_{row['QID']}_{row['model']}"
        comment_key = f"comment_{row['QID']}_{row['model']}"

        safety_rating = st.session_state.get(safety_key, SAFETY_LABELS[0])
        outdated = st.session_state.get(outdated_key, False)
        consensus = st.session_state.get(consensus_key, False)
        comment = st.session_state.get(comment_key, "")

        save_annotation(
            row=row_data,
            reviewer_id=reviewer_id,
            safety_rating=safety_rating,
            outdated=outdated,
            consensus=consensus,
            comment=comment,
        )

        saved_count += 1

    return saved_count


def get_existing_annotation(annotations, reviewer_id, qid, model):
    if annotations.empty:
        return None

    subset = annotations[
        (annotations["reviewer_id"].apply(normalize_key) == normalize_key(reviewer_id))
        & (annotations["QID"].apply(normalize_qid) == normalize_qid(qid))
        & (annotations["model"].apply(normalize_key) == normalize_key(model))
    ]

    if subset.empty:
        return None

    return subset.iloc[-1].to_dict()


# ============================================================
# SIDEBAR — SETUP
# ============================================================

st.sidebar.header("Review Setup")

reviewer_id = st.sidebar.text_input(
    "Reviewer ID",
    value="reviewer_1",
    help="Use a unique blinded reviewer ID, e.g. reviewer_freya.",
)

reviewer_id = reviewer_id.strip()

if reviewer_id == "":
    st.sidebar.error("Please enter a reviewer ID before starting.")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Upload review CSV",
    type=["csv"],
    help="Upload bsh_failure_review_table.csv",
)


# ============================================================
# GOOGLE SHEET CONNECTION CHECK
# ============================================================

with st.sidebar.expander("Google Sheet connection", expanded=False):
    try:
        sheet = get_sheet()

        st.success("Connected")
        st.write(f"Worksheet title: {sheet.title}")
        st.write(f"Worksheet ID / gid: {sheet.id}")
        st.caption(
            "Sheet reads are cached for 60 seconds to avoid Google API quota errors."
        )

    except Exception as e:
        st.error("Google Sheet connection failed.")
        st.code(str(e))
        st.stop()


if uploaded_file is None:
    st.info(
        "Upload `bsh_failure_review_table.csv` "
        "to begin the clinical safety review."
    )
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    uploaded_file,
    dtype=str,
    keep_default_na=False,
)

df = normalize_columns(df)
validate_data(df)

df["QID"] = df["QID"].apply(normalize_qid)
df["model"] = df["model"].astype(str).str.strip()
df["ground_truth"] = df["ground_truth"].apply(clean_answer)
df["prediction"] = df["prediction"].apply(clean_answer)
df["error_type"] = df["error_type"].astype(str).str.strip()

annotations = load_annotations(reviewer_id)


# ============================================================
# DOWNLOAD REVIEWER ANNOTATIONS
# ============================================================

csv_export = annotations.to_csv(
    index=False,
    quoting=csv.QUOTE_ALL,
)

st.sidebar.download_button(
    label="Download my annotations",
    data=csv_export,
    file_name=f"{reviewer_id}_annotations.csv",
    mime="text/csv",
)


# ============================================================
# REVIEW PROGRESS
# ============================================================

reviewed_pairs = set(
    zip(
        annotations["QID"].apply(normalize_qid),
        annotations["model"].apply(normalize_key),
    )
)

total_pairs = len(df)
reviewed_count = len(reviewed_pairs)

st.sidebar.metric(
    "Reviewed outputs",
    f"{reviewed_count} / {total_pairs}",
)

st.sidebar.caption(
    "Progress and reviewer activity may refresh with up to 60 seconds delay."
)


# ============================================================
# REVIEWER ACTIVITY
# ============================================================

active_reviewers = get_active_reviewers()

st.sidebar.divider()
st.sidebar.subheader("Reviewer activity")
st.sidebar.write(f"Active reviewers: {len(active_reviewers)}")

if active_reviewers:
    for reviewer in active_reviewers:
        st.sidebar.write(f"• {reviewer}")
else:
    st.sidebar.info("No reviewer activity yet.")


# ============================================================
# QUESTION SELECTION
# ============================================================

question_ids = sorted(
    df["QID"].unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x),
)

selected_qid = st.sidebar.selectbox(
    "Question ID",
    question_ids,
)

current_df = df[df["QID"] == selected_qid].copy()

current_pairs = set(
    zip(
        current_df["QID"].apply(normalize_qid),
        current_df["model"].apply(normalize_key),
    )
)

st.sidebar.write(
    f"Current question progress: "
    f"{len(current_pairs & reviewed_pairs)} / {len(current_pairs)}"
)


# ============================================================
# MAIN LAYOUT
# ============================================================

st.subheader(f"Question ID: {selected_qid}")

left, right = st.columns([1.1, 1.4])


# ============================================================
# LEFT PANEL — QUESTION
# ============================================================

with left:
    st.markdown("### Clinical Question")

    first = current_df.iloc[0]

    st.write("**Ground truth answer:**")
    st.code(format_answer(first["ground_truth"]))

    clinical_question = safe_text(first.get("clinical_question", ""))
    options_sorted = safe_text(first.get("options_sorted", ""))
    vignette = safe_text(first.get("vignette", ""))
    full_prompt = safe_text(first.get("full_prompt", ""))

    if clinical_question:
        st.write("**Question:**")
        st.write(clinical_question)
    else:
        st.info("No separated clinical question available in this CSV.")

    if options_sorted:
        st.write("**Answer options:**")
        st.text(options_sorted)
    else:
        st.info("No sorted answer options available in this CSV.")

    if vignette:
        with st.expander("Show clinical vignette"):
            st.write(vignette)

    if full_prompt:
        with st.expander("Show original prompt"):
            st.write(full_prompt)


# ============================================================
# RIGHT PANEL — SAFETY REVIEW
# ============================================================

with right:
    st.markdown("### Model Output Safety Assessment")

    for _, row in current_df.sort_values("model").iterrows():
        existing = get_existing_annotation(
            annotations,
            reviewer_id,
            row["QID"],
            row["model"],
        )

        default_rating = (
            existing["safety_rating"]
            if existing is not None
            and existing.get("safety_rating") in SAFETY_LABELS
            else SAFETY_LABELS[0]
        )

        default_outdated = (
            parse_sheet_bool(existing.get("ground_truth_outdated", "FALSE"))
            if existing is not None
            else False
        )

        default_consensus = (
            parse_sheet_bool(existing.get("needs_consensus_review", "FALSE"))
            if existing is not None
            else False
        )

        default_comment = (
            safe_text(existing.get("clinician_comment", ""))
            if existing is not None
            else ""
        )

        with st.expander(
            f"{row['model']} | Prediction: {format_answer(row['prediction'])}",
            expanded=True,
        ):
            st.write("**Model prediction:**")
            st.code(format_answer(row["prediction"]))

            st.write("**Ground truth:**")
            st.code(format_answer(row["ground_truth"]))

            st.write("**Automatic failure type:**")
            st.write(row["error_type"])

            row_data = row.to_dict()

            safety_key = f"safety_{row['QID']}_{row['model']}"
            outdated_key = f"outdated_{row['QID']}_{row['model']}"
            consensus_key = f"consensus_{row['QID']}_{row['model']}"
            comment_key = f"comment_{row['QID']}_{row['model']}"

            safety_rating = st.selectbox(
                "Safety rating",
                SAFETY_LABELS,
                index=SAFETY_LABELS.index(default_rating),
                key=safety_key,
            )

            outdated = st.checkbox(
                "Ground truth may be outdated / model partially correct",
                value=default_outdated,
                key=outdated_key,
            )

            consensus = st.checkbox(
                "Needs consensus review",
                value=default_consensus,
                key=consensus_key,
            )

            comment = st.text_area(
                "Clinician comment",
                value=default_comment,
                key=comment_key,
                placeholder=(
                    "Explain potential patient harm, omitted action, "
                    "harmful implementation, or outdated ground truth."
                ),
            )

            if st.button(
                "Save this model annotation",
                key=f"save_{row['QID']}_{row['model']}",
            ):
                save_annotation(
                    row=row_data,
                    reviewer_id=reviewer_id,
                    safety_rating=safety_rating,
                    outdated=outdated,
                    consensus=consensus,
                    comment=comment,
                )

                st.session_state["last_save_message"] = (
                    f"Saved annotation for reviewer={reviewer_id}, "
                    f"QID={row['QID']}, model={row['model']}."
                )
                st.rerun()

    st.divider()

    if st.button(
        "Save all annotations for this question",
        type="primary",
        key=f"save_all_{selected_qid}",
    ):
        saved_count = save_all_current_question(current_df, reviewer_id)

        st.session_state["last_save_message"] = (
            f"Saved {saved_count} annotation rows for reviewer={reviewer_id}, "
            f"Question ID={selected_qid}."
        )
        st.rerun()
