from datetime import datetime
import re

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

REQUIRED_COLUMNS = {
    "QID",
    "model",
    "ground_truth",
    "prediction",
    "error_type",
}

# Exact Google Sheet ID from your URL:
# https://docs.google.com/spreadsheets/d/1e0MGEdJAXGe3TwPz_JTu5rGR65Af4Q4tX_bMAFQqhVY/edit
SPREADSHEET_ID = "1e0MGEdJAXGe3TwPz_JTu5rGR65Af4Q4tX_bMAFQqhVY"

# Your Google Sheet URL ends with gid=0, so we write to that exact tab.
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
    """Normalize reviewer/model keys for matching duplicate rows."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_qid(value) -> str:
    """Normalize QID for matching, e.g. 41878.0 -> 41878."""
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
    Normalize MCQ answers, preserving leading zeros and original order.
    Examples:
        3.0          -> "3"
        "0 1 2 4"    -> "0124"
        "[0,1,2]"    -> "012"
        NaN          -> ""
    """
    try:
        if pd.isna(answer):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(answer).strip()

    if text == "":
        return ""

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
    """
    Format a cleaned answer for display.
    Single digit "0" -> "0".
    Multi-digit "023" -> "0 2 3".
    """
    cleaned = clean_answer(answer)

    if cleaned == "":
        return "No answer"

    if re.fullmatch(r"\d+", cleaned):
        return " ".join(list(cleaned))

    return cleaned


def safe_text(value) -> str:
    """Return clean display text, guarding against NaN and non-strings."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def to_sheet_bool(value) -> str:
    """Write booleans consistently as TRUE/FALSE text."""
    return "TRUE" if bool(value) else "FALSE"


def parse_sheet_bool(value) -> bool:
    """Read booleans robustly from Google Sheets."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def empty_annotation_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def load_annotations(reviewer_id: str) -> pd.DataFrame:
    """Load all annotations for a given reviewer from Google Sheets."""
    sheet = get_sheet()
    records = sheet.get_all_records()

    if not records:
        return empty_annotation_table()

    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()

    if "reviewer_id" not in df.columns:
        return empty_annotation_table()

    return df[
        df["reviewer_id"].apply(normalize_key) == normalize_key(reviewer_id)
    ].copy()


def save_annotation(
    row,
    reviewer_id,
    safety_rating,
    outdated,
    consensus,
    comment,
):
    """
    Save or overwrite one annotation in Google Sheets.

    Unique key:
        reviewer_id + QID + model

    If duplicates already exist for this key, the first one is updated
    and the extra duplicate rows are deleted.
    """
    sheet = get_sheet()
    records = sheet.get_all_records()

    reviewer_key = normalize_key(reviewer_id)
    qid_key = normalize_qid(row["QID"])
    model_key = normalize_key(row["model"])

    matching_rows = []

    for idx, record in enumerate(records, start=2):
        if (
            normalize_key(record.get("reviewer_id", "")) == reviewer_key
            and normalize_qid(record.get("QID", "")) == qid_key
            and normalize_key(record.get("model", "")) == model_key
        ):
            matching_rows.append(idx)

    new_row = [
        reviewer_key,
        datetime.now().isoformat(timespec="seconds"),
        qid_key,
        model_key,
        clean_answer(row["ground_truth"]),
        clean_answer(row["prediction"]),
        normalize_key(row["error_type"]),
        safety_rating,
        to_sheet_bool(outdated),
        to_sheet_bool(consensus),
        normalize_key(comment),
    ]

    if matching_rows:
        first_row = matching_rows[0]
        cell_range = f"A{first_row}:K{first_row}"

        sheet.update(
            cell_range,
            [new_row],
            value_input_option="RAW",
        )

        # Delete duplicate rows for the same reviewer/QID/model.
        # Delete from bottom to top so row numbers remain valid.
        for duplicate_row in reversed(matching_rows[1:]):
            sheet.delete_rows(duplicate_row)

    else:
        sheet.append_row(
            new_row,
            value_input_option="RAW",
        )


def save_all_current_question(current_df: pd.DataFrame, reviewer_id: str) -> int:
    """
    Save all model annotations for the currently selected QID.

    Example:
        If the selected QID has LLM-1, LLM-2, LLM-3, LLM-4,
        this saves all four rows.
    """
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
    """Retrieve the most recent annotation for a reviewer/QID/model."""
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


def get_active_reviewers() -> list:
    """Return sorted list of all reviewers who have submitted annotations."""
    sheet = get_sheet()
    all_records = sheet.get_all_records()

    return sorted(
        set(
            normalize_key(r.get("reviewer_id", ""))
            for r in all_records
            if normalize_key(r.get("reviewer_id", ""))
        )
    )


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
        st.write(f"Rows in worksheet: {len(sheet.get_all_values())}")
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

df = pd.read_csv(uploaded_file, dtype=str)
df = normalize_columns(df)
validate_data(df)

df["QID"] = df["QID"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()
df["ground_truth"] = df["ground_truth"].apply(clean_answer)
df["prediction"] = df["prediction"].apply(clean_answer)
df["error_type"] = df["error_type"].astype(str).str.strip()

annotations = load_annotations(reviewer_id)


# ============================================================
# DOWNLOAD REVIEWER ANNOTATIONS
# ============================================================

csv_export = annotations.to_csv(index=False)

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
