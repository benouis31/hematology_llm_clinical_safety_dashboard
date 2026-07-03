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

SHEET_NAME = "Clinical_Safety_Annotations"
WORKSHEET_NAME = "annotations"


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Clinical Safety Review Dashboard",
    layout="wide",
)

st.title("Clinical Safety Review Dashboard")
st.caption("Blinded clinician review of LLM failures in hematology MCQs")

st.success(
    "Annotations are saved automatically after each change. "
    "You can also click 'Save annotation' manually."
)


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

    return client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)


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


def clean_answer(answer) -> str:
    """
    Normalize MCQ answers, preserving leading zeros and original order.
    Examples:
        3.0         -> "3"
        "0 1 2 4"   -> "0124"
        "[0,1,2]"   -> "012"
        NaN         -> ""
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
    Handles "0", empty strings, and NaN safely.
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
        df["reviewer_id"].astype(str) == str(reviewer_id)
    ].copy()


def save_annotation(
    row,
    reviewer_id,
    safety_rating,
    outdated,
    consensus,
    comment,
):
    """Save or overwrite an annotation in Google Sheets."""
    sheet = get_sheet()
    records = sheet.get_all_records()

    row_exists = False
    existing_row_number = None

    for idx, record in enumerate(records, start=2):
        if (
            str(record.get("reviewer_id", "")) == str(reviewer_id)
            and str(record.get("QID", "")) == str(row["QID"])
            and str(record.get("model", "")) == str(row["model"])
        ):
            row_exists = True
            existing_row_number = idx
            break

    new_row = [
        str(reviewer_id),
        datetime.now().isoformat(timespec="seconds"),
        str(row["QID"]),
        str(row["model"]),
        clean_answer(row["ground_truth"]),
        clean_answer(row["prediction"]),
        str(row["error_type"]),
        safety_rating,
        str(bool(outdated)),
        str(bool(consensus)),
        str(comment),
    ]

    if row_exists:
        cell_range = f"A{existing_row_number}:K{existing_row_number}"
        sheet.update(cell_range, [new_row])
    else:
        sheet.append_row(new_row, value_input_option="USER_ENTERED")


def autosave_annotation(
    row,
    reviewer_id,
    safety_key,
    outdated_key,
    consensus_key,
    comment_key,
    saved_key,
):
    """
    Autosave annotation whenever the reviewer changes an input.
    This uses the existing Google Sheets save_annotation() function.
    """
    safety_rating = st.session_state.get(safety_key, SAFETY_LABELS[0])
    outdated = st.session_state.get(outdated_key, False)
    consensus = st.session_state.get(consensus_key, False)
    comment = st.session_state.get(comment_key, "")

    save_annotation(
        row=row,
        reviewer_id=reviewer_id,
        safety_rating=safety_rating,
        outdated=outdated,
        consensus=consensus,
        comment=comment,
    )

    st.session_state[saved_key] = datetime.now().strftime("%H:%M:%S")


def get_existing_annotation(annotations, reviewer_id, qid, model):
    """Retrieve the most recent annotation for a reviewer/QID/model."""
    if annotations.empty:
        return None

    subset = annotations[
        (annotations["reviewer_id"].astype(str) == str(reviewer_id))
        & (annotations["QID"].astype(str) == str(qid))
        & (annotations["model"].astype(str) == str(model))
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
            str(r.get("reviewer_id", ""))
            for r in all_records
            if str(r.get("reviewer_id", "")).strip()
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

uploaded_file = st.sidebar.file_uploader(
    "Upload review CSV",
    type=["csv"],
    help="Upload bsh_failure_review_table.csv",
)

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

df["QID"] = df["QID"].astype(str)
df["model"] = df["model"].astype(str)
df["ground_truth"] = df["ground_truth"].apply(clean_answer)
df["prediction"] = df["prediction"].apply(clean_answer)
df["error_type"] = df["error_type"].astype(str)

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
        annotations["QID"].astype(str),
        annotations["model"].astype(str),
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
        current_df["QID"].astype(str),
        current_df["model"].astype(str),
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
            str(existing.get("ground_truth_outdated", "False")).lower() == "true"
            if existing is not None
            else False
        )

        default_consensus = (
            str(existing.get("needs_consensus_review", "False")).lower() == "true"
            if existing is not None
            else False
        )

        default_comment = (
            str(existing.get("clinician_comment", ""))
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
            saved_key = f"saved_{row['QID']}_{row['model']}"

            autosave_args = (
                row_data,
                reviewer_id,
                safety_key,
                outdated_key,
                consensus_key,
                comment_key,
                saved_key,
            )

            safety_rating = st.selectbox(
                "Safety rating",
                SAFETY_LABELS,
                index=SAFETY_LABELS.index(default_rating),
                key=safety_key,
                on_change=autosave_annotation,
                args=autosave_args,
            )

            outdated = st.checkbox(
                "Ground truth may be outdated / model partially correct",
                value=default_outdated,
                key=outdated_key,
                on_change=autosave_annotation,
                args=autosave_args,
            )

            consensus = st.checkbox(
                "Needs consensus review",
                value=default_consensus,
                key=consensus_key,
                on_change=autosave_annotation,
                args=autosave_args,
            )

            comment = st.text_area(
                "Clinician comment",
                value=default_comment,
                key=comment_key,
                placeholder=(
                    "Explain potential patient harm, omitted action, "
                    "harmful implementation, or outdated ground truth."
                ),
                on_change=autosave_annotation,
                args=autosave_args,
            )

            if saved_key in st.session_state:
                st.caption(f"Autosaved at {st.session_state[saved_key]}")

            if st.button(
                "Save annotation",
                key=f"save_{row['QID']}_{row['model']}",
            ):
                save_annotation(
                    row=row,
                    reviewer_id=reviewer_id,
                    safety_rating=safety_rating,
                    outdated=outdated,
                    consensus=consensus,
                    comment=comment,
                )
                st.success("Annotation saved permanently.")
