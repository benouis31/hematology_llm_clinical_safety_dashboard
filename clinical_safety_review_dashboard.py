from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import streamlit as st
import gspread

from google.oauth2.service_account import Credentials
import json


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
    "Annotations are saved permanently after clicking 'Save annotation'."
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
    if pd.isna(answer):
        return ""

    text = str(answer).strip()

    if text == "":
        return ""

    try:
        value = float(text)

        if value.is_integer():
            text = str(int(value))

    except Exception:
        pass

    digits = re.findall(r"\d", text)

    if not digits:
        return text

    return "".join(digits)


def format_answer(answer) -> str:
    cleaned = clean_answer(answer)

    if cleaned == "":
        return "No answer"

    if cleaned.isdigit():
        return " ".join(list(cleaned))

    return cleaned


def safe_text(value) -> str:
    if pd.isna(value):
        return ""

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
        reviewer_id,
        datetime.now().isoformat(timespec="seconds"),
        str(row["QID"]),
        str(row["model"]),
        clean_answer(row["ground_truth"]),
        clean_answer(row["prediction"]),
        str(row["error_type"]),
        safety_rating,
        str(bool(outdated)),
        str(bool(consensus)),
        comment,
    ]

    if row_exists:

        cell_range = (
            f"A{existing_row_number}:K{existing_row_number}"
        )

        sheet.update(
            cell_range,
            [new_row],
        )

    else:

        sheet.append_row(
            new_row,
            value_input_option="USER_ENTERED",
        )


def get_existing_annotation(
    annotations,
    reviewer_id,
    qid,
    model,
):
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


# ============================================================
# SIDEBAR — SETUP
# ============================================================

st.sidebar.header("Review Setup")

reviewer_id = st.sidebar.text_input(
    "Reviewer ID",
    value="reviewer_1",
    help="Use a unique blinded reviewer ID.",
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

df = pd.read_csv(uploaded_file)

df = normalize_columns(df)

validate_data(df)

df["QID"] = df["QID"].astype(str)

df["model"] = df["model"].astype(str)

df["ground_truth"] = df["ground_truth"].apply(
    clean_answer
)

df["prediction"] = df["prediction"].apply(
    clean_answer
)

df["error_type"] = df["error_type"].astype(str)

annotations = load_annotations(reviewer_id)


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

all_annotations = get_sheet().get_all_records()

active_reviewers = sorted(
    list(
        set(
            str(r.get("reviewer_id", ""))
            for r in all_annotations
            if str(r.get("reviewer_id", "")).strip()
        )
    )
)

st.sidebar.divider()

st.sidebar.subheader("Reviewer activity")

st.sidebar.write(
    f"Active reviewers: {len(active_reviewers)}"
)

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

current_df = df[
    df["QID"] == selected_qid
].copy()

current_pairs = set(
    zip(
        current_df["QID"].astype(str),
        current_df["model"].astype(str),
    )
)

st.sidebar.write(
    f"Current question progress: "
    f"{len(current_pairs & reviewed_pairs)} "
    f"/ {len(current_pairs)}"
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

    st.code(
        format_answer(first["ground_truth"])
    )

    clinical_question = safe_text(
        first.get("clinical_question", "")
    )

    options_sorted = safe_text(
        first.get("options_sorted", "")
    )

    vignette = safe_text(
        first.get("vignette", "")
    )

    full_prompt = safe_text(
        first.get("full_prompt", "")
    )

    if clinical_question:

        st.write("**Question:**")

        st.write(clinical_question)

    if options_sorted:

        st.write("**Answer options:**")

        st.text(options_sorted)

    if vignette:

        with st.expander(
            "Show clinical vignette"
        ):
            st.write(vignette)

    if full_prompt:

        with st.expander(
            "Show original prompt"
        ):
            st.write(full_prompt)


# ============================================================
# RIGHT PANEL — SAFETY REVIEW
# ============================================================

with right:

    st.markdown(
        "### Model Output Safety Assessment"
    )

    for _, row in current_df.sort_values(
        "model"
    ).iterrows():

        existing = get_existing_annotation(
            annotations,
            reviewer_id,
            row["QID"],
            row["model"],
        )

        default_rating = (
            existing["safety_rating"]
            if existing is not None
            and existing.get(
                "safety_rating"
            ) in SAFETY_LABELS
            else SAFETY_LABELS[0]
        )

        default_outdated = (
            str(
                existing.get(
                    "ground_truth_outdated",
                    "False",
                )
            ).lower()
            == "true"
            if existing is not None
            else False
        )

        default_consensus = (
            str(
                existing.get(
                    "needs_consensus_review",
                    "False",
                )
            ).lower()
            == "true"
            if existing is not None
            else False
        )

        default_comment = (
            existing.get(
                "clinician_comment",
                "",
            )
            if existing is not None
            else ""
        )

        with st.expander(
            f"{row['model']} | "
            f"Prediction: "
            f"{format_answer(row['prediction'])}",
            expanded=True,
        ):

            st.write("**Model prediction:**")

            st.code(
                format_answer(
                    row["prediction"]
                )
            )

            st.write("**Ground truth:**")

            st.code(
                format_answer(
                    row["ground_truth"]
                )
            )

            st.write(
                "**Automatic failure type:**"
            )

            st.write(row["error_type"])

            safety_rating = st.selectbox(
                "Safety rating",
                SAFETY_LABELS,
                index=SAFETY_LABELS.index(
                    default_rating
                ),
                key=(
                    f"safety_"
                    f"{row['QID']}_"
                    f"{row['model']}"
                ),
            )

            outdated = st.checkbox(
                (
                    "Ground truth may be "
                    "outdated / "
                    "model partially correct"
                ),
                value=default_outdated,
                key=(
                    f"outdated_"
                    f"{row['QID']}_"
                    f"{row['model']}"
                ),
            )

            consensus = st.checkbox(
                "Needs consensus review",
                value=default_consensus,
                key=(
                    f"consensus_"
                    f"{row['QID']}_"
                    f"{row['model']}"
                ),
            )

            comment = st.text_area(
                "Clinician comment",
                value=default_comment,
                key=(
                    f"comment_"
                    f"{row['QID']}_"
                    f"{row['model']}"
                ),
                placeholder=(
                    "Explain potential "
                    "patient harm, "
                    "omitted action, "
                    "harmful implementation, "
                    "or outdated "
                    "ground truth."
                ),
            )

            if st.button(
                "Save annotation",
                key=(
                    f"save_"
                    f"{row['QID']}_"
                    f"{row['model']}"
                ),
            ):

                save_annotation(
                    row=row,
                    reviewer_id=reviewer_id,
                    safety_rating=safety_rating,
                    outdated=outdated,
                    consensus=consensus,
                    comment=comment,
                )

                st.success(
                    (
                        "Annotation saved "
                        "permanently."
                    )
                )
