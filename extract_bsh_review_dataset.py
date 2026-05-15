# ============================================================
# EXTRACT BSH FAILURE REVIEW DATASET FROM DOCX
# ============================================================

import re
import argparse
from pathlib import Path

import pandas as pd
from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph


DEFAULT_MODELS = ["LLM-1", "LLM-2", "LLM-3", "LLM-4"]


def normalize_answer(x):
    if x is None:
        return ""
    digits = re.findall(r"\d", str(x))
    return "".join(sorted(digits))


def iter_block_items(parent):
    if isinstance(parent, DocumentObject):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element

    for child in parent_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def classify_error(gt, pred):
    gt_set = set(str(gt))
    pred_set = set(str(pred))

    if pred_set == gt_set:
        return "Correct"
    if len(pred_set) == 0:
        return "No answer"
    if pred_set.issubset(gt_set):
        return "Under-selection"
    if gt_set.issubset(pred_set):
        return "Over-selection"
    if len(gt_set & pred_set) > 0:
        return "Mixed partial error"
    return "Completely wrong"


def extract_options(prompt_text):
    """
    Extract and sort MCQ options by option number.
    Returns text formatted as:
    0. ...
    1. ...
    2. ...
    """
    if "Options:" not in prompt_text:
        return ""

    options_text = prompt_text.split("Options:", 1)[1].strip()

    pattern = r"(?m)^\s*(\d)\.\s*(.*?)(?=^\s*\d\.|\Z)"
    matches = re.findall(pattern, options_text, flags=re.DOTALL)

    options = {}
    for number, text in matches:
        clean_text = " ".join(text.split())
        options[int(number)] = clean_text

    sorted_lines = []
    for idx in sorted(options):
        sorted_lines.append(f"{idx}. {options[idx]}")

    return "\n".join(sorted_lines)


def split_vignette_and_question(prompt_text):
    """
    Separates the long clinical vignette from the actual final MCQ question.
    """
    if "Options:" in prompt_text:
        before_options = prompt_text.split("Options:", 1)[0].strip()
    else:
        before_options = prompt_text.strip()

    # Remove instruction prefix
    marker = "Multiple-choice questions may have more than one answer."
    if marker in before_options:
        before_options = before_options.split(marker, 1)[1].strip()

    lines = [line.strip() for line in before_options.splitlines() if line.strip()]

    if not lines:
        return "", ""

    question_start_idx = None

    question_mark_keywords = [
        "which of the following",
        "complete the following",
        "which are",
        "what does",
        "what kind",
        "according to",
        "therapeutic options",
        "known complications",
        "clinical features",
    ]

    for i in range(len(lines) - 1, -1, -1):
        lower = lines[i].lower()
        if "?" in lines[i] or any(k in lower for k in question_mark_keywords):
            question_start_idx = i
            break

    if question_start_idx is None:
        if len(lines) >= 2:
            vignette = "\n".join(lines[:-1])
            question = lines[-1]
        else:
            vignette = ""
            question = lines[0]
    else:
        vignette = "\n".join(lines[:question_start_idx])
        question = "\n".join(lines[question_start_idx:])

    return vignette.strip(), question.strip()


def extract_prompt_text(paragraphs, start_idx):
    """
    Extract prompt paragraph following the word 'Prompt:'.
    """
    if start_idx + 1 >= len(paragraphs):
        return ""

    return paragraphs[start_idx + 1].text.strip()


def table_to_predictions(table, model_names):
    predictions = []

    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]

        if len(cells) < 2:
            continue

        model = cells[0]
        pred = normalize_answer(cells[1])

        if model in model_names:
            predictions.append((model, pred))

    return predictions


def parse_docx(docx_path, model_names):
    doc = Document(docx_path)

    blocks = list(iter_block_items(doc))
    rows = []

    current_qid = None
    current_gt = None
    current_predictions = None

    for idx, block in enumerate(blocks):

        if isinstance(block, Paragraph):
            text = block.text.strip()

            if text.startswith("QID:"):
                current_qid = text.replace("QID:", "").strip()

            elif text.startswith("Ground Truth:"):
                current_gt = normalize_answer(text.replace("Ground Truth:", "").strip())

            elif text == "Prompt:":
                prompt_text = ""

                # Find next non-empty paragraph as prompt
                for j in range(idx + 1, len(blocks)):
                    if isinstance(blocks[j], Paragraph):
                        candidate = blocks[j].text.strip()
                        if candidate:
                            prompt_text = candidate
                            break

                vignette, clinical_question = split_vignette_and_question(prompt_text)
                options_sorted = extract_options(prompt_text)

                if current_qid and current_gt and current_predictions:
                    for model, pred in current_predictions:
                        failed = int(pred != current_gt)
                        error_type = classify_error(current_gt, pred)

                        rows.append({
                            "QID": current_qid,
                            "model": model,
                            "ground_truth": current_gt,
                            "prediction": pred,
                            "failed": failed,
                            "error_type": error_type,
                            "vignette": vignette,
                            "clinical_question": clinical_question,
                            "options_sorted": options_sorted,
                            "full_prompt": prompt_text,
                        })

                current_qid = None
                current_gt = None
                current_predictions = None

        elif isinstance(block, Table):
            predictions = table_to_predictions(block, model_names)
            if predictions:
                current_predictions = predictions

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No data extracted. Check DOCX structure or model names.")

    df = (
        df.sort_values(["QID", "model"])
        .drop_duplicates(subset=["QID", "model"], keep="first")
        .reset_index(drop=True)
    )

    return df


def build_shared_failures(df):
    return (
        df.groupby("QID")["failed"]
        .sum()
        .reset_index(name="n_models_failed")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input DOCX file")
    parser.add_argument("--output", default="bsh_failure_review", help="Output folder")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_docx(input_path, args.models)
    shared = build_shared_failures(df)

    review_file = output_dir / "bsh_failure_review_table.csv"
    long_file = output_dir / "bsh_failure_long_table.csv"
    shared_file = output_dir / "bsh_shared_failures.csv"

    df.to_csv(review_file, index=False)
    df.to_csv(long_file, index=False)
    shared.to_csv(shared_file, index=False)

    print("\nExtraction complete.")
    print(f"Rows: {len(df)}")
    print(f"Questions: {df['QID'].nunique()}")
    print(f"Empty prediction rate: {df['prediction'].eq('').mean() * 100:.1f}%")
    print(f"Saved: {review_file}")
    print(f"Saved: {long_file}")
    print(f"Saved: {shared_file}")


if __name__ == "__main__":
    main()


'''
python extract_bsh_review_dataset.py \
  --input bsh_case_reports_fails.docx \
  --output bsh_failure_review
'''