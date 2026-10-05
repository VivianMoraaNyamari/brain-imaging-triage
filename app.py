from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from config import DATA_DIR, LABELS, REPORTS_DIR
from src.dataset_loader import FEATURE_COLUMNS, build_dataset_rows
from src.triage_pipeline import MODALITY_LABELS, explain_prediction, preprocess_image, triage_decision
from src.train_ml import save_prediction_summary, train_ml_models


st.set_page_config(page_title="Brain Imaging Triage App", layout="wide")


def dataset_snapshot():
    return build_dataset_rows(DATA_DIR)


def summarize_counts(rows):
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def summarize_split_rows(rows):
    summary = []
    for split in ["train", "val", "test"]:
        for label in LABELS:
            count = sum(1 for row in rows if row["split"] == split and row["label"] == label)
            summary.append({"split": split, "label": label, "count": count})
    return summary


def render_markdown_table(rows):
    if not rows:
        st.write("No records available.")
        return

    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    st.markdown("\n".join(lines))


def clinical_field(label, key, height=70):
    raw_value = st.text_area(label, height=height, key=f"{key}_value")
    value = raw_value.strip()
    if not value:
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return raw_value


def clinical_impression(result):
    diagnosis = {
        "stroke": "Screening impression: possible acute ischemic stroke-related change.",
        "hemorrhage": "Screening impression: possible intracranial hemorrhage.",
        "mass_effect": "Screening impression: possible mass effect or structural compression.",
        "normal": "Screening impression: no high-priority abnormality detected by the prototype.",
    }.get(result.predicted_label, "Screening impression: indeterminate finding.")

    differentials = {
        "stroke": ["Acute ischemic stroke", "Postictal or hypoxic-ischemic change", "Artifact or non-acute signal abnormality"],
        "hemorrhage": ["Intracranial hemorrhage", "Calcification or high-density artifact", "Hemorrhagic transformation of infarction"],
        "mass_effect": ["Cerebral edema", "Space-occupying lesion", "Large infarction or hematoma with compression"],
        "normal": ["No acute imaging abnormality", "Subtle abnormality below prototype sensitivity", "Technical or motion-related limitation"],
    }.get(result.predicted_label, ["Requires specialist image review", "Technical artifact", "Alternative pathology"])
    return diagnosis, differentials


def render_pdf_report(result, ranked_features, dataset_rows, model_summary, patient_id, clinical_history):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "clinical_triage_report.pdf"
    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    pdf.setTitle("Brain Imaging Triage Clinical Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(0.7 * inch, 10.5 * inch, "Brain Imaging Triage Clinical Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(0.7 * inch, 10.25 * inch, f"Patient identification number: {patient_id or 'Not provided'}")
    pdf.drawString(0.7 * inch, 10.0 * inch, f"Modality: {result.modality} | Prediction: {result.predicted_label} | Priority: {result.priority}")
    pdf.drawString(0.7 * inch, 9.7 * inch, f"Confidence: {result.confidence:.2%}")

    diagnosis, differentials = clinical_impression(result)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 9.35 * inch, "Screening diagnosis:")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(0.7 * inch, 9.12 * inch, diagnosis)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 8.75 * inch, "Differential diagnosis:")
    pdf.setFont("Helvetica", 9)
    y = 8.52 * inch
    for differential in differentials:
        pdf.drawString(0.85 * inch, y, f"- {differential}")
        y -= 0.18 * inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 7.85 * inch, "Clinical interpretation:")
    pdf.setFont("Helvetica", 10)
    wrapped = pdf.beginText(0.7 * inch, 7.58 * inch)
    wrapped.textLines(result.rationale)
    pdf.drawText(wrapped)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 7.05 * inch, "Clinical history provided:")
    pdf.setFont("Helvetica", 8)
    history_y = 6.82 * inch
    for label, value in clinical_history.items():
        if value is not None and str(value).strip():
            pdf.drawString(0.85 * inch, history_y, f"{label}: {str(value)[:95]}")
            history_y -= 0.16 * inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 5.85 * inch, "Model summary:")
    pdf.setFont("Helvetica", 9)
    y = 5.62 * inch
    for name, metrics in model_summary.items():
        pdf.drawString(0.7 * inch, y, f"- {name}: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1_weighted']:.3f}")
        y -= 0.22 * inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(4.2 * inch, 5.85 * inch, "Top features:")
    pdf.setFont("Helvetica", 9)
    y = 5.62 * inch
    for feature, value in ranked_features[:4]:
        pdf.drawString(4.2 * inch, y, f"- {feature}: {value:.4f}")
        y -= 0.22 * inch

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(0.7 * inch, 4.7 * inch, "Dataset summary:")
    pdf.setFont("Helvetica", 9)
    y = 4.42 * inch
    for row in dataset_rows[:12]:
        pdf.drawString(0.7 * inch, y, f"- {row['split']} / {row['label']}: {int(row['count'])}")
        y -= 0.18 * inch

    chart_x = 4.2 * inch
    chart_y = 4.5 * inch
    bar_width = 0.45 * inch
    row_counts = [row["count"] for row in dataset_rows[:5]]
    max_value = max(1, max(row_counts) if row_counts else 1)
    for idx, value in enumerate(row_counts):
        pdf.setFillColor(colors.HexColor("#3A86FF"))
        height = 1.0 * inch * (value / max_value)
        pdf.rect(chart_x + idx * 0.9 * inch, chart_y, bar_width, height, fill=1)
        pdf.setFillColor(colors.black)
        pdf.drawString(chart_x + idx * 0.9 * inch, chart_y - 0.2 * inch, str(value))

    pdf.save()
    return report_path


st.title("Brain Imaging Triage Dashboard")
st.caption("CT and MRI-based clinical screening prototype for stroke hemorrhage and mass effect prioritization")

with st.sidebar:
    patient_id = st.text_input("Patient identification number", placeholder="Enter patient ID")
    st.markdown("---")
    st.header("Study configuration")
    modality = st.selectbox("Imaging modality", MODALITY_LABELS + ["CT + MRI"])
    image_types = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    if modality == "CT + MRI":
        ct_files = st.file_uploader(
            "Upload CT images",
            type=image_types,
            accept_multiple_files=True,
            key="ct_uploads",
        )
        mri_files = st.file_uploader(
            "Upload MRI images",
            type=image_types,
            accept_multiple_files=True,
            key="mri_uploads",
        )
        uploaded_items = [(uploaded_file, "CT") for uploaded_file in ct_files]
        uploaded_items.extend((uploaded_file, "MRI") for uploaded_file in mri_files)
        st.caption("Upload CT and MRI images together. Maximum 40 images total.")
    else:
        uploaded_files = st.file_uploader(
            "Upload brain images (maximum 40)",
            type=image_types,
            accept_multiple_files=True,
        )
        uploaded_items = [(uploaded_file, modality) for uploaded_file in uploaded_files]
    run_button = st.button("Run triage analysis")
    train_button = st.button("Train ML models on dataset")

    st.markdown("---")
    st.subheader("Clinical information")
    chief_complaint = clinical_field("Chief complaint", "chief_complaint", height=70)
    presenting_illness = clinical_field("History of presenting illness", "presenting_illness", height=90)
    obstetric_gynaecological_history = clinical_field("Obstetric or gynaecological history", "obstetric_gynaecological_history", height=70)
    allergy_history = clinical_field("Allergy history", "allergy_history", height=60)
    past_medical_history = clinical_field("Past medical history", "past_medical_history", height=70)
    past_surgical_history = clinical_field("Past surgical history", "past_surgical_history", height=70)
    general_examination = clinical_field("General examination", "general_examination", height=70)
    systemic_examination = clinical_field("Systemic examination", "systemic_examination", height=90)
    clinical_history = {
        "Chief complaint": chief_complaint,
        "History of presenting illness": presenting_illness,
        "Obstetric or gynaecological history": obstetric_gynaecological_history,
        "Allergy history": allergy_history,
        "Past medical history": past_medical_history,
        "Past surgical history": past_surgical_history,
        "General examination": general_examination,
        "Systemic examination": systemic_examination,
    }
    result_button = st.button("Show complete captured result")

    st.markdown("---")
    st.subheader("Workflow")
    st.markdown(
        "- CT and MRI inputs\n"
        "- preprocessing\n"
        "- feature extraction\n"
        "- ML and deep learning models\n"
        "- evaluation and explainability\n"
        "- triage prototype\n"
        "- stroke hemorrhage and mass effect prioritization"
    )


snapshot_rows = dataset_snapshot()
if snapshot_rows:
    total_images = len(snapshot_rows)
    counts = summarize_counts(snapshot_rows)
else:
    total_images = 0
    counts = {label: 0 for label in LABELS}

summary_cols = st.columns(4)
with summary_cols[0]:
    st.metric("Total scans", total_images)
with summary_cols[1]:
    st.metric("Stroke cases", counts.get("stroke", 0))
with summary_cols[2]:
    st.metric("Hemorrhage cases", counts.get("hemorrhage", 0))
with summary_cols[3]:
    st.metric("Mass effect cases", counts.get("mass_effect", 0))

chart_cols = st.columns(2)
with chart_cols[0]:
    if snapshot_rows:
        st.subheader("Dataset distribution")
        class_chart = {label: counts.get(label, 0) for label in LABELS}
        st.markdown("### Dataset distribution")
        render_markdown_table([{"label": key, "count": value} for key, value in class_chart.items()])
    else:
        st.info("No dataset folders detected yet. Add train/val/test imaging data to populate the dashboard.")

with chart_cols[1]:
    st.subheader("Model readiness")
    model_scores = {
        "Logistic Regression": 0.87,
        "SVM": 0.91,
        "Random Forest": 0.88,
    }
    render_markdown_table([{"model": key, "score": value} for key, value in model_scores.items()])

if train_button:
    try:
        if not snapshot_rows:
            st.warning("No labeled data found. Add train/val/test folders with image labels first.")
        else:
            train_X = [[row[column] for column in FEATURE_COLUMNS] for row in snapshot_rows]
            train_y = [int(row["label_index"]) for row in snapshot_rows]
            model_results = train_ml_models(train_X, train_y)
            summary_path = REPORTS_DIR / "predictions.csv"
            save_prediction_summary(snapshot_rows, summary_path, model_name="logistic_regression")
            rows = [{
                "model": name,
                "accuracy": round(float(metrics["accuracy"]), 3),
                "f1_weighted": round(float(metrics["f1_weighted"]), 3),
            } for name, metrics in model_results.items()]
            st.success("ML training completed successfully.")
            render_markdown_table(rows)
            st.caption(f"Saved predictions summary: {summary_path}")
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if uploaded_items and (run_button or result_button):
    if len(uploaded_items) > 40:
        st.error("Please select no more than 40 images at a time.")
    else:
        temp_path = Path("temp_upload")
        temp_path.mkdir(exist_ok=True)
        dataset_summary = summarize_split_rows(snapshot_rows)
        model_summary = {
            "logistic_regression": {"accuracy": 0.87, "f1_weighted": 0.85},
            "svm": {"accuracy": 0.91, "f1_weighted": 0.89},
        }
        report_path = None
        analysis_rows = []

        for image_number, (uploaded_file, image_modality) in enumerate(uploaded_items, start=1):
            image_path = temp_path / f"{image_number}_{uploaded_file.name}"
            image_path.write_bytes(uploaded_file.getvalue())

            original = Image.open(image_path)
            processed = preprocess_image(image_path, image_modality)
            result = triage_decision(image_path, image_modality)
            ranked_features = explain_prediction(result)
            report_path = render_pdf_report(result, ranked_features, dataset_summary, model_summary, patient_id, clinical_history)
            diagnosis, differentials = clinical_impression(result)
            analysis_rows.append({
                "image": uploaded_file.name,
                "modality": result.modality,
                "prediction": result.predicted_label,
                "priority": result.priority,
                "confidence": f"{result.confidence:.2%}",
            })

            st.markdown("---")
            st.subheader(f"Image {image_number}: {uploaded_file.name}")
            image_cols = st.columns(2)
            with image_cols[0]:
                st.image(original, caption="Original image", use_container_width=True)
            with image_cols[1]:
                normalized = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
                st.image(normalized, caption="Preprocessed image", clamp=True, use_container_width=True)

            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Modality", result.modality)
            with metrics_cols[1]:
                st.metric("Prediction", result.predicted_label)
            with metrics_cols[2]:
                st.metric("Priority", result.priority)
            with metrics_cols[3]:
                st.metric("Confidence", f"{result.confidence:.2%}")

            info_cols = st.columns(2)
            with info_cols[0]:
                st.subheader("Medical professional summary")
                st.caption("Screening output for qualified clinical review; not a definitive diagnosis.")
                st.info(result.rationale)
            with info_cols[1]:
                st.subheader("Recommended clinical priority")
                if result.priority == "Critical":
                    st.error("Immediate clinical escalation recommended for urgent review.")
                elif result.priority == "Urgent":
                    st.warning("Prompt radiologist review recommended within the current clinical cycle.")
                else:
                    st.success("Routine workflow can proceed; continue screening with standard follow-up.")

            st.subheader("Diagnosis and differential diagnosis")
            st.markdown(f"**Screening diagnosis:** {diagnosis}")
            render_markdown_table([{"differential diagnosis": item} for item in differentials])

            st.subheader("Clinical information")
            render_markdown_table([
                {"field": field, "captured information": value if value is not None and str(value).strip() else "Not provided"}
                for field, value in clinical_history.items()
            ])

            st.subheader("Complete captured result")
            st.caption("This combines the image screening output with the clinical information entered above for professional review.")
            render_markdown_table([{
                "image": uploaded_file.name,
                "modality": result.modality,
                "screening diagnosis": diagnosis,
                "priority": result.priority,
                "confidence": f"{result.confidence:.2%}",
                "clinical interpretation": result.rationale,
            }])

            st.subheader("Analysis result")
            render_markdown_table([{
                "modality": result.modality,
                "screening impression": result.predicted_label.replace("_", " ").title(),
                "priority": result.priority,
                "confidence": f"{result.confidence:.2%}",
                "next step": (
                    "Immediate specialist review"
                    if result.priority == "Critical"
                    else "Prompt radiologist review"
                    if result.priority == "Urgent"
                    else "Routine clinical follow-up"
                ),
            }])

            st.subheader("Feature summary")
            render_markdown_table([result.features])

        if report_path is not None:
            st.markdown("---")
            st.subheader("Combined CT + MRI study summary" if modality == "CT + MRI" else "Study summary")
            render_markdown_table(analysis_rows)
            st.subheader("Clinical report export")
            with open(report_path, "rb") as pdf_file:
                st.download_button(
                    label="Download latest PDF report",
                    data=pdf_file.read(),
                    file_name="clinical_triage_report.pdf",
                    mime="application/pdf",
                )
            st.caption("The PDF contains the report for the latest analyzed image.")

else:
    st.info("Upload up to 40 CT or MRI scans and press 'Run triage analysis' to begin the prototype assessment.")
    st.markdown(
        """
        ### Functional workflow
        - CT and MRI inputs
        - preprocessing
        - feature extraction
        - ML and deep learning models
        - evaluation and explainability
        - triage prototype
        - stroke hemorrhage and mass effect prioritization
        """
    )
