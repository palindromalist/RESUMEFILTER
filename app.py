import os
import tempfile
import pandas as pd
import streamlit as st

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from explainer import CandidateExplainer
from jd_auditor import JDAuditor
from jd_extractor import JDSkillExtractor
from parser import extract_skills, extract_text_from_pdf, normalize_text
from ranker import CandidateRanker

# -----------------------------------------------------------------------------
# Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Shortlisting Engine", page_icon="⚡", layout="wide"
)

st.title("⚡ Smart Shortlisting Engine")
st.caption(
    "Upload a Job Description PDF and Candidate Resumes to extract skills,"
    " audit bias, and generate XAI-ranked leaderboards."
)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header(" Engine Controls")
alpha = st.sidebar.slider(
    "Keyword Weight (BM25 α)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Higher α prioritizes exact skill matches. Lower α favors SBERT semantic context.",
)

st.sidebar.markdown(
    f"**Weighting Split:**\n- BM25 (Keywords): **{int(alpha * 100)}%**\n- SBERT"
    f" (Semantics): **{int((1 - alpha) * 100)}%**"
)

# -----------------------------------------------------------------------------
# Step 1 & 2: User Inputs via File Uploaders
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("1. Job Description")
  jd_file = st.file_uploader(
      "Upload Job Description PDF", type=["pdf"], key="jd_upload"
  )

with col2:
  st.subheader("2. Candidate Resumes")
  resume_files = st.file_uploader(
      "Upload Resumes (PDFs)",
      type=["pdf"],
      accept_multiple_files=True,
      key="resumes_upload",
  )

st.markdown("---")

# Helper function to generate exportable Pandas DataFrame
def get_leaderboard_dataframe(ranked_results: list[dict]) -> pd.DataFrame:
  table_data = []
  for cand in ranked_results:
    table_data.append({
        "Rank": cand["rank"],
        "Candidate Name": cand["candidate_name"],
        "Fit Score (%)": cand["final_score"],
        "BM25 Score (%)": cand["keyword_score"],
        "SBERT Score (%)": cand["semantic_score"],
        "Verified Skills": ", ".join(cand["matched_skills"]),
        "Missing Skills": ", ".join(cand["missing_skills"]),
    })
  return pd.DataFrame(table_data)


# -----------------------------------------------------------------------------
# Step 3: Trigger & Execution Pipeline
# -----------------------------------------------------------------------------
if st.button("🚀 Process & Rank Candidates", type="primary"):
  # Validation: Ensure files are uploaded
  if not jd_file:
    st.error("Please upload a Job Description PDF to continue.")
    st.stop()

  if not resume_files:
    st.error("Please upload at least one Candidate Resume PDF to evaluate.")
    st.stop()

  # 1. Parse uploaded JD PDF
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
    tmp_jd.write(jd_file.getvalue())
    tmp_jd_path = tmp_jd.name

  raw_jd_text = extract_text_from_pdf(tmp_jd_path)
  cleaned_jd = normalize_text(raw_jd_text)
  os.remove(tmp_jd_path)

  if not cleaned_jd.strip():
    st.error(
        "Could not extract text from the Job Description PDF. Make sure it is"
        " not a scanned image with restricted permissions."
    )
    st.stop()

  # 2. Audit JD Phrasing
  auditor = JDAuditor()
  audit_results = auditor.audit_jd(cleaned_jd)

  if audit_results["total_flags"] > 0:
    with st.expander(
        f"⚠️ Job Description Accessibility Audit ({audit_results['total_flags']}"
        " Flags)",
        expanded=False,
    ):
      for flag in audit_results["flags"]:
        st.warning(f"**[{flag['category']}]**: {flag['issue']}")
        st.info(f"Fix Suggestion: {flag['recommendation']}")

  # 3. Dynamic Skill Extraction
  extractor = JDSkillExtractor()
  extraction_data = extractor.extract_required_skills(cleaned_jd)
  req_skills = extraction_data["extracted_skills"]

  st.success(
      f"Extracted {len(req_skills)} Required Skills from JD:"
      f" `{', '.join(req_skills)}`"
  )

  # 4. Parse Uploaded Candidate Resumes
  candidates_data = []
  for file in resume_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_res:
      tmp_res.write(file.getvalue())
      tmp_res_path = tmp_res.name

    raw_res = extract_text_from_pdf(tmp_res_path)
    cleaned_res = normalize_text(raw_res)
    extracted_cand_skills = extract_skills(cleaned_res)

    candidates_data.append({
        "file_name": file.name,
        "cleaned_text": cleaned_res,
        "skills": extracted_cand_skills,
    })
    os.remove(tmp_res_path)

  # 5. Hybrid Ranking
  ranker = CandidateRanker(alpha=alpha)
  ranked_results = ranker.rank_candidates(cleaned_jd, candidates_data, req_skills)

  # -----------------------------------------------------------------------------
  # Step 4: Display Tabular Results & XAI Leaderboard
  # -----------------------------------------------------------------------------
  st.subheader("3. Shortlist Leaderboard")

  df_leaderboard = get_leaderboard_dataframe(ranked_results)

  # Display interactive table
  st.dataframe(df_leaderboard, use_container_width=True)

  # Download button for recruiters/judges
  csv_data = df_leaderboard.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Download Results (CSV)",
      data=csv_data,
      file_name="candidate_shortlist.csv",
      mime="text/csv",
  )

  st.subheader("4. Detailed Candidate XAI Reports")
  explainer = CandidateExplainer()

  for cand in ranked_results:
    with st.expander(
        f"Rank #{cand['rank']}: {cand['candidate_name']} — Match Score:"
        f" {cand['final_score']}%",
        expanded=True,
    ):
      st.markdown(explainer.generate_explanation(cand))