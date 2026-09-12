import glob
import os
import pandas as pd

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from explainer import CandidateExplainer
from jd_extractor import JDSkillExtractor
from parser import extract_skills, extract_text_from_pdf, normalize_text
from ranker import CandidateRanker

USER_JD = """
About the role:
We are seeking an App & Web Developer.

Requirements:
- Strong knowledge of React, JavaScript, REST API, Git, and SQL.
- Familiarity with Flutter or Python is a plus.

Responsibilities:
- Build modular web apps and manage API integrations.
"""


def get_leaderboard_dataframe(ranked_results: list[dict]) -> pd.DataFrame:
  """Converts candidate results list into a structured Pandas DataFrame."""
  table_data = []
  for cand in ranked_results:
    table_data.append({
        "Rank": cand["rank"],
        "Candidate Name": cand["candidate_name"],
        "Overall Fit Score (%)": cand["final_score"],
        "BM25 Keyword Match (%)": cand["keyword_score"],
        "SBERT Semantic Match (%)": cand["semantic_score"],
        "Verified Skills": ", ".join(cand["matched_skills"]),
        "Missing Skills": ", ".join(cand["missing_skills"]),
    })
  return pd.DataFrame(table_data)


def run_pipeline():
  # 1. Dynamically extract required skills from the user-provided JD
  print("Step 1: Extracting required skills from Job Description...")
  extractor = JDSkillExtractor()
  extraction_result = extractor.extract_required_skills(USER_JD)
  REQUIRED_SKILLS = extraction_result["extracted_skills"]

  print(
      f"-> Extracted Skills from JD ({len(REQUIRED_SKILLS)}):"
      f" {REQUIRED_SKILLS}\n"
  )

  # 2. Parse candidate PDF resumes
  print("Step 2: Parsing resumes...")
  pdf_files = [
      f
      for f in glob.glob("*.pdf")
      if f.lower() not in ["sample_jd.pdf", "job_description_fullstack.pdf"]
  ]

  candidates_data = []
  for file_path in pdf_files:
    raw = extract_text_from_pdf(file_path)
    cleaned = normalize_text(raw)
    skills = extract_skills(cleaned)

    candidates_data.append({
        "file_name": file_path,
        "cleaned_text": cleaned,
        "skills": skills,
    })

  if not candidates_data:
    print("❌ No resume PDFs found in the project directory to evaluate.")
    return

  # 3. Rank candidates
  print(f"\nStep 3: Ranking {len(candidates_data)} candidate(s)...")
  ranker = CandidateRanker(alpha=0.4)
  ranked_results = ranker.rank_candidates(
      USER_JD, candidates_data, REQUIRED_SKILLS
  )

  # 4. Generate Top 3 Report
  print("\nStep 4: Generating explanation report...")
  explainer = CandidateExplainer()
  report = explainer.generate_top_3_report(ranked_results)
  print("\n" + report)

  # 5. Leaderboard Export (Pandas + CSV)
  df_leaderboard = get_leaderboard_dataframe(ranked_results)

  print("\n" + "=" * 80)
  print("BATCH SHORTLIST LEADERBOARD (Top Candidates)")
  print("=" * 80)
  print(df_leaderboard.to_string(index=False))

  df_leaderboard.to_csv("shortlist_leaderboard.csv", index=False)
  print("\n✓ Saved full results to 'shortlist_leaderboard.csv'")


if __name__ == "__main__":
  run_pipeline()