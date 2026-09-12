import glob
from explainer import CandidateExplainer
from jd_pdf_parser import JDPDFParser
from parser import extract_skills, extract_text_from_pdf, normalize_text
from ranker import CandidateRanker


def run_pdf_to_pdf_pipeline(jd_pdf_filename: str):
  print("=" * 60)
  print("STEP 1: PARSING JOB DESCRIPTION PDF")
  print("=" * 60)

  # 1. Parse your Sample_JD.pdf & extract skills dynamically
  jd_parser = JDPDFParser()
  jd_data = jd_parser.parse_jd_pdf(jd_pdf_filename)

  jd_text = jd_data["cleaned_text"]
  required_skills = jd_data["required_skills"]

  print(
      f"✓ Successfully Parsed JD! Extracted ({len(required_skills)}) Required"
      " Skills:"
  )
  print(f"  {required_skills}\n")

  # 2. Find and parse candidate resume PDFs (excluding the JD file itself)
  print("=" * 60)
  print("STEP 2: PARSING CANDIDATE RESUME PDFs")
  print("=" * 60)

  all_pdfs = glob.glob("*.pdf")
  resume_files = [f for f in all_pdfs if f != jd_pdf_filename]

  candidates_data = []
  for file_path in resume_files:
    raw = extract_text_from_pdf(file_path)
    cleaned = normalize_text(raw)
    skills = extract_skills(cleaned)

    candidates_data.append({
        "file_name": file_path,
        "cleaned_text": cleaned,
        "skills": skills,
    })

  print(f"✓ Parsed {len(candidates_data)} candidate resume(s).\n")

  # 3. Rank candidates using Hybrid BM25 + SBERT
  print("=" * 60)
  print("STEP 3: RUNNING HYBRID BM25 + SBERT RANKER")
  print("=" * 60)
  ranker = CandidateRanker(alpha=0.4)
  ranked_results = ranker.rank_candidates(
      jd_text, candidates_data, required_skills
  )

  # 4. Generate candidate evaluation report
  explainer = CandidateExplainer()
  report = explainer.generate_top_3_report(ranked_results)

  print("\n" + report)


if __name__ == "__main__":
  # Targets your sample JD file
  TARGET_JD_PDF = "Sample_JD.pdf"

  try:
    run_pdf_to_pdf_pipeline(TARGET_JD_PDF)
  except FileNotFoundError:
    print(
        f"Error: Could not find '{TARGET_JD_PDF}'. Check if the file is"
        " named 'Sample_JD.pdf' or 'Sample_JD' in PyCharm!"
    )