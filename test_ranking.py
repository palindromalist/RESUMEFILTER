from parser import (
    SKILL_TAXONOMY,
    extract_skills,
    extract_text_from_pdf,
    normalize_text,
)
from ranker import CandidateRanker

# 1. Sample Job Description for Full Stack Intern
SAMPLE_JD = """
TechNova Solutions is looking for a Junior Full Stack Developer Intern.
Required Skills: React, Node.js, Express.js, MongoDB, JavaScript, REST API, Git, SQL.
Responsibilities: Build REST APIs, maintain MongoDB database schemas, and create responsive UI with React.
"""

REQUIRED_JD_SKILLS = [
    "react",
    "node.js",
    "express.js",
    "mongodb",
    "javascript",
    "rest api",
    "git",
    "sql",
]

# 2. Extract Kavya Menon's Data
pdf_file = (
    "App_Developer_Resume_2_Kavya_Menon.pdf"
)
raw_text = extract_text_from_pdf(pdf_file)
cleaned_text = normalize_text(raw_text)
skills = extract_skills(cleaned_text)

kavya_data = {
    "file_name": "Kavya Menon",
    "cleaned_text": cleaned_text,
    "skills": skills,
}

# 3. Test Hybrid Ranker Engine
ranker = CandidateRanker(alpha=0.4)
results = ranker.rank_candidates(
    SAMPLE_JD, [kavya_data], required_skills=REQUIRED_JD_SKILLS
)

print("\n" + "=" * 50)
print("RANKING ENGINE OUTPUT PREVIEW")
print("=" * 50)
for r in results:
  print(f"Rank {r['rank']}: {r['candidate_name']}")
  print(f"  - Final Score: {r['final_score']}/100")
  print(f"  - Keyword Score (BM25): {r['keyword_score']}/100")
  print(f"  - Semantic Score (SBERT): {r['semantic_score']}/100")
  print(f"  - Matched Skills: {r['matched_skills']}")
  print(f"  - Missing Required Skills: {r['missing_skills']}")