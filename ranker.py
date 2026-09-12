import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


class CandidateRanker:

  def __init__(
      self,
      model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
      alpha: float = 0.4,
  ):
    """alpha: Weight given to Keyword Matching (BM25).

    (1 - alpha): Weight given to Semantic Matching (SBERT).
    """
    print("Loading Sentence Transformer embedding model...")
    self.model = SentenceTransformer(model_name)
    self.alpha = alpha

  def compute_keyword_scores(
      self, jd_text: str, resumes_text: list[str]
  ) -> list[float]:
    """Calculates BM25 keyword overlap scores normalized between 0 and 1."""
    tokenized_resumes = [r.lower().split() for r in resumes_text]
    bm25 = BM25Okapi(tokenized_resumes)

    jd_tokens = jd_text.lower().split()
    raw_scores = bm25.get_scores(jd_tokens)

    # Min-Max Normalization to scale scores from 0.0 to 1.0
    max_s, min_s = max(raw_scores), min(raw_scores)
    if max_s == min_s:
      return [1.0] * len(raw_scores)

    return [(s - min_s) / (max_s - min_s) for s in raw_scores]

  def compute_semantic_scores(
      self, jd_text: str, resumes_text: list[str]
  ) -> list[float]:
    """Calculates Cosine Similarity scores using SBERT vector embeddings."""
    jd_embedding = self.model.encode(jd_text, convert_to_tensor=True)
    resume_embeddings = self.model.encode(
        resumes_text, convert_to_tensor=True
    )

    # Compute cosine similarity matrix
    cosine_scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
    return cosine_scores.cpu().numpy().tolist()

  def rank_candidates(
      self,
      jd_text: str,
      candidates_data: list[dict],
      required_skills: list[str],
  ) -> list[dict]:
    """Combines BM25 & Semantic scores, generates candidate rankings and skill match gaps."""
    resumes_text = [c["cleaned_text"] for c in candidates_data]

    kw_scores = self.compute_keyword_scores(jd_text, resumes_text)
    sem_scores = self.compute_semantic_scores(jd_text, resumes_text)

    ranked_list = []
    for idx, candidate in enumerate(candidates_data):
      # Weighted Hybrid Formula: Score = (alpha * BM25) + ((1 - alpha) * SBERT)
      final_score = (self.alpha * kw_scores[idx]) + (
          (1 - self.alpha) * sem_scores[idx]
      )

      # Determine Matched vs Missing Core Skills
      candidate_skills = candidate.get("skills", [])
      matched_skills = [
          s for s in required_skills if s.lower() in candidate_skills
      ]
      missing_skills = [
          s for s in required_skills if s.lower() not in candidate_skills
      ]

      ranked_list.append({
          "candidate_name": candidate.get("file_name", f"Candidate_{idx+1}"),
          "final_score": round(final_score * 100, 2),
          "keyword_score": round(kw_scores[idx] * 100, 2),
          "semantic_score": round(sem_scores[idx] * 100, 2),
          "matched_skills": matched_skills,
          "missing_skills": missing_skills,
      })

    # Sort candidates descending by final score
    ranked_list.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign Rank numbers
    for rank, cand in enumerate(ranked_list, start=1):
      cand["rank"] = rank

    return ranked_list


import pandas as pd


def get_leaderboard_dataframe(ranked_results: list[dict]) -> pd.DataFrame:
  """Converts ranked candidate results into a structured Pandas DataFrame suitable for CSV export or tabular display."""
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

  df = pd.DataFrame(table_data)
  return df