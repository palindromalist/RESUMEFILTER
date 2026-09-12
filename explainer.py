class CandidateExplainer:

  def __init__(self, jd_title: str = "Full Stack Developer Intern"):
    self.jd_title = jd_title

  def generate_explanation(self, candidate_data: dict) -> str:
    """Generates a structured natural language explanation for why a candidate ranked at their position."""
    rank = candidate_data.get("rank", 1)
    name = candidate_data.get("candidate_name", "Candidate")
    final_score = candidate_data.get("final_score", 0.0)
    kw_score = candidate_data.get("keyword_score", 0.0)
    sem_score = candidate_data.get("semantic_score", 0.0)
    matched = candidate_data.get("matched_skills", [])
    missing = candidate_data.get("missing_skills", [])

    explanation = []
    explanation.append(
        f"### Rank #{rank}: {name} (Overall Score: {final_score:.1f}%)\n"
    )

    # Core Verdict Summary
    if final_score >= 75:
      verdict = (
          "Strong Fit — Demonstrates heavy overlap in key technologies and"
          " project experience."
      )
    elif final_score >= 50:
      verdict = (
          "Moderate Fit — Has relevant transferrable skills but missing a few"
          " core tools."
      )
    else:
      verdict = (
          "Weak Fit — Significant skill gaps relative to explicit requirements."
      )

    explanation.append(f"**Fit Verdict:** {verdict}\n")

    # Score Breakdown Analysis
    explanation.append("**Score Breakdown:**")
    explanation.append(
        f"- **Keyword Match (BM25):** {kw_score:.1f}% (Direct skill term"
        " overlap)"
    )
    explanation.append(
        f"- **Semantic Context (SBERT):** {sem_score:.1f}% (Contextual similarity"
        " of responsibilities & project experience)\n"
    )

    # Skill Gap Breakdown
    explanation.append("**Key Skill Analysis:**")
    if matched:
      matched_str = ", ".join([s.title() for s in matched])
      explanation.append(
          f"- **Verified Matching Skills ({len(matched)}):** {matched_str}"
      )
    else:
      explanation.append(
          "- **Verified Matching Skills:** None explicitly matched."
      )

    if missing:
      missing_str = ", ".join([s.title() for s in missing])
      explanation.append(
          f"- **Missing/Unverified Skills ({len(missing)}):** {missing_str}"
      )
    else:
      explanation.append(
          "- **Missing/Unverified Skills:** None (100% core skill coverage)."
      )

    return "\n".join(explanation)

  def generate_top_3_report(self, ranked_candidates: list[dict]) -> str:
    """Compiles individual explanations into a clean report for the top 3 candidates."""
    top_3 = ranked_candidates[:3]
    report = ["# TOP CANDIDATES SHORTLIST EXPLANATION REPORT\n", "=" * 60 + "\n"]

    for candidate in top_3:
      report.append(self.generate_explanation(candidate))
      report.append("\n" + "-" * 60 + "\n")

    return "\n".join(report)