import re


class JDAuditor:

  def __init__(self):
    # Regex pattern to catch overly rigid experience requirements
    self.rigid_years_pattern = r"\b(10\+|8\+|5\+)\s*(years|yrs)\b"

    # Specific tool lock-in warnings
    self.single_tool_locks = {
        "react": (
            "Demanding 'React' strictly instead of 'Modern Frontend Framework'"
        ),
        "aws": "Demanding 'AWS' strictly instead of 'Cloud Platforms'",
        "postgres": (
            "Demanding 'PostgreSQL' strictly instead of 'Relational Databases"
            " (SQL)'"
        ),
    }

  def audit_jd(self, jd_text: str) -> dict:
    """Audits job description text for overly restrictive language or tool lock-in."""
    jd_lower = jd_text.lower()
    flags = []

    # Check for hard year constraints
    years_matches = re.findall(self.rigid_years_pattern, jd_lower)
    if years_matches:
      matched_phrases = [m[0] + " " + m[1] for m in years_matches]
      flags.append({
          "category": "Overly Narrow Constraints",
          "severity": "High",
          "issue": (
              f"Requires high hard-year thresholds ({', '.join(matched_phrases)})."
          ),
          "recommendation": (
              "Focus on competency and demonstrated projects rather than"
              " arbitrary year minimums."
          ),
      })

    # Check for tool lock-in without fallback phrasing
    for tool, warning in self.single_tool_locks.items():
      if (
          re.search(r"\b" + tool + r"\b", jd_lower)
          and "equivalent" not in jd_lower
          and "or" not in jd_lower
      ):
        flags.append({
            "category": "Narrow Skill Lock-In",
            "severity": "Medium",
            "issue": warning,
            "recommendation": (
                "Add 'or equivalent practical experience' to broaden your"
                " candidate pool."
            ),
        })

    return {"total_flags": len(flags), "flags": flags}