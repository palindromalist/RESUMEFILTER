import re
from parser import SKILL_TAXONOMY


class JDSkillExtractor:
  """Parses user-input Job Description text to isolate the required skills section

  and extract canonical skills dynamically.
  """

  # Common section headers used in JDs for skills/requirements
  SKILL_SECTION_HEADERS = [
      "required skills",
      "technical skills",
      "requirements",
      "what you need",
      "qualifications",
      "key skills",
      "tech stack",
      "skills required",
      "must have",
  ]

  def isolate_skills_section(self, jd_text: str) -> str:
    """Finds and extracts text following skill-related section headings."""
    lines = jd_text.split("\n")
    skills_section_lines = []
    capturing = False

    for line in lines:
      line_clean = line.strip().lower()

      # Check if this line acts as a Skill Section Header
      if any(
          line_clean.startswith(header) or line_clean.endswith(header + ":")
          for header in self.SKILL_SECTION_HEADERS
      ):
        capturing = True
        continue

      # Stop capturing if a new unrelated section starts (e.g., Responsibilities, Benefits)
      elif capturing and any(
          header in line_clean
          for header in [
              "responsibilities",
              "what you'll do",
              "about us",
              "benefits",
              "perks",
              "education",
          ]
      ):
        break

      if capturing:
        skills_section_lines.append(line)

    isolated_text = "\n".join(skills_section_lines).strip()

    # Fallback: If no explicit section header was found, use full JD text
    return isolated_text if isolated_text else jd_text

  def extract_required_skills(self, jd_text: str) -> dict:
    """Dynamically extracts canonical required skills from user-provided JD text."""
    target_text = self.isolate_skills_section(jd_text)
    target_text_lower = target_text.lower()

    detected_skills = set()

    for canonical_skill, synonyms in SKILL_TAXONOMY.items():
      for synonym in synonyms:
        # Match using exact word boundary regex
        pattern = r"\b" + re.escape(synonym) + r"\b"
        if re.search(pattern, target_text_lower):
          detected_skills.add(canonical_skill)
          break

    return {
        "isolated_section_preview": target_text[:200],
        "extracted_skills": sorted(list(detected_skills)),
    }