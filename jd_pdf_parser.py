from jd_extractor import JDSkillExtractor
from parser import extract_text_from_pdf, normalize_text


class JDPDFParser:
  """Parses a Job Description PDF, cleans its text, and dynamically extracts required skills."""

  def __init__(self):
    self.extractor = JDSkillExtractor()

  def parse_jd_pdf(self, jd_pdf_path: str) -> dict:
    """Reads a JD PDF file and returns the clean text + extracted core skills."""
    print(f"Reading Job Description PDF: '{jd_pdf_path}'...")

    # Reuse your robust pdfplumber + normalization pipeline
    raw_text = extract_text_from_pdf(jd_pdf_path)
    cleaned_text = normalize_text(raw_text)

    if not cleaned_text.strip():
      raise ValueError(
          f"Could not extract any text from '{jd_pdf_path}'. Check if the PDF"
          " is corrupted or image-only."
      )

    # Dynamically extract core skills from the parsed JD text
    extraction_data = self.extractor.extract_required_skills(cleaned_text)
    extracted_skills = extraction_data["extracted_skills"]

    return {
        "jd_path": jd_pdf_path,
        "cleaned_text": cleaned_text,
        "required_skills": extracted_skills,
    }