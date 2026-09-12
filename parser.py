import re
import pdfplumber

# 1. Canonical skill mapping dictionary
SKILL_TAXONOMY = {
    "node.js": ["nodejs", "node.js", "node js", "node"],
    "express.js": ["express", "expressjs", "express.js"],
    "react": ["react", "reactjs", "react.js", "react native"],
    "mongodb": ["mongodb", "mongo", "mongo db"],
    "python": ["python", "py"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "sql": ["sql", "postgresql", "postgres", "mysql", "sqlite"],
    "docker": ["docker", "containerization"],
    "aws": ["aws", "amazon web services"],
    "rest api": ["rest api", "restful api", "rest", "web apis"],
    "git": ["git", "github", "gitlab"],
    "flutter": ["flutter", "dart"],
    "android": ["android", "android studio", "kotlin", "java"],
    "ios": ["ios", "swift", "xcode"]
}

# 2. Recognized Section Header Keywords
SECTION_HEADERS = {
    "skills": ["skills", "technical skills", "technologies", "core competencies", "tools"],
    "experience": ["experience", "work experience", "employment history", "internships", "work history"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "education": ["education", "academic qualification", "qualifications", "education & background"]
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Step 1: Extract plain text from PDF."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def normalize_text(raw_text: str) -> str:
    """Step 2: Clean up bad formatting, bullet points, and odd spaces."""
    if not raw_text:
        return ""
    text = re.sub(r'[\u2022\u2023\u25B6\u25C0\u25E6\u25A0\u25A1\u25CF\u25CB\u2013\u2014|]', ' ', raw_text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def extract_skills(text: str) -> list[str]:
    """Step 3: Match technologies against taxonomy using regex."""
    text_lower = text.lower()
    matched_skills = set()

    for canonical_skill, synonyms in SKILL_TAXONOMY.items():
        for synonym in synonyms:
            pattern = r'\b' + re.escape(synonym) + r'\b'
            if re.search(pattern, text_lower):
                matched_skills.add(canonical_skill)
                break

    return sorted(list(matched_skills))


def parse_sections(text: str) -> dict[str, str]:
    """Step 4: Segment text into section blocks."""
    lines = text.split('\n')
    current_section = "general"
    sections = {
        "general": "",
        "skills": "",
        "experience": "",
        "projects": "",
        "education": ""
    }

    for line in lines:
        line_clean = line.strip().lower()
        matched_header = None

        for section_name, keywords in SECTION_HEADERS.items():
            if any(line_clean == kw or line_clean.startswith(kw + ":") for kw in keywords):
                matched_header = section_name
                break

        if matched_header:
            current_section = matched_header
        else:
            sections[current_section] += line + " "

    return {k: v.strip() for k, v in sections.items() if v.strip()}


if __name__ == "__main__":
    # Your updated sample PDF path
    test_file = "App_Developer_Resume_2_Kavya_Menon.pdf"

    try:
        print(f"Reading file: {test_file}...\n")
        raw = extract_text_from_pdf(test_file)
        cleaned = normalize_text(raw)
        skills = extract_skills(cleaned)
        sections = parse_sections(cleaned)

        print("=" * 60)
        print("PARSER TEST RESULTS FOR KAVYA MENON")
        print("=" * 60)
        print(f"\nExtracted Core Skills ({len(skills)}):")
        print(skills)

        print(f"\nDetected Sections ({len(sections)}):")
        for sec_name in sections.keys():
            print(f"- {sec_name.upper()}")

        print("\nText Preview (First 250 characters):")
        print(cleaned[:250])

    except FileNotFoundError:
        print(f"File '{test_file}' not found. Ensure it is placed directly inside your PyCharm project folder.")