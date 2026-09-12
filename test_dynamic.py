from jd_extractor import JDSkillExtractor

# Example raw user input job description
RAW_USER_JD = """
About Us:
CloudScale Tech is hiring a Mobile & Web Developer.

Required Skills:
- Flutter / Dart or React Native
- Strong background in Python, Node.js, and Express.js
- Database management with SQL (PostgreSQL/MySQL) and MongoDB
- REST API integration & Git version control

Responsibilities:
- Build cross-platform apps and backend services.
"""

extractor = JDSkillExtractor()
result = extractor.extract_required_skills(RAW_USER_JD)

print("=" * 50)
print("DYNAMIC JD SKILL EXTRACTION RESULTS")
print("=" * 50)
print(f"Isolated Section Preview:\n{result['isolated_section_preview']}\n")
print(f"Dynamically Extracted Required Skills ({len(result['extracted_skills'])}):")
print(result["extracted_skills"])