import os
import tempfile
import pandas as pd
import streamlit as st

# Suppress Hugging Face warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from explainer import CandidateExplainer
from jd_auditor import JDAuditor
from jd_extractor import JDSkillExtractor
from parser import extract_skills, extract_text_from_pdf, normalize_text
from ranker import CandidateRanker

# -----------------------------------------------------------------------------
# Page Configuration & Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Shortlist Desk", page_icon="🗂", layout="wide"
)

# -----------------------------------------------------------------------------
# Custom styling
# -----------------------------------------------------------------------------
# This block of CSS just re-skins the default Streamlit look (different fonts,
# colors, card borders, etc.) so the page doesn't look like an out-of-the-box
# Streamlit demo. It doesn't change any logic below -- purely visual.
# -----------------------------------------------------------------------------
# Color palette (defined once as plain variables, so every hex code below
# traces back to one of these four names -- easy to tweak later).
#   CREAM  = page background (warm off-white)
#   NOIR   = main text color (near-black, high contrast against cream)
#   TAUPE  = primary accent -- buttons, headers, borders
#   BLUSH  = secondary accent, used sparingly -- top-candidate highlight,
#            hover states
# -----------------------------------------------------------------------------
CREAM = "#F5F0E6"
NOIR = "#1A1A1A"
TAUPE = "#7A6C5D"
BLUSH = "#E8C4C0"

st.markdown(
    f"""
    <style>
    /* An editorial serif for headings + a plain sans for body text reads
       more like a printed program than a typical app/dashboard font pairing. */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Source Sans 3', sans-serif;
        color: {NOIR};
    }}

    h1, h2, h3, .desk-header h1 {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
    }}

    /* Overall app background -- flat cream, no gradient/glow, so it reads
       like paper rather than a "tech product" screen. */
    .stApp {{
        background-color: {CREAM};
        color: {NOIR};
    }}

    /* Header / title area */
    .desk-header {{
        padding: 1.4rem 1.6rem;
        border-bottom: 3px solid {TAUPE};
        margin-bottom: 1.4rem;
    }}
    .desk-header h1 {{
        margin: 0;
        font-size: 2.1rem;
        color: {NOIR};
        letter-spacing: 0.2px;
    }}
    .desk-header p {{
        margin: 0.35rem 0 0 0;
        color: {TAUPE};
        font-size: 1rem;
        font-style: italic;
    }}

    /* Section labels -- small printed "tag" rather than a glowing chip */
    .section-tag {{
        display: inline-block;
        background: transparent;
        color: {TAUPE};
        border-bottom: 2px solid {TAUPE};
        padding: 0 0.1rem 0.15rem 0.1rem;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.72rem;
        margin-bottom: 0.6rem;
    }}

    /* Body text, labels, captions */
    p, label, .stMarkdown, .stCaption {{
        color: {NOIR};
    }}

    /* File uploader boxes */
    [data-testid="stFileUploaderDropzone"] {{
        background: #FFFFFF;
        border: 1.5px dashed {TAUPE};
        border-radius: 4px;
    }}

    /* Buttons -- taupe fill, blush on hover instead of a brighter/glowing tone */
    .stButton > button {{
        background: {TAUPE};
        color: {CREAM};
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1.3rem;
    }}
    .stButton > button:hover {{
        background: {BLUSH};
        color: {NOIR};
    }}

    /* Download button gets the same treatment */
    .stDownloadButton > button {{
        background: {TAUPE};
        color: {CREAM};
        border: none;
        border-radius: 4px;
        font-weight: 600;
    }}
    .stDownloadButton > button:hover {{
        background: {BLUSH};
        color: {NOIR};
    }}

    /* Dataframe / table container */
    [data-testid="stDataFrame"] {{
        border: 1px solid {TAUPE};
        border-radius: 4px;
        overflow: hidden;
    }}

    /* Expanders (scorecards) -- taupe header, cream body, so the top-ranked
       card can be picked out later with a blush border (see render_scorecard) */
    .streamlit-expanderHeader {{
        background: #EFE8D8;
        color: {NOIR};
        border-radius: 4px;
        font-weight: 600;
    }}
    details {{
        border: 1px solid {TAUPE};
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }}

    /* Rank #1's card gets a blush left-edge so it stands out at a glance --
       this class is added only to the top card, see render_scorecard() */
    .top-rank-card details {{
        border-left: 5px solid {BLUSH};
    }}

    hr {{
        border-color: {TAUPE};
        opacity: 0.4;
    }}

    /* Info / warning / success boxes -- recolor so they don't clash with the
       cream palette (Streamlit's defaults are blue/red/green by default) */
    [data-testid="stAlert"] {{
        background: #EFE8D8;
        color: {NOIR};
        border-left: 4px solid {TAUPE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="desk-header">
        <h1>Shortlist Desk</h1>
        <p>A ranked, explainable read on every resume against one job description.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# NOTE: There used to be a sidebar here that let the user drag a slider to
# change how much weight goes to keyword matching vs. semantic matching.
# We removed that entirely -- the split is now fixed at semantic 80% /
# keyword 20% and is not shown or editable anywhere in the UI. See the
# "CandidateRanker(alpha=0.2)" line further down for the hardcoded value
# (alpha is the keyword weight, so alpha=0.2 means keyword=20%, semantic=80%).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 1 & 2: User Inputs via File Uploaders
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
  st.markdown('<span class="section-tag">STEP 1</span>', unsafe_allow_html=True)
  st.subheader("Job Description")
  jd_file = st.file_uploader(
      "Upload Job Description PDF", type=["pdf"], key="jd_upload"
  )

with col2:
  st.markdown('<span class="section-tag">STEP 2</span>', unsafe_allow_html=True)
  st.subheader("Candidate Resumes")
  resume_files = st.file_uploader(
      "Upload Resumes (PDFs)",
      type=["pdf"],
      accept_multiple_files=True,
      key="resumes_upload",
  )

st.markdown("---")


# Helper function to generate exportable Pandas DataFrame
def get_leaderboard_dataframe(ranked_results: list[dict]) -> pd.DataFrame:
  table_data = []
  for cand in ranked_results:
    table_data.append({
        "Rank": cand["rank"],
        "Candidate Name": cand["candidate_name"],
        "Fit Score (%)": cand["final_score"],
        "BM25 Score (%)": cand["keyword_score"],
        "SBERT Score (%)": cand["semantic_score"],
        "Verified Skills": ", ".join(cand["matched_skills"]),
        "Missing Skills": ", ".join(cand["missing_skills"]),
    })
  return pd.DataFrame(table_data)


# Small helper so we don't repeat the same scorecard-rendering code in
# multiple places (top-3 loop, and the rank-search box). Renders one
# candidate's card in the same format everywhere.
def render_scorecard(cand: dict, explainer: "CandidateExplainer"):
  # Rank #1 gets wrapped in a div with the "top-rank-card" class, which the
  # CSS above uses to draw a blush-colored left edge -- a small, deliberate
  # use of the secondary accent color rather than sprinkling it everywhere.
  is_top_rank = cand["rank"] == 1
  if is_top_rank:
    st.markdown('<div class="top-rank-card">', unsafe_allow_html=True)

  with st.expander(
      f"Rank #{cand['rank']}: {cand['candidate_name']} — Match Score:"
      f" {cand['final_score']}%",
      expanded=True,
  ):
    st.markdown(explainer.generate_explanation(cand))

  if is_top_rank:
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Step 3: Trigger & Execution Pipeline
# -----------------------------------------------------------------------------
# NOTE ON STRUCTURE: This button's job is now ONLY to compute results and save
# them into st.session_state. Nothing is displayed inside this "if" block
# anymore. Why? Streamlit re-runs your whole script top-to-bottom on every
# interaction (including clicking "Look Up Candidate" further down). If we
# displayed the table/top-3 cards *inside* this "if st.button(...)" block,
# they would vanish the moment the user clicked a different button, because
# on that re-run this particular button is no longer being pressed.
# By saving everything to session_state here, and displaying it in a
# separate section below (that just reads from session_state), the leaderboard
# and top-3 cards stay visible no matter which button the user clicks next.
if st.button("Process & Rank Candidates", type="primary"):
  # Validation: Ensure files are uploaded
  if not jd_file:
    st.error("Please upload a Job Description PDF to continue.")
    st.stop()

  if not resume_files:
    st.error("Please upload at least one Candidate Resume PDF to evaluate.")
    st.stop()

  # 1. Parse uploaded JD PDF
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
    tmp_jd.write(jd_file.getvalue())
    tmp_jd_path = tmp_jd.name

  raw_jd_text = extract_text_from_pdf(tmp_jd_path)
  cleaned_jd = normalize_text(raw_jd_text)
  os.remove(tmp_jd_path)

  if not cleaned_jd.strip():
    st.error(
        "Could not extract text from the Job Description PDF. Make sure it is"
        " not a scanned image with restricted permissions."
    )
    st.stop()

  # 2. Audit JD Phrasing
  auditor = JDAuditor()
  audit_results = auditor.audit_jd(cleaned_jd)

  # 3. Dynamic Skill Extraction
  extractor = JDSkillExtractor()
  extraction_data = extractor.extract_required_skills(cleaned_jd)
  req_skills = extraction_data["extracted_skills"]

  # 4. Parse Uploaded Candidate Resumes
  candidates_data = []
  for file in resume_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_res:
      tmp_res.write(file.getvalue())
      tmp_res_path = tmp_res.name

    raw_res = extract_text_from_pdf(tmp_res_path)
    cleaned_res = normalize_text(raw_res)
    extracted_cand_skills = extract_skills(cleaned_res)

    candidates_data.append({
        "file_name": file.name,
        "cleaned_text": cleaned_res,
        "skills": extracted_cand_skills,
    })
    os.remove(tmp_res_path)

  # 5. Hybrid Ranking
  # alpha is the KEYWORD weight (see ranker.py: final_score = alpha * keyword
  # + (1 - alpha) * semantic). We want semantic:keyword = 80:20, so alpha
  # (keyword weight) = 0.2, which makes (1 - alpha) = 0.8 the semantic weight.
  # final_score = 0.2 * keyword_score + 0.8 * semantic_score
  # There is no slider or control for this anywhere in the UI.
  ranker = CandidateRanker(alpha=0.2)
  ranked_results = ranker.rank_candidates(cleaned_jd, candidates_data, req_skills)

  # Save EVERYTHING we'll need to display into session_state. This is what
  # lets the table, top-3 cards, and lookup box all survive later re-runs.
  st.session_state["ranked_results"] = ranked_results
  st.session_state["req_skills"] = req_skills
  st.session_state["audit_results"] = audit_results

  # A little confirmation so the user knows the click actually did something.
  st.success(f"Ranked {len(ranked_results)} candidates against the job description.")


# -----------------------------------------------------------------------------
# Step 4: Display Results (Leaderboard, Top-3 Cards, Lookup)
# -----------------------------------------------------------------------------
# Everything below reads from st.session_state instead of being nested inside
# the button's "if" block. This means it stays on screen across re-runs --
# including after the user looks up a different rank in the search box.
# -----------------------------------------------------------------------------
if "ranked_results" in st.session_state:
  ranked_results = st.session_state["ranked_results"]
  req_skills = st.session_state["req_skills"]
  audit_results = st.session_state["audit_results"]
  explainer = CandidateExplainer()

  # JD skill extraction summary
  st.success(
      f"Extracted {len(req_skills)} Required Skills from JD:"
      f" `{', '.join(req_skills)}`"
  )

  # JD bias/phrasing audit (only shown if something was actually flagged)
  if audit_results["total_flags"] > 0:
    with st.expander(
        f"Job Description Accessibility Audit ({audit_results['total_flags']}"
        " Flags)",
        expanded=False,
    ):
      for flag in audit_results["flags"]:
        st.warning(f"**[{flag['category']}]**: {flag['issue']}")
        st.info(f"Fix Suggestion: {flag['recommendation']}")

  # --- Leaderboard table -----------------------------------------------------
  st.subheader("3. Shortlist Leaderboard")

  df_leaderboard = get_leaderboard_dataframe(ranked_results)

  # Display interactive table -- this still shows EVERY candidate, unchanged.
  st.dataframe(df_leaderboard, use_container_width=True)

  # Download button for recruiters/judges
  csv_data = df_leaderboard.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="Download Results (CSV)",
      data=csv_data,
      file_name="candidate_shortlist.csv",
      mime="text/csv",
  )

  # --- Top-3 scorecards --------------------------------------------------
  # Only rank 1, 2, and 3 get an individual scorecard here. This section is
  # ALWAYS shown as long as we have results in session_state -- it does not
  # disappear when the user looks up a different rank below, because it no
  # longer lives inside the "Process & Rank Candidates" button's "if" block.
  st.subheader("4. Detailed Candidate XAI Reports (Top 3)")

  top_3 = ranked_results[:3]
  for cand in top_3:
    render_scorecard(cand, explainer)

  # -----------------------------------------------------------------------
  # Search bar to look up ANY candidate by rank number.
  # This does NOT replace the top-3 section above -- both are visible at
  # the same time. Looking up rank 7, for example, just adds one more
  # scorecard below, without hiding ranks 1-3.
  # -----------------------------------------------------------------------
  st.markdown("---")
  st.markdown('<span class="section-tag">LOOKUP</span>', unsafe_allow_html=True)
  st.subheader("Find a Specific Candidate by Rank")

  total_candidates = len(ranked_results)

  rank_to_find = st.number_input(
      f"Enter a rank between 1 and {total_candidates}",
      min_value=1,
      max_value=max(total_candidates, 1),  # avoid max_value=0 crashing the widget
      value=1,
      step=1,
  )

  if st.button("Look Up Candidate"):
    # Guard against an out-of-range rank instead of letting the app crash.
    if rank_to_find < 1 or rank_to_find > total_candidates:
      st.warning(
          f"There's no candidate at rank {rank_to_find}. Please enter a"
          f" number between 1 and {total_candidates}."
      )
    elif rank_to_find <= 3:
      # Ranks 1-3 are already shown above in the Top 3 section -- let the
      # user know instead of just silently reprinting the same card.
      st.info(
          f"Rank {rank_to_find} is already shown above in the Top 3 section."
      )
    else:
      # Ranks start at 1, but Python lists start at 0, so we subtract 1.
      found_candidate = ranked_results[rank_to_find - 1]
      render_scorecard(found_candidate, explainer)
else:
  # Nothing has been processed yet this session.
  st.info(
      "Upload a Job Description and resumes above, then click 'Process &"
      " Rank Candidates' to see the leaderboard and top candidates."
  )
