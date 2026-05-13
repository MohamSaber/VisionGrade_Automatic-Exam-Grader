import streamlit as st
from scratch_image import encode_image, invert_binary, load_grayscale_from_bytes, save_image

# ── Project imports ──────────────────────────────────────────
from grader import grade_answers, parse_answer_key
from main import (
    stage1_load, stage2_deskew_hough, stage3_denoise,
    stage4_clahe, stage5_threshold, remove_small_noise,
    stage6_morphology, parse_ocr_results
)
from ocr_engine import recognize_text_from_image

# ════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "VisionGrade — Exam Grader",
    page_icon   = "📝",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ════════════════════════════════════════════════════════════
# Custom CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a2f3e;
}
h1, h2, h3, h4 { color: #e8eaf0; font-family: 'Segoe UI', sans-serif; }
p, label, span  { color: #a8b0c0; font-family: 'Segoe UI', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a2236 0%, #0d1b2a 50%, #162032 100%);
    border: 1px solid #2a3f5f;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero p {
    color: #8899aa;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Cards ── */
.card {
    background: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7a99;
    margin-bottom: 6px;
}

/* ── Score metrics ── */
.metric-box {
    background: #1a2030;
    border: 1px solid #2a3550;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6b7a99;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1;
}
.metric-sub {
    font-size: 0.82rem;
    color: #6b7a99;
    margin-top: 6px;
}

/* ── Pass/Fail badge ── */
.badge-pass {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 999px;
    padding: 4px 18px;
    font-size: 0.9rem;
    font-weight: 600;
}
.badge-fail {
    display: inline-block;
    background: #450a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 999px;
    padding: 4px 18px;
    font-size: 0.9rem;
    font-weight: 600;
}

/* ── Score bar ── */
.score-bar-bg {
    background: #1e2535;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin-top: 12px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Stage pipeline ── */
.stage-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.stage-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #60a5fa;
    flex-shrink: 0;
}
.stage-name {
    font-size: 0.85rem;
    color: #c8d0e0;
    flex: 1;
}
.stage-done {
    font-size: 0.78rem;
    color: #4ade80;
    font-weight: 600;
}

/* ── Question table ── */
.q-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    background: #1a2030;
    border: 1px solid #232a3a;
}
.q-num {
    font-size: 0.82rem;
    font-weight: 700;
    color: #60a5fa;
    width: 36px;
    flex-shrink: 0;
}
.q-recognized {
    flex: 1;
    font-size: 0.88rem;
    color: #c8d0e0;
}
.q-correct {
    flex: 1;
    font-size: 0.88rem;
    color: #8899aa;
}
.q-sim {
    font-size: 0.82rem;
    color: #a8b0c0;
    width: 50px;
    text-align: right;
}
.q-badge-ok  {
    background:#14532d; color:#4ade80;
    border:1px solid #166534; border-radius:6px;
    padding:2px 10px; font-size:0.78rem; font-weight:600;
    width: 70px; text-align:center; flex-shrink:0;
}
.q-badge-err {
    background:#450a0a; color:#f87171;
    border:1px solid #7f1d1d; border-radius:6px;
    padding:2px 10px; font-size:0.78rem; font-weight:600;
    width: 70px; text-align:center; flex-shrink:0;
}

/* ── Image section ── */
.img-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7a99;
    margin-bottom: 8px;
    text-align: center;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #1a2030;
    border: 1px solid #2a3040;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.sidebar-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a5568;
    margin-bottom: 10px;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #2a2f3e;
    margin: 24px 0;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1a2030;
    border: 2px dashed #2a3f5f;
    border-radius: 12px;
    padding: 12px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3550; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Helper: parse answer key text
# ════════════════════════════════════════════════════════════
def parse_key(text: str) -> dict:
    key = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            digits = "".join(c for c in k if c.isdigit())
            if digits:
                key[int(digits)] = v.strip()
    return key


# ════════════════════════════════════════════════════════════
# Helper: score bar color
# ════════════════════════════════════════════════════════════
def bar_color(pct: int) -> str:
    if pct >= 80: return "#4ade80"
    if pct >= 50: return "#facc15"
    return "#f87171"


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px'>
        <div style='font-size:2rem'>📝</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e8eaf0'>VisionGrade</div>
        <div style='font-size:0.78rem; color:#6b7a99'>Exam Grader & Image Enhancer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Answer Key</div>', unsafe_allow_html=True)
    ans_key_text = st.text_area(
        label     = "One answer per line (Q1: answer)",
        value     = "Q1: apple\nQ2: 1423\nQ3: Cairo\nQ4: blue\nQ5: 99",
        height    = 180,
        label_visibility = "collapsed"
    )
    st.caption("Format: `Q1: answer` — one per line")

    st.markdown("<hr style='border-color:#2a2f3e; margin:20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Pipeline Stages</div>', unsafe_allow_html=True)

    pipeline_stages = [
        ("Stage 2", "Deskew — rotation fix"),
        ("Stage 3", "Denoise — Gaussian + Median"),
        ("Stage 4", "Lighting normalization"),
        ("Stage 5", "Adaptive threshold"),
        ("Stage 5b","Remove small noise"),
        ("Stage 6", "Morphology — clean text"),
    ]
    for (num, desc) in pipeline_stages:
        st.markdown(f"""
        <div class="stage-row">
            <div class="stage-dot"></div>
            <div class="stage-name"><b style='color:#60a5fa'>{num}</b> · {desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2f3e; margin:20px 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; color:#3a4560; text-align:center">Computer Vision Project · 2025</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>VisionGrade</h1>
    <p>Upload a handwritten answer sheet — the system enhances it, reads the answers, and grades them automatically.</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ════════════════════════════════════════════════════════════
col_up, col_btn = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Drop your answer sheet here",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    run_btn = st.button("✨ Enhance & Grade", disabled=uploaded_file is None)


# ════════════════════════════════════════════════════════════
# SHOW ORIGINAL IMAGE PREVIEW IMMEDIATELY AFTER UPLOAD
# ════════════════════════════════════════════════════════════
if uploaded_file and not run_btn:
    file_bytes = uploaded_file.read()
    original_img = load_grayscale_from_bytes(file_bytes)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="img-label">Preview — Original Upload</div>', unsafe_allow_html=True)
    _, pc, _ = st.columns([1, 2, 1])
    with pc:
        st.image(original_img, use_container_width=True)


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE — runs when button is clicked
# ════════════════════════════════════════════════════════════
if uploaded_file and run_btn:
    file_bytes = uploaded_file.read()
    original_img = load_grayscale_from_bytes(file_bytes)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── PART A: Enhancement ──────────────────────────────────
    with st.status("Running enhancement pipeline...", expanded=True) as status:
        st.write("Stage 2 — Deskew (rotation fix)...")
        s2 = stage2_deskew_hough(original_img)

        st.write("Stage 3 — Denoise (Gaussian + Median)...")
        s3 = stage3_denoise(s2)

        st.write("Stage 4 - background lighting normalization...")
        s4 = stage4_clahe(s3)

        st.write("Stage 5 — Adaptive thresholding...")
        s5 = stage5_threshold(s4)
        s5_clean = remove_small_noise(s5)
        ocr_img = invert_binary(s5_clean)

        st.write("Stage 6 - extra-clean display output...")
        enhanced_img = stage6_morphology(s5_clean)

        status.update(label="Enhancement complete", state="complete", expanded=False)

    # ── PART B: OCR & Grading ────────────────────────────────
    with st.status("Running OCR and grading...", expanded=True) as status:
        st.write("Saving enhanced image for OCR engine...")
        temp_path = "temp_enhanced.jpg"
        save_image(temp_path, ocr_img)

        st.write("Recognizing text with OCR engine...")
        ocr_results = recognize_text_from_image(temp_path)

        st.write("Parsing answer key...")
        correct_ans = parse_key(ans_key_text)

        st.write("Matching recognized answers against key...")
        recognized_ans = parse_ocr_results(ocr_results, num_questions=len(correct_ans))
        grade_report   = grade_answers(recognized_ans, correct_ans)

        status.update(label="Grading complete", state="complete", expanded=False)

    # ════════════════════════════════════════════════════════
    # RESULTS SECTION
    # ════════════════════════════════════════════════════════
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    score_pct  = grade_report['score_percent']
    correct_n  = grade_report['correct_count']
    total_n    = grade_report['total_questions']
    passed     = score_pct >= 50

    # ── Score overview ───────────────────────────────────────
    st.markdown("### 📊 Grade Summary")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Score</div>
            <div class="metric-value" style="color:{bar_color(score_pct)}">{score_pct}%</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill"
                     style="width:{score_pct}%; background:{bar_color(score_pct)}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Correct Answers</div>
            <div class="metric-value">{correct_n}</div>
            <div class="metric-sub">out of {total_n} questions</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Wrong Answers</div>
            <div class="metric-value" style="color:#f87171">{total_n - correct_n}</div>
            <div class="metric-sub">questions missed</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        badge = '<span class="badge-pass">✓ PASS</span>' if passed else '<span class="badge-fail">✗ FAIL</span>'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Result</div>
            <div style="margin-top:12px">{badge}</div>
            <div class="metric-sub" style="margin-top:10px">Pass threshold: 50%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Image comparison ─────────────────────────────────────
    st.markdown("### 🖼️ Image Enhancement")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown('<div class="img-label">Original — Noisy Input</div>',
                    unsafe_allow_html=True)
        st.image(original_img, use_container_width=True)

    with img_col2:
        st.markdown('<div class="img-label">Enhanced — Clean Output</div>',
                    unsafe_allow_html=True)
        st.image(enhanced_img, use_container_width=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown("### Preprocessing Stages")
    stage_outputs = [
        ("Stage 1 - Grayscale input", original_img),
        ("Stage 2 - Deskew", s2),
        ("Stage 3 - Denoise", s3),
        ("Stage 4 - Lighting normalized", s4),
        ("Stage 5 - Text threshold", s5),
        ("Stage 5b - Small noise removed", s5_clean),
        ("OCR-safe image", ocr_img),
        ("Stage 6 - Final clean output", enhanced_img),
    ]

    stage_cols = st.columns(3)
    for idx, (label, image) in enumerate(stage_outputs):
        with stage_cols[idx % 3]:
            st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Per-question breakdown ───────────────────────────────
    st.markdown("### 📝 Question Breakdown")

    # Table header
    st.markdown("""
    <div style="display:flex; gap:12px; padding:6px 16px; margin-bottom:4px">
        <div style="width:36px; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; color:#3a4a6a">Q#</div>
        <div style="flex:1; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; color:#3a4a6a">Recognized</div>
        <div style="flex:1; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; color:#3a4a6a">Correct Answer</div>
        <div style="width:50px; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; color:#3a4a6a; text-align:right">Match</div>
        <div style="width:70px; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; color:#3a4a6a; text-align:center">Result</div>
    </div>
    """, unsafe_allow_html=True)

    details = grade_report.get('details', [])

    if details:
        for row in details:
            # Exact keys from grader.py:
            # question, recognized_answer, correct_answer, is_correct, normalized_recognized
            q_num    = row.get('question', '?')
            recog    = row.get('recognized_answer', row.get('recognized', ''))
            correct  = row.get('correct_answer',    row.get('correct',    '?'))
            ok       = row.get('is_correct', False)
            norm_rec = row.get('normalized_recognized', '')

            badge        = '<span class="q-badge-ok">✓ Correct</span>' if ok else '<span class="q-badge-err">✗ Wrong</span>'
            recog_color  = "#4ade80" if ok else "#f87171"
            recog_html   = recog if recog else "<span style='color:#3a4a6a'>—</span>"
            norm_html    = f"<div style='font-size:0.72rem;color:#4a5568;margin-top:3px'>normalized: <i>{norm_rec}</i></div>" if norm_rec else ""

            st.markdown(f"""
            <div class="q-row">
                <div class="q-num">Q{q_num}</div>
                <div class="q-recognized" style="color:{recog_color}">{recog_html}{norm_html}</div>
                <div class="q-correct">{correct}</div>
                <div class="q-sim">—</div>
                {badge}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No detailed breakdown available from grader.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Download enhanced image ──────────────────────────────
    st.markdown("### 💾 Download")
    _, dl_col, _ = st.columns([2, 1, 2])
    with dl_col:
        success, buf = True, encode_image(enhanced_img, fmt="JPEG")
        if success:
            st.download_button(
                label     = "⬇ Download Enhanced Image",
                data      = buf,
                file_name = "enhanced_output.jpg",
                mime      = "image/jpeg",
                use_container_width=True
            )
