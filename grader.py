"""
grader.py — Answer sheet grading logic.
Parses answer key, compares with OCR results, outputs grade.
Includes fuzzy matching to handle common OCR recognition errors.
"""
import re


def parse_answer_key(file_path):
    """Parse an answer key text file.
    
    Expected format (one per line):
        Q1 : apple
        Q2 : 1423
        Q3 : Cairo
    
    Returns:
        dict mapping question number (int) to correct answer (str)
    """
    answers = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Support formats: "Q1 : answer", "Q1: answer", "Q1 answer"
            if ":" in line:
                parts = line.split(":", 1)
                q_part = parts[0].strip()
                a_part = parts[1].strip()
            else:
                parts = line.split(None, 1)
                q_part = parts[0].strip()
                a_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract question number
            q_num = ""
            for ch in q_part:
                if ch.isdigit():
                    q_num += ch
            
            if q_num:
                answers[int(q_num)] = a_part
    
    return answers


def normalize_ocr_text(text):
    """Normalize OCR text to handle common recognition errors.
    
    Common EasyOCR mistakes on handwriting:
    - Extra spaces within words: "Cair 0" -> "Cair0"
    - Pipe | read as 1: "|43" -> "143"
    - Trailing/leading junk characters
    - Case differences
    
    Returns:
        Cleaned, lowercase text for comparison
    """
    cleaned = text.lower().strip()
    
    # Replace common OCR character confusions

    cleaned = re.sub(r'[^a-z0-9]', '', cleaned)
    cleaned = cleaned.replace("|", "1")   # pipe -> one
    cleaned = cleaned.replace("{", "(")
    cleaned = cleaned.replace("}", ")")
    cleaned = cleaned.replace("[", "(")
    cleaned = cleaned.replace("]", ")")
    
    # Remove all spaces (handles "Cair 0" -> "Cair0")
    cleaned = cleaned.replace(" ", "")
    
    # Lowercase for case-insensitive comparison
    cleaned = cleaned.lower()
    
    return cleaned


def strings_match_with_ocr_confusion(s1, s2):
    """Check if two normalized strings match when accounting for 
    common digit/letter confusions in OCR.
    
    OCR commonly confuses these character pairs:
    - 0 <-> o (zero vs letter O)
    - 1 <-> l <-> i (one vs lowercase L vs uppercase I)
    - 5 <-> s
    - 8 <-> b
    - 2 <-> z
    - 6 <-> g
    - 9 <-> q
    
    Both strings should already be normalized (lowercase, no spaces).
    """
    if len(s1) != len(s2):
        return False
    
    # Characters that OCR commonly mixes up
    confusion_groups = [
        {'0', 'o'},
        {'1', 'l', 'i'},
        {'5', 's'},
        {'8', 'b'},
        {'2', 'z'},
        {'6', 'g', '9'},
        {'9', 'q', 'g'},
    ]
    
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            continue
        # Check if they belong to the same confusion group
        matched = False
        for group in confusion_groups:
            if c1 in group and c2 in group:
                matched = True
                break
        if not matched:
            return False
    
    return True


def levenshtein_distance(s1, s2):
    """Compute edit distance for short OCR strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (c1 != c2)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def fuzzy_match(recognized, correct, case_sensitive=False):
    """Check if recognized text matches the correct answer with OCR tolerance.
    
    Applies multiple matching strategies to handle common OCR errors:
    
    1. Exact match (after basic cleanup)
    2. Normalized match (strip spaces, fix pipe->1)
    3. OCR confusion match (0<->o, 1<->l, etc.)
    4. Containment check (for partial recognition)
    
    Returns:
        True if the texts are considered a match
    """
    # Basic cleanup
    rec_clean = recognized.strip()
    cor_clean = correct.strip()
    
    if not case_sensitive:
        rec_lower = rec_clean.lower()
        cor_lower = cor_clean.lower()
    else:
        rec_lower = rec_clean
        cor_lower = cor_clean
    
    # 1. Exact match
    if rec_lower == cor_lower:
        return True
    
    # 2. Normalized match (handles "Cair 0" spaces, "|" -> "1")
    rec_norm = normalize_ocr_text(recognized)
    cor_norm = normalize_ocr_text(correct)
    
    if rec_norm == cor_norm:
        return True
    
    # 3. OCR digit/letter confusion match (0<->o, 1<->l, etc.)
    if strings_match_with_ocr_confusion(rec_norm, cor_norm):
        return True

    if len(cor_norm) >= 4 and len(rec_norm) >= 4:
        if levenshtein_distance(rec_norm, cor_norm) <= 1:
            return True
    
    if cor_norm and cor_norm in rec_norm:
        return True
    
    # 4. Containment check — if one is a substring of the other
    #    (handles cases where OCR adds or drops a character)
    if len(cor_norm) >= 3 and len(rec_norm) >= 3:
        if cor_norm in rec_norm or rec_norm in cor_norm:
            return True
    
    return False


def grade_answers(recognized_answers, correct_answers, case_sensitive=False):
    """Compare recognized answers with correct answers and compute grade.
    
    Uses fuzzy matching to handle common OCR recognition errors.
    
    Args:
        recognized_answers: dict {question_num: recognized_text}
        correct_answers: dict {question_num: correct_text}
        case_sensitive: whether comparison is case-sensitive
    
    Returns:
        dict with:
            - total_questions: number of questions
            - correct_count: number correct
            - score_percent: percentage score
            - details: list of per-question results
    """
    details = []
    correct_count = 0
    total = len(correct_answers)
    
    for q_num in sorted(correct_answers.keys()):
        correct = correct_answers[q_num]
        recognized = recognized_answers.get(q_num, "")
        
        # Use fuzzy matching for OCR tolerance
        is_correct = fuzzy_match(recognized, correct, case_sensitive)
        
        if is_correct:
            correct_count += 1
        
        details.append({
            "question": q_num,
            "correct_answer": correct,
            "recognized_answer": recognized,
            "is_correct": is_correct,
            "normalized_recognized": normalize_ocr_text(recognized),
            "normalized_correct": normalize_ocr_text(correct),
        })
    
    score_percent = (correct_count / total * 100) if total > 0 else 0
    
    return {
        "total_questions": total,
        "correct_count": correct_count,
        "score_percent": round(score_percent, 1),
        "details": details
    }


def print_grade_report(result):
    """Print a formatted grade report to console."""
    print("=" * 50)
    print("         EXAM GRADING REPORT")
    print("=" * 50)
    print()
    
    for item in result["details"]:
        status = "CORRECT" if item["is_correct"] else "WRONG"
        mark = "+" if item["is_correct"] else "x"
        print(f"  Q{item['question']:>2}: {mark} {status}")
        print(f"        Expected:   {item['correct_answer']}")
        print(f"        Recognized: {item['recognized_answer']}")
        if not item["is_correct"]:
            print(f"        (normalized: \"{item['normalized_recognized']}\" vs \"{item['normalized_correct']}\")")
        print()
    
    print("-" * 50)
    print(f"  Score: {result['correct_count']} / {result['total_questions']}"
          f"  ({result['score_percent']}%)")
    print("=" * 50)
