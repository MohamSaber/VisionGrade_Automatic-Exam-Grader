"""
Automatic Exam Answer Sheet Grader

Part A: image enhancement implemented from scratch with NumPy.
Part B: OCR recognition and grading against an answer key.
"""
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from grader import parse_answer_key, grade_answers, print_grade_report
from ocr_engine import recognize_text_from_image
from scratch_image import (
    adaptive_text_threshold,
    background_normalize,
    clean_document_output,
    estimate_skew_angle,
    gaussian_blur,
    invert_binary,
    load_grayscale,
    median_filter,
    remove_small_components,
    rotate_image,
    save_image,
)


# ---------------------------------------------------------------------------
# PART A: SCRATCH IMAGE ENHANCEMENT PIPELINE
# ---------------------------------------------------------------------------

def stage1_load(path: str):
    try:
        gray = load_grayscale(path)
    except Exception as exc:
        raise FileNotFoundError(f"Cannot load: {path}") from exc
    print("[Stage 1] Loaded image and converted to grayscale")
    return gray


def stage2_deskew(img):
    """Deskew by scanning angles and choosing the sharpest text-line projection."""
    angle = estimate_skew_angle(img, max_angle=10, step=0.5)
    rotated = rotate_image(img, -angle)
    print(f"[Stage 2] Rotation corrected: {-angle:.2f} degrees")
    return rotated


def stage2_deskew_hough(img):
    """Compatibility wrapper for the old UI name; implemented from scratch."""
    return stage2_deskew(img)


def stage3_denoise(img):
    denoise = median_filter(img, size=3)
    blur = gaussian_blur(denoise)
    print("[Stage 3] Median denoise + Gaussian blur applied")
    return blur


def stage4_clahe(img):
    print("[Stage 4] Background lighting normalized")
    return background_normalize(img, block_size=75, strength=3.5)


def stage5_threshold(img):
    print("[Stage 5] Adaptive text threshold applied")
    return adaptive_text_threshold(img, block_size=51, offset=30)


def remove_small_noise(img):
    return remove_small_components(img, min_area=80)


def stage6_morphology(img):
    print("[Stage 6] Extra-clean display output generated")
    return clean_document_output(img, min_area=300)


def run_scratch_pipeline(image_path):
    print("\n" + "=" * 50)
    print("  PART A: IMAGE FILTERING AND ENHANCEMENT (FROM SCRATCH)")
    print("=" * 50)

    os.makedirs("output", exist_ok=True)
    t0 = time.time()

    s1 = stage1_load(image_path)
    s2 = stage2_deskew_hough(s1)
    s3 = stage3_denoise(s2)
    s4 = stage4_clahe(s3)
    s5 = stage5_threshold(s4)
    s5_clean = remove_small_noise(s5)
    s6 = stage6_morphology(s5_clean)
    s_ocr = invert_binary(s5_clean)

    save_image("output/1_gray.jpg", s1)
    save_image("output/2_deskew.jpg", s2)
    save_image("output/3_denoise.jpg", s3)
    save_image("output/4_clahe.jpg", s4)
    save_image("output/5_threshold.jpg", s5)
    save_image("output/5b_ocr_safe.jpg", s_ocr)

    final_path = "output/6_final.jpg"
    save_image(final_path, s6)
    ocr_path = "output/5b_ocr_safe.jpg"

    print(f"\nPipeline completed in {time.time() - t0:.2f}s")
    print(f"Saved extra-clean display image to: {final_path}")
    print(f"Saved OCR-safe image to: {ocr_path}")
    return ocr_path


# ---------------------------------------------------------------------------
# PART B: ROBUST OCR PARSING AND GRADING
# ---------------------------------------------------------------------------

def get_bbox_center_y(bbox):
    return sum(p[1] for p in bbox) / len(bbox)


def get_bbox_center_x(bbox):
    return sum(p[0] for p in bbox) / len(bbox)


def compute_adaptive_threshold(results):
    if len(results) < 2:
        return 50
    centers = sorted([get_bbox_center_y(bbox) for bbox, _, _ in results])
    gaps = sorted(
        [
            centers[i + 1] - centers[i]
            for i in range(len(centers) - 1)
            if centers[i + 1] - centers[i] > 5
        ]
    )
    if not gaps:
        return 50
    return max(30, gaps[len(gaps) // 2] / 2.5)


def group_detections_into_rows(results, y_threshold=None):
    if not results:
        return []
    if y_threshold is None:
        y_threshold = compute_adaptive_threshold(results)

    items = [
        (get_bbox_center_y(bbox), get_bbox_center_x(bbox), bbox, text, conf)
        for bbox, text, conf in results
    ]
    items.sort(key=lambda x: x[0])

    rows = []
    current_row = [items[0]]
    current_y = items[0][0]
    for item in items[1:]:
        if abs(item[0] - current_y) <= y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
            current_y = item[0]
    rows.append(current_row)

    sorted_rows = []
    for row in rows:
        row.sort(key=lambda x: x[1])
        sorted_rows.append([(item[2], item[3], item[4]) for item in row])
    return sorted_rows


SUBSCRIPT_MAP = {"s": "5", "z": "3", "l": "1", "i": "1"}


def parse_ocr_results(results, num_questions=5):
    rows = group_detections_into_rows(results)
    recognized = {}
    pending_rows = []

    for i, row in enumerate(rows):
        row_texts = [text for _, text, _ in row]
        combined = " ".join(row_texts)

        q_match = re.search(r"[QqKkR]\s*(\d+)", combined)
        if q_match:
            q_num = int(q_match.group(1))
            if q_num > num_questions and str(q_num)[0].isdigit():
                first_digit = int(str(q_num)[0])
                if 1 <= first_digit <= num_questions:
                    q_num = first_digit
            answer = re.sub(r"^[\s;:),]+", "", combined[q_match.end():]).strip()
            if answer:
                recognized[q_num] = answer
            continue

        q_match = re.search(r"[Qq]([a-zA-Z])", combined)
        if q_match and q_match.group(1).lower() in SUBSCRIPT_MAP:
            q_num = int(SUBSCRIPT_MAP[q_match.group(1).lower()])
            answer = re.sub(r"^[\s;:),]+", "", combined[q_match.end():]).strip()
            if answer:
                recognized[q_num] = answer
            continue

        pending_rows.append((i, combined))

    still_pending = []
    for i, combined in pending_rows:
        num_match = re.match(r"(\d+)\s+(.+)", combined.strip())
        if num_match:
            q_num = int(num_match.group(1))
            answer = re.sub(r"^[\s;:),]+", "", num_match.group(2)).strip()
            if q_num not in recognized and q_num <= num_questions:
                recognized[q_num] = answer
                continue
        still_pending.append((i, combined))

    missing_q = sorted(set(range(1, num_questions + 1)) - set(recognized.keys()))
    unassigned_answers = []
    for i, combined in still_pending:
        text = combined.strip()
        if re.search(r"[QqKk]", text):
            q_match = re.search(r"[QqKk]", text)
            answer = re.sub(r"^[\s;:),]+", "", text[q_match.end():]).strip()
        else:
            answer = text

        if answer:
            unassigned_answers.append((i, answer))

    for idx, (_, answer) in enumerate(unassigned_answers):
        if idx < len(missing_q):
            recognized[missing_q[idx]] = answer

    return recognized


def run_grading_pipeline(processed_image_path, answer_key_path):
    print("\n" + "=" * 50)
    print("  PART B: HANDWRITING RECOGNITION AND GRADING")
    print("=" * 50)

    print("\n[1/3] Parsing answer key...")
    correct_answers = parse_answer_key(answer_key_path)
    num_questions = len(correct_answers)

    print(f"[2/3] Running OCR on processed image: {processed_image_path}")
    results = recognize_text_from_image(processed_image_path)
    recognized_answers = parse_ocr_results(results, num_questions=num_questions)

    print("\n[3/3] Grading answers...")
    result = grade_answers(recognized_answers, correct_answers)
    print_grade_report(result)
    return result


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    image_path = "test2.jpeg"
    answer_key_path = "answer_key.txt"

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    if len(sys.argv) >= 3:
        answer_key_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(answer_key_path):
        print(f"Error: Answer key file not found: {answer_key_path}")
        sys.exit(1)

    print("=" * 50)
    print("  AUTOMATIC EXAM ANSWER SHEET GRADER")
    print("=" * 50)
    print(f"  Image Input: {image_path}")
    print(f"  Answer Key:  {answer_key_path}")

    final_processed_path = run_scratch_pipeline(image_path)
    run_grading_pipeline(final_processed_path, answer_key_path)


if __name__ == "__main__":
    main()
