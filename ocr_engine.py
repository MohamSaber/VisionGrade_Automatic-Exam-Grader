from rapidocr_onnxruntime import RapidOCR
from scratch_image import load_grayscale, resize_bilinear

# Global OCR engine
_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = RapidOCR()

    return _reader


def recognize_text_from_image(image_path):
    reader = get_reader()

    try:
        img = load_grayscale(image_path)
    except Exception as exc:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize large images for faster OCR
    h, w = img.shape
    max_width = 1800

    if w > max_width:
        scale = max_width / w
        img = resize_bilinear(img, scale)

    # OCR
    result, _ = reader(img)

    formatted_results = []

    if result:
        for line in result:
            bbox = line[0]
            text = line[1]
            conf = line[2]

            if conf < 0.35:
                continue

            formatted_results.append((bbox, text, conf))

    return formatted_results


# Example usage
if __name__ == "__main__":
    results = recognize_text_from_image("image.jpg")

    for bbox, text, conf in results:
        print(f"Text: {text}")
        print(f"Confidence: {conf}")
        print(f"BBox: {bbox}")
        print("-" * 50)
