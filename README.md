# VisionGrade — Automatic Exam Answer Sheet Grader
### Text Image Preprocessing & Enhancement Pipeline

---

## What it does
Takes a **noisy text image** as input and outputs a **step-by-step visualization** showing every enhancement stage plus the final clean image.

---

## CV Techniques Used

| Rule | Stage | Technique |
|------|-------|-----------|
| Rule 2 | Stage 2 | Gaussian Filter — removes blur and Gaussian noise |
| Rule 2 | Stage 3 | Mean Filter — removes salt-and-pepper noise |
| Rule 2 | Stage 4 | Low-pass FFT Filter — removes periodic noise |
| Rule 2 | Stage 5 | CLAHE — fixes uneven lighting |
| Rule 2 | Stage 6 | Contrast Stretching + Gamma Correction |
| Rule 2 | Stage 7 | Deskew — corrects rotation/tilt |
| Rule 1 | Stage 8a | Sobel Edge Detection |
| Rule 1 | Stage 8b | Prewitt Edge Detection |
| Rule 1 | Stage 8c | Canny Edge Detection |
| Both   | Stage 9 | Otsu Thresholding — final clean binary image |

---

## Setup & Run

```bash
pip install -r requirements.txt

# Run on your own image
python enhancer.py input/your_image.jpg

# Run demo (creates synthetic noisy image automatically)
python enhancer.py
```

---

## Output

All results saved to `output/` folder:

| File | What it shows |
|------|--------------|
| `pipeline_all_stages.jpg` | Full step-by-step grid (main output) |
| `01_original.jpg` | Original noisy input |
| `02_gaussian.jpg` | After Gaussian filter |
| `03_mean.jpg` | After Mean filter |
| `04_lowpass_fft.jpg` | After Low-pass FFT filter |
| `05_clahe.jpg` | After CLAHE lighting fix |
| `06_contrast_gamma.jpg` | After contrast + gamma |
| `07_deskewed.jpg` | After rotation correction |
| `08a_sobel.jpg` | Sobel edge map |
| `08b_prewitt.jpg` | Prewitt edge map |
| `08c_canny.jpg` | Canny edge map |
| `09_final_binary.jpg` | Final clean binary image |
