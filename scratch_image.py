import math
from collections import deque

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def load_grayscale(path):
    img = Image.open(path)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def load_grayscale_from_bytes(data):
    from io import BytesIO

    img = Image.open(BytesIO(data))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def save_image(path, img):
    Image.fromarray(_as_uint8(img)).save(path)


def encode_image(img, fmt="JPEG"):
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(_as_uint8(img)).save(buf, format=fmt)
    return buf.getvalue()


def resize_bilinear(img, scale):
    if scale == 1:
        return img.copy()
    h, w = img.shape
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    y = np.linspace(0, h - 1, nh)
    x = np.linspace(0, w - 1, nw)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0

    top = (1 - wx)[None, :] * img[y0[:, None], x0[None, :]] + wx[None, :] * img[y0[:, None], x1[None, :]]
    bot = (1 - wx)[None, :] * img[y1[:, None], x0[None, :]] + wx[None, :] * img[y1[:, None], x1[None, :]]
    out = (1 - wy)[:, None] * top + wy[:, None] * bot
    return _as_uint8(out)


def otsu_threshold(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    best_var = -1.0
    best_t = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return best_t


def estimate_skew_angle(img, max_angle=10, step=0.5):
    work = img
    max_dim = max(work.shape)
    if max_dim > 450:
        work = resize_bilinear(work, 450 / max_dim)
    binary = (work < otsu_threshold(work)).astype(np.uint8)
    best_angle = 0.0
    best_score = -1.0
    angles = np.arange(-max_angle, max_angle + step, step)
    for angle in angles:
        rotated = rotate_image(binary * 255, angle, fill=0)
        projection = rotated.sum(axis=1).astype(np.float64)
        score = np.var(np.diff(projection))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def rotate_image(img, angle_degrees, fill=None):
    h, w = img.shape
    if fill is None:
        fill = int(np.median(img[[0, -1], :]))
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = xx - cx
    y = yy - cy
    src_x = cos_a * x + sin_a * y + cx
    src_y = -sin_a * x + cos_a * y + cy
    return bilinear_sample(img, src_x, src_y, fill)


def bilinear_sample(img, x, y, fill):
    h, w = img.shape
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    top = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
    bot = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
    out = (1 - wy) * top + wy * bot
    out[~valid] = fill
    return _as_uint8(out)


def median_filter(img, size=3):
    pad = size // 2
    padded = np.pad(img, pad, mode="edge")
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2)).astype(np.uint8)


def gaussian_blur(img):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    return convolve(img, kernel)


def convolve(img, kernel):
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(img.astype(np.float32), ((py, py), (px, px)), mode="edge")
    windows = sliding_window_view(padded, (kh, kw))
    out = np.tensordot(windows, kernel, axes=((-1, -2), (0, 1)))
    return _as_uint8(out)


def box_mean(img, size):
    pad = size // 2
    padded = np.pad(img.astype(np.float32), pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / (size * size)


def background_normalize(img, block_size=75, strength=3.0):
    background = box_mean(img, block_size)
    ink_response = np.maximum(background - img.astype(np.float32), 0)
    normalized = 255 - ink_response * strength
    return _as_uint8(normalized)


def clahe(img, clip_limit=1.2, tile_grid=(8, 8)):
    h, w = img.shape
    tiles_y, tiles_x = tile_grid
    tile_h = math.ceil(h / tiles_y)
    tile_w = math.ceil(w / tiles_x)
    maps = np.zeros((tiles_y, tiles_x, 256), dtype=np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, y1 = ty * tile_h, min((ty + 1) * tile_h, h)
            x0, x1 = tx * tile_w, min((tx + 1) * tile_w, w)
            tile = img[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
            limit = max(1.0, clip_limit * tile.size / 256.0)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            maps[ty, tx] = np.clip(255 * cdf / cdf[-1], 0, 255)

    out = np.empty_like(img, dtype=np.float32)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, y1 = ty * tile_h, min((ty + 1) * tile_h, h)
            x0, x1 = tx * tile_w, min((tx + 1) * tile_w, w)
            tile = img[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = maps[ty, tx, tile]
    return _as_uint8(out)


def adaptive_threshold_mean(img, block_size=11, c=3):
    pad = block_size // 2
    padded = np.pad(img.astype(np.float32), pad, mode="edge")
    windows = sliding_window_view(padded, (block_size, block_size))
    local_mean = windows.mean(axis=(-1, -2))
    return np.where(img < local_mean - c, 255, 0).astype(np.uint8)


def adaptive_text_threshold(img, block_size=51, offset=40):
    local_mean = box_mean(img, block_size)
    return np.where(img < local_mean - offset, 255, 0).astype(np.uint8)


def invert_binary(img):
    return np.where(img > 0, 0, 255).astype(np.uint8)


def remove_border_components(img, margin_x_ratio=0.16, margin_y_ratio=0.06, max_area=1500):
    binary = img > 0
    h, w = binary.shape
    margin_x = int(w * margin_x_ratio)
    margin_y = int(h * margin_y_ratio)
    visited = np.zeros_like(binary, dtype=bool)
    out = np.zeros_like(img)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for sy in range(h):
        for sx in range(w):
            if not binary[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            pixels = []
            min_y = max_y = sy
            min_x = max_x = sx
            while q:
                y, x = q.popleft()
                pixels.append((y, x))
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x)
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            area = len(pixels)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            near_border = (
                center_x < margin_x
                or center_x > w - margin_x
                or center_y < margin_y
                or center_y > h - margin_y
            )
            if near_border and area < max_area:
                continue
            ys, xs = zip(*pixels)
            out[ys, xs] = 255
    return out


def clean_document_output(img, min_area=300):
    cleaned = remove_small_components(img, min_area=min_area)
    cleaned = open_image(close_image(cleaned, kernel_size=2, iterations=1), kernel_size=2, iterations=1)
    cleaned = remove_border_components(cleaned)
    return invert_binary(cleaned)


def remove_small_components(img, min_area=50):
    binary = img > 0
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    out = np.zeros_like(img)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for sy in range(h):
        for sx in range(w):
            if not binary[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            pixels = []
            while q:
                y, x = q.popleft()
                pixels.append((y, x))
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(pixels) >= min_area:
                ys, xs = zip(*pixels)
                out[ys, xs] = 255
    return out


def erode(img, kernel_size=2, iterations=1):
    out = img.copy()
    for _ in range(iterations):
        padded = np.pad(out > 0, ((0, kernel_size - 1), (0, kernel_size - 1)), mode="edge")
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        out = np.where(windows.all(axis=(-1, -2)), 255, 0).astype(np.uint8)
    return out


def dilate(img, kernel_size=2, iterations=1):
    out = img.copy()
    for _ in range(iterations):
        padded = np.pad(out > 0, ((0, kernel_size - 1), (0, kernel_size - 1)), mode="edge")
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        out = np.where(windows.any(axis=(-1, -2)), 255, 0).astype(np.uint8)
    return out


def close_image(img, kernel_size=2, iterations=1):
    return erode(dilate(img, kernel_size, iterations), kernel_size, iterations)


def open_image(img, kernel_size=2, iterations=1):
    return dilate(erode(img, kernel_size, iterations), kernel_size, iterations)


def _as_uint8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
