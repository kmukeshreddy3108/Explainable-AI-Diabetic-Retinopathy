"""
Synthetic fundus image generator for testing and demonstration.

Generates realistic-looking retinal fundus images with controllable
ICDR grade (0–4) and quality impairments (blur, dark, bright, glare, poor FOV).
These are NOT real medical images — they are for pipeline testing only.
"""

import os
import cv2
import numpy as np


def generate_synthetic_fundus(
    grade: int = 0,
    quality: str = "good",
    img_size: int = 512,
    seed: int | None = None,
) -> np.ndarray:
    """
    Synthesise a retinal fundus image for testing.

    Parameters
    ----------
    grade : int
        ICDR DR severity grade (0–4).
    quality : str
        Quality impairment: 'good', 'blurry', 'dark', 'bright', 'glare', 'poor_fov'.
    img_size : int
        Output image dimension (square).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        BGR uint8 image of shape (img_size, img_size, 3).
    """
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    center = (img_size // 2, img_size // 2)
    radius = int(img_size * 0.45)

    # ── 1. Base fundus circle & background mask ──────────────────────────
    y, x = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius

    if quality == "poor_fov":
        offset_center = (int(img_size * 0.35), int(img_size * 0.4))
        dist_from_center = np.sqrt(
            (x - offset_center[0]) ** 2 + (y - offset_center[1]) ** 2
        )
        mask = dist_from_center <= radius

    # Reddish-orange gradient
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    red_channel = 180 - 60 * norm_dist + np.random.normal(0, 5, (img_size, img_size))
    green_channel = 90 - 40 * norm_dist + np.random.normal(0, 4, (img_size, img_size))
    blue_channel = 25 - 15 * norm_dist + np.random.normal(0, 2, (img_size, img_size))

    img[:, :, 2] = red_channel   # R
    img[:, :, 1] = green_channel # G
    img[:, :, 0] = blue_channel  # B
    img = np.clip(img, 0, 255)

    # ── 2. Optic disc (bright yellowish-white ellipse) ───────────────────
    disc_center = (int(center[0] - radius * 0.4), int(center[1]))
    disc_radius = int(radius * 0.18)
    cv2.circle(img, disc_center, disc_radius, (180, 235, 250), -1)
    cv2.circle(img, disc_center, int(disc_radius * 0.5), (210, 245, 255), -1)

    # ── 3. Fovea / macula (darker region) ────────────────────────────────
    fovea_center = (int(center[0] + radius * 0.25), int(center[1]))
    fovea_radius = int(radius * 0.15)
    fovea_mask = (
        np.sqrt((x - fovea_center[0]) ** 2 + (y - fovea_center[1]) ** 2)
        <= fovea_radius
    )
    img[fovea_mask, 2] *= 0.6
    img[fovea_mask, 1] *= 0.5

    # ── 4. Blood vessels (branching tree from optic disc) ────────────────
    vessel_img = np.zeros((img_size, img_size), dtype=np.uint8)
    angles = [-0.7, -0.2, 0.2, 0.7]
    for angle in angles:
        pts = [disc_center]
        curr_x, curr_y = disc_center
        step = radius * 0.1
        for s in range(6):
            curr_x += int(step * np.cos(angle) + np.random.randint(-5, 6))
            curr_y += int(
                step * np.sin(angle) * (1 if s % 2 == 0 else -1)
                + np.random.randint(-5, 6)
            )
            pts.append((curr_x, curr_y))
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        thickness = max(1, int(radius * 0.035))
        cv2.polylines(vessel_img, [pts_arr], False, 255, thickness)

        if len(pts) > 3:
            sub_pt1 = pts[2]
            sub_pt2 = (
                sub_pt1[0] + int(radius * 0.3 * np.cos(angle + 0.5)),
                sub_pt1[1] + int(radius * 0.3 * np.sin(angle + 0.5)),
            )
            cv2.line(vessel_img, sub_pt1, sub_pt2, 255, max(1, thickness - 2))

    vessel_mask_pixels = vessel_img > 0
    img[vessel_mask_pixels, 2] *= 0.35
    img[vessel_mask_pixels, 1] *= 0.20
    img[vessel_mask_pixels, 0] *= 0.10

    # ── 5. DR lesions based on ICDR grade ────────────────────────────────
    if grade >= 1:
        # Microaneurysms (small dark-red dots)
        for _ in range(8 * grade):
            rx = int(center[0] + np.random.uniform(-0.6, 0.6) * radius)
            ry = int(center[1] + np.random.uniform(-0.6, 0.6) * radius)
            cv2.circle(img, (rx, ry), np.random.randint(2, 4), (15, 10, 80), -1)

    if grade >= 2:
        # Hard exudates (bright yellow-white flecks near macula)
        for _ in range(12 * (grade - 1)):
            ex = int(fovea_center[0] + np.random.uniform(-0.35, 0.35) * radius)
            ey = int(fovea_center[1] + np.random.uniform(-0.35, 0.35) * radius)
            cv2.circle(
                img, (ex, ey), np.random.randint(3, 7), (180, 245, 255), -1
            )

    if grade >= 3:
        # Hemorrhages (blot hemorrhages — dark-red irregular shapes)
        for _ in range(10 * (grade - 2)):
            hx = int(center[0] + np.random.uniform(-0.6, 0.6) * radius)
            hy = int(center[1] + np.random.uniform(-0.6, 0.6) * radius)
            h_size = np.random.randint(6, 14)
            cv2.ellipse(
                img,
                (hx, hy),
                (h_size, int(h_size * np.random.uniform(0.5, 0.9))),
                np.random.randint(0, 180),
                0,
                360,
                (10, 10, 70),
                -1,
            )

    if grade == 4:
        # Neovascularisation (fine frond-like vessels near optic disc)
        # NOTE: illustrative only — matches blueprint's caution against over-claiming
        for _ in range(8):
            n_angle = np.random.uniform(0, 2 * np.pi)
            n_len = np.random.randint(15, 45)
            nx2 = int(disc_center[0] + n_len * np.cos(n_angle))
            ny2 = int(disc_center[1] + n_len * np.sin(n_angle))
            cv2.line(img, disc_center, (nx2, ny2), (20, 15, 90), 1)

    # Black outside the fundus circle
    img[~mask] = 0
    img = np.clip(img, 0, 255)

    # ── 6. Quality impairments ───────────────────────────────────────────
    if quality == "blurry":
        img = cv2.GaussianBlur(img, (31, 31), 10.0)
    elif quality == "dark":
        img = img * 0.3
    elif quality == "bright":
        img = np.clip(img * 1.8 + 40, 0, 255)
    elif quality == "glare":
        glare_center = (
            int(center[0] + radius * 0.3),
            int(center[1] - radius * 0.3),
        )
        cv2.circle(img, glare_center, int(radius * 0.3), (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (15, 15), 5.0)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_sample_dataset(output_dir: str) -> list[tuple[str, int, str]]:
    """
    Generate a full suite of test images covering all quality conditions
    and ICDR grades.  Returns list of (filepath, grade, quality) tuples.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = [
        ("normal_grade0_good.png", 0, "good"),
        ("mild_grade1_good.png", 1, "good"),
        ("moderate_grade2_good.png", 2, "good"),
        ("severe_grade3_good.png", 3, "good"),
        ("proliferative_grade4_good.png", 4, "good"),
        ("reject_blurry.png", 2, "blurry"),
        ("reject_dark.png", 2, "dark"),
        ("reject_glare.png", 1, "glare"),
        ("reject_poor_fov.png", 0, "poor_fov"),
    ]

    generated = []
    for idx, (fname, grade, qual) in enumerate(samples):
        fpath = os.path.join(output_dir, fname)
        img = generate_synthetic_fundus(grade=grade, quality=qual, img_size=512, seed=42 + idx)
        cv2.imwrite(fpath, img)
        generated.append((fpath, grade, qual))

    return generated


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/samples"
    paths = generate_sample_dataset(out_dir)
    print(f"Generated {len(paths)} sample fundus images in {out_dir}")
