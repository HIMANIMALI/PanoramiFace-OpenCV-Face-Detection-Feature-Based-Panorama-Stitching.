import cv2
import numpy as np
import argparse
import json
import os
from typing import List
from utils import show_image


def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panorama construction"
    )
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result"
    )
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panorama image"
    )
    return parser.parse_args()


def stitch(inp_path: str, imgmark: str, N: int = 4, savepath: str = '') -> np.ndarray:
    
    # 1) Load images
    imgs = [cv2.imread(os.path.join(inp_path, f"{imgmark}_{i}.png")) for i in range(1, N+1)]

    # 2) ORB + matcher
    orb = cv2.ORB_create(2000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 3) Detect keypoints and descriptors
    kps, descs = [], []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        kps.append(kp)
        descs.append(des)

    # 4) Build raw overlap matrix with a tighter threshold
    overlap = np.zeros((N, N), dtype=int)
    MATCH_THRESHOLD = 30  # require at least 30 good matches to declare overlap
    for i in range(N):
        for j in range(N):
            matches = bf.knnMatch(descs[i], descs[j], k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            overlap[i, j] = 1 if len(good) >= MATCH_THRESHOLD else 0

    # 5) Enforce symmetry and self-overlap
    for i in range(N):
        overlap[i, i] = 1
        for j in range(i+1, N):
            val = max(overlap[i, j], overlap[j, i])
            overlap[i, j] = overlap[j, i] = val

    # --- Panorama stitching ---
    # Compute homographies to the first image
    H_matrices = [np.eye(3)]
    for i in range(1, N):
        matches = bf.knnMatch(descs[i], descs[0], k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        src = np.float32([kps[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kps[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        H_matrices.append(H)

    # Determine canvas size
    h0, w0 = imgs[0].shape[:2]
    corners = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
    all_corners = [corners]
    for i in range(1, N):
        warped = cv2.perspectiveTransform(corners, H_matrices[i])
        all_corners.append(warped)
    all_corners = np.concatenate(all_corners, axis=0)
    min_xy = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    max_xy = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    offset = [-min_xy[0], -min_xy[1]]
    canvas_w = max_xy[0] - min_xy[0]
    canvas_h = max_xy[1] - min_xy[1]

    # Warp and blend
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        H = H_matrices[i].copy()
        H[0, 2] += offset[0]
        H[1, 2] += offset[1]
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h))
        mask = (warped > 0)
        panorama[mask] = warped[mask]

    # Save final panorama
    cv2.imwrite(savepath, panorama)

    return overlap


if __name__ == "__main__":
    args = parse_args()
    overlap_mat = stitch(args.input_path, 't2', N=4, savepath=args.output_panaroma)
    with open(args.output_overlap, 'w') as f:
        json.dump(overlap_mat.tolist(), f)