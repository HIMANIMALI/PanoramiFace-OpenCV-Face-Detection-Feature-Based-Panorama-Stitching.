import cv2
import numpy as np
import argparse
import json
import os
import sys
import math

from typing import Dict, List
from utils import show_image

#_ — Load Haar cascade classifier once —_
cascade_file_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_file_path)
if face_detector.empty():
    raise IOError(f"Could not load Haar cascade from {cascade_file_path}")


def detect_faces(img: np.ndarray) -> List[List[float]]:
    
    face_boxes: List[List[float]] = []

    # to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect
    detections = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # format
    for x, y, w, h in detections:
        face_boxes.append([float(x), float(y), float(w), float(h)])

    return face_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/validation_folder/images",
        help="path to validation or test folder"
    )
    parser.add_argument(
        "--output", type=str, default="./result_task1.json",
        help="path to save JSON results"
    )
    return parser.parse_args()


def save_results(result_dict, filename):
    with open(filename, "w") as f:
        json.dump(result_dict, f, indent=4)


def check_output_format(faces, img, img_name):
    if not isinstance(faces, list):
        print(f'Wrong output type for image {img_name}!')
        return False
    for i, face in enumerate(faces):
        if not (isinstance(face, list) and len(face) == 4 and all(isinstance(n, float) for n in face)):
            print(f'Wrong BOX in {img_name}, face {i}!')
            return False
        x, y, w, h = face
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            print(f'Warning: BOX {i} in {img_name} out of bounds!')
    return True


def batch_detection(img_dir):
    results = {}
    for fname in sorted(os.listdir(img_dir)):
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(img)
        if not check_output_format(faces, img, fname):
            sys.exit(2)
        results[fname] = faces
    return results


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    detections = batch_detection(args.input_path)
    save_results(detections, args.output)


if __name__ == "__main__":
    main()
