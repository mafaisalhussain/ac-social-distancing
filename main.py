"""
Assassin's Creed Social Distancing Detector
Lab: Social Distancing Detection using Pose Estimation
Jetson Nano — Dusty's jetson-inference poseNet + OpenCV AC HUD
"""

import sys
import cv2
import time
import argparse
import jetson.inference
import jetson.utils
from overlay import ACOverlay
from distance import check_distances
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    DISTANCE_THRESHOLD, POSENET_MODEL, KEYPOINT_THRESHOLD
)

def main():
    print("\n[ANIMUS] Synchronizing with the Grid...")
    print(f"[ANIMUS] Loading poseNet model: {POSENET_MODEL}")

    # Load poseNet from jetson-inference
    net = jetson.inference.poseNet(POSENET_MODEL, threshold=KEYPOINT_THRESHOLD)

    # Open camera via jetson.utils
    camera = jetson.utils.videoSource(f"/dev/video{CAMERA_INDEX}")
    display = None  # we use OpenCV window instead

    overlay = ACOverlay(FRAME_WIDTH, FRAME_HEIGHT)

    print("[ANIMUS] Eagle Vision ONLINE")
    print("[CONTROLS] Q = Quit | E = Toggle Eagle Vision | S = Screenshot | R = Reset log\n")

    eagle_vision_on = True
    frame_count     = 0
    fps_time        = time.time()
    fps             = 0.0

    while True:
        # Capture CUDA image from jetson.utils
        cuda_img = camera.Capture()
        if cuda_img is None:
            print("[ERROR] Failed to capture frame.")
            break

        # Run poseNet inference
        poses = net.Process(cuda_img)

        # Convert CUDA image to OpenCV BGR numpy array
        bgr_img = jetson.utils.cudaToNumpy(cuda_img)
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_RGBA2BGR)

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - fps_time)
            fps_time = now

        # Compute distances + find violations
        persons, violations = check_distances(poses, bgr_img.shape[1], bgr_img.shape[0])

        # Draw AC HUD
        output = overlay.draw(bgr_img, poses, persons, violations, fps, eagle_vision_on)

        cv2.imshow("ANIMUS // SOCIAL DISTANCING MONITOR", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[ANIMUS] Desynchronizing...")
            break
        elif key == ord('e') or key == ord('E'):
            eagle_vision_on = not eagle_vision_on
            print(f"[EAGLE VISION] {'ON' if eagle_vision_on else 'OFF'}")
        elif key == ord('s') or key == ord('S'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"[SCREENSHOT] Saved -> {fname}")
        elif key == ord('r') or key == ord('R'):
            overlay.reset_log()
            print("[LOG] Cleared.")

    cv2.destroyAllWindows()
    print("[ANIMUS] Session ended. Nothing is true, everything is permitted.\n")

if __name__ == "__main__":
    main()
