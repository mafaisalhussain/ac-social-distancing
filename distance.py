"""
distance.py — Extract hip centers from poseNet poses and compute
Euclidean distances between all detected people.
"""

import math
from config import DISTANCE_THRESHOLD, LEFT_HIP, RIGHT_HIP, KEYPOINT_THRESHOLD


class Person:
    """One detected person with their hip center and pose."""
    def __init__(self, pose, center, idx):
        self.pose   = pose    # jetson.inference poseNet pose object
        self.center = center  # (cx, cy) pixel coords
        self.idx    = idx


class Violation:
    """Two people closer than the threshold."""
    def __init__(self, person_a, person_b, distance):
        self.person_a = person_a
        self.person_b = person_b
        self.distance = distance


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_hip_center(pose, frame_w, frame_h):
    """
    Compute the midpoint between left_hip and right_hip keypoints.
    Falls back to bounding box center if hips not visible.
    """
    try:
        left_kp  = pose.FindKeypoint(LEFT_HIP)
        right_kp = pose.FindKeypoint(RIGHT_HIP)

        left_ok  = left_kp  >= 0
        right_ok = right_kp >= 0

        if left_ok and right_ok:
            lx = pose.Keypoints[left_kp].x
            ly = pose.Keypoints[left_kp].y
            rx = pose.Keypoints[right_kp].x
            ry = pose.Keypoints[right_kp].y
            return (int((lx + rx) / 2), int((ly + ry) / 2))

        if left_ok:
            return (int(pose.Keypoints[left_kp].x), int(pose.Keypoints[left_kp].y))

        if right_ok:
            return (int(pose.Keypoints[right_kp].x), int(pose.Keypoints[right_kp].y))

    except Exception:
        pass

    # Fallback: bounding box center at 60% height (hip level)
    try:
        x1 = int(pose.Left)
        y1 = int(pose.Top)
        x2 = int(pose.Right)
        y2 = int(pose.Bottom)
        cx = (x1 + x2) // 2
        cy = int(y1 + (y2 - y1) * 0.60)
        return (cx, cy)
    except Exception:
        return (frame_w // 2, frame_h // 2)


def check_distances(poses, frame_w, frame_h):
    """
    Build Person list from poseNet poses and find all violating pairs.
    Returns: (persons list, violations list)
    """
    persons = []
    for i, pose in enumerate(poses):
        center = get_hip_center(pose, frame_w, frame_h)
        persons.append(Person(pose, center, i))

    violations = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            d = euclidean(persons[i].center, persons[j].center)
            if d < DISTANCE_THRESHOLD:
                violations.append(Violation(persons[i], persons[j], d))

    return persons, violations
