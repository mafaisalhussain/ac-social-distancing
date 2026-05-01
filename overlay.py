
"""
overlay.py — Assassin's Creed Eagle Vision HUD for Social Distancing Detector
"""

import cv2
import numpy as np
import math
import time
from collections import deque
from config import (
    COLOR_GOLD, COLOR_SAFE, COLOR_VIOLATION,
    COLOR_HUD_TEXT, COLOR_GOLD_DIM,
    DISTANCE_THRESHOLD, SKELETON_PAIRS, KEYPOINT_THRESHOLD
)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(img, text, pos, font=FONT_SMALL, scale=0.45,
              color=None, thickness=1, shadow=True):
    color = color or COLOR_HUD_TEXT
    x, y = pos
    if shadow:
        cv2.putText(img, text, (x+1, y+1), font, scale,
                    (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel_bg(img, x, y, w, h, alpha=0.72):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (10, 8, 5), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_GOLD, 1, cv2.LINE_AA)
    cv2.line(img, (x, y), (x+w, y), COLOR_GOLD, 2, cv2.LINE_AA)


class ACOverlay:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.event_log        = deque(maxlen=8)
        self.session_start    = time.time()
        self.total_violations = 0
        self.prev_viol_count  = 0

    def reset_log(self):
        self.event_log.clear()

    def _log(self, text, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        self.event_log.appendleft(f"[{ts}][{level}] {text}")

    def draw(self, frame, poses, persons, violations, fps, eagle_vision_on):
        output = frame.copy()
        if eagle_vision_on:
            output = self._eagle_tint(output)
        if len(violations) > self.prev_viol_count:
            self.total_violations += len(violations) - self.prev_viol_count
            self._log(f"PROXIMITY BREACH - {len(violations)} PAIR(S)", "ALERT")
        self.prev_viol_count = len(violations)
        violating_ids = set()
        for v in violations:
            violating_ids.add(v.person_a.idx)
            violating_ids.add(v.person_b.idx)
        for person in persons:
            color = COLOR_VIOLATION if person.idx in violating_ids else COLOR_SAFE
            self._draw_skeleton(output, person.pose, color)
            self._draw_center(output, person.center, color, person.idx)
        for v in violations:
            self._draw_violation_line(output, v)
        self._draw_top_bar(output, fps, persons, violations)
        self._draw_left_panel(output, persons, violations, violating_ids)
        self._draw_event_log(output)
        self._draw_bottom_bar(output)
        self._draw_crosshair(output)
        self._draw_corner_decor(output)
        if violations:
            self._draw_alert_banner(output, violations)
        return output

    def _eagle_tint(self, frame):
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tint   = np.full_like(frame, (25, 15, 5), dtype=np.uint8)
        out    = cv2.addWeighted(gray_b, 0.55, tint, 0.15, 0)
        return cv2.addWeighted(out, 0.85, frame, 0.15, 0)

    def _draw_skeleton(self, img, pose, color):
        for (a, b) in SKELETON_PAIRS:
            try:
                ka = pose.FindKeypoint(a)
                kb = pose.FindKeypoint(b)
                if ka < 0 or kb < 0:
                    continue
                kpa = pose.Keypoints[ka]
                kpb = pose.Keypoints[kb]
                pa = (int(kpa.x), int(kpa.y))
                pb = (int(kpb.x), int(kpb.y))
                cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
            except Exception:
                continue
        for kp in pose.Keypoints:
            pt = (int(kp.x), int(kp.y))
            cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, (0,0,0), 1, cv2.LINE_AA)

    def _draw_center(self, img, center, color, idx):
        cx, cy = center
        cv2.circle(img, (cx, cy), 14, color, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy),  4, color, -1, cv2.LINE_AA)
        cv2.line(img, (cx-10, cy), (cx-5, cy),  color, 1, cv2.LINE_AA)
        cv2.line(img, (cx+5,  cy), (cx+10, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy-10), (cx, cy-5),  color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy+5),  (cx, cy+10), color, 1, cv2.LINE_AA)
        draw_text(img, f"S{idx+1:02d}", (cx+16, cy+5), scale=0.38, color=color)

    def _draw_violation_line(self, img, v):
        a = v.person_a.center
        b = v.person_b.center
        flash = abs(math.sin(time.time() * 4))
        col   = tuple(int(c * (0.5 + 0.5*flash)) for c in COLOR_VIOLATION)
        for p1, p2 in self._dash_points(a, b):
            cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
        mx = (a[0] + b[0]) // 2
        my = (a[1] + b[1]) // 2
        draw_text(img, f"{int(v.distance)}px", (mx+6, my), scale=0.42, color=COLOR_VIOLATION)

    def _dash_points(self, p1, p2, dash=12, gap=6):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(math.sqrt(dx*dx + dy*dy), 1)
        ux, uy = dx/length, dy/length
        segs, pos = [], 0
        while pos < length:
            s = (int(p1[0]+ux*pos), int(p1[1]+uy*pos))
            e = (int(p1[0]+ux*min(pos+dash,length)), int(p1[1]+uy*min(pos+dash,length)))
            segs.append((s, e))
            pos += dash + gap
        return segs

    def _draw_top_bar(self, img, fps, persons, violations):
        W = img.shape[1]
        draw_panel_bg(img, 0, 0, W, 38, alpha=0.80)
        draw_text(img, "ANIMUS // SOCIAL DISTANCING MONITOR", (12, 24), font=FONT, scale=0.52, color=COLOR_GOLD)
        elapsed = int(time.time() - self.session_start)
        mid_txt = f"SYNC: {elapsed:04d}s    FPS: {fps:4.1f}"
        tw = cv2.getTextSize(mid_txt, FONT_SMALL, 0.45, 1)[0][0]
        draw_text(img, mid_txt, (W//2 - tw//2, 24), scale=0.45)
        status = f"PERSONS: {len(persons):02d}   VIOLATIONS: {len(violations):02d}"
        tw2 = cv2.getTextSize(status, FONT_SMALL, 0.45, 1)[0][0]
        col = COLOR_VIOLATION if violations else COLOR_GOLD
        draw_text(img, status, (W - tw2 - 12, 24), scale=0.45, color=col)
        cv2.line(img, (0, 38), (W, 38), COLOR_GOLD, 1, cv2.LINE_AA)

    def _draw_left_panel(self, img, persons, violations, violating_ids):
        px, py, pw = 10, 48, 240
        ph = 42 + max(len(persons), 1) * 22 + 28
        draw_panel_bg(img, px, py, pw, ph)
        draw_text(img, "// SUBJECT ANALYSIS", (px+8, py+16), scale=0.40, color=COLOR_GOLD)
        cv2.line(img, (px+4, py+20), (px+pw-4, py+20), COLOR_GOLD_DIM, 1)
        y = py + 34
        for person in persons:
            is_v  = person.idx in violating_ids
            color = COLOR_VIOLATION if is_v else COLOR_SAFE
            tag   = "BREACH" if is_v else "SAFE  "
            cv2.circle(img, (px+12, y-4), 4, color, -1, cv2.LINE_AA)
            draw_text(img, f"SUBJECT {person.idx+1:02d}", (px+22, y), scale=0.40, color=color)
            draw_text(img, tag, (px+pw-62, y), scale=0.38, color=color)
            y += 22
        if not persons:
            draw_text(img, "NO SUBJECTS DETECTED", (px+12, y), scale=0.36, color=(60,60,60))
        cv2.line(img, (px+4, py+ph-18), (px+pw-4, py+ph-18), COLOR_GOLD_DIM, 1)
        draw_text(img, f"TOTAL BREACHES: {self.total_violations:03d}", (px+8, py+ph-5), scale=0.38, color=COLOR_GOLD)

    def _draw_event_log(self, img):
        H, W = img.shape[:2]
        pw, ph = 270, 175
        px, py = W - pw - 10, 48
        draw_panel_bg(img, px, py, pw, ph)
        draw_text(img, "// EVENT LOG", (px+8, py+16), scale=0.40, color=COLOR_GOLD)
        cv2.line(img, (px+4, py+20), (px+pw-4, py+20), COLOR_GOLD_DIM, 1)
        y = py + 34
        for entry in list(self.event_log)[:6]:
            col     = COLOR_VIOLATION if "ALERT" in entry else (100, 160, 180)
            display = entry[:38] if len(entry) > 38 else entry
            draw_text(img, display, (px+8, y), scale=0.32, color=col)
            y += 22
        if not self.event_log:
            draw_text(img, "MONITORING...", (px+8, y), scale=0.36, color=(60,60,60))

    def _draw_bottom_bar(self, img):
        H, W = img.shape[:2]
        draw_panel_bg(img, 0, H-28, W, 28, alpha=0.80)
        cv2.line(img, (0, H-28), (W, H-28), COLOR_GOLD, 1, cv2.LINE_AA)
        ctrl = "[Q] QUIT    [E] EAGLE VISION    [S] SCREENSHOT    [R] RESET LOG"
        tw   = cv2.getTextSize(ctrl, FONT_SMALL, 0.38, 1)[0][0]
        draw_text(img, ctrl, (W//2 - tw//2, H-10), scale=0.38, color=(100,130,150))
        draw_text(img, "NOTHING IS TRUE, EVERYTHING IS PERMITTED", (10, H-10), scale=0.34, color=COLOR_GOLD)

    def _draw_crosshair(self, img):
        H, W = img.shape[:2]
        cx, cy, s, g = W//2, H//2, 18, 6
        for p1, p2 in [((cx-s,cy),(cx-g,cy)),((cx+g,cy),(cx+s,cy)),((cx,cy-s),(cx,cy-g)),((cx,cy+g),(cx,cy+s))]:
            cv2.line(img, p1, p2, COLOR_GOLD, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 2, COLOR_GOLD, -1, cv2.LINE_AA)

    def _draw_corner_decor(self, img):
        H, W, s, t = img.shape[0], img.shape[1], 24, 2
        cv2.line(img, (0,H-28),  (s,H-28),   COLOR_GOLD, t, cv2.LINE_AA)
        cv2.line(img, (0,H-28),  (0,H-28-s), COLOR_GOLD, t, cv2.LINE_AA)
        cv2.line(img, (W,H-28),  (W-s,H-28), COLOR_GOLD, t, cv2.LINE_AA)
        cv2.line(img, (W,H-28),  (W,H-28-s), COLOR_GOLD, t, cv2.LINE_AA)

    def _draw_alert_banner(self, img, violations):
        H, W  = img.shape[:2]
        flash = abs(math.sin(time.time() * 3.5))
        bh, by = 44, H//2 - 22
        ov = img.copy()
        cv2.rectangle(ov, (0, by), (W, by+bh), (0,0,120), -1)
        cv2.addWeighted(ov, 0.55+0.30*flash, img, 1-(0.55+0.30*flash), 0, img)
        cv2.line(img, (0,by),    (W,by),    COLOR_VIOLATION, 2, cv2.LINE_AA)
        cv2.line(img, (0,by+bh), (W,by+bh), COLOR_VIOLATION, 2, cv2.LINE_AA)
        n   = len(violations)
        msg = f"!! SOCIAL DISTANCING VIOLATED - {n} BREACH{'ES' if n>1 else ''} DETECTED !!"
        tw  = cv2.getTextSize(msg, FONT, 0.65, 2)[0][0]
        r   = int(180 + 75*flash)
        cv2.putText(img, msg, (W//2-tw//2+2, by+30), FONT, 0.65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, msg, (W//2-tw//2,   by+28), FONT, 0.65, (0,0,r), 2, cv2.LINE_AA)

