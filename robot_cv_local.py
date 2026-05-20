import cv2
import numpy as np
import argparse
import math
import yaml
from collections import deque


# =========================
# Завантаження конфігурації
# =========================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MIN_AREA = config["MIN_AREA"]
FPS = config["FPS"]
PREDICT_TIME = config["PREDICT_TIME"]
STOP_THRESHOLD = config["STOP_THRESHOLD"]


# =========================
# Background Subtractor
# =========================

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)


# =========================
# 1. detect_robot()
# =========================

def detect_robot(frame):

    fgmask = bg_subtractor.apply(frame)

    # Прибираємо шум
    _, fgmask = cv2.threshold(
        fgmask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)

        if area > MIN_AREA:

            return cv2.boundingRect(c)

    return None


# =========================
# 2. classify_state()
# =========================

def classify_state(history):

    if len(history) < 2:
        return "stopped"

    dx = history[-1][0] - history[-2][0]
    dy = history[-1][1] - history[-2][1]

    speed = math.sqrt(dx * dx + dy * dy)

    if speed < STOP_THRESHOLD:
        return "stopped"

    return "moving"


# =========================
# 3. classify_direction()
# =========================

def classify_direction(dx, dy):

    if abs(dx) < STOP_THRESHOLD and abs(dy) < STOP_THRESHOLD:
        return "rotation"

    if abs(dx) > abs(dy):

        if dx > 0:
            return "right"
        else:
            return "left"

    else:

        if dy > 0:
            return "forward"
        else:
            return "backward"


# =========================
# 4. predict_position()
# =========================

def predict_position(center, velocity, t=3.0):

    vx, vy = velocity

    px = int(center[0] + vx * FPS * t)
    py = int(center[1] + vy * FPS * t)

    # Обмеження кадром
    px = max(0, min(639, px))
    py = max(0, min(479, py))

    return (px, py)


# =========================
# 5. draw_overlay()
# =========================

def draw_overlay(
    frame,
    bbox,
    state,
    direction,
    prediction
):

    x, y, w, h = bbox

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Текст
    cv2.putText(
        frame,
        f"STATE: {state}",
        (x, y - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"DIRECTION: {direction}",
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Prediction point
    cv2.circle(
        frame,
        prediction,
        8,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        "Prediction (3 sec)",
        (prediction[0] + 10, prediction[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {FPS}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="test.avi"
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Помилка відкриття відео")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_video = cv2.VideoWriter(
        'artifact_1_local.avi',
        fourcc,
        FPS,
        (640, 480)
    )

    history = deque(maxlen=10)

    print("Запуск локального тесту...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        bbox = detect_robot(frame)

        if bbox:

            x, y, w, h = bbox

            cx = x + w // 2
            cy = y + h // 2

            center = (cx, cy)

            history.append(center)

            dx, dy = 0, 0

            if len(history) >= 2:

                dx = history[-1][0] - history[-2][0]
                dy = history[-1][1] - history[-2][1]

            state = classify_state(history)

            direction = classify_direction(dx, dy)

            prediction = predict_position(
                center,
                (dx, dy),
                PREDICT_TIME
            )

            frame = draw_overlay(
                frame,
                bbox,
                state,
                direction,
                prediction
            )

        out_video.write(frame)

        cv2.imshow(
            "Robot CV Local",
            frame
        )

        key = cv2.waitKey(30)

        if key == 27:
            break

    cap.release()

    out_video.release()

    cv2.destroyAllWindows()

    print("Готово!")
    print("Створено artifact_1_local.avi")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()