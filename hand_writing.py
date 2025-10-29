import cv2
import mediapipe as mp
import numpy as np
import time
import random

# --------- Settings ---------
DRAW_COLOR = (0, 0, 0)              
CANVAS_BG_COLOR = (255, 255, 255)
DRAW_THICKNESS = 6
ERASER_RADIUS = 50
SMOOTHING = 0.35
MAX_MISS_FRAMES = 5
PREVIEW_SIZE = 120
CONFIRM_HOLD_TIME = 1.0
# ----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
miss_count = 0

confirmed_previews = []
confirm_start_time = None
confirming = False

# Animation state
particles = []  # each = [x, y, vx, vy, emoji, life]


def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []
    fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    for tip in tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
    return fingers


def palm_center_px(hand_landmarks, img_w, img_h):
    idxs = [0, 5, 9, 13, 17]
    xs = [hand_landmarks.landmark[i].x for i in idxs]
    ys = [hand_landmarks.landmark[i].y for i in idxs]
    return int(np.mean(xs) * img_w), int(np.mean(ys) * img_h)


def save_whole_drawing(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    xs, ys, xe, ye = [], [], [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x); ys.append(y); xe.append(x + w); ye.append(y + h)
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xe), max(ye)
    cropped = canvas[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None
    h_c, w_c = cropped.shape[:2]
    scale = PREVIEW_SIZE / max(h_c, w_c)
    new_w, new_h = int(w_c * scale), int(h_c * scale)
    resized = cv2.resize(cropped, (new_w, new_h))
    return resized


# --- Shape Detection ---
def detect_shape(img):
    """Very basic heuristic-based detector for heart and smile."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 200:
        return "unknown"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / float(h)
    # Heart tends to have many contour points and is roughly symmetric
    if len(approx) > 8 and 0.8 < aspect < 1.3:
        return "heart"
    # Smile: roughly circular outline
    if len(approx) > 5 and 0.9 < aspect < 1.2:
        return "smile"
    return "unknown"


# --- Animation System ---
def spawn_effect(shape, frame_w):
    global particles
    if shape == "heart":
        emoji = "❤️"
        color = (0, 0, 255)
        for _ in range(20):
            x = random.randint(0, frame_w)
            y = random.randint(-50, 0)
            vx = random.uniform(-1, 1)
            vy = random.uniform(2, 4)
            life = random.randint(60, 120)
            particles.append([x, y, vx, vy, emoji, life, color])
    elif shape == "smile":
        emoji = "😊"
        color = (0, 255, 255)
        for _ in range(20):
            x = random.randint(0, frame_w)
            y = random.randint(-50, 0)
            vx = random.uniform(-1, 1)
            vy = random.uniform(2, 4)
            life = random.randint(60, 120)
            particles.append([x, y, vx, vy, emoji, life, color])


def update_particles(frame):
    global particles
    new_particles = []
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[5] -= 1
        if p[5] > 0:
            cv2.putText(frame, p[4], (int(p[0]), int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, p[6], 2)
            new_particles.append(p)
    particles = new_particles


# ---- Main Loop ----
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.full((h, w, 3), CANVAS_BG_COLOR, dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    drawing_this_frame = False
    now = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_state = fingers_up(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            # ☝️ Write
            if finger_state == [0, 1, 0, 0, 0]:
                drawing_this_frame = True
                miss_count = 0
                confirming = False
                confirm_start_time = None

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                smoothed_x = int(prev_x + (ix - prev_x) * SMOOTHING)
                smoothed_y = int(prev_y + (iy - prev_y) * SMOOTHING)
                cv2.line(canvas, (prev_x, prev_y), (smoothed_x, smoothed_y),
                         DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
                prev_x, prev_y = smoothed_x, smoothed_y

            # ✋ Erase
            elif finger_state == [1, 1, 1, 1, 1]:
                cx, cy = palm_center_px(hand_landmarks, w, h)
                cv2.circle(canvas, (cx, cy), ERASER_RADIUS, CANVAS_BG_COLOR, -1)
                cv2.circle(frame, (cx, cy), ERASER_RADIUS, (0, 255, 0), 2)
                prev_x, prev_y = 0, 0
                confirming = False
                confirm_start_time = None

            # 👍 Confirm
            elif finger_state == [1, 0, 0, 0, 0]:
                if confirm_start_time is None:
                    confirm_start_time = now
                    confirming = True
                elif now - confirm_start_time > CONFIRM_HOLD_TIME:
                    preview = save_whole_drawing(canvas)
                    if preview is not None:
                        shape = detect_shape(preview)
                        spawn_effect(shape, w)
                        confirmed_previews.append(preview)
                        if len(confirmed_previews) > 5:
                            confirmed_previews.pop(0)
                        canvas[:] = CANVAS_BG_COLOR
                    confirming = False
                    confirm_start_time = None
                    prev_x, prev_y = 0, 0
            else:
                miss_count += 1
                if miss_count > MAX_MISS_FRAMES:
                    prev_x, prev_y = 0, 0
                confirming = False
                confirm_start_time = None
    else:
        miss_count += 1
        if miss_count > MAX_MISS_FRAMES:
            prev_x, prev_y = 0, 0
        confirming = False
        confirm_start_time = None

    # Merge canvas + frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    strokes_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, strokes_fg)

    # Update animations
    update_particles(combined)

    # Show saved previews
    margin = 10
    y_offset = margin
    for img in reversed(confirmed_previews[-4:]):
        h_p, w_p, _ = img.shape
        x_offset = w - w_p - margin
        combined[y_offset:y_offset + h_p, x_offset:x_offset + w_p] = img
        cv2.rectangle(combined, (x_offset - 2, y_offset - 2),
                      (x_offset + w_p + 2, y_offset + h_p + 2), (0, 180, 0), 2)
        y_offset += h_p + margin

    # Text feedback
    if confirming and confirm_start_time is not None:
        progress = min(1.0, (now - confirm_start_time) / CONFIRM_HOLD_TIME)
        cv2.putText(combined, f"Confirming... {int(progress * 100)}%",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
    else:
        cv2.putText(combined, "Mode: Write ☝ | Erase ✋ | Confirm 👍 (hold)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    cv2.imshow("Air Writing (Shape Detection + Live Effects)", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
