import cv2
import mediapipe as mp
import numpy as np
import time

# ==== Setup ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ==== Constants ====
MODE_KEYBOARD = "keyboard"
MODE_DRAWER = "drawer"
HOVER_TIME = 2.0  # hover delay for keys
DRAW_COLOR = (0, 0, 0)
ERASER_RADIUS = 40
DRAW_THICKNESS = 6
SMOOTHING = 0.3
PREVIEW_SIZE = 120
CONFIRM_HOLD_TIME = 1.2
SAVE_BAR_WIDTH = 300
TEXT_BG_COLOR = (242, 161, 155)

# ==== State ====
mode = None
hover_start = None
hover_key = None
typed_text = ""
confirmed_previews = []
canvas = None
prev_x, prev_y = 0, 0
confirm_start_time = None
confirming = False
save_progress = 0.0
pointer_state = "normal"

# ==== Helpers ====
def draw_button(frame, x, y, w, h, label, active=False):
    color = (0, 200, 0) if active else (230, 230, 230)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 2)
    cv2.putText(frame, label, (x + 15, y + h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def get_keyboard_buttons(frame_width):
    rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM"), ["SPACE", "DEL"]]
    button_size = (70, 70)
    button_margin = 10
    start_y = 220
    buttons = []
    y = start_y
    for row in rows:
        row_width = len(row) * (button_size[0] + button_margin)
        start_x = (frame_width - row_width) // 2
        x = start_x
        for key in row:
            buttons.append((x, y, button_size[0], button_size[1], key))
            x += button_size[0] + button_margin
        y += button_size[1] + button_margin
    return buttons

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

def save_preview_from_text(text):
    if text.strip() == "":
        return None
    img = np.full((PREVIEW_SIZE, PREVIEW_SIZE * 2, 3), 255, np.uint8)
    cv2.putText(img, text, (10, PREVIEW_SIZE // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img

def save_preview_from_canvas(canvas):
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

def draw_save_bar(frame, progress, h):
    bar_x, bar_y = 50, h - 40
    cv2.rectangle(frame, (bar_x, bar_y - 20), (bar_x + SAVE_BAR_WIDTH, bar_y + 10), (200, 200, 200), 2)
    fill_w = int(SAVE_BAR_WIDTH * min(progress, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y - 20), (bar_x + fill_w, bar_y + 10), (0, 255, 0), -1)
    cv2.putText(frame, "Saving...", (bar_x + SAVE_BAR_WIDTH + 20, bar_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_progress_ring(frame, center, radius, progress, color=(0,255,255)):
    start_angle = -90
    end_angle = int(360 * progress) - 90
    cv2.circle(frame, center, radius, (200,200,200), 2)
    if progress > 0:
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 3)

def get_pointer_color(state):
    if state == "normal": return (0,255,0)
    elif state == "hover": return (0,255,255)
    elif state == "click": return (0,0,255)
    return (255,255,255)

# ==== Main Loop ====
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.full((h, w, 3), 255, np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None
    finger_state = None
    now = time.time()
    pointer_state = "normal"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_state = fingers_up(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            fingertip = (int(index_tip.x * w), int(index_tip.y * h))
            current_hand = hand_landmarks
            break

    # --- Left Buttons ---
    btn_keyboard = (20, 150, 160, 60)
    btn_drawer = (20, 230, 160, 60)
    draw_button(frame, *btn_keyboard, "Keyboard", mode == MODE_KEYBOARD)
    draw_button(frame, *btn_drawer, "Air Drawer", mode == MODE_DRAWER)

    hovered = None
    if fingertip:
        fx, fy = fingertip
        if btn_keyboard[0] < fx < btn_keyboard[0] + btn_keyboard[2] and btn_keyboard[1] < fy < btn_keyboard[1] + btn_keyboard[3]:
            hovered = "Keyboard"
        elif btn_drawer[0] < fx < btn_drawer[0] + btn_drawer[2] and btn_drawer[1] < fy < btn_drawer[1] + btn_drawer[3]:
            hovered = "Air Drawer"

        if hovered:
            pointer_state = "hover"
            if hover_key != hovered:
                hover_key = hovered
                hover_start = now
            elif now - hover_start >= HOVER_TIME:
                mode = MODE_KEYBOARD if hovered == "Keyboard" else MODE_DRAWER
                typed_text = ""
                canvas[:] = 255
                pointer_state = "click"
                hover_key = None
                hover_start = None
            else:
                progress = (now - hover_start) / HOVER_TIME
                draw_progress_ring(frame, (fx, fy), 25, progress)
        else:
            hover_key = None
            hover_start = None

    # === Keyboard Mode ===
    if mode == MODE_KEYBOARD:
        buttons = get_keyboard_buttons(w)
        if fingertip:
            fx, fy = fingertip
            key_hovered = None
            for x, y, bw, bh, key in buttons:
                is_hover = x < fx < x + bw and y < fy < y + bh
                draw_button(frame, x, y, bw, bh, key, is_hover)
                if is_hover:
                    pointer_state = "hover"
                    key_hovered = key
                    if hover_key != key:
                        hover_key = key
                        hover_start = now
                    else:
                        progress = (now - hover_start) / HOVER_TIME
                        draw_progress_ring(frame, (fx, fy), 25, progress)
                        if progress >= 1.0:
                            pointer_state = "click"
                            if key == "SPACE":
                                typed_text += " "
                            elif key == "DEL":
                                typed_text = typed_text[:-1]
                            else:
                                typed_text += key
                            hover_key = None
                            hover_start = None
                    break
            if not key_hovered:
                hover_key = None
                hover_start = None
        # Show typed text
        if typed_text.strip() != "":
            (text_w, text_h), _ = cv2.getTextSize(typed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x, text_y = (w // 2 - text_w // 2 - 20), 150 - text_h // 2
            cv2.rectangle(frame, (text_x, text_y - 10),
                          (text_x + text_w + 40, text_y + text_h + 20), TEXT_BG_COLOR, -1)
            cv2.putText(frame, typed_text, (text_x + 20, text_y + text_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # Save gesture
        if finger_state == [1, 0, 0, 0, 0]:
            if confirm_start_time is None:
                confirm_start_time = now
            save_progress = (now - confirm_start_time) / CONFIRM_HOLD_TIME
            draw_save_bar(frame, save_progress, h)
            if now - confirm_start_time > CONFIRM_HOLD_TIME:
                preview = save_preview_from_text(typed_text)
                if preview is not None:
                    confirmed_previews.append(preview)
                    if len(confirmed_previews) > 5:
                        confirmed_previews.pop(0)
                    typed_text = ""
                confirm_start_time = None
                save_progress = 0
        else:
            confirm_start_time = None
            save_progress = 0

    # === Drawer Mode ===
    elif mode == MODE_DRAWER and finger_state:
        ix, iy = fingertip if fingertip else (0, 0)
        if finger_state == [0, 1, 0, 0, 0]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = ix, iy
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), DRAW_COLOR, DRAW_THICKNESS)
            prev_x, prev_y = ix, iy
        elif finger_state == [1, 1, 1, 1, 1]:
            cx, cy = palm_center_px(current_hand, w, h)
            cv2.circle(canvas, (cx, cy), ERASER_RADIUS, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), ERASER_RADIUS, (0, 255, 0), 2)
            prev_x, prev_y = 0, 0
        elif finger_state == [1, 0, 0, 0, 0]:
            if confirm_start_time is None:
                confirm_start_time = now
            save_progress = (now - confirm_start_time) / CONFIRM_HOLD_TIME
            draw_save_bar(frame, save_progress, h)
            if now - confirm_start_time > CONFIRM_HOLD_TIME:
                preview = save_preview_from_canvas(canvas)
                if preview is not None:
                    confirmed_previews.append(preview)
                    if len(confirmed_previews) > 5:
                        confirmed_previews.pop(0)
                    canvas[:] = 255
                confirm_start_time = None
                save_progress = 0
        else:
            prev_x, prev_y = 0, 0
            confirm_start_time = None
            save_progress = 0

        # Merge drawing
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        strokes_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        frame = cv2.add(frame_bg, strokes_fg)

    # === Confirmed Previews ===
    y_offset = 20
    for img in reversed(confirmed_previews[-4:]):
        h_p, w_p, _ = img.shape
        x_offset = w - w_p - 20
        frame[y_offset:y_offset + h_p, x_offset:x_offset + w_p] = img
        cv2.rectangle(frame, (x_offset - 2, y_offset - 2),
                      (x_offset + w_p + 2, y_offset + h_p + 2), (0, 150, 0), 2)
        y_offset += h_p + 10

    # === Draw fingertip pointer ===
    if fingertip:
        pointer_color = get_pointer_color(pointer_state)
        cv2.circle(frame, fingertip, 10, pointer_color, -1)
        cv2.circle(frame, fingertip, 20, pointer_color, 2)

    cv2.imshow("Air Drawer + Keyboard System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
