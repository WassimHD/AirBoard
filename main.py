import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from llama4 import *

# ==== Setup ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# ==== Constants ====
MODE_KEYBOARD = "keyboard"
MODE_DRAWER = "drawer"
HOVER_TIME = 2.0
DRAW_COLOR = (0, 0, 0)
ERASER_RADIUS = 40
DRAW_THICKNESS = 6
CONFIRM_HOLD_TIME = 1.2
SAVE_BAR_WIDTH = 300
TEXT_BG_COLOR = (242, 161, 155)
POINTER_COLOR = (0, 255, 255)
POINTER_RADIUS = 20
PREVIEW_TARGET_SIZE = (120, 80)  # smaller, compact preview frames

# ==== State ====
mode = None
hover_start = None
hover_key = None
typed_text = ""
confirmed_previews = []
canvas = None
prev_x, prev_y = 0, 0
confirm_start_time = None
save_progress = 0.0
ai_response = ""
loading = False
clear_draw_hover_start = None

# ==== Helpers ====
def draw_button(frame, x, y, w, h, label, active=False, color_override=None):
    color = color_override if color_override else ((0, 200, 0) if active else (230, 230, 230))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 2)
    cv2.putText(frame, label, (x + 15, y + h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def get_keyboard_buttons(frame_width):
    rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM"), ["SPACE", "DEL", "Search", "Clear"]]
    button_size = (70, 70)
    margin = 10
    start_y = 220
    buttons = []
    y = start_y
    for row in rows:
        row_width = len(row) * (button_size[0] + margin)
        start_x = (frame_width - row_width) // 2
        x = start_x
        for key in row:
            buttons.append((x, y, button_size[0], button_size[1], key))
            x += button_size[0] + margin
        y += button_size[1] + margin
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

def draw_save_bar(frame, progress, h):
    bar_x, bar_y = 50, h - 40
    cv2.rectangle(frame, (bar_x, bar_y - 20), (bar_x + SAVE_BAR_WIDTH, bar_y + 10), (200, 200, 200), 2)
    fill_w = int(SAVE_BAR_WIDTH * min(progress, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y - 20), (bar_x + fill_w, bar_y + 10), (0, 255, 0), -1)
    cv2.putText(frame, "Saving...", (bar_x + SAVE_BAR_WIDTH + 20, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def crop_sketch_region(img):
    """Crop the drawing area around actual ink pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y + h, x:x + w]
        return cv2.resize(cropped, PREVIEW_TARGET_SIZE)
    else:
        # fallback: empty white preview
        return np.full((PREVIEW_TARGET_SIZE[1], PREVIEW_TARGET_SIZE[0], 3), 255, np.uint8)

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
    results = mp_hands.Hands.process(hands, rgb)
    fingertip, finger_state, current_hand = None, None, None
    now = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_state = fingers_up(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            fingertip = (int(index_tip.x * w), int(index_tip.y * h))
            current_hand = hand_landmarks
            break

    # === Left menu ===
    btn_keyboard = (20, 150, 160, 60)
    btn_drawer = (20, 230, 160, 60)
    btn_clear_draws = (20, 310, 160, 60)
    draw_button(frame, *btn_keyboard, "Keyboard", mode == MODE_KEYBOARD)
    draw_button(frame, *btn_drawer, "Air Drawer", mode == MODE_DRAWER)
    draw_button(frame, *btn_clear_draws, "Clear Draws", False, (255, 180, 130))

    # === Mode Switching & Clear Drawings ===
    if fingertip:
        fx, fy = fingertip
        # Mode switches
        if btn_keyboard[0] < fx < btn_keyboard[0] + btn_keyboard[2] and btn_keyboard[1] < fy < btn_keyboard[1] + btn_keyboard[3]:
            if mode != MODE_KEYBOARD:
                mode = MODE_KEYBOARD
                typed_text = ""
                canvas[:] = 255
        elif btn_drawer[0] < fx < btn_drawer[0] + btn_drawer[2] and btn_drawer[1] < fy < btn_drawer[1] + btn_drawer[3]:
            if mode != MODE_DRAWER:
                mode = MODE_DRAWER
                typed_text = ""
                canvas[:] = 255
        # Clear saved drawings
        if btn_clear_draws[0] < fx < btn_clear_draws[0] + btn_clear_draws[2] and btn_clear_draws[1] < fy < btn_clear_draws[1] + btn_clear_draws[3]:
            if clear_draw_hover_start is None:
                clear_draw_hover_start = now
            elif now - clear_draw_hover_start >= HOVER_TIME:
                confirmed_previews.clear()
                clear_draw_hover_start = None
        else:
            clear_draw_hover_start = None

    # === Keyboard Mode ===
    if mode == MODE_KEYBOARD:
        buttons = get_keyboard_buttons(w)
        if fingertip:
            fx, fy = fingertip
            for x, y, bw, bh, key in buttons:
                is_hover = x < fx < x + bw and y < fy < y + bh
                color_override = (180, 100, 255) if key == "Search" else ((0, 165, 255) if key == "Clear" else None)
                draw_button(frame, x, y, bw, bh, key, is_hover, color_override)

                if is_hover:
                    if hover_key != key:
                        hover_key = key
                        hover_start = now
                    elif now - hover_start >= HOVER_TIME:
                        if key == "SPACE":
                            typed_text += " "
                        elif key == "DEL":
                            typed_text = typed_text[:-1]
                        elif key == "Search":
                            if not loading and typed_text.strip():
                                def run_query():
                                    global ai_response, loading
                                    loading = True
                                    ai_response = llama4_scout(typed_text)
                                    loading = False
                                threading.Thread(target=run_query, daemon=True).start()
                        elif key == "Clear":
                            typed_text = ""
                            ai_response = ""
                            loading = False
                        else:
                            typed_text += key
                        hover_key, hover_start = None, None

        # Display typed text
        if typed_text.strip():
            (tw, th), _ = cv2.getTextSize(typed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx, ty = (w // 2 - tw // 2 - 20), 150 - th // 2
            cv2.rectangle(frame, (tx, ty - 10), (tx + tw + 40, ty + th + 20), TEXT_BG_COLOR, -1)
            cv2.putText(frame, typed_text, (tx + 20, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # AI Response / Loading
        resp_x, resp_y = w - 400, 20
        if loading:
            cv2.rectangle(frame, (resp_x - 10, resp_y - 10), (w - 20, resp_y + 60), (200, 200, 255), -1)
            cv2.putText(frame, "Loading...", (resp_x, resp_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 200), 2)
        elif ai_response:
            wrapped, line = [], ""
            for word in ai_response.split():
                if len(line + word) < 35:
                    line += word + " "
                else:
                    wrapped.append(line)
                    line = word + " "
            wrapped.append(line)
            y_line = resp_y + 20
            cv2.rectangle(frame, (resp_x - 10, resp_y - 10), (w - 20, resp_y + 25 * len(wrapped) + 20), (200, 255, 200), -1)
            for line in wrapped:
                cv2.putText(frame, line.strip(), (resp_x, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
                y_line += 25

    # === Drawer Mode ===
    elif mode == MODE_DRAWER and finger_state:
        ix, iy = fingertip if fingertip else (0, 0)
        if finger_state == [0, 1, 0, 0, 0]:  # Draw
            cv2.circle(frame, (ix, iy), POINTER_RADIUS, POINTER_COLOR, 4)
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = ix, iy
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), DRAW_COLOR, DRAW_THICKNESS)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = 0, 0
        if finger_state == [1, 1, 1, 1, 1]:  # Erase
            cx, cy = palm_center_px(current_hand, w, h)
            cv2.circle(canvas, (cx, cy), ERASER_RADIUS, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), ERASER_RADIUS, (0, 255, 0), 2)
        if finger_state == [1, 0, 0, 0, 0]:  # Save
            if confirm_start_time is None:
                confirm_start_time = now
            save_progress = (now - confirm_start_time) / CONFIRM_HOLD_TIME
            draw_save_bar(frame, save_progress, h)
            if now - confirm_start_time > CONFIRM_HOLD_TIME:
                cropped_preview = crop_sketch_region(canvas)
                confirmed_previews.append(cropped_preview)
                canvas[:] = 255
                confirm_start_time, save_progress = None, 0
        else:
            confirm_start_time, save_progress = None, 0

    # === Merge canvas ===
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.add(cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
                    cv2.bitwise_and(canvas, canvas, mask=mask))

    # === Show saved previews (bottom-right, smaller, cropped) ===
    y_offset = h - 20
    for img in reversed(confirmed_previews[-6:]):
        h_p, w_p, _ = img.shape
        y_offset -= (h_p + 8)
        x_offset = w - w_p - 20
        if y_offset < 0: break
        frame[y_offset:y_offset + h_p, x_offset:x_offset + w_p] = img
        cv2.rectangle(frame, (x_offset - 2, y_offset - 2), (x_offset + w_p + 2, y_offset + h_p + 2), (0, 150, 0), 2)

    cv2.imshow("Air Drawer + Keyboard System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
