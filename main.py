import cv2
import mediapipe as mp

# ================= Setup =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ================= Variables =================
last_digit_detected = 0      # last digit (excluding thumbs)
current_number_str = ""      # multi-digit number being composed
final_numbers = []           # finalized numbers list

# ================= Functions =================
def detect_hand_face(lm_list):
    palm_indices = [0, 1, 5, 9, 13, 17]
    palm_z = [lm_list[i][2] for i in palm_indices]
    wrist_z = lm_list[0][2]
    return sum(palm_z)/len(palm_z) < wrist_z

def count_fingers(lm_list, hand_label, is_palm):
    tips_ids = [8, 12, 16, 20]  # index to pinky
    other_fingers = []
    for tip_id in tips_ids:
        tip_y = lm_list[tip_id][1]
        pip_y = lm_list[tip_id-2][1]
        finger_up = 1 if (tip_y < pip_y if is_palm else tip_y > pip_y) else 0
        other_fingers.append(finger_up)
    # Thumb separately for gesture detection
    if hand_label=="Right":
        thumb_up = 1 if lm_list[4][0] > lm_list[3][0] else 0
    else:
        thumb_up = 1 if lm_list[4][0] < lm_list[3][0] else 0
    return sum(other_fingers), thumb_up

# ================= Main Loop =================
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_count = right_count = 0
    left_thumb = right_thumb = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = image.shape
            lm_list = [(int(lm.x*w), int(lm.y*h), lm.z) for lm in hand_landmarks.landmark]
            hand_label = results.multi_handedness[idx].classification[0].label
            is_palm = detect_hand_face(lm_list)
            count, thumb_up = count_fingers(lm_list, hand_label, is_palm)
            if hand_label=="Left":
                left_count = count
                left_thumb = thumb_up
            else:
                right_count = count
                right_thumb = thumb_up

    # Total fingers for digit (0-9), ignoring thumbs
    total_fingers = min(left_count + right_count, 9)
    if total_fingers != 0:
        last_digit_detected = total_fingers

    # ===== Check gestures =====
    # Save digit → one thumb up (exclusive)
    if (left_thumb ^ right_thumb):
        if len(current_number_str)==0 or current_number_str[-1] != str(last_digit_detected):
            current_number_str += str(last_digit_detected)

    # Finalize number → two thumbs up
    if left_thumb and right_thumb:
        if len(current_number_str) > 0:
            final_numbers.append(int(current_number_str))
            current_number_str = ""

    # ===== Display =====
    cv2.putText(image, f"Last Digit: {last_digit_detected}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    cv2.putText(image, f"Current Number: {current_number_str}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255),3)
    cv2.putText(image, f"Saved Numbers: {final_numbers}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)

    cv2.imshow("Finger Number Entry", image)
    if cv2.waitKey(5) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
