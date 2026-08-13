import cv2
import math
import time

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_HANDS,
    FRAME_MARGIN,
    CURSOR_SMOOTHING,
    CURSOR_DEAD_ZONE,
    PINCH_THRESHOLD,
    PINCH_RELEASE_THRESHOLD,
    RIGHT_CLICK_COOLDOWN,
)

from hand_tracking.hand_detector import HandDetector
from mouse_control.mouse_controller import MouseController


def map_value(value, input_min, input_max, output_min, output_max):
    """Map a value from one range to another."""

    return (
        (value - input_min)
        * (output_max - output_min)
        / (input_max - input_min)
        + output_min
    )


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def is_finger_extended(hand_landmarks, tip_id, pip_id):
    """
    Check whether a finger is extended.

    For index, middle, ring and pinky,
    the fingertip should be above the PIP joint.
    """

    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]

    return tip.y < pip.y


def main():

    # ==========================================
    # WEBCAM
    # ==========================================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        return

    # ==========================================
    # HAND DETECTOR
    # ==========================================

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE,
    )

    # ==========================================
    # MOUSE CONTROLLER
    # ==========================================

    mouse = MouseController(
        smoothing=CURSOR_SMOOTHING,
        dead_zone=CURSOR_DEAD_ZONE
    )

    screen_width, screen_height = (
        mouse.get_screen_size()
    )

    # ==========================================
    # GESTURE STATES
    # ==========================================

    # Left click
    pinch_active = False

    # Right click
    right_click_active = False

    # Last right click time
    last_right_click_time = 0

    # ==========================================
    # START MESSAGE
    # ==========================================

    print()
    print("====================================")
    print("          AI VIRTUAL MOUSE")
    print("====================================")
    print()
    print("Index finger        -> Cursor")
    print("Thumb + Index pinch -> Left Click")
    print("Index + Middle      -> Right Click")
    print()
    print("Cursor smoothing    -> ENABLED")
    print(
        f"Dead zone           -> {CURSOR_DEAD_ZONE}"
    )
    print()
    print("Press ESC to exit.")
    print("====================================")
    print()

    # ==========================================
    # MAIN LOOP
    # ==========================================

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # ======================================
        # MIRROR CAMERA
        # ======================================

        frame = cv2.flip(frame, 1)

        # ======================================
        # HAND DETECTION
        # ======================================

        frame, results = detector.find_hands(frame)

        # ======================================
        # HAND DETECTED
        # ======================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # ==================================
            # INDEX FINGER
            # Landmark 8
            # ==================================

            index_tip = (
                hand_landmarks.landmark[8]
            )

            index_x = int(
                index_tip.x * frame_width
            )

            index_y = int(
                index_tip.y * frame_height
            )

            # ==================================
            # THUMB
            # Landmark 4
            # ==================================

            thumb_tip = (
                hand_landmarks.landmark[4]
            )

            thumb_x = int(
                thumb_tip.x * frame_width
            )

            thumb_y = int(
                thumb_tip.y * frame_height
            )

            # ==================================
            # DRAW INDEX
            # ==================================

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                cv2.FILLED,
            )

            # ==================================
            # DRAW THUMB
            # ==================================

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                cv2.FILLED,
            )

            # ==================================
            # PINCH DISTANCE
            # ==================================

            pinch_distance = calculate_distance(
                thumb_x,
                thumb_y,
                index_x,
                index_y,
            )

            # ==================================
            # PINCH LINE
            # ==================================

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 255, 0),
                2,
            )

            # ==================================
            # INTERACTION AREA
            # ==================================

            cv2.rectangle(
                frame,
                (FRAME_MARGIN, FRAME_MARGIN),
                (
                    frame_width - FRAME_MARGIN,
                    frame_height - FRAME_MARGIN,
                ),
                (255, 0, 255),
                2,
            )

            # ==================================
            # LIMIT INDEX POSITION
            # ==================================

            index_x = max(
                FRAME_MARGIN,
                min(
                    index_x,
                    frame_width - FRAME_MARGIN
                ),
            )

            index_y = max(
                FRAME_MARGIN,
                min(
                    index_y,
                    frame_height - FRAME_MARGIN
                ),
            )

            # ==================================
            # CAMERA → SCREEN MAPPING
            # ==================================

            screen_x = map_value(
                index_x,
                FRAME_MARGIN,
                frame_width - FRAME_MARGIN,
                0,
                screen_width,
            )

            screen_y = map_value(
                index_y,
                FRAME_MARGIN,
                frame_height - FRAME_MARGIN,
                0,
                screen_height,
            )

            # ==================================
            # MOVE CURSOR
            #
            # Smoothing + dead zone are handled
            # inside MouseController.
            # ==================================

            mouse.move(
                screen_x,
                screen_y
            )

            # ==================================
            # LEFT CLICK
            # Thumb + Index pinch
            # ==================================

            if pinch_distance < PINCH_THRESHOLD:

                if not pinch_active:

                    mouse.left_click()

                    pinch_active = True

                    print("LEFT CLICK")

            # ==================================
            # PINCH RELEASE
            # ==================================

            elif (
                pinch_distance
                > PINCH_RELEASE_THRESHOLD
            ):

                pinch_active = False

            # ==================================
            # FINGER DETECTION
            # ==================================

            index_extended = is_finger_extended(
                hand_landmarks,
                8,
                6,
            )

            middle_extended = is_finger_extended(
                hand_landmarks,
                12,
                10,
            )

            ring_extended = is_finger_extended(
                hand_landmarks,
                16,
                14,
            )

            pinky_extended = is_finger_extended(
                hand_landmarks,
                20,
                18,
            )

            # ==================================
            # RIGHT CLICK GESTURE
            #
            # Index   = Extended
            # Middle  = Extended
            # Ring    = Folded
            # Pinky   = Folded
            # Pinch   = Not active
            # ==================================

            right_click_gesture = (
                index_extended
                and middle_extended
                and not ring_extended
                and not pinky_extended
                and pinch_distance
                > PINCH_RELEASE_THRESHOLD
            )

            # ==================================
            # RIGHT CLICK
            # ==================================

            current_time = time.time()

            if right_click_gesture:

                if not right_click_active:

                    if (
                        current_time
                        - last_right_click_time
                        > RIGHT_CLICK_COOLDOWN
                    ):

                        mouse.right_click()

                        last_right_click_time = (
                            current_time
                        )

                        right_click_active = True

                        print("RIGHT CLICK")

            else:

                right_click_active = False

            # ==================================
            # STATUS
            # ==================================

            cv2.putText(
                frame,
                "CURSOR: ACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # ==================================
            # PINCH DISTANCE
            # ==================================

            cv2.putText(
                frame,
                f"Pinch Distance: {int(pinch_distance)}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # ==================================
            # LEFT CLICK STATUS
            # ==================================

            if pinch_active:

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            # ==================================
            # RIGHT CLICK STATUS
            # ==================================

            elif right_click_active:

                cv2.putText(
                    frame,
                    "RIGHT CLICK",
                    (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            # ==================================
            # NORMAL MOVEMENT
            # ==================================

            else:

                cv2.putText(
                    frame,
                    "GESTURE: MOVE",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # ==================================
            # STABILIZATION STATUS
            # ==================================

            cv2.putText(
                frame,
                "STABILIZATION: ON",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        # ======================================
        # NO HAND DETECTED
        # ======================================

        else:

            # Reset states
            pinch_active = False
            right_click_active = False

            cv2.putText(
                frame,
                "CURSOR: INACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                "GESTURE: NONE",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "STABILIZATION: ON",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        # ======================================
        # TITLE
        # ======================================

        cv2.putText(
            frame,
            "AI VIRTUAL MOUSE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # ======================================
        # DISPLAY
        # ======================================

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # ======================================
        # ESC TO EXIT
        # ======================================

        if cv2.waitKey(1) & 0xFF == 27:

            break

    # ==========================================
    # CLEANUP
    # ==========================================

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()