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
    DOUBLE_CLICK_INTERVAL,
    RIGHT_CLICK_COOLDOWN,
    SCROLL_THRESHOLD,
    SCROLL_SPEED,
)

from hand_tracking.hand_detector import HandDetector
from mouse_control.mouse_controller import MouseController


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def map_value(
    value,
    input_min,
    input_max,
    output_min,
    output_max
):
    """Map camera coordinates to screen coordinates."""

    return (
        (value - input_min)
        * (output_max - output_min)
        / (input_max - input_min)
        + output_min
    )


def calculate_distance(
    x1,
    y1,
    x2,
    y2
):
    """Calculate Euclidean distance."""

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def is_finger_extended(
    hand_landmarks,
    tip_id,
    pip_id
):
    """
    Check whether a finger is extended.
    """

    tip = hand_landmarks.landmark[tip_id]

    pip = hand_landmarks.landmark[pip_id]

    return tip.y < pip.y


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # WEBCAM
    # ========================================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    # ========================================================
    # HAND DETECTOR
    # ========================================================

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE,
    )

    # ========================================================
    # MOUSE CONTROLLER
    # ========================================================

    mouse = MouseController(
        smoothing=CURSOR_SMOOTHING,
        dead_zone=CURSOR_DEAD_ZONE
    )

    screen_width, screen_height = (
        mouse.get_screen_size()
    )

    # ========================================================
    # CLICK STATES
    # ========================================================

    # Current pinch state

    pinch_active = False

    # Time when pinch was released

    last_pinch_release_time = 0

    # Waiting to determine single/double click

    pending_click = False

    # ========================================================
    # RIGHT CLICK STATE
    # ========================================================

    right_click_active = False

    last_right_click_time = 0

    # ========================================================
    # SCROLL STATE
    # ========================================================

    scroll_active = False

    previous_scroll_y = None

    # ========================================================
    # START MESSAGE
    # ========================================================

    print()

    print("==============================================")

    print("             AI VIRTUAL MOUSE")

    print("==============================================")

    print()

    print("INDEX FINGER")

    print("    -> Move Cursor")

    print()

    print("THUMB + INDEX PINCH")

    print("    -> Left Click")

    print()

    print("DOUBLE PINCH")

    print("    -> Double Click")

    print()

    print("INDEX + MIDDLE + MOVE")

    print("    -> Scroll")

    print()

    print("INDEX + MIDDLE HOLD")

    print("    -> Right Click")

    print()

    print("ESC")

    print("    -> Exit")

    print()

    print("==============================================")

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # HAND DETECTION
        # ====================================================

        frame, results = detector.find_hands(
            frame
        )

        # ====================================================
        # HAND FOUND
        # ====================================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # =================================================
            # INDEX FINGER
            # =================================================

            index_tip = (
                hand_landmarks.landmark[8]
            )

            index_x = int(
                index_tip.x
                * frame_width
            )

            index_y = int(
                index_tip.y
                * frame_height
            )

            # =================================================
            # THUMB
            # =================================================

            thumb_tip = (
                hand_landmarks.landmark[4]
            )

            thumb_x = int(
                thumb_tip.x
                * frame_width
            )

            thumb_y = int(
                thumb_tip.y
                * frame_height
            )

            # =================================================
            # MIDDLE FINGER
            # =================================================

            middle_tip = (
                hand_landmarks.landmark[12]
            )

            middle_x = int(
                middle_tip.x
                * frame_width
            )

            middle_y = int(
                middle_tip.y
                * frame_height
            )

            # =================================================
            # DRAW LANDMARKS
            # =================================================

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                (middle_x, middle_y),
                8,
                (0, 165, 255),
                cv2.FILLED
            )

            # =================================================
            # PINCH DISTANCE
            # =================================================

            pinch_distance = calculate_distance(
                thumb_x,
                thumb_y,
                index_x,
                index_y
            )

            # =================================================
            # PINCH LINE
            # =================================================

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 255, 0),
                2
            )

            # =================================================
            # INTERACTION AREA
            # =================================================

            cv2.rectangle(
                frame,
                (
                    FRAME_MARGIN,
                    FRAME_MARGIN
                ),
                (
                    frame_width - FRAME_MARGIN,
                    frame_height - FRAME_MARGIN
                ),
                (255, 0, 255),
                2
            )

            # =================================================
            # LIMIT INDEX
            # =================================================

            index_x = max(
                FRAME_MARGIN,
                min(
                    index_x,
                    frame_width - FRAME_MARGIN
                )
            )

            index_y = max(
                FRAME_MARGIN,
                min(
                    index_y,
                    frame_height - FRAME_MARGIN
                )
            )

            # =================================================
            # CAMERA → SCREEN
            # =================================================

            screen_x = map_value(
                index_x,
                FRAME_MARGIN,
                frame_width - FRAME_MARGIN,
                0,
                screen_width
            )

            screen_y = map_value(
                index_y,
                FRAME_MARGIN,
                frame_height - FRAME_MARGIN,
                0,
                screen_height
            )

            # =================================================
            # FINGER STATES
            # =================================================

            index_extended = is_finger_extended(
                hand_landmarks,
                8,
                6
            )

            middle_extended = is_finger_extended(
                hand_landmarks,
                12,
                10
            )

            ring_extended = is_finger_extended(
                hand_landmarks,
                16,
                14
            )

            pinky_extended = is_finger_extended(
                hand_landmarks,
                20,
                18
            )

            # =================================================
            # TWO FINGER GESTURE
            # =================================================

            two_finger_gesture = (
                index_extended
                and middle_extended
                and not ring_extended
                and not pinky_extended
                and pinch_distance
                > PINCH_RELEASE_THRESHOLD
            )

            # =================================================
            # SCROLL
            # =================================================

            if two_finger_gesture:

                average_y = (
                    index_y + middle_y
                ) / 2

                if previous_scroll_y is None:

                    previous_scroll_y = (
                        average_y
                    )

                else:

                    scroll_delta = (
                        previous_scroll_y
                        - average_y
                    )

                    if (
                        abs(scroll_delta)
                        >= SCROLL_THRESHOLD
                    ):

                        scroll_amount = int(
                            scroll_delta
                            * SCROLL_SPEED
                            / 10
                        )

                        if scroll_amount != 0:

                            mouse.scroll(
                                scroll_amount
                            )

                            scroll_active = True

                            previous_scroll_y = (
                                average_y
                            )

                            print(
                                "SCROLL:",
                                scroll_amount
                            )

                    else:

                        scroll_active = False

            else:

                previous_scroll_y = None

                scroll_active = False

            # =================================================
            # PINCH START
            # =================================================

            current_time = time.time()

            if pinch_distance < PINCH_THRESHOLD:

                # Pinch has just started

                if not pinch_active:

                    pinch_active = True

                    print(
                        "PINCH DETECTED"
                    )

            # =================================================
            # PINCH RELEASE
            # =================================================

            elif (
                pinch_distance
                > PINCH_RELEASE_THRESHOLD
            ):

                # Pinch was active and is now released

                if pinch_active:

                    pinch_active = False

                    current_time = time.time()

                    # -----------------------------------------
                    # Check previous pinch
                    # -----------------------------------------

                    if (
                        pending_click
                        and (
                            current_time
                            - last_pinch_release_time
                            <= DOUBLE_CLICK_INTERVAL
                        )
                    ):

                        # DOUBLE CLICK

                        mouse.double_click()

                        print(
                            "DOUBLE CLICK"
                        )

                        pending_click = False

                        last_pinch_release_time = 0

                    else:

                        # Wait to see whether
                        # another pinch happens.

                        pending_click = True

                        last_pinch_release_time = (
                            current_time
                        )

                        print(
                            "PINCH RELEASE"
                        )

            # =================================================
            # SINGLE CLICK CONFIRMATION
            # =================================================

            if pending_click:

                if (
                    time.time()
                    - last_pinch_release_time
                    > DOUBLE_CLICK_INTERVAL
                ):

                    mouse.left_click()

                    print(
                        "LEFT CLICK"
                    )

                    pending_click = False

                    last_pinch_release_time = 0

            # =================================================
            # RIGHT CLICK
            # =================================================

            current_time = time.time()

            if two_finger_gesture:

                if not scroll_active:

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

                            print(
                                "RIGHT CLICK"
                            )

                else:

                    right_click_active = False

            else:

                right_click_active = False

            # =================================================
            # CURSOR MOVEMENT
            # =================================================

            if not two_finger_gesture:

                mouse.move(
                    screen_x,
                    screen_y
                )

            # =================================================
            # STATUS
            # =================================================

            cv2.putText(
                frame,
                "CURSOR: ACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # =================================================
            # PINCH DISTANCE
            # =================================================

            cv2.putText(
                frame,
                f"Pinch: {int(pinch_distance)}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # =================================================
            # GESTURE STATUS
            # =================================================

            if pinch_active:

                cv2.putText(
                    frame,
                    "GESTURE: PINCH",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            elif scroll_active:

                cv2.putText(
                    frame,
                    "GESTURE: SCROLLING",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            elif right_click_active:

                cv2.putText(
                    frame,
                    "GESTURE: RIGHT CLICK",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            elif pending_click:

                cv2.putText(
                    frame,
                    "WAITING FOR CLICK",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "GESTURE: MOVE",
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # STABILIZATION
            # =================================================

            cv2.putText(
                frame,
                "STABILIZATION: ON",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            pinch_active = False

            right_click_active = False

            scroll_active = False

            previous_scroll_y = None

            # ------------------------------------------------
            # If a click is pending and hand disappears,
            # still allow the single click to complete.
            # ------------------------------------------------

            if pending_click:

                if (
                    time.time()
                    - last_pinch_release_time
                    > DOUBLE_CLICK_INTERVAL
                ):

                    mouse.left_click()

                    print(
                        "LEFT CLICK"
                    )

                    pending_click = False

                    last_pinch_release_time = 0

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            cv2.putText(
                frame,
                "CURSOR: INACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "GESTURE: NONE",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(
            frame,
            "AI VIRTUAL MOUSE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # ====================================================
        # ESC
        # ====================================================

        if (
            cv2.waitKey(1)
            & 0xFF
            == 27
        ):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()