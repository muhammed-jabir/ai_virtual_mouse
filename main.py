import cv2
import math
import time
import keyboard

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_HANDS,
    FRAME_MARGIN,
    CURSOR_SMOOTHING,
    PINCH_THRESHOLD,
    PINCH_RELEASE_THRESHOLD,
    RIGHT_CLICK_COOLDOWN,
    CURSOR_DEAD_ZONE,
)

from hand_tracking.hand_detector import HandDetector
from mouse_control.mouse_controller import MouseController
from eye_tracking.eye_tracker import EyeTracker


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def map_value(
    value,
    input_min,
    input_max,
    output_min,
    output_max
):

    if input_max == input_min:
        return output_min

    return (
        (value - input_min)
        * (output_max - output_min)
        /
        (input_max - input_min)
        + output_min
    )


def calculate_distance(
    x1,
    y1,
    x2,
    y2
):

    return math.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


def is_finger_extended(
    hand_landmarks,
    tip_id,
    pip_id
):

    tip = hand_landmarks.landmark[tip_id]
    pip = hand_landmarks.landmark[pip_id]

    return tip.y < pip.y


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # CAMERA
    # ======================================================

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

    # ======================================================
    # HAND DETECTOR
    # ======================================================

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE
    )

    # ======================================================
    # EYE TRACKER
    # ======================================================

    eye_tracker = EyeTracker(
        max_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    )

    # ======================================================
    # MOUSE
    # ======================================================

    mouse = MouseController(
        smoothing=CURSOR_SMOOTHING,
        dead_zone=CURSOR_DEAD_ZONE
    )

    screen_width, screen_height = (
        mouse.get_screen_size()
    )

    # ======================================================
    # STATES
    # ======================================================

    # Hand cursor ON by default
    hand_enabled = True

    # Eye cursor OFF by default
    eye_enabled = False

    # Hybrid OFF by default
    hybrid_enabled = False

    # Left click
    pinch_active = False

    # Right click
    right_click_active = False

    last_right_click_time = 0

    # Scroll
    previous_scroll_y = None

    # Eye smoothing
    previous_eye_x = None
    previous_eye_y = None

    # ======================================================
    # KEYBOARD STATE
    # ======================================================

    previous_f6 = False
    previous_f7 = False
    previous_f8 = False

    # ======================================================
    # START MESSAGE
    # ======================================================

    print()
    print("==============================================")
    print("              AI VIRTUAL MOUSE")
    print("==============================================")
    print()
    print("HAND MODE")
    print("Index finger        -> Cursor")
    print("Thumb + Index       -> Left Click")
    print("Index + Middle      -> Right Click")
    print("Hand movement       -> Scroll")
    print()
    print("EYE + HAND CONTROLS")
    print("F6 -> Hybrid Eye + Finger")
    print("F7 -> Eye Cursor")
    print("F8 -> Hand Cursor")
    print("ESC -> Exit")
    print()
    print("Eye control starts OFF for safety.")
    print("==============================================")
    print()

    # ======================================================
    # MAIN LOOP
    # ======================================================

    while True:

        # ==================================================
        # GLOBAL KEYBOARD CONTROLS
        # ==================================================

        f6_pressed = keyboard.is_pressed("f6")
        f7_pressed = keyboard.is_pressed("f7")
        f8_pressed = keyboard.is_pressed("f8")

        # ----------------------------------------------
        # F6 = HYBRID
        # ----------------------------------------------

        if f6_pressed and not previous_f6:

            hybrid_enabled = True
            eye_enabled = True
            hand_enabled = True

            mouse.reset_position()

            previous_eye_x = None
            previous_eye_y = None

            print(
                "MODE: HYBRID EYE + HAND"
            )

        # ----------------------------------------------
        # F7 = EYE
        # ----------------------------------------------

        if f7_pressed and not previous_f7:

            hybrid_enabled = False
            eye_enabled = True
            hand_enabled = False

            mouse.reset_position()

            previous_eye_x = None
            previous_eye_y = None

            print(
                "MODE: EYE CURSOR"
            )

        # ----------------------------------------------
        # F8 = HAND
        # ----------------------------------------------

        if f8_pressed and not previous_f8:

            hybrid_enabled = False
            eye_enabled = False
            hand_enabled = True

            mouse.reset_position()

            previous_eye_x = None
            previous_eye_y = None

            print(
                "MODE: HAND CURSOR"
            )

        previous_f6 = f6_pressed
        previous_f7 = f7_pressed
        previous_f8 = f8_pressed

        # ==================================================
        # CAMERA FRAME
        # ==================================================

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # ==================================================
        # MIRROR
        # ==================================================

        frame = cv2.flip(
            frame,
            1
        )

        frame_height, frame_width, _ = (
            frame.shape
        )

        # ==================================================
        # HAND DETECTION
        # ==================================================

        frame, results = detector.find_hands(
            frame
        )

        # ==================================================
        # EYE DETECTION
        # ==================================================

        eye_results = eye_tracker.find_face(
            frame
        )

        # ==================================================
        # VARIABLES
        # ==================================================

        finger_screen_x = None
        finger_screen_y = None

        eye_screen_x = None
        eye_screen_y = None

        # ==================================================
        # HAND PROCESSING
        # ==================================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # ==================================================
            # INDEX
            # ==================================================

            index_tip = (
                hand_landmarks.landmark[8]
            )

            index_x = int(
                index_tip.x *
                frame_width
            )

            index_y = int(
                index_tip.y *
                frame_height
            )

            # ==================================================
            # THUMB
            # ==================================================

            thumb_tip = (
                hand_landmarks.landmark[4]
            )

            thumb_x = int(
                thumb_tip.x *
                frame_width
            )

            thumb_y = int(
                thumb_tip.y *
                frame_height
            )

            # ==================================================
            # DRAW INDEX
            # ==================================================

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                cv2.FILLED
            )

            # ==================================================
            # DRAW THUMB
            # ==================================================

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                cv2.FILLED
            )

            # ==================================================
            # PINCH
            # ==================================================

            pinch_distance = calculate_distance(
                thumb_x,
                thumb_y,
                index_x,
                index_y
            )

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 255, 0),
                2
            )

            # ==================================================
            # INTERACTION AREA
            # ==================================================

            cv2.rectangle(
                frame,
                (FRAME_MARGIN, FRAME_MARGIN),
                (
                    frame_width - FRAME_MARGIN,
                    frame_height - FRAME_MARGIN
                ),
                (255, 0, 255),
                2
            )

            # ==================================================
            # CLAMP INDEX
            # ==================================================

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

            # ==================================================
            # HAND → SCREEN
            # ==================================================

            finger_screen_x = map_value(
                index_x,
                FRAME_MARGIN,
                frame_width - FRAME_MARGIN,
                0,
                screen_width - 1
            )

            finger_screen_y = map_value(
                index_y,
                FRAME_MARGIN,
                frame_height - FRAME_MARGIN,
                0,
                screen_height - 1
            )

            # ==================================================
            # FINGER CURSOR
            # ==================================================

            if hand_enabled and not hybrid_enabled:

                mouse.move(
                    finger_screen_x,
                    finger_screen_y
                )

            # ==================================================
            # LEFT CLICK
            # ==================================================

            if pinch_distance < PINCH_THRESHOLD:

                if not pinch_active:

                    mouse.left_click()

                    pinch_active = True

                    print(
                        "LEFT CLICK"
                    )

            elif (
                pinch_distance >
                PINCH_RELEASE_THRESHOLD
            ):

                pinch_active = False

            # ==================================================
            # FINGER STATES
            # ==================================================

            index_extended = (
                is_finger_extended(
                    hand_landmarks,
                    8,
                    6
                )
            )

            middle_extended = (
                is_finger_extended(
                    hand_landmarks,
                    12,
                    10
                )
            )

            ring_extended = (
                is_finger_extended(
                    hand_landmarks,
                    16,
                    14
                )
            )

            pinky_extended = (
                is_finger_extended(
                    hand_landmarks,
                    20,
                    18
                )
            )

            # ==================================================
            # RIGHT CLICK
            # ==================================================

            right_click_gesture = (
                index_extended
                and
                middle_extended
                and
                not ring_extended
                and
                not pinky_extended
                and
                pinch_distance >
                PINCH_RELEASE_THRESHOLD
            )

            current_time = time.time()

            if right_click_gesture:

                if not right_click_active:

                    if (
                        current_time -
                        last_right_click_time
                        >
                        RIGHT_CLICK_COOLDOWN
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

            # ==================================================
            # SCROLL
            # ==================================================
            #
            # Index + middle extended
            # while ring + pinky folded
            #
            # Moving the hand vertically scrolls.
            #

            scroll_gesture = (
                index_extended
                and
                middle_extended
                and
                not ring_extended
                and
                not pinky_extended
                and
                pinch_distance >
                PINCH_RELEASE_THRESHOLD
            )

            if scroll_gesture:

                if previous_scroll_y is None:

                    previous_scroll_y = index_y

                else:

                    scroll_delta = (
                        index_y -
                        previous_scroll_y
                    )

                    if abs(scroll_delta) > 5:

                        scroll_amount = int(
                            -scroll_delta / 12
                        )

                        if scroll_amount != 0:

                            mouse.scroll(
                                scroll_amount
                            )

                        previous_scroll_y = (
                            index_y
                        )

            else:

                previous_scroll_y = None

        else:

            pinch_active = False
            right_click_active = False
            previous_scroll_y = None

        # ======================================================
        # EYE PROCESSING
        # ======================================================

        if (
            eye_enabled
            and
            eye_results.multi_face_landmarks
        ):

            face_landmarks = (
                eye_results.multi_face_landmarks[0]
            )

            # ==================================================
            # DRAW EYES
            # ==================================================

            eye_tracker.draw_eye_landmarks(
                frame,
                face_landmarks
            )

            # ==================================================
            # GET GAZE
            # ==================================================

            (
                gaze_x,
                gaze_y,
                direction,
                gaze_ratio
            ) = eye_tracker.get_gaze_position(
                frame,
                face_landmarks
            )

            # ==================================================
            # EYE X
            # ==================================================

            eye_screen_x = (
                gaze_x *
                (screen_width - 1)
            )

            # ==================================================
            # EYE Y
            # ==================================================
            #
            # IMPORTANT:
            # We use the iris position relative to
            # the face rather than raw camera Y.
            #

            left_iris_y = (
                face_landmarks.landmark[468].y
            )

            right_iris_y = (
                face_landmarks.landmark[473].y
            )

            average_iris_y = (
                left_iris_y +
                right_iris_y
            ) / 2

            # Normalize approximate eye movement.
            #
            # This is intentionally conservative.
            #

            eye_y_normalized = map_value(
                average_iris_y,
                0.30,
                0.70,
                0.0,
                1.0
            )

            eye_y_normalized = max(
                0.0,
                min(
                    1.0,
                    eye_y_normalized
                )
            )

            eye_screen_y = (
                eye_y_normalized *
                (screen_height - 1)
            )

            # ==================================================
            # EYE SMOOTHING
            # ==================================================

            if previous_eye_x is None:

                previous_eye_x = eye_screen_x

            if previous_eye_y is None:

                previous_eye_y = eye_screen_y

            previous_eye_x = (
                previous_eye_x * 0.8
                +
                eye_screen_x * 0.2
            )

            previous_eye_y = (
                previous_eye_y * 0.8
                +
                eye_screen_y * 0.2
            )

            eye_screen_x = (
                previous_eye_x
            )

            eye_screen_y = (
                previous_eye_y
            )

            # ==================================================
            # EYE ONLY
            # ==================================================

            if (
                eye_enabled
                and
                not hybrid_enabled
                and
                not hand_enabled
            ):

                mouse.move_eye(
                    eye_screen_x,
                    eye_screen_y
                )

            # ==================================================
            # HYBRID
            # ==================================================

            if hybrid_enabled:

                if (
                    finger_screen_x is not None
                    and
                    finger_screen_y is not None
                ):

                    mouse.move_hybrid(
                        eye_screen_x,
                        eye_screen_y,
                        finger_screen_x,
                        finger_screen_y,
                        eye_weight=0.7,
                        finger_weight=0.3
                    )

                else:

                    mouse.move_eye(
                        eye_screen_x,
                        eye_screen_y
                    )

            # ==================================================
            # EYE UI
            # ==================================================

            cv2.putText(
                frame,
                f"GAZE: {direction}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"GAZE RATIO: {gaze_ratio:.2f}",
                (20, 315),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ======================================================
        # MODE DISPLAY
        # ======================================================

        if hybrid_enabled:

            mode_text = (
                "MODE: HYBRID EYE + HAND"
            )

            mode_color = (
                0,
                255,
                255
            )

        elif eye_enabled:

            mode_text = (
                "MODE: EYE CURSOR"
            )

            mode_color = (
                255,
                255,
                0
            )

        else:

            mode_text = (
                "MODE: HAND CURSOR"
            )

            mode_color = (
                0,
                255,
                0
            )

        cv2.putText(
            frame,
            mode_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mode_color,
            2
        )

        # ======================================================
        # HAND STATUS
        # ======================================================

        hand_status = (
            "HAND: ON"
            if hand_enabled
            else
            "HAND: OFF"
        )

        eye_status = (
            "EYE: ON"
            if eye_enabled
            else
            "EYE: OFF"
        )

        cv2.putText(
            frame,
            hand_status,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            eye_status,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ======================================================
        # GESTURE STATUS
        # ======================================================

        if pinch_active:

            gesture_text = (
                "GESTURE: LEFT CLICK"
            )

        elif right_click_active:

            gesture_text = (
                "GESTURE: RIGHT CLICK"
            )

        else:

            gesture_text = (
                "GESTURE: MOVE / SCROLL"
            )

        cv2.putText(
            frame,
            gesture_text,
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ======================================================
        # CONTROLS
        # ======================================================

        cv2.putText(
            frame,
            "F6: HYBRID | F7: EYE | F8: HAND | ESC: EXIT",
            (20, frame_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ======================================================
        # TITLE
        # ======================================================

        cv2.putText(
            frame,
            "AI VIRTUAL MOUSE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ======================================================
        # DISPLAY
        # ======================================================

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # ======================================================
        # ESC
        # ======================================================

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            break

    # ======================================================
    # CLEANUP
    # ======================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print(
        "AI Virtual Mouse stopped."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()