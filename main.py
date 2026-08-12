import cv2
import math

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
    """Calculate distance between two points."""

    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
    )


def main():

    # --------------------------------
    # Webcam
    # --------------------------------

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

    # --------------------------------
    # Hand detector
    # --------------------------------

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE,
    )

    # --------------------------------
    # Mouse controller
    # --------------------------------

    mouse = MouseController(
        smoothing=CURSOR_SMOOTHING
    )

    screen_width, screen_height = mouse.get_screen_size()

    # --------------------------------
    # Pinch state
    # --------------------------------

    pinch_active = False

    print("AI Virtual Mouse started.")

    print("Index finger → Cursor")

    print("Thumb + Index pinch → Left Click")

    print("Press ESC to exit.")

    # --------------------------------
    # Main loop
    # --------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # --------------------------------
        # Detect hand
        # --------------------------------

        frame, results = detector.find_hands(frame)

        # --------------------------------
        # Hand detected
        # --------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # --------------------------------
            # INDEX FINGER
            # Landmark 8
            # --------------------------------

            index_tip = (
                hand_landmarks.landmark[8]
            )

            index_x = int(
                index_tip.x * frame_width
            )

            index_y = int(
                index_tip.y * frame_height
            )

            # --------------------------------
            # THUMB
            # Landmark 4
            # --------------------------------

            thumb_tip = (
                hand_landmarks.landmark[4]
            )

            thumb_x = int(
                thumb_tip.x * frame_width
            )

            thumb_y = int(
                thumb_tip.y * frame_height
            )

            # --------------------------------
            # Draw index
            # --------------------------------

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                cv2.FILLED,
            )

            # --------------------------------
            # Draw thumb
            # --------------------------------

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                cv2.FILLED,
            )

            # --------------------------------
            # Calculate pinch distance
            # --------------------------------

            pinch_distance = calculate_distance(
                thumb_x,
                thumb_y,
                index_x,
                index_y,
            )

            # --------------------------------
            # Draw line between fingers
            # --------------------------------

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 255, 0),
                2,
            )

            # --------------------------------
            # Cursor interaction area
            # --------------------------------

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

            # Keep index inside area
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

            # --------------------------------
            # Map to screen
            # --------------------------------

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

            # --------------------------------
            # Move cursor
            # --------------------------------

            mouse.move(
                screen_x,
                screen_y
            )

            # --------------------------------
            # PINCH DETECTION
            # --------------------------------

            if (
                pinch_distance
                < PINCH_THRESHOLD
            ):

                # Only click once
                if not pinch_active:

                    mouse.left_click()

                    pinch_active = True

                    print("LEFT CLICK")

            # --------------------------------
            # PINCH RELEASE
            # --------------------------------

            elif (
                pinch_distance
                > PINCH_RELEASE_THRESHOLD
            ):

                pinch_active = False

            # --------------------------------
            # UI STATUS
            # --------------------------------

            cv2.putText(
                frame,
                "CURSOR: ACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Pinch Distance: {int(pinch_distance)}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

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

        else:

            # If hand disappears,
            # reset pinch state.

            pinch_active = False

            cv2.putText(
                frame,
                "CURSOR: INACTIVE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # --------------------------------
        # Title
        # --------------------------------

        cv2.putText(
            frame,
            "AI VIRTUAL MOUSE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # --------------------------------
        # Display
        # --------------------------------

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # ESC
        if cv2.waitKey(1) & 0xFF == 27:

            break

    # --------------------------------
    # Cleanup
    # --------------------------------

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()