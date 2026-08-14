import cv2

from eye_tracking.eye_tracker import EyeTracker
from mouse_control.mouse_controller import MouseController


def map_value(
    value,
    input_min,
    input_max,
    output_min,
    output_max
):

    return (
        (value - input_min)
        * (output_max - output_min)
        /
        (input_max - input_min)
        + output_min
    )


def main():

    # ==========================================
    # CAMERA
    # ==========================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    # ==========================================
    # EYE TRACKER
    # ==========================================

    eye_tracker = EyeTracker(
        max_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    )

    # ==========================================
    # MOUSE
    # ==========================================

    mouse = MouseController(
        smoothing=8,
        dead_zone=2
    )

    screen_width, screen_height = (
        mouse.get_screen_size()
    )

    print()
    print("==============================")
    print("       EYE CURSOR TEST")
    print("==============================")
    print()
    print("Move your eyes LEFT / RIGHT.")
    print()
    print("The cursor should move.")
    print()
    print("Press ESC to exit.")
    print()

    # ==========================================
    # FACE POSITION
    # ==========================================

    previous_face_y = None

    # ==========================================
    # MAIN LOOP
    # ==========================================

    while True:

        success, frame = cap.read()

        if not success:

            break

        frame = cv2.flip(
            frame,
            1
        )

        # ======================================
        # FACE DETECTION
        # ======================================

        results = eye_tracker.find_face(
            frame
        )

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            # ==================================
            # DRAW
            # ==================================

            eye_tracker.draw_eye_landmarks(
                frame,
                face_landmarks
            )

            # ==================================
            # GAZE
            # ==================================

            (
                gaze_x,
                gaze_y,
                direction,
                gaze_ratio
            ) = eye_tracker.get_gaze_position(
                frame,
                face_landmarks
            )

            # ==================================
            # SCREEN X
            # ==================================

            screen_x = (
                gaze_x
                * screen_width
            )

            # ==================================
            # FACE CENTER Y
            # ==================================

            nose = (
                face_landmarks.landmark[1]
            )

            face_y = (
                nose.y
                * frame.shape[0]
            )

            # ==================================
            # INITIALIZE
            # ==================================

            if previous_face_y is None:

                previous_face_y = face_y

            # ==================================
            # FACE MOVEMENT Y
            # ==================================

            face_delta_y = (
                face_y
                - previous_face_y
            )

            # ==================================
            # CONVERT FACE MOVEMENT
            # ==================================

            current_mouse_y = (
                screen_height / 2
            )

            if abs(face_delta_y) > 2:

                current_mouse_y += (
                    face_delta_y
                    * 25
                )

            # ==================================
            # CLAMP
            # ==================================

            current_mouse_y = max(
                0,
                min(
                    screen_height,
                    current_mouse_y
                )
            )

            # ==================================
            # MOVE CURSOR
            # ==================================

            mouse.move(
                screen_x,
                current_mouse_y
            )

            # ==================================
            # DISPLAY
            # ==================================

            cv2.putText(
                frame,
                f"GAZE: {direction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Ratio: {gaze_ratio:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Screen X: {int(screen_x)}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EYE CURSOR: ACTIVE",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            previous_face_y = face_y

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ======================================
        # DISPLAY
        # ======================================

        cv2.imshow(
            "AI Virtual Mouse - Eye Cursor",
            frame
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == 27
        ):

            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()