import cv2

from eye_tracking.eye_tracker import EyeTracker


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

    print()
    print("==============================")
    print("       GAZE TEST")
    print("==============================")
    print()
    print("Look LEFT")
    print("Look CENTER")
    print("Look RIGHT")
    print()
    print("Press ESC to exit.")
    print()

    while True:

        success, frame = cap.read()

        if not success:

            break

        # Mirror

        frame = cv2.flip(
            frame,
            1
        )

        # ======================================
        # FACE
        # ======================================

        results = eye_tracker.find_face(
            frame
        )

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            # ==================================
            # DRAW LANDMARKS
            # ==================================

            eye_tracker.draw_eye_landmarks(
                frame,
                face_landmarks
            )

            # ==================================
            # GAZE
            # ==================================

            (
                direction,
                gaze_ratio,
                left_iris,
                right_iris
            ) = eye_tracker.get_gaze_direction(
                frame,
                face_landmarks
            )

            # ==================================
            # DRAW IRIS
            # ==================================

            cv2.circle(
                frame,
                left_iris,
                6,
                (0, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                right_iris,
                6,
                (0, 0, 255),
                cv2.FILLED
            )

            # ==================================
            # DISPLAY DIRECTION
            # ==================================

            cv2.putText(
                frame,
                f"GAZE: {direction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
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
            "AI Virtual Mouse - Gaze Test",
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