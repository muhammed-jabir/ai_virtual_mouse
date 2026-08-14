
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
    print("       EYE TRACKING TEST")
    print("==============================")
    print()
    print("Look at the camera.")
    print("Eye landmarks should appear.")
    print()
    print("Press ESC to exit.")
    print()

    # ==========================================
    # LOOP
    # ==========================================

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        # Mirror

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

        # ======================================
        # FACE FOUND
        # ======================================

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            # ----------------------------------
            # Draw eye landmarks
            # ----------------------------------

            eye_tracker.draw_eye_landmarks(
                frame,
                face_landmarks
            )

            # ----------------------------------
            # Eye centers
            # ----------------------------------

            left_eye, right_eye = (
                eye_tracker.get_eye_centers(
                    frame,
                    face_landmarks
                )
            )

            # ----------------------------------
            # Draw centers
            # ----------------------------------

            cv2.circle(
                frame,
                left_eye,
                6,
                (0, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                right_eye,
                6,
                (0, 0, 255),
                cv2.FILLED
            )

            # ----------------------------------
            # Status
            # ----------------------------------

            cv2.putText(
                frame,
                "EYES: DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "EYES: NOT DETECTED",
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
            "AI Virtual Mouse - Eye Test",
            frame
        )

        # ======================================
        # ESC
        # ======================================

        if (
            cv2.waitKey(1)
            & 0xFF
            == 27
        ):

            break

    # ==========================================
    # CLEANUP
    # ==========================================

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()