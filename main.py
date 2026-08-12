import cv2

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_HANDS,
)

from hand_tracking.hand_detector import HandDetector


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = HandDetector(
        max_hands=MAX_HANDS,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE,
    )

    print("Virtual Mouse started.")
    print("Press ESC to exit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, results = detector.find_hands(frame)

        # Display information
        cv2.putText(
            frame,
            "AI VIRTUAL MOUSE - Hand Tracking",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if results.multi_hand_landmarks:
            cv2.putText(
                frame,
                "Hand: DETECTED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Hand: NOT DETECTED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("AI Virtual Mouse", frame)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()