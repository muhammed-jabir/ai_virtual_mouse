import cv2
import mediapipe as mp


class EyeTracker:

    def __init__(
        self,
        max_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):

        # ==========================================
        # MEDIAPIPE FACE MESH
        # ==========================================

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # ==========================================
        # DRAWING
        # ==========================================

        self.mp_drawing = mp.solutions.drawing_utils

    # ==============================================
    # PROCESS FRAME
    # ==============================================

    def find_face(self, frame):

        # Convert BGR → RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Process face

        results = self.face_mesh.process(
            rgb_frame
        )

        return results

    # ==============================================
    # DRAW EYE LANDMARKS
    # ==============================================

    def draw_eye_landmarks(
        self,
        frame,
        face_landmarks
    ):

        # ------------------------------------------------
        # LEFT EYE
        # ------------------------------------------------

        left_eye_indices = [
            33,
            133,
            159,
            145,
            160,
            144,
            158,
            153
        ]

        # ------------------------------------------------
        # RIGHT EYE
        # ------------------------------------------------

        right_eye_indices = [
            362,
            263,
            386,
            374,
            387,
            373,
            385,
            380
        ]

        # ------------------------------------------------
        # Draw left eye
        # ------------------------------------------------

        for index in left_eye_indices:

            landmark = face_landmarks.landmark[index]

            x = int(
                landmark.x
                * frame.shape[1]
            )

            y = int(
                landmark.y
                * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                cv2.FILLED
            )

        # ------------------------------------------------
        # Draw right eye
        # ------------------------------------------------

        for index in right_eye_indices:

            landmark = face_landmarks.landmark[index]

            x = int(
                landmark.x
                * frame.shape[1]
            )

            y = int(
                landmark.y
                * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                cv2.FILLED
            )

    # ==============================================
    # GET EYE CENTERS
    # ==============================================

    def get_eye_centers(
        self,
        frame,
        face_landmarks
    ):

        # ==========================================
        # LEFT EYE
        # ==========================================

        left_indices = [
            33,
            133,
            159,
            145
        ]

        # ==========================================
        # RIGHT EYE
        # ==========================================

        right_indices = [
            362,
            263,
            386,
            374
        ]

        # ==========================================
        # LEFT CENTER
        # ==========================================

        left_x = 0
        left_y = 0

        for index in left_indices:

            landmark = (
                face_landmarks.landmark[index]
            )

            left_x += (
                landmark.x
                * frame.shape[1]
            )

            left_y += (
                landmark.y
                * frame.shape[0]
            )

        left_x /= len(left_indices)
        left_y /= len(left_indices)

        # ==========================================
        # RIGHT CENTER
        # ==========================================

        right_x = 0
        right_y = 0

        for index in right_indices:

            landmark = (
                face_landmarks.landmark[index]
            )

            right_x += (
                landmark.x
                * frame.shape[1]
            )

            right_y += (
                landmark.y
                * frame.shape[0]
            )

        right_x /= len(right_indices)
        right_y /= len(right_indices)

        return (
            int(left_x),
            int(left_y)
        ), (
            int(right_x),
            int(right_y)
        )