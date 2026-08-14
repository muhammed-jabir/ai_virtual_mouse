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

    # ==============================================
    # PROCESS FRAME
    # ==============================================

    def find_face(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        return results

    # ==============================================
    # GET LANDMARK PIXEL POSITION
    # ==============================================

    def get_point(
        self,
        frame,
        face_landmarks,
        landmark_id
    ):

        landmark = (
            face_landmarks.landmark[landmark_id]
        )

        x = int(
            landmark.x
            * frame.shape[1]
        )

        y = int(
            landmark.y
            * frame.shape[0]
        )

        return x, y

    # ==============================================
    # DRAW EYE LANDMARKS
    # ==============================================

    def draw_eye_landmarks(
        self,
        frame,
        face_landmarks
    ):

        # Eye boundary landmarks

        eye_indices = [
            # Left eye
            33,
            133,
            159,
            145,

            # Right eye
            362,
            263,
            386,
            374,

            # Left iris
            468,
            469,
            470,
            471,
            472,

            # Right iris
            473,
            474,
            475,
            476,
            477
        ]

        for index in eye_indices:

            x, y = self.get_point(
                frame,
                face_landmarks,
                index
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                cv2.FILLED
            )

    # ==============================================
    # GET IRIS CENTER
    # ==============================================

    def get_iris_centers(
        self,
        frame,
        face_landmarks
    ):

        # ==========================================
        # LEFT IRIS
        # ==========================================

        left_iris_indices = [
            468,
            469,
            470,
            471,
            472
        ]

        # ==========================================
        # RIGHT IRIS
        # ==========================================

        right_iris_indices = [
            473,
            474,
            475,
            476,
            477
        ]

        # ==========================================
        # LEFT IRIS CENTER
        # ==========================================

        left_x = 0
        left_y = 0

        for index in left_iris_indices:

            x, y = self.get_point(
                frame,
                face_landmarks,
                index
            )

            left_x += x
            left_y += y

        left_x /= len(
            left_iris_indices
        )

        left_y /= len(
            left_iris_indices
        )

        # ==========================================
        # RIGHT IRIS CENTER
        # ==========================================

        right_x = 0
        right_y = 0

        for index in right_iris_indices:

            x, y = self.get_point(
                frame,
                face_landmarks,
                index
            )

            right_x += x
            right_y += y

        right_x /= len(
            right_iris_indices
        )

        right_y /= len(
            right_iris_indices
        )

        return (
            int(left_x),
            int(left_y)
        ), (
            int(right_x),
            int(right_y)
        )

    # ==============================================
    # GET EYE BOUNDARIES
    # ==============================================

    def get_eye_boundaries(
        self,
        frame,
        face_landmarks
    ):

        # ==========================================
        # LEFT EYE
        # ==========================================

        left_corner = self.get_point(
            frame,
            face_landmarks,
            33
        )

        left_inner = self.get_point(
            frame,
            face_landmarks,
            133
        )

        # ==========================================
        # RIGHT EYE
        # ==========================================

        right_corner = self.get_point(
            frame,
            face_landmarks,
            362
        )

        right_inner = self.get_point(
            frame,
            face_landmarks,
            263
        )

        return (
            left_corner,
            left_inner
        ), (
            right_corner,
            right_inner
        )

    # ==============================================
    # CALCULATE GAZE
    # ==============================================

    def get_gaze_direction(
        self,
        frame,
        face_landmarks
    ):

        (
            left_iris,
            right_iris
        ) = self.get_iris_centers(
            frame,
            face_landmarks
        )

        (
            left_eye,
            right_eye
        ) = self.get_eye_boundaries(
            frame,
            face_landmarks
        )

        # ==========================================
        # LEFT EYE RATIO
        # ==========================================

        left_x_min = min(
            left_eye[0][0],
            left_eye[1][0]
        )

        left_x_max = max(
            left_eye[0][0],
            left_eye[1][0]
        )

        left_range = (
            left_x_max
            - left_x_min
        )

        if left_range == 0:

            left_ratio = 0.5

        else:

            left_ratio = (
                left_iris[0]
                - left_x_min
            ) / left_range

        # ==========================================
        # RIGHT EYE RATIO
        # ==========================================

        right_x_min = min(
            right_eye[0][0],
            right_eye[1][0]
        )

        right_x_max = max(
            right_eye[0][0],
            right_eye[1][0]
        )

        right_range = (
            right_x_max
            - right_x_min
        )

        if right_range == 0:

            right_ratio = 0.5

        else:

            right_ratio = (
                right_iris[0]
                - right_x_min
            ) / right_range

        # ==========================================
        # AVERAGE
        # ==========================================

        gaze_ratio = (
            left_ratio
            + right_ratio
        ) / 2

        # ==========================================
        # CLASSIFY GAZE
        # ==========================================

        if gaze_ratio < 0.40:

            direction = "LEFT"

        elif gaze_ratio > 0.60:

            direction = "RIGHT"

        else:

            direction = "CENTER"

        return (
            direction,
            gaze_ratio,
            left_iris,
            right_iris
        )