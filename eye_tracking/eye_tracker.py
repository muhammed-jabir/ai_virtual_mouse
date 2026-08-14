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

        # ==============================================
    # CALCULATE GAZE
    # ==============================================

    def get_gaze_direction(
        self,
        frame,
        face_landmarks
    ):

        # ==========================================
        # IRIS CENTERS
        # ==========================================

        (
            left_iris,
            right_iris
        ) = self.get_iris_centers(
            frame,
            face_landmarks
        )

        # ==========================================
        # LEFT EYE CORNERS
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
        # RIGHT EYE CORNERS
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

        # ==========================================
        # LEFT EYE RATIO
        # ==========================================

        left_min = min(
            left_corner[0],
            left_inner[0]
        )

        left_max = max(
            left_corner[0],
            left_inner[0]
        )

        left_width = (
            left_max - left_min
        )

        if left_width > 0:

            left_ratio = (
                left_iris[0] - left_min
            ) / left_width

        else:

            left_ratio = 0.5

        # ==========================================
        # RIGHT EYE RATIO
        # ==========================================

        right_min = min(
            right_corner[0],
            right_inner[0]
        )

        right_max = max(
            right_corner[0],
            right_inner[0]
        )

        right_width = (
            right_max - right_min
        )

        if right_width > 0:

            right_ratio = (
                right_iris[0] - right_min
            ) / right_width

        else:

            right_ratio = 0.5

        # ==========================================
        # AVERAGE BOTH EYES
        # ==========================================

        gaze_ratio = (
            left_ratio +
            right_ratio
        ) / 2

        # ==========================================
        # DEBUG
        # ==========================================

        print(
            f"Left: {left_ratio:.2f} | "
            f"Right: {right_ratio:.2f} | "
            f"Average: {gaze_ratio:.2f}"
        )

        # ==========================================
        # GAZE CLASSIFICATION
        # ==========================================

        if gaze_ratio < 0.42:

            direction = "LEFT"

        elif gaze_ratio > 0.58:

            direction = "RIGHT"

        else:

            direction = "CENTER"

        return (
            direction,
            gaze_ratio,
            left_iris,
            right_iris
        )
        
        # ==============================================
    # GET GAZE POSITION
    # ==============================================

    def get_gaze_position(
        self,
        frame,
        face_landmarks
    ):

        (
            direction,
            gaze_ratio,
            left_iris,
            right_iris
        ) = self.get_gaze_direction(
            frame,
            face_landmarks
        )

        # ==========================================
        # NORMALIZE GAZE
        # ==========================================
        #
        # Based on your webcam results:
        #
        # approximately:
        # LEFT  = 0.42
        # CENTER = 0.47
        # RIGHT = 0.50
        #
        # We expand this range.
        #

        min_ratio = 0.40
        max_ratio = 0.55

        gaze_x = (
            gaze_ratio - min_ratio
        ) / (
            max_ratio - min_ratio
        )

        # Keep between 0 and 1

        gaze_x = max(
            0.0,
            min(
                1.0,
                gaze_x
            )
        )

        # ==========================================
        # VERTICAL GAZE
        # ==========================================

        left_y = left_iris[1]
        right_y = right_iris[1]

        gaze_y = (
            left_y + right_y
        ) / 2

        # ==========================================
        # Return
        # ==========================================

        return (
            gaze_x,
            gaze_y,
            direction,
            gaze_ratio
        )