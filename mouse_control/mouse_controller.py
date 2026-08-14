import pyautogui


class MouseController:

    def __init__(
        self,
        smoothing=5,
        dead_zone=3
    ):

        # ==========================================
        # SCREEN SIZE
        # ==========================================

        self.screen_width, self.screen_height = (
            pyautogui.size()
        )

        # ==========================================
        # SETTINGS
        # ==========================================

        self.smoothing = smoothing
        self.dead_zone = dead_zone

        # ==========================================
        # PREVIOUS CURSOR POSITION
        # ==========================================

        self.previous_x = None
        self.previous_y = None

        # ==========================================
        # HYBRID CURSOR STATE
        # ==========================================

        self.hybrid_x = None
        self.hybrid_y = None

        # ==========================================
        # PYAutoGUI SAFETY
        # ==========================================

        pyautogui.PAUSE = 0.01

    # ==============================================
    # GET SCREEN SIZE
    # ==============================================

    def get_screen_size(self):

        return (
            self.screen_width,
            self.screen_height
        )

    # ==============================================
    # RESET CURSOR STATE
    # ==============================================

    def reset_position(self):

        self.previous_x = None
        self.previous_y = None

        self.hybrid_x = None
        self.hybrid_y = None

    # ==============================================
    # CLAMP SCREEN POSITION
    # ==============================================

    def clamp_position(self, x, y):

        x = max(
            0,
            min(
                int(x),
                self.screen_width - 1
            )
        )

        y = max(
            0,
            min(
                int(y),
                self.screen_height - 1
            )
        )

        return x, y

    # ==============================================
    # NORMAL HAND / FINGER CURSOR
    # ==============================================

    def move(self, x, y):

        # ------------------------------------------
        # Clamp
        # ------------------------------------------

        x, y = self.clamp_position(
            x,
            y
        )

        # ------------------------------------------
        # First movement
        # ------------------------------------------

        if (
            self.previous_x is None
            or self.previous_y is None
        ):

            self.previous_x = x
            self.previous_y = y

            pyautogui.moveTo(
                x,
                y
            )

            return

        # ------------------------------------------
        # Movement difference
        # ------------------------------------------

        delta_x = (
            x - self.previous_x
        )

        delta_y = (
            y - self.previous_y
        )

        # ------------------------------------------
        # Dead zone
        # ------------------------------------------

        if (
            abs(delta_x) < self.dead_zone
            and
            abs(delta_y) < self.dead_zone
        ):

            return

        # ------------------------------------------
        # Smooth movement
        # ------------------------------------------

        smooth_x = (
            self.previous_x
            +
            delta_x / self.smoothing
        )

        smooth_y = (
            self.previous_y
            +
            delta_y / self.smoothing
        )

        # ------------------------------------------
        # Clamp
        # ------------------------------------------

        smooth_x, smooth_y = (
            self.clamp_position(
                smooth_x,
                smooth_y
            )
        )

        # ------------------------------------------
        # Move cursor
        # ------------------------------------------

        pyautogui.moveTo(
            smooth_x,
            smooth_y
        )

        # ------------------------------------------
        # Save position
        # ------------------------------------------

        self.previous_x = smooth_x
        self.previous_y = smooth_y

    # ==============================================
    # HYBRID EYE + FINGER CURSOR
    # ==============================================

    def move_hybrid(
        self,
        eye_x,
        eye_y,
        finger_x,
        finger_y,
        eye_weight=0.7,
        finger_weight=0.3
    ):

        # ==========================================
        # VALIDATE WEIGHTS
        # ==========================================

        total_weight = (
            eye_weight +
            finger_weight
        )

        if total_weight <= 0:

            eye_weight = 0.7
            finger_weight = 0.3
            total_weight = 1.0

        eye_weight /= total_weight
        finger_weight /= total_weight

        # ==========================================
        # COMBINE EYE + FINGER
        # ==========================================

        target_x = (
            eye_x * eye_weight
            +
            finger_x * finger_weight
        )

        target_y = (
            eye_y * eye_weight
            +
            finger_y * finger_weight
        )

        # ==========================================
        # CLAMP
        # ==========================================

        target_x, target_y = (
            self.clamp_position(
                target_x,
                target_y
            )
        )

        # ==========================================
        # INITIALIZE
        # ==========================================

        if (
            self.hybrid_x is None
            or self.hybrid_y is None
        ):

            self.hybrid_x = target_x
            self.hybrid_y = target_y

            pyautogui.moveTo(
                target_x,
                target_y
            )

            return

        # ==========================================
        # DIFFERENCE
        # ==========================================

        delta_x = (
            target_x -
            self.hybrid_x
        )

        delta_y = (
            target_y -
            self.hybrid_y
        )

        # ==========================================
        # DEAD ZONE
        # ==========================================

        if (
            abs(delta_x) < self.dead_zone
            and
            abs(delta_y) < self.dead_zone
        ):

            return

        # ==========================================
        # SMOOTH HYBRID MOVEMENT
        # ==========================================

        smooth_x = (
            self.hybrid_x
            +
            delta_x / self.smoothing
        )

        smooth_y = (
            self.hybrid_y
            +
            delta_y / self.smoothing
        )

        # ==========================================
        # CLAMP
        # ==========================================

        smooth_x, smooth_y = (
            self.clamp_position(
                smooth_x,
                smooth_y
            )
        )

        # ==========================================
        # MOVE
        # ==========================================

        pyautogui.moveTo(
            smooth_x,
            smooth_y
        )

        # ==========================================
        # SAVE
        # ==========================================

        self.hybrid_x = smooth_x
        self.hybrid_y = smooth_y

    # ==============================================
    # EYE ONLY CURSOR
    # ==============================================

    def move_eye(
        self,
        x,
        y
    ):

        # ==========================================
        # Clamp
        # ==========================================

        x, y = self.clamp_position(
            x,
            y
        )

        # ==========================================
        # Initialize
        # ==========================================

        if (
            self.hybrid_x is None
            or self.hybrid_y is None
        ):

            self.hybrid_x = x
            self.hybrid_y = y

            pyautogui.moveTo(
                x,
                y
            )

            return

        # ==========================================
        # Difference
        # ==========================================

        delta_x = (
            x -
            self.hybrid_x
        )

        delta_y = (
            y -
            self.hybrid_y
        )

        # ==========================================
        # Dead zone
        # ==========================================

        if (
            abs(delta_x) < self.dead_zone
            and
            abs(delta_y) < self.dead_zone
        ):

            return

        # ==========================================
        # Smooth
        # ==========================================

        smooth_x = (
            self.hybrid_x
            +
            delta_x / self.smoothing
        )

        smooth_y = (
            self.hybrid_y
            +
            delta_y / self.smoothing
        )

        # ==========================================
        # Clamp
        # ==========================================

        smooth_x, smooth_y = (
            self.clamp_position(
                smooth_x,
                smooth_y
            )
        )

        # ==========================================
        # Move
        # ==========================================

        pyautogui.moveTo(
            smooth_x,
            smooth_y
        )

        # ==========================================
        # Save
        # ==========================================

        self.hybrid_x = smooth_x
        self.hybrid_y = smooth_y

    # ==============================================
    # LEFT CLICK
    # ==============================================

    def left_click(self):

        pyautogui.click()

    # ==============================================
    # RIGHT CLICK
    # ==============================================

    def right_click(self):

        pyautogui.rightClick()

    # ==============================================
    # DOUBLE CLICK
    # ==============================================

    def double_click(self):

        pyautogui.doubleClick()

    # ==============================================
    # SCROLL
    # ==============================================

    def scroll(self, amount):

        pyautogui.scroll(
            int(amount)
        )

    # ==============================================
    # HORIZONTAL SCROLL
    # ==============================================

    def horizontal_scroll(self, amount):

        pyautogui.hscroll(
            int(amount)
        )

    # ==============================================
    # DRAG
    # ==============================================

    def mouse_down(self):

        pyautogui.mouseDown()

    def mouse_up(self):

        pyautogui.mouseUp()

    # ==============================================
    # CURRENT CURSOR POSITION
    # ==============================================

    def get_position(self):

        return pyautogui.position()