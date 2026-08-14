import pyautogui


class MouseController:

    def __init__(
        self,
        smoothing=5,
        dead_zone=3
    ):

        # ==========================================
        # SCREEN INFORMATION
        # ==========================================

        self.screen_width, self.screen_height = (
            pyautogui.size()
        )

        # ==========================================
        # CURSOR SETTINGS
        # ==========================================

        self.smoothing = smoothing
        self.dead_zone = dead_zone

        # Previous cursor position
        self.previous_x = None
        self.previous_y = None

    # ==============================================
    # CURSOR MOVEMENT
    # ==============================================

    def move(self, x, y):
        """
        Move the Windows cursor using
        smoothing and dead-zone filtering.
        """

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
                int(x),
                int(y)
            )

            return

        # ------------------------------------------
        # Calculate movement
        # ------------------------------------------

        delta_x = x - self.previous_x
        delta_y = y - self.previous_y

        # ------------------------------------------
        # Dead zone
        # ------------------------------------------
        # Ignore very small movements caused
        # by MediaPipe tracking jitter.

        if (
            abs(delta_x) < self.dead_zone
            and abs(delta_y) < self.dead_zone
        ):

            return

        # ------------------------------------------
        # Smooth movement
        # ------------------------------------------

        smooth_x = (
            self.previous_x
            + delta_x / self.smoothing
        )

        smooth_y = (
            self.previous_y
            + delta_y / self.smoothing
        )

        # ------------------------------------------
        # Move actual Windows cursor
        # ------------------------------------------

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        # ------------------------------------------
        # Save position
        # ------------------------------------------

        self.previous_x = smooth_x
        self.previous_y = smooth_y

    # ==============================================
    # LEFT CLICK
    # ==============================================

    def left_click(self):
        """
        Perform one left mouse click.
        """

        pyautogui.click()

    # ==============================================
    # RIGHT CLICK
    # ==============================================

    def right_click(self):
        """
        Perform one right mouse click.
        """

        pyautogui.rightClick()

    # ==============================================
    # SCROLL
    # ==============================================

    def scroll(self, amount):
        """
        Scroll vertically.

        Positive value  -> Scroll UP
        Negative value  -> Scroll DOWN
        """

        if amount == 0:
            return

        pyautogui.scroll(
            int(amount)
        )

    # ==============================================
    # SCREEN SIZE
    # ==============================================

    def get_screen_size(self):
        """
        Return screen width and height.
        """

        return (
            self.screen_width,
            self.screen_height
        )