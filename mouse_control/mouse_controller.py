import pyautogui


class MouseController:

    def __init__(
        self,
        smoothing=5,
        dead_zone=3
    ):

        self.screen_width, self.screen_height = (
            pyautogui.size()
        )

        self.smoothing = smoothing
        self.dead_zone = dead_zone

        self.previous_x = None
        self.previous_y = None

    def move(self, x, y):

        # --------------------------------
        # First movement
        # --------------------------------

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

        # --------------------------------
        # Calculate movement
        # --------------------------------

        delta_x = x - self.previous_x
        delta_y = y - self.previous_y

        # --------------------------------
        # Dead zone
        # --------------------------------
        # Ignore tiny movements caused
        # by hand-tracking jitter.

        if (
            abs(delta_x) < self.dead_zone
            and abs(delta_y) < self.dead_zone
        ):

            return

        # --------------------------------
        # Smooth movement
        # --------------------------------

        smooth_x = (
            self.previous_x
            + delta_x / self.smoothing
        )

        smooth_y = (
            self.previous_y
            + delta_y / self.smoothing
        )

        # --------------------------------
        # Move cursor
        # --------------------------------

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        # --------------------------------
        # Save position
        # --------------------------------

        self.previous_x = smooth_x
        self.previous_y = smooth_y

    def left_click(self):

        pyautogui.click()

    def right_click(self):

        pyautogui.rightClick()

    def get_screen_size(self):

        return (
            self.screen_width,
            self.screen_height
        )