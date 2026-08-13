import pyautogui


class MouseController:

    def __init__(self, smoothing=5):

        self.screen_width, self.screen_height = pyautogui.size()

        self.smoothing = smoothing

        self.previous_x = 0
        self.previous_y = 0

    def move(self, x, y):

        smooth_x = (
            self.previous_x
            + (x - self.previous_x) / self.smoothing
        )

        smooth_y = (
            self.previous_y
            + (y - self.previous_y) / self.smoothing
        )

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        self.previous_x = smooth_x
        self.previous_y = smooth_y

    def left_click(self):

        pyautogui.click()

    def right_click(self):

        pyautogui.rightClick()

    def get_screen_size(self):

        return self.screen_width, self.screen_height