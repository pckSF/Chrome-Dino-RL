from __future__ import annotations

import base64
import io
import time

import cv2
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


class Interface:
    """Interface for interaction between the agent and the chrome dino game.

    An interface that uses a selenium webdriver to start a chrome instance in
    which the chrome-dino game is run. Provides methods that enable interaction
    between the agent and the game.

    Attributes:
        driver: The selenium chrome instance
        runner: A link to the game in the driver instance
        get_canvas_js: A string containing a javascript to extract the image from
            the div class="runner-canvas"
    """

    def __init__(self) -> None:
        """Initialises the chrome interface.

        - Uses chrome and fetch the latest webdriver
        - Loads the local chrome://dino game
        - Reads the get_canvas.js script into a class variable
        """
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument("--mute-audio")
        self.driver = webdriver.Chrome(
            ChromeDriverManager().install(), chrome_options=self.chrome_options
        )
        self.driver.set_window_position(x=10, y=10)
        self.driver.set_window_size(800, 400)
        try:
            self.driver.get("chrome://dino/")
        except Exception:
            print("Ignoring WebDriverException: net::ERR_INTERNET_DISCONNECTED")
        self.driver.execute_script("Runner.config.ACCELERATION = 0")
        self.runner = self.driver.find_element_by_xpath("//body[@id='t']")
        self.get_canvas_js = open("scs/get_canvas.js").read()

    def re_start(self) -> np.ndarray:
        """Simulates a SPACE keypress to start or restart the game"""
        self.action(3)
        return self.get_canvas()

    def get_score(self) -> int:
        """
        Gets and return sthe current score by accessing its variables with javascript
        """
        score_array = self.driver.execute_script(
            "return Runner.instance_.distanceMeter.digits"
        )
        return int("".join(score_array))

    def check_crashed(self) -> bool:
        """Gets and returns the playing and crashed gamestates"""
        # playing = self.driver.execute_script("return Runner.instance_.playing")
        crashed = self.driver.execute_script("return Runner.instance_.crashed")
        return crashed

    def action(self, action: int) -> tuple[np.ndarray, int, bool]:
        """
        Performs one of the possible actions:
            0 = do nothing / run straight:
            1 = simulate ARROW_UP keypress = jump
            2 = simulate ARROW_DOwn keypress = crouch
            else = simulate SPACE keypress = jump

        Args:
            action: An integer representing one of three actions

        Returns:
            Environment parameters after action has been executed
        """
        if action == 0:
            pass
        elif action == 1:
            self.runner.send_keys(Keys.ARROW_UP)
        elif action == 2:
            self.runner.send_keys(Keys.ARROW_DOWN)
        else:
            self.runner.send_keys(Keys.SPACE)
        time.sleep(0.2)
        state = self.get_canvas()
        reward = self.get_score()
        terminal = self.check_crashed()
        return state, reward, terminal

    def get_canvas(self) -> np.ndarray:
        """
        Extracts the canvas by executing get_canvas.js
        Decodes canvas into a numpy array and returns the reduced image.
        Applies preprocessing in form of of cutting pixel values above and below
        given threshold to clean the image of background noise.
        """
        base64_canvas = self.driver.execute_script(self.get_canvas_js)
        decoded_canvas = base64.b64decode(base64_canvas)
        canvas_image = np.array(Image.open(io.BytesIO(decoded_canvas)))
        reduced_dim_image = canvas_image[25:260, 50:400, 0]
        reduced_dim_image[reduced_dim_image > 100] = 0
        reduced_dim_image[reduced_dim_image > 0] = 1
        rescaled_image = cv2.resize(reduced_dim_image, (100, 50))
        return rescaled_image[np.newaxis, :]

    def close(self) -> None:
        """Closes the active chrome session"""
        self.driver.close()
