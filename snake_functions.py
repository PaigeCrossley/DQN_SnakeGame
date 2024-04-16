from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from collections import deque
import numpy as np

def newGame(): # Initialize the game
    driver = webdriver.Chrome()

    driver.get('snake_game.html')

    WebDriverWait(driver, 120).until(page_loaded)

    return driver

def imageProcessing(image): # Crop, resize and grayscale the images
    left = 400
    top = 400
    right = 850
    bottom = 850

    img = image.crop((left, top, right, bottom))
    img1 = img.resize((45, 45))
    img1 = img1.convert("L")

    return img1

# stackImages code came from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb
def stackImages(stackedImages, img, isNewEpisode):
    if isNewEpisode:
        stackedImages  =  deque([np.zeros((45,45), dtype=np.intc) for i in range(4)], maxlen=4)

        stackedImages.append(img)
        stackedImages.append(img)
        stackedImages.append(img)
        stackedImages.append(img)

    else:
        stackedImages.append(img)

    stackedState = np.stack(stackedImages, axis=2)

    return stackedImages, stackedState

def page_loaded(driver): # Check that the page is fully loaded
    return ((driver.execute_script("return document.readyState") == "complete") and EC.presence_of_element_located((By.ID, 'score'))(driver))