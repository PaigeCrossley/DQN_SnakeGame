from selenium.common.exceptions import UnexpectedAlertPresentException, NoSuchWindowException, ElementNotInteractableException, NoAlertPresentException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from itertools import count
import time
from collections import deque
from PIL import Image
from io import BytesIO
import torch
import numpy as np

from snake_functions import *
from DQN import *

def main():

    stackedImages  =  deque([np.zeros((45,45), dtype=np.intc) for i in range(4)], maxlen=4)
    isNewEpisode = True
    
    driver = newGame()

    num_episodes = 100

    f = open('logs.txt', 'w')
    for i_episode in range(num_episodes + 1):
        f.write(f'iteration: {i_episode}\n')
        while True:
            f.write('while true...\n')
            try:
                f.write('try...\n')
                screenshot = driver.get_screenshot_as_png()

                img = Image.open(BytesIO(screenshot))

                img = imageProcessing(img)

                stackedImages, stackedState = stackImages(stackedImages, img, isNewEpisode)
                isNewEpisode = False

                state = torch.tensor(stackedState, dtype=torch.float32, device=device).unsqueeze(0)

                f.write(str('for t...\n'))
                for t in count():
                    score_element = driver.execute_script("return document.getElementById('score').innerText")
                    f.write('score_element\n')
                    if score_element is not None:
                        score = int(score_element)

                    f.write(f'score int: {score}, {type(score)}\n')

                    action = select_action(state)
                    key = actions[action.item()]

                    body_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    f.write('body_element\n')
                    if body_element is not None:
                        f.write('not None\n')
                        body_element.send_keys(actions[action.item()])

                    screenshot = driver.get_screenshot_as_png()
                    img_next = Image.open(BytesIO(screenshot))
                    img_next_processed = imageProcessing(img_next)
                    stackedImages, stackedState_next = stackImages(stackedImages, img_next_processed, isNewEpisode)

                    next_state = torch.tensor(stackedState_next, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    new_score_element = driver.execute_script("return document.getElementById('score').innerText")
                    f.write('new_score_element\n')
                    if new_score_element is not None:
                        new_score = int(new_score_element)

                    f.write(f'new score int: {new_score}, {type(new_score)}\n')

                    reward = torch.tensor([computeReward(new_score > score)], device=device)

                    # Store the transition in memory
                    memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)

            except UnexpectedAlertPresentException: # An alert pops up when the game ends, this handles the punishment & duration recording
                f.write(f'game over {time.time()}')
                print(f'Game Over: {i_episode}')
                reward = torch.tensor([computeReward(0)], device=device)
                # Store the transition in memory
                memory.push(state, action, next_state, reward)
                episode_durations.append(t + 1)
                f.write(f'{scores}\n')
                scores.append(new_score if new_score else score if score else 0)
                f.write(f'{scores}\n')
                plot_durations_scores()

                try:
                    driver.switch_to.alert.accept()
                    break
                except NoAlertPresentException:
                    break

            except ElementNotInteractableException:
                f.write('element not interactable... trying again\n')
                continue

            except NoSuchWindowException: # Handles what happens when I manually close the game window
                f.write('no such window exception\n')
                return False

    f.write('Complete')
    f.close()
    plot_durations_scores(show_result=True)
    plt.ioff()
    plt.show()
    driver.close()

if __name__ == "__main__":
    main()