import cv2
import numpy as np

DIM = 2
def gray_progress_bar(image,
                      stage_clear,
                      time_remaining,
                      keys):
    # Resize
    scale_image = cv2.convertScaleAbs(
        cv2.resize(
            image, (84*DIM, 84*DIM), interpolation=cv2.INTER_CUBIC), alpha=(
            255.0 / 1.0))
    # Add state zone
    state_image = cv2.rectangle(scale_image, (0, 75*DIM), (84*DIM, 84*DIM), (0, 0, 0), -1)
    # Add floor progress bar
    progress_bar = int((stage_clear / 26) * 84*DIM)
    state_image = cv2.rectangle(
        state_image, (0, 75*DIM), (progress_bar, 78*DIM), (70, 70, 70), -1)
    # Add time remaining progress bar
    progress_bar = int((time_remaining / 10000) * 84*DIM)
    state_image = cv2.rectangle(
        state_image, (0, 78*DIM), (progress_bar, 81*DIM), (140, 140, 140), -1)
    # keys state
    if keys > 0:
        state_image = cv2.rectangle(
            state_image, (0, 81*DIM), (84*DIM, 84*DIM), (255, 255, 255), -1)
    # Gray
    gray_image = cv2.cvtColor(state_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('progress', cv2.resize(gray_image, (840, 840), interpolation=cv2.INTER_CUBIC))
    # cv2.waitKey(20)
    return gray_image


def rgb_progress_bar(image,
                     stage_clear,
                     time_remaining,
                     keys):
    # Resize
    scale_image = cv2.convertScaleAbs(image, alpha=(255.0 / 1.0))
    # Add state zone
    state_image = cv2.rectangle(
        scale_image, (0, 150), (168, 168), (0, 0, 0), -1)
    # Add floor progress bar
    progress_bar = int((stage_clear / 26) * 168)
    state_image = cv2.rectangle(
        state_image, (0, 150), (progress_bar, 156), (70, 70, 70), -1)
    # Add time remaining progress bar
    progress_bar = int((time_remaining / 10000) * 168)
    state_image = cv2.rectangle(
        state_image, (0, 156), (progress_bar, 162), (140, 140, 140), -1)
    # keys state
    if keys > 0:
        state_image = cv2.rectangle(
            state_image, (0, 162), (168, 168), (255, 255, 255), -1)

    # cv2.imshow('progress', cv2.resize(state_image, (840, 840), interpolation=cv2.INTER_CUBIC))
    # cv2.waitKey(20)

    return state_image
