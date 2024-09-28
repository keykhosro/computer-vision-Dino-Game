# Dino Game Controller Using Hand Gestures
This repository contains Python code for controlling the Chrome Dino game using a webcam and hand gestures. The project is divided into three main parts:

## Hand Detection (part1.py):
Utilizes skin detection and deep learning techniques inspired by the research paper "Hand Gesture Recognition with Skin Detection and Deep Learning Method" by Hanwen Huang et al. The implementation improves upon the original paper's method, refining the hand detection model to address issues such as varying skin tones, lighting conditions, and noise in the background.

## Gesture Recognition (part2.py):
Determines whether the hand is open or closed using edge detection and the Hough Transform to count the number of prominent lines in the image. Open hand gestures initiate a jump in the game, while a closed hand does not trigger any action.

## Game Control (part3.py):
Integrates hand detection and gesture recognition to send commands to the Dino game based on the identified gestures. The program captures the player's hand as a sample to adapt to varying lighting and individual skin tones, making it robust to environmental changes.

## How to Run:
Open Chrome and navigate to chrome://dino.
Run the script part3.py and follow the on-screen instructions to calibrate the hand sample.
Use hand gestures (open hand for jumping) to control the Dino and enjoy the game!
## Key Features:
Real-time hand detection using webcam input.
Gesture recognition without complex machine learning models.
Improved accuracy in diverse environments.
Feel free to explore and contribute!

