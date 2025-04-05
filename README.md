# Hand Gesture Controller

A Python-based hand gesture recognition system using **MediaPipe** and **OpenCV**. It maps hand gestures to keyboard actions for hands-free control, making it ideal for gaming, presentations, or accessibility applications.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Calibration Mode](#calibration-mode)
6. [Debug Mode](#debug-mode)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

- **Real-time Gesture Detection**: Detects hand gestures using a webcam.
- **Keyboard Mapping**: Maps gestures to specific keyboard keys (e.g., arrow keys, spacebar).
- **Sound Feedback**: Provides optional sound feedback for detected gestures.
- **Calibration Mode**: Allows users to fine-tune detection thresholds and other settings.
- **Debug Mode**: Displays real-time information about finger positions and gesture detection.
- **Logging**: Logs gesture detections and actions to a text file for analysis.
- **Visual Indicators**: Overlays directional arrows on the video feed to indicate detected gestures.

---

## Installation

### Prerequisites
- Python 3.x
- A webcam

### Install Dependencies
Install the required Python libraries using `pip`:

\`\`\`bash
pip install opencv-python mediapipe numpy pyautogui pygame
\`\`\`

### Clone the Repository
Clone this repository to your local machine:

\`\`\`bash
git clone https://github.com/dkpython7/Game-Hand-Gesture-Controller.git
cd Game-Hand-Gesture-Controller
\`\`\`

---

## Usage

1. **Run the Application**:
   Execute the script using Python:

   \`\`\`bash
   python game.py
   \`\`\`

2. **Perform Gestures**:
   - Place your hand in front of the webcam.
   - Perform one of the supported gestures:
     - **Thumb to Palm**: Simulates the "up" arrow key.
     - **Closed Fist**: Simulates the "down" arrow key.
     - **Index Finger Only**: Simulates the "left" arrow key.
     - **Index and Middle Fingers**: Simulates the "right" arrow key.
     - **All Fingers Extended**: Simulates the "space" key.

3. **Exit**:
   Press \`ESC\` to quit the application.

---

## Configuration

The configuration file (\`gesture_config.json\`) contains the following parameters:

| Parameter                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| \`gestures\`                 | Maps gestures to actions (e.g., \`thumb_to_palm → up\`).                     |
| \`keys\`                     | Maps actions to keyboard keys (e.g., \`up → up arrow\`).                     |
| \`cooldown\`                 | Minimum time (in seconds) between consecutive actions.                     |
| \`detection_threshold\`      | Sensitivity for gesture recognition.                                       |
| \`thumb_to_palm_threshold\`  | Distance threshold for detecting the thumb-to-palm gesture.                |
| \`finger_detection_threshold\` | Threshold for detecting extended fingers.                                |
| \`sound_enabled\`            | Enables/disables sound feedback.                                           |

You can edit this file manually or adjust settings through calibration mode.

---

## Calibration Mode

To fine-tune detection parameters:
1. Press \`C\` to enter calibration mode.
2. Use the following keys to adjust settings:
   - \`+\`/\`-\`: Adjust detection threshold.
   - \`[\`/\`]\`: Adjust cooldown time.
   - \`{\`/\`}\`: Adjust thumb-to-palm threshold.
   - \`S\`: Save updated settings to \`gesture_config.json\`.
   - \`T\`: Toggle sound feedback.
   - \`D\`: Toggle debug mode.

Press \`C\` again to exit calibration mode.

---

## Debug Mode

Debug mode displays additional information on the video feed:
- Finger status (up/down).
- Raw and smoothed gesture detection results.

To toggle debug mode, press \`D\`.

---

## Troubleshooting

1. **Gesture Not Detected**:
   - Ensure proper lighting and positioning of your hand.
   - Adjust detection thresholds in calibration mode.

2. **Incorrect Action Triggered**:
   - Increase the detection threshold or cooldown time.

3. **No Sound Feedback**:
   - Ensure sound files (\`up.wav\`, \`down.wav\`, etc.) are present in the \`sounds\` directory.

4. **Performance Issues**:
   - Reduce the camera resolution or detection confidence.

---

## Contributing

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss your ideas.
