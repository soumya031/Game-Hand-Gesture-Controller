import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import json
import os
import pygame
from datetime import datetime

class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [3, 7, 11, 15, 19]
        self.mcp_ids = [1, 5, 9, 13, 17]
        self.lm_list = []
        self.current_gesture = None
        self.last_gesture_time = time.time()
        self.gesture_history = []
        self.history_size = 5
        pygame.mixer.init()
        self.sounds_dir = 'sounds'
        if not os.path.exists(self.sounds_dir):
            os.makedirs(self.sounds_dir)
        self.sounds = {
            'up': pygame.mixer.Sound(f'{self.sounds_dir}/up.wav') if os.path.exists(f'{self.sounds_dir}/up.wav') else None,
            'down': pygame.mixer.Sound(f'{self.sounds_dir}/down.wav') if os.path.exists(f'{self.sounds_dir}/down.wav') else None,
            'left': pygame.mixer.Sound(f'{self.sounds_dir}/left.wav') if os.path.exists(f'{self.sounds_dir}/left.wav') else None,
            'right': pygame.mixer.Sound(f'{self.sounds_dir}/right.wav') if os.path.exists(f'{self.sounds_dir}/right.wav') else None,
            'neutral': pygame.mixer.Sound(f'{self.sounds_dir}/neutral.wav') if os.path.exists(f'{self.sounds_dir}/neutral.wav') else None
        }
        self.default_config = {
            'gestures': {
                'fist': 'down',
                'thumb_to_palm': 'up',
                'two_fingers': 'right',
                'one_finger': 'left',
                'all_fingers': 'neutral'
            },
            'keys': {
                'up': 'up',
                'down': 'down',
                'left': 'left',
                'right': 'right',
                'neutral': 'space'
            },
            'cooldown': 0.3,
            'detection_threshold': 0.6,
            'sound_enabled': True,
            'thumb_to_palm_threshold': 0.3,
            'finger_detection_threshold': 0.6
        }
        self.config_path = 'gesture_config.json'
        self.config = self.load_config()
        self.calibration_mode = False
        self.debug_mode = True
        self.p_time = 0
        self.running = True
        self.log_file = f"gesture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.create_visual_indicators()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                for key, value in self.default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading config: {e}. Using default.")
                return self.default_config
        else:
            with open(self.config_path, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            return self.default_config

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print("Settings saved successfully.")

    def create_visual_indicators(self):
        self.icons_dir = 'icons'
        if not os.path.exists(self.icons_dir):
            os.makedirs(self.icons_dir)
        self.indicators = {
            'up': self.create_arrow_indicator('up'),
            'down': self.create_arrow_indicator('down'),
            'left': self.create_arrow_indicator('left'),
            'right': self.create_arrow_indicator('right'),
            'neutral': self.create_neutral_indicator()
        }

    def create_arrow_indicator(self, direction):
        icon = np.zeros((80, 80, 3), dtype=np.uint8)
        if direction == 'up':
            points = np.array([[40, 10], [10, 70], [70, 70]])
            color = (0, 255, 0)
        elif direction == 'down':
            points = np.array([[40, 70], [10, 10], [70, 10]])
            color = (0, 0, 255)
        elif direction == 'left':
            points = np.array([[10, 40], [70, 10], [70, 70]])
            color = (255, 0, 0)
        elif direction == 'right':
            points = np.array([[70, 40], [10, 10], [10, 70]])
            color = (0, 255, 255)
        cv2.fillPoly(icon, [points], color)
        return icon

    def create_neutral_indicator(self):
        icon = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.circle(icon, (40, 40), 30, (0, 200, 200), -1)
        return icon

    def overlay_indicator(self, frame, direction):
        indicator = self.indicators[direction]
        h, w = indicator.shape[:2]
        if direction == 'up':
            pos_x, pos_y = frame.shape[1] - w - 20, 20
        elif direction == 'down':
            pos_x, pos_y = frame.shape[1] - w - 20, frame.shape[0] - h - 20
        elif direction == 'left':
            pos_x, pos_y = 20, frame.shape[0] // 2 - h // 2
        elif direction == 'right':
            pos_x, pos_y = frame.shape[1] - w - 20, frame.shape[0] // 2 - h // 2
        elif direction == 'neutral':
            pos_x, pos_y = frame.shape[1] // 2 - w // 2, 20
        roi = frame[pos_y:pos_y+h, pos_x:pos_x+w]
        if roi.shape[0] == h and roi.shape[1] == w:
            gray = cv2.cvtColor(indicator, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            indicator_masked = cv2.bitwise_and(indicator, indicator, mask=mask)
            roi_masked = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            result = cv2.add(roi_masked, indicator_masked)
            frame[pos_y:pos_y+h, pos_x:pos_x+w] = result

    def find_hands(self, img):
        if img is None:
            print("Error: No image captured from the camera.")
            return img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, 
                    hand_lms, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
        return img

    def find_position(self, img):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, c = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if id in self.tip_ids:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]
        thumb_tip = self.lm_list[4]
        thumb_mcp = self.lm_list[2]
        pinky_mcp = self.lm_list[17]
        dist_tip_to_pinky = math.hypot(thumb_tip[1] - pinky_mcp[1], thumb_tip[2] - pinky_mcp[2])
        dist_mcp_to_pinky = math.hypot(thumb_mcp[1] - pinky_mcp[1], thumb_mcp[2] - pinky_mcp[2])
        if dist_tip_to_pinky > dist_mcp_to_pinky:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.pip_ids[id]][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def is_thumb_to_palm(self):
        if len(self.lm_list) < 21:
            return False
        palm_x = sum(self.lm_list[id][1] for id in self.mcp_ids) / 5
        palm_y = sum(self.lm_list[id][2] for id in self.mcp_ids) / 5
        thumb_tip_x, thumb_tip_y = self.lm_list[4][1], self.lm_list[4][2]
        hand_width = math.hypot(
            self.lm_list[5][1] - self.lm_list[17][1],
            self.lm_list[5][2] - self.lm_list[17][2]
        )
        thumb_to_palm_dist = math.hypot(thumb_tip_x - palm_x, thumb_tip_y - palm_y) / hand_width
        fingers = self.fingers_up()
        other_fingers_down = sum(fingers[1:]) == 0
        threshold = self.config.get('thumb_to_palm_threshold', 0.3)
        return thumb_to_palm_dist < threshold and other_fingers_down

    def detect_gesture(self):
        if not self.lm_list or len(self.lm_list) < 21:
            return None
        fingers = self.fingers_up()
        if self.is_thumb_to_palm():
            return 'thumb_to_palm'
        if sum(fingers) == 0:
            return 'fist'
        if fingers == [0, 1, 0, 0, 0]:
            return 'one_finger'
        if fingers == [0, 1, 1, 0, 0]:
            return 'two_fingers'
        if sum(fingers) == 5:
            return 'all_fingers'
        if fingers[0] == 0 and sum(fingers[1:]) == 4:
            return 'all_fingers'
        return None

    def smooth_gesture(self, gesture):
        if gesture is not None:
            self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        if not self.gesture_history:
            return None
        counts = {}
        for g in self.gesture_history:
            counts[g] = counts.get(g, 0) + 1
        most_common = max(counts, key=counts.get) if counts else None
        max_count = counts.get(most_common, 0)
        threshold = self.history_size * self.config.get('detection_threshold', 0.6)
        if max_count >= threshold:
            return most_common
        return None

    def log_gesture(self, gesture, action):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - Detected: {gesture}, Action: {action}\n")

    def execute_action(self, gesture):
        if not gesture:
            return None
        current_time = time.time()
        if current_time - self.last_gesture_time < self.config.get('cooldown', 0.3):
            return None
        gesture_mapping = self.config.get('gestures', {})
        action = gesture_mapping.get(gesture)
        if not action:
            return None
        key_mapping = self.config.get('keys', {})
        key = key_mapping.get(action)
        if key:
            self.log_gesture(gesture, action)
            print(f"Executing action: {action} (key: {key}) from gesture: {gesture}")
            if self.config.get('sound_enabled', True) and action in self.sounds and self.sounds[action]:
                self.sounds[action].play()
            pyautogui.press(key)
            self.last_gesture_time = current_time
            return action
        return None

    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

    def draw_calibration_ui(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "CALIBRATION MODE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset = 70
        cv2.putText(frame, f"Detection threshold: {self.config.get('detection_threshold', 0.6):.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, f"Cooldown time: {self.config.get('cooldown', 0.3):.2f}s", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, f"Sound enabled: {self.config.get('sound_enabled', True)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, f"Thumb to palm threshold: {self.config.get('thumb_to_palm_threshold', 0.3):.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, f"Finger detection threshold: {self.config.get('finger_detection_threshold', 0.6):.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 50
        cv2.putText(frame, "Controls:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "T: Toggle sound", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "+/-: Adjust detection threshold", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "[/]: Adjust cooldown time", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "{/}: Adjust thumb to palm threshold", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "S: Save settings", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "C: Exit calibration mode", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "D: Toggle debug mode", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 50
        cv2.putText(frame, "Gesture Guide:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "Thumb to palm: UP", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "Closed fist: DOWN", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "Index+Middle fingers: RIGHT", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "Only Index finger: LEFT", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        cv2.putText(frame, "All fingers extended: NEUTRAL/FORWARD", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def handle_calibration_keys(self, key):
        if key == ord('t'):
            self.config['sound_enabled'] = not self.config.get('sound_enabled', True)
        elif key == ord('+'):
            self.config['detection_threshold'] = min(1.0, self.config.get('detection_threshold', 0.6) + 0.05)
        elif key == ord('-'):
            self.config['detection_threshold'] = max(0.1, self.config.get('detection_threshold', 0.6) - 0.05)
        elif key == ord(']'):
            self.config['cooldown'] = min(1.0, self.config.get('cooldown', 0.3) + 0.05)
        elif key == ord('['):
            self.config['cooldown'] = max(0.05, self.config.get('cooldown', 0.3) - 0.05)
        elif key == ord('}'):
            self.config['thumb_to_palm_threshold'] = min(0.5, self.config.get('thumb_to_palm_threshold', 0.3) + 0.05)
        elif key == ord('{'):
            self.config['thumb_to_palm_threshold'] = max(0.05, self.config.get('thumb_to_palm_threshold', 0.3) - 0.05)
        elif key == ord('s'):
            self.save_config()
        elif key == ord('c'):
            self.toggle_calibration_mode()
        elif key == ord('d'):
            self.toggle_debug_mode()

    def draw_debug_info(self, frame):
        if not self.debug_mode or not self.lm_list:
            return frame
        h, w, _ = frame.shape
        y_offset = h - 170
        fingers = self.fingers_up()
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        cv2.rectangle(frame, (10, y_offset-20), (200, y_offset+130), (0, 0, 0, 128), -1)
        cv2.putText(frame, "DEBUG INFO", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        for i, (name, status) in enumerate(zip(finger_names, fingers)):
            color = (0, 255, 0) if status == 1 else (0, 0, 255)
            cv2.putText(frame, f"{name}: {'UP' if status == 1 else 'DOWN'}", (20, y_offset + 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Raw gesture: {self.detect_gesture()}", (20, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Smooth gesture: {self.current_gesture}", (20, y_offset + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return frame

    def run(self):
        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to capture frame from camera. Retrying...")
                    continue
                frame = cv2.flip(frame, 1)
                frame = self.find_hands(frame)
                self.find_position(frame)
                raw_gesture = self.detect_gesture()
                smooth_gesture = self.smooth_gesture(raw_gesture)
                if smooth_gesture and smooth_gesture != self.current_gesture:
                    action = self.execute_action(smooth_gesture)
                    if action:
                        self.overlay_indicator(frame, action)
                        self.current_gesture = smooth_gesture
                c_time = time.time()
                fps = 1 / (c_time - self.p_time) if c_time != self.p_time else 0
                self.p_time = c_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mode_text = "CALIBRATION MODE" if self.calibration_mode else "CONTROL MODE"
                cv2.putText(frame, mode_text, (frame.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if self.calibration_mode else (0, 255, 0), 2)
                frame = self.draw_debug_info(frame)
                if self.calibration_mode:
                    self.draw_calibration_ui(frame)
                cv2.imshow("Hand Gesture Controller", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.running = False
                elif key == ord('c'):
                    self.toggle_calibration_mode()
                elif key == ord('d'):
                    self.toggle_debug_mode()
                if self.calibration_mode:
                    self.handle_calibration_keys(key)
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            print("Cleaning up resources...")
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("Application terminated.")

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    print("Starting Hand Gesture Controller...")
    print("Press 'C' to enter calibration mode")
    print("Press 'D' to toggle debug info")
    print("Press 'ESC' to exit")
    controller = HandGestureController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("Application interrupted by user.")
    finally:
        controller.cleanup()

if __name__ == "__main__":
    main()