import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from collections import deque
import math

class RobotCVNode(Node):
    def __init__(self):
        super().__init__('robot_cv_node')
        self.bridge = CvBridge()
        
        # Завантаження налаштувань (Артефакт №4)
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        # Параметри з конфігу
        self.MIN_AREA = self.config["MIN_AREA"]
        self.FPS = self.config["FPS"]
        self.PREDICT_TIME = self.config["PREDICT_TIME"]
        self.STOP_THRESHOLD = self.config["STOP_THRESHOLD"]

        # Підписка на топік стенду (Крок Б.2)
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        # Ініціалізація алгоритмів
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.history = deque(maxlen=10)
        
        self.get_logger().info("CV Node started. Listening to /image_raw...")

    # --- ТВОЇ ФУНКЦІЇ З LOCAL (Адаптовані під клас) ---

    def detect_robot(self, frame):
        fgmask = self.bg_subtractor.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > self.MIN_AREA:
                return cv2.boundingRect(c)
        return None

    def classify_state(self):
        if len(self.history) < 2: return "stopped"
        dx = self.history[-1][0] - self.history[-2][0]
        dy = self.history[-1][1] - self.history[-2][1]
        speed = math.sqrt(dx**2 + dy**2)
        return "moving" if speed >= self.STOP_THRESHOLD else "stopped"

    def classify_direction(self, dx, dy):
        if abs(dx) < self.STOP_THRESHOLD and abs(dy) < self.STOP_THRESHOLD:
            return "rotation"
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "forward" if dy > 0 else "backward"

    def predict_position(self, center, dx, dy):
        px = int(center[0] + dx * self.FPS * self.PREDICT_TIME)
        py = int(center[1] + dy * self.FPS * self.PREDICT_TIME)
        return (max(0, min(639, px)), max(0, min(479, py)))

    def draw_overlay(self, frame, bbox, state, direction, prediction):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"STATE: {state}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"DIR: {direction}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, prediction, 8, (0, 0, 255), -1)
        cv2.putText(frame, "3s Predict", (prediction[0] + 10, prediction[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    # --- ОСНОВНИЙ ЦИКЛ ОБРОБКИ ---

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")
            return

        bbox = self.detect_robot(frame)
        if bbox:
            x, y, w, h = bbox
            center = (x + w // 2, y + h // 2)
            self.history.append(center)

            dx, dy = 0, 0
            if len(self.history) >= 2:
                dx = self.history[-1][0] - self.history[-2][0]
                dy = self.history[-1][1] - self.history[-2][1]

            state = self.classify_state()
            direction = self.classify_direction(dx, dy)
            prediction = self.predict_position(center, dx, dy)
            
            frame = self.draw_overlay(frame, bbox, state, direction, prediction)

        cv2.imshow('Robot CV - Stand Test', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RobotCVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()