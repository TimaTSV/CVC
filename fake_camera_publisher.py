import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FakeCameraPublisher(Node):
    def __init__(self):
        super().__init__('fake_camera_publisher')
        self.publisher = self.create_publisher(Image, '/image_raw', 10) # Публікація в топік згідно з Кроком Б.1
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('test.avi') # Використання синтетичного відео з Етапу А
        self.timer = self.create_timer(0.05, self.publish_frame) # Частота 20 FPS (0.05с)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Циклічне відтворення для безперервного тесту
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)