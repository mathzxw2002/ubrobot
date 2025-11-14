#!/usr/bin/env python3
# encoding: utf-8

#import ros lib
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mediapipe as mp
#import define msg
from yahboomcar_msgs.msg import PointArray

#import common lib
import cv2 as cv
import numpy as np
import time
import os
print("import done")


class HandDetector(Node):
    def __init__(self, name, mode=False, maxHands=2, detectorCon=0.5, trackCon=0.5):
        super().__init__(name)
        self.mpHand = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHand.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectorCon,
            min_tracking_confidence=trackCon)
        self.lmDrawSpec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=6)
        self.drawSpec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        # Create publisher to publish hand points
        self.pub_point = self.create_publisher(PointArray, '/mediapipe/points', 1000)
        
        # Initialize CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the image topic
        self.create_subscription(Image, '/ascamera_hp60c/camera_publisher/rgb0/image', self.image_callback, 10)

        self.pTime = 0  # Initialize pTime to calculate FPS
        self.exit_flag = False  # Flag to indicate exit condition

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame, img = self.pubHandsPoint(frame, draw=False)
        
        # Display FPS and combine frames
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        text = "FPS : " + str(int(fps))
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        
        # Combine the frames (original and landmark drawn image)
        dist = self.frame_combine(frame, img)
        
        # Check if 'q' is pressed to exit the program
        if cv.waitKey(1) & 0xFF == ord('q'):
            self.exit_flag = True
        
        # Display the final image
        cv.imshow('dist', dist)

    def pubHandsPoint(self, frame, draw=True):
        pointArray = PointArray()
        img = np.zeros(frame.shape, np.uint8)
        img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for i in range(len(self.results.multi_hand_landmarks)):
                if draw: self.mpDraw.draw_landmarks(frame, self.results.multi_hand_landmarks[i], self.mpHand.HAND_CONNECTIONS, self.lmDrawSpec, self.drawSpec)
                self.mpDraw.draw_landmarks(img, self.results.multi_hand_landmarks[i], self.mpHand.HAND_CONNECTIONS, self.lmDrawSpec, self.drawSpec)
                for id, lm in enumerate(self.results.multi_hand_landmarks[i].landmark):
                    point = Point()
                    point.x, point.y, point.z = lm.x, lm.y, lm.z
                    pointArray.points.append(point)
        
        self.pub_point.publish(pointArray)
        return frame, img
    
    def frame_combine(self, frame, src):
        if len(frame.shape) == 3:
            frameH, frameW = frame.shape[:2]
            srcH, srcW = src.shape[:2]
            dst = np.zeros((max(frameH, srcH), frameW + srcW, 3), np.uint8)
            dst[:, :frameW] = frame[:, :]
            dst[:, frameW:] = src[:, :]
        else:
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            frameH, frameW = frame.shape[:2]
            imgH, imgW = src.shape[:2]
            dst = np.zeros((frameH, frameW + imgW), np.uint8)
            dst[:, :frameW] = frame[:, :]
            dst[:, frameW:] = src[:, :]
        return dst

    def run(self):
        # Custom loop for handling ROS 2 callback and OpenCV events
        while rclpy.ok() and not self.exit_flag:
            rclpy.spin_once(self)  # Process one callback
            if self.exit_flag:
                break
        cv.destroyAllWindows()


def main():
    print("start it")
    rclpy.init()
    hand_detector = HandDetector('hand_detector')

    try:
        hand_detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        hand_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

