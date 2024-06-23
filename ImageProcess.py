#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class ImageProcess():
    def __init__(self):
        # 영상 사이즈는 가로세로 640 x 480
        self.Width = 640
        self.Height = 480
        #ROI 영역 : 세로 420 ~ 460 만큼 잘라서 사용
        self.Offset = 330
        self.Gap = 40
        # rospy.init_node("DrivingLaneNode")
        # self.pro=ImageProcess()
        # self.pub = rospy.Publisher("/lane_map_classic", Image, queue_size=1) #
        # self.bridge=CvBridge() #
        # self.sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        # self.image = None # 초기 이미지를 None으로 설정합니다.
        # self.lane = Image()
        # self.is_started = False # `start` 함수가 호출되었는지 확인하기 위한 플래그를 추가합니다.
        
    # def callback(self, data):
    #     self.image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    #     if not self.is_started and self.image is not None:
    #         self.start() # 이미지 데이터를 받은 후에 `start` 함수를 호출합니다.
    #         self.is_started = True # `start` 함수가 호출되었음을 표시합니다.



    
    # draw lines
    def draw_lines(self,img, lines):
        for line in lines:
            x1, y1, x2, y2 = line[0]
        return img

    # draw rectangle

    def draw_rectangle(self,img, lpos, rpos, offset=0):
        center = int((lpos + rpos) / 2)
        lpos = int(lpos)
        rpos = int(rpos)
    
        cv2.rectangle(img, (lpos - 5, 15 + offset), (lpos + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (rpos - 5, 15 + offset), (rpos + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (center - 5, 15 + offset), (center + 5, 25 + offset), (0, 255, 0), 2)
        cv2.rectangle(img, (330, 15 + offset), (340, 25 + offset), (0, 0, 255), 2)
        return img


    def divide_left_right(self, lines):
        # 기울기 절대값이 0 ~ 10 인것만 추출
        low_slope_threshold = 0
        high_slope_threshold = 10

        slopes = []
        new_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                slope = 0
            else:
                slope = float(y2 - y1) / float(x2 - x1)
            if(abs(slope) > low_slope_threshold) and (abs(slope) < high_slope_threshold):
                slopes.append(slope)
                new_lines.append(line[0])

            
        # divide lines left to right

        left_lines = []
        right_lines = []

        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = Line
            # 화면에 왼쪽/오른쪽에 있는 선분 중에서 기울기가 음수 / 양수 인것들만 모음
            if(slope < 0) and (x2 < self.Width/2 - 90):
                left_lines.append([Line.tolist()])
            elif (slope > 0) and (x1 > self.Width/2 + 90):
                right_lines.append([Line.tolist()])
        return left_lines, right_lines

    # 기울기와 y절편의 평균값 구하기

    def get_line_params(self,lines):
        # sum of x, y, m
        x_sum = 0.0
        y_sum = 0.0
        m_sum = 0.0

        size = len(lines)

        if size == 0:
            return 0, 0
        
        for line in lines :
            x1, y1, x2, y2 = line[0]
            
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)
        m = m_sum / size
        b = y_avg - m * x_avg

        return m, b


    def get_line_pos(self,img, lines, left=False, right=False):
        m, b = self.get_line_params(lines)

        if m == 0 and b == 0:
            if left: 
                pos = 0
            if right:
                pos = self.Width
        else:
            # y 값을 ROI의 세로 중간값으로 지정하여 대입
            y = self.Gap / 2
            pos = (y - b) / m

            # y 값을 맨 끝 값들로 정해줬을 때의 x값 구함
            b += self.Offset
            x1 = (self.Height - b) /float(m)
            x2 = ((self.Height/2) - b) / float(m)

            cv2.line(img, (int(x1), self.Height), (int(x2), int(self.Height/2)), (255,0,0), 3)
        return img, pos


    def process_image(self,frame):
        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blur
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


        # canny edge
        low_threshold = 60
        high_threshold = 70
        edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

        # HoughLinesP
        roi = edge_img[self.Offset : self.Offset+self.Gap, 0 : self.Width]
        all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30,30,10)

        # divide left, right lines
        if all_lines is None:
            return (0, 640), frame
        left_lines, right_lines = self.divide_left_right(all_lines)

        # get center of lines

        frame, lpos = self.get_line_pos(frame, left_lines, left=True)
        frame, rpos = self.get_line_pos(frame, right_lines, right=True)
        lpos = int(lpos)
        rpos = int(rpos)
        # draw lines

        frame = self.draw_lines(frame, left_lines)
        frame = self.draw_lines(frame, right_lines)

        # draw rectangle
        frame = self.draw_rectangle(frame, lpos, rpos, offset=self.Offset)

        return (lpos, rpos), frame
