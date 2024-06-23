#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from lane_hj.msg import SteeringMsg
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
import math
import time

Width = 1640
Height = 590
Offset=450

DELTA_T = 0.05


class LaneDetection:
    def __init__(self):
        self.bridge = CvBridge()

        model_path = "/home/frozen/catkin_ws/src/lane_hj_2/models/culane_18.pth"
        model_type = ModelType.CULANE
        use_gpu = True

        self.lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

        #self.image_sub = rospy.Subscriber("usb_cam/image_raw",Image,self.callback)
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed",CompressedImage,self.callback)
        self.image_pub = rospy.Publisher("lane_hj_image",Image,queue_size=1)
        self.steering_pub=rospy.Publisher("/lane_steer",SteeringMsg,queue_size=1)
        self.final_center_x = None
        self.steering_angle =0
        self.prev_error=0
        self.output_img = None
        rospy.loginfo("---**---lane detection started---**---")
    
    def process_lane(self, lanes_points, visualization_img):
        arctan = self.calculate_arctan(lanes_points)
        dev = np.round(arctan - np.mean(arctan))
        var = np.mean(dev**2)

        print("분산 : ",var)
    
        if var > 100 :
            #편차가 매우 커 값 사용 불가 교차점 이미지 중앙점으로 줌
            cross_x = 0
            pass
        elif var > 50 :
            #편차가 다소 큰 상태여서 이상치 제거하고 선형 회귀 후 y중앙일때의 x값을 교차점으로
            lanes_points = self.remove_outlier(lanes_points, dev)
            slope, intercept = self.train_huber_regression(visualization_img, lanes_points, (0, 0, 255))
            cross_x = (430-intercept)/slope
        else :
            #편차가 작은 편이라서 선형 회귀 후 y중앙일때의prev_error = 0  # 이전 오차 x값을 교차점으로
            slope, intercept = self.train_huber_regression(visualization_img, lanes_points, (0, 0, 255))
            cross_x = (430-intercept)/slope
        return cross_x

    def remove_outlier(self,lanes_points, data):
        data_mean = np.mean(np.abs(data))
        outlier_index = np.where(np.abs(data)< data_mean)
        cleaned_lanes_points = np.delete(lanes_points, outlier_index, axis=0)
        return cleaned_lanes_points


    def calculate_arctan(self,points):
        x, y = np.transpose(points)
        dx = np.diff(x)
        dy = np.diff(y)
        arctan_values = np.round(np.degrees(np.arctan2(dy, dx)), 1)
        return arctan_values

    # huber regression 선형회귀 모델 적용
    def train_huber_regression(self, output_img, points, color):
        points_x = points[:, 0]
        points_y = points[:, 1]

        # Huber Regression 모델 생성
        huber_reg = HuberRegressor()

        # X와 y를 2D 배열로 변환
        X = points_x.reshape(-1,1)
        y = points_y.reshape(-1,)
        # 모델 학습
        huber_reg.fit(X, y)

        slope = huber_reg.coef_[0]
        intercept = huber_reg.intercept_

        if slope > 2000000000:
            slope = 2000000000
        elif slope < -2000000000:
            slope = -2000000000

        cv2.line(output_img, (int((590 - intercept) / slope), 590), (int((295 - intercept) / slope), 295), color, 2)
        return slope, intercept
    
    def roi_image(self, image):
        (x,y),(w,h) = (0,0),(1280,720) 
        roi_img = image[y:y+h, x:x+w]

        return roi_img

    def white_filter(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 200]) 
        white_upper = np.array([90, 80, 255])

        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        white_masked = cv2.bitwise_and(image, image, mask=white_mask)

        return white_masked
    
    def stopline(self, cv_image):
        lineTF = False
        cap=self.roi_image(cv_image) 

        white_src = self.white_filter(cap) 

        white_dst = cv2.Canny(white_src, 300, 600, None, 3) 

        white_cdst = cv2.cvtColor(white_dst, cv2.COLOR_GRAY2BGR) 
        white_cdstP = np.copy(white_cdst) 

        stop_lines = cv2.HoughLines(white_dst, 1, 5*np.pi / 180, 150, None, 0, 0, 88, 93) 

        if stop_lines is not None:
            for i in range(0, len(stop_lines)):
                rho = stop_lines[0][0][0] 
                theta = stop_lines[0][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))) 
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))) 
                cv2.line(white_cdst, pt1, pt2, (30, 225, 225), 2, cv2.LINE_AA) 

        stop_linesP = cv2.HoughLinesP(white_dst, 1, 15*np.pi /180, 2, None, 150, 1) 
        if stop_linesP is not None:
            lineTF = True
            for i in range(0, len(stop_linesP)):
                l = stop_linesP[i][0]
                cv2.line(white_cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

            return lineTF

    def callback(self,data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # 이미지에 테두리 추가
        left_right_padding = (1640 - 1280) // 2
        top_bottom_padding = (590 - 720) // 2

        if top_bottom_padding < 0: 
            img = cv_image[abs(top_bottom_padding):720-abs(top_bottom_padding), :, :]
            top_bottom_padding = 0

        bordered_img = cv2.copyMakeBorder(img, top=top_bottom_padding, bottom=top_bottom_padding, 
                                        left=left_right_padding, right=left_right_padding, 
                                        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])


        output_img = self.lane_detector.detect_lanes(bordered_img)
        IsStopLine = self.stopline(cv_image)
        if IsStopLine == True:
            self.steering_angle = 14
            steering_msg = SteeringMsg()
            steering_msg.steer_angle = self.steering_angle
            self.steering_pub.publish(steering_msg)
            time.sleep(3)

        self.output_img = output_img 

        #lane_centers = []
        coordinates = self.lane_detector.lanes_points
        #chaewon
        center_x, dcsion_y, offset_x, dcsion_line_w = 820, 430, 300, 250

        L1_x_left, L1_x_right = int(center_x - offset_x - dcsion_line_w/2), int(center_x - offset_x + dcsion_line_w/2)
        L2_x_left, L2_x_right = int(center_x + offset_x - dcsion_line_w/2), int(center_x + offset_x + dcsion_line_w/2)

        cv2.line(output_img, (center_x , dcsion_y + 50), (center_x , dcsion_y - 50), (0, 255, 255), 2)
        cv2.line(output_img, (L1_x_left, dcsion_y), (L1_x_right, dcsion_y), (255, 0, 0), 2)
        cv2.line(output_img, (L2_x_left, dcsion_y), (L2_x_right, dcsion_y), (255, 0, 0), 2)



        
        #print("len(coord) normal")

        left_lane = np.array(coordinates[1])
        right_lane = np.array(coordinates[2])


        cv2.circle(self.output_img, (Width//2, Offset), 5, (0, 0, 255), -1)  # Red circle for the center of the image
        filtered_error=0

        if left_lane.size==0 and right_lane.size==0 : #양쪽 차선 다 감지 x
            #이전 프레임 값 사용
            print("---both lane X---")
            steering_msg = SteeringMsg()
            steering_msg.steer_angle = self.prev_error
            self.steering_pub.publish(steering_msg)
            return
        else :
            if left_lane.size != 0 and right_lane.size != 0:  # 양쪽 차선 감지
                #usable?
                L1_x = self.process_lane(left_lane, output_img)
                L2_x = self.process_lane(right_lane, output_img)

                if L1_x==0 and L2_x ==0 :
                    steering_msg = SteeringMsg()
                    steering_msg.steer_angle = self.prev_error
                    self.steering_pub.publish(steering_msg)
                    print("---both lane unusable---")
                    return
                else :
                    if L1_x==0 :
                        print("---RRRRRRRR usable---")
                        ct_error= int(L2_x-180) - center_x
                        filtered_error = int(self.prev_error*0.9 + ct_error * 0.05)
                    elif L2_x==0 :
                        print("---LLLLLLL usable---")
                        ct_error= int( L1_x + 180) - center_x
                        filtered_error  = int(self.prev_error *0.9 + ct_error * 0.05)
                    else :
                        print("wowowowowowowowowowowowowowowowowowow")
                        ct_error = int((L1_x + L2_x ) / 2) - center_x
                        filtered_error= int(self.prev_error*0.9 + ct_error * 0.05)
                cv2.line(output_img, (center_x, dcsion_y ), (center_x + filtered_error, dcsion_y), (0, 255, 0), 5) # center에다가 움직일 steering 값 만큼 그어주는거에요
                # cv2.line(output_img, (center_x, dcsion_y+30 ), (center_x  + ct_error , dcsion_y+30), (122, 122, 255), 5) # center에다가 ct_error 오차값 그어주는거에여
            else :
                if left_lane.size!=0: #only왼쪽 감지
                    #usable?
                    L1_x = self.process_lane(left_lane, output_img)
                    if L1_x==0 : #left unusable
                        ct_error =  self.prev_error
                        filtered_error  = int(self.prev_error *0.9 + ct_error * 0.05)
                        pass
                    else :
                    
                        print("LLLLLLLLLLLLLLLLLLLLLLL")
                        ct_error= int( L1_x + 180) - center_x
                        print(ct_error)
                        filtered_error  = int(self.prev_error *0.9 + ct_error * 0.05)

                elif right_lane.size!=0:  #only오른쪽 감지
                    #usable?
                    print("RRRRRRRRRRRRRRRRRR")
                    L2_x = self.process_lane(right_lane, output_img)
                    if L2_x==0 : #right unusable
                        pass
                    else:
                        ct_error= int(L2_x-180) - center_x
                        filtered_error = int(self.prev_error*0.9 + ct_error * 0.05)

                cv2.line(output_img, (center_x, dcsion_y ), (center_x + filtered_error, dcsion_y), (0, 255, 0), 5) # center에다가 움직일 steering 값 만큼 그어주는거에요
                # cv2.line(output_img, (center_x, dcsion_y+30 ), (center_x  + ct_error , dcsion_y+30), (122, 122, 255), 5) # center에다가 ct_error 오차값 그어주는거에여
        

        # PD 제어 게인 설정
        k_p = 0.01  # P 제어 게인
        k_d = 0.0001  # D 제어 게인
        
        # P 제어 계산
        p_control = k_p * filtered_error
        
        
        # D 제어 계산
        d_control = k_d * (filtered_error - self.prev_error)
        
        
        # 총 제어 값 계산RRRRRRRR
        control = p_control + d_control

        if self.prev_error-5<control<self.prev_error+5 :
            self.steering_angle=max(min(control, 30), -30)
            self.prev_error = self.steering_angle
        else : 
            control=self.prev_error
            self.steering_angle=max(min(control, 30), -30)
            self.prev_error = self.steering_angle
        cv2.line(output_img, (center_x, dcsion_y ), (int(center_x + control), dcsion_y), (0, 255, 0), 5) # center에다가 움직일 steering 값 만큼 그어주는거에요

        print(" control : ", control)
        print("----------------------------------------")



        # # only lpf
        # if self.prev_error-5<filtered_error<self.prev_error+5 or self.prev_error==0:
        #     #스티어링 값을 최대값으로 제한
        #     self.steering_angle = max(min(filtered_error, 30), -30)
        #     self.prev_error = self.steering_angle
        # else : 
        #     filtered_error=self.prev_error
        #     self.steering_angle = max(min(filtered_error, 30), -30)
        #     self.prev_error = self.steering_angle
        # cv2.line(output_img, (center_x, dcsion_y ), (int(center_x + self.steering_angle), dcsion_y), (0, 255, 0), 5) # center에다가 움직일 steering 값 만큼 그어주는거에요

        print(" filtered_error : ", filtered_error)
        print("----------------------------------------")

        print("steering angle : ", self.steering_angle)
        print("----------------------------------------")

       
        # Here, create the SteeringMsg and publish it
        steering_msg = SteeringMsg()
        steering_msg.steer_angle = self.steering_angle
        self.steering_pub.publish(steering_msg)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.output_img, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('lane_detection', anonymous=True)
    ld = LaneDetection()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
    

if __name__ == '__main__':
    main()
