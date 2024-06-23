#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
#from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from lane_hj.msg import SteeringMsg

# 영상 사이즈는 가로세로 640 x 480
Width = 1280
Height = 720
# ROI 영역 : 세로 420 ~ 460 만큼 잘라서 사용
Offset = 
Gap = 40

DELTA_T = 0.05

class PID:

    def __init__(self, kp, ki, kd):

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.err_sum = 0.0
        self.delta_err = 0.0
        self.last_err = 0.0

    def getPID(self, error):

        err = error
        self.err_sum += err * DELTA_T
        self.delta_err = err - self.last_err
        self.last_err = err

        self.p = self.kp * err
        self.i = self.ki * self.err_sum
        self.d = self.kd * (self.delta_err * DELTA_T)

        self.u = self.p + self.i + self.d

        return self.u


class LaneDetection:
    def __init__(self):
        self.bridge = CvBridge()

        model_path = "/home/frozen/catkin_ws/src/lane_hj/models/tusimple_18.pth"
        model_type = ModelType.TUSIMPLE
        use_gpu = True

        # UltrafastLaneDetector Class에서 갖고오는 값
        self.lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

        self.image_sub = rospy.Subscriber("usb_cam/image_raw",Image,self.callback)
        self.image_pub = rospy.Publisher("lane_hj_image",Image,queue_size=1)
        #self.center_pub = rospy.Publisher("lane_center",Float32MultiArray,queue_size=1)
        self.steering_pub=rospy.Publisher("/lane_steer",SteeringMsg,queue_size=1) #chaewon
        self.final_center = None
        self.output_img = None
        rospy.loginfo("---**---lane detection started---**---")

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #output_img = self.lane_detector.detect_lanes(cv_image)
        output_img = self.lane_detector.detect_lanes(cv_image)
        self.output_img = output_img 

        # calculate the middle
        lane_centers = []
        coordinates = self.lane_detector.lanes_points
        for points in coordinates:
            if points:
                points_array = np.array(points)
                lane_center = np.mean(points_array, axis=0)
                lane_centers.append(lane_center)

        print("The centers of the lanes are at:", lane_centers)

        # Find the center value for each lane with another lane
        for i in range(len(lane_centers)):
            other_lanes = lane_centers[:i] + lane_centers[i+1:]
            center_avg = np.mean(other_lanes, axis=0)
            print(f"The center of lane {i+1} with other lanes is at: {center_avg}")

        #all_lanes = np.array(lane_centers)
        #self.final_center = np.mean(all_lanes, axis=0)
               # In case there are no lane centers, we set the final center as None
        if len(lane_centers) == 0:
            self.final_center = None
        else:
            all_lanes = np.array(lane_centers)
            final_center = np.mean(all_lanes, axis=0)
            self.final_center = final_center
            print("The final center value is at:", final_center)
        #print("The final center value is at:", final_center)

        print("----------------------------------------")

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.output_img, "bgr8"))
            #array_for_publishing.data = final_center.tolist()
            #self.center_pub.publish(array_for_publishing)
        except CvBridgeError as e:
            print(e)

    def draw_steer(self,image, steer_angle):
        global Width, Height, arrow_pic

        arrow_pic = cv2.imread('/home/frozen/catkin_ws/src/lane_hj/steer_arrow.png', cv2.IMREAD_COLOR)

        origin_Height = arrow_pic.shape[0]
        origin_Width = arrow_pic.shape[1]
        steer_wheel_center = origin_Height * 0.76
        arrow_Height = Height/2
        arrow_Width = (arrow_Height * 462)/728

        matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (steer_angle) * 2.5, 0.7)    
        arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
        arrow_pic = cv2.resize(arrow_pic, dsize=(int(arrow_Width), int(arrow_Height)), interpolation=cv2.INTER_AREA)

        gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

        #arrow_roi = image[int(arrow_Height): int(Height), int(Width/2 - arrow_Width/2) : int(Width/2 + arrow_Width/2)]

        arrow_pic_resized_height, arrow_pic_resized_width = arrow_pic.shape[:2] # resized arrow_pic의 높이와 너비를 얻습니다.

        # arrow_roi를 설정할 때 arrow_pic의 크기에 따라 설정합니다.
        arrow_roi = image[Height - arrow_pic_resized_height: Height, Width//2 - arrow_pic_resized_width//2 : Width//2 + arrow_pic_resized_width//2]

        arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
        res = cv2.add(arrow_roi, arrow_pic)
        image[Height - arrow_pic_resized_height: Height, Width//2 - arrow_pic_resized_width//2 : Width//2 + arrow_pic_resized_width//2] = arrow_roi


        #arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
        
        #image[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

        cv2.imshow('steer', image)

def main():
    rospy.init_node('lane_detection', anonymous=True)
    ld = LaneDetection()
    
    #chaewon
    global Width, Height

    angle_list = []
        
    steering = PID(0.25, 0, 0.03)

    center=0
    while not rospy.is_shutdown():

        if ld.final_center is not None: 
            center =  (ld.final_center[0] + ld.final_center[1]) / 2

        angle = 335 - center
        angle = steering.getPID(angle)

        angle_list.append(angle)


        if(len(angle_list) > 10):
            #print('-----------------')
            avg_angle = 0.0
            for i in range(len(angle_list) - 1, len(angle_list) - 11, -1):
                #print(1)
                avg_angle += angle_list[i]
    
            avg_angle = avg_angle/10
            angle = avg_angle
            #print('-----------------')


        if angle > 20:
            angle = 20
        elif angle < -20:
            angle = -20

        # Here, create the SteeringMsg and publish it
        steering_msg = SteeringMsg()
        steering_msg.steer_angle = angle # assuming SteeringMsg has a field called steer_angle
        ld.steering_pub.publish(steering_msg)



        if ld.output_img is not None:  # output_img가 None이 아닌지 확인
            ld.draw_steer(ld.output_img, angle)



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
    

if __name__ == '__main__':
    main()