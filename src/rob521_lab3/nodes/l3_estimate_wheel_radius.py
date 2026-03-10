#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

INT32_MAX = 2**31
DRIVEN_DISTANCE = 0.75 #in meters
TICKS_PER_ROTATION = 4096

class WheelRadiusEstimator():
    def __init__(self):
        rospy.init_node('wheel_radius_estimator', anonymous=True) # Initialize node

        #Subscriber bank
        rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback)
        rospy.Subscriber("sensor_state", SensorState, self.sensor_callback) #Subscribe to the sensor state msg

        #Publisher bank
        self.reset_pub = rospy.Publisher('reset', Empty, queue_size=1)

        #Initialize variables
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.del_left_encoder = 0
        self.del_right_encoder = 0
        self.is_moving = False #Moving or not moving
        self.lock = threading.Lock()

        #Reset the robot 
        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)
        rospy.loginfo('Ready to start wheel radius calibration!')

    def safe_del_phi(self, a, b):
        #Need to check if the encoder storage variable has overflowed
        diff = b - a
        if diff < -INT32_MAX: #Overflowed
            diff = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
        elif diff > INT32_MAX - 1: #Underflowed
            diff = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
        else:
            diff = b - a  
        return diff

    def sensor_callback(self, msg: SensorState):
        #Retrieve the encoder data form the sensor state msg
        with self.lock:
            if self.left_encoder_prev is None or self.right_encoder_prev is None: 
                self.left_encoder_prev = msg.left_encoder #int32
                self.right_encoder_prev = msg.right_encoder #int32
            else:
                #Calculate and integrate the change in encoder value
                self.del_left_encoder += self.safe_del_phi(self.left_encoder_prev, msg.left_encoder)
                self.del_right_encoder += self.safe_del_phi(self.right_encoder_prev, msg.right_encoder)

                #Store the new encoder values
                self.left_encoder_prev = msg.left_encoder #int32
                self.right_encoder_prev = msg.right_encoder #int32

    def cmd_vel_callback(self, msg: Twist):
        input_velocity_mag = np.linalg.norm([msg.linear.x, msg.linear.y, msg.linear.z])
        if not self.is_moving and np.abs(input_velocity_mag) > 0:
            self.is_moving = True #Set state to moving
            rospy.loginfo('Starting Calibration Procedure')

        elif self.is_moving and np.isclose(input_velocity_mag, 0):
            self.is_moving = False #Set the state to stopped

            # # YOUR CODE HERE!!!
            # Calculate the radius of the wheel based on encoder measurements

            # Explanation: Distance = encoder_ticks / ticks_per_rotation * (2 * pi * radius)
            # Therefore radius = distance / (2 * pi * (encoder_ticks / ticks_per_rotation))
            # Assuming both wheels have same radius, we can taken average of left/right ticks
            avg_ticks = (self.del_left_encoder + self.del_right_encoder) / 2
            radius = DRIVEN_DISTANCE / (2 * np.pi * (avg_ticks / TICKS_PER_ROTATION))

            rospy.loginfo(f'Calibrated Radius: {radius} m')

            #Reset the robot and calibration routine
            with self.lock:
                self.left_encoder_prev = None
                self.right_encoder_prev = None
                self.del_left_encoder = 0
                self.del_right_encoder = 0
            reset_msg = Empty()
            self.reset_pub.publish(reset_msg)
            rospy.loginfo('Resetted the robot to calibrate again!')


if __name__ == '__main__':
    estimator = WheelRadiusEstimator() #create instance
    rospy.spin()
