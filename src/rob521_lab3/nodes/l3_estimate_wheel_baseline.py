#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

INT32_MAX = 2**31
NUM_ROTATIONS = 3 
TICKS_PER_ROTATION = 4096
WHEEL_RADIUS = 0.066 / 2 #In meters


class WheelBaselineEstimator():
    def __init__(self):
        rospy.init_node('wheel_baseline_estimator', anonymous=True) # Initialize node

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
        self.last_moving_msg = rospy.Time.now()

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
            if self.left_encoder_prev is None or self.left_encoder_prev is None: 
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
        if np.abs(msg.angular.z) > 1e-6:
            self.last_moving_msg = rospy.Time.now()
            if not self.is_moving:
                self.is_moving = True
                rospy.loginfo('Starting Calibration Procedure')
                return

        if not self.is_moving:
            return

        # Must have stopped for more than 3 seconds, to prevent unintentional stopping present in bag
        if (rospy.Time.now() - self.last_moving_msg).to_sec() > 3.0:
            self.is_moving = False

            # # YOUR CODE HERE!!!
            # Calculate the radius of the wheel based on encoder measurements

            # Explanation: Circumference of circle is pi * diameter. Diameter represents
            # the separation and circumference represents the distance traveled by the wheel.
            # If the robot does n rotations, then each wheel must move distance n * pi * diameter
            # From encoder, we have distance = encoder_ticks / ticks_per_rotation * (2 * pi * radius)
            # Therefore, diameter = distance / (n * pi) for each wheel which we can combine together 
            # to get separation = (distanceLeft + distanceRight) / (2 * n * pi)
            dist_left = np.abs((self.del_left_encoder / TICKS_PER_ROTATION) * (2 * np.pi * WHEEL_RADIUS))
            dist_right = np.abs((self.del_right_encoder / TICKS_PER_ROTATION) * (2 * np.pi * WHEEL_RADIUS))
            separation = (dist_left + dist_right) / (2 * NUM_ROTATIONS * np.pi)

            rospy.loginfo(f'Calibrated Separation: {separation} m')

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
    estimator = WheelBaselineEstimator() #create instance
    rospy.spin()
