#!/usr/bin/env python3

from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
    InterbotixRobotNode
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

from threading import Thread
from typing import Optional

from interbotix_common_modules.common_robot.exceptions import InterbotixException
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import Image, JointState

import numpy as np
np.set_printoptions(suppress=True,precision=4)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Bool
from std_srvs.srv import Trigger

import threading
import time

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
CONTROL_DT = 0.1 #15hz
CONTROL_DT_DURATION = Duration(seconds=0, nanoseconds= CONTROL_DT * S_TO_NS)
SLEEP_DT_DURATION = Duration(seconds=0, nanoseconds= S_TO_NS)

def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
    initial_pose,
) -> None:
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
    torque_on(follower_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    # start_poses = [ 
    #     # [-0.42644668, -0.10124274,  0.58444667, -0.59518456,  0.641204  , 0.3666214],
    #     [-0.42644668, -0.50124274,  0.50444667, -0.11518456,  0.741204  , 0.3666214],
    #     [ 0.20190297, -0.61512631,  0.50467968,  0.11965051,  0.78539819, -0.03067962]
    # ]
    start_poses = initial_pose
    move_arms(
        [follower_bot_left, follower_bot_right],
        start_poses,
        moving_time=4.0,
    )
    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [1.62, 1.62],
        moving_time=0.5
    )

    return True

class AlohaRobot(InterbotixRobotNode):
    def __init__(self):
        super().__init__('aloha_node')

        self.follower_bot_left = InterbotixManipulatorXS(
            robot_model='vx300s',
            robot_name='follower_left',
            node=self,
            iterative_update_fk=False,
        )
        self.follower_bot_right = InterbotixManipulatorXS(
            robot_model='vx300s',
            robot_name='follower_right',
            node=self,
            iterative_update_fk=False,
        )
        
        # Teleoperation loop
        self.gripper_left_command = JointSingleCommand(name='gripper')
        self.gripper_right_command = JointSingleCommand(name='gripper')

        robot_startup(self)


        task_name = 'ziploc'
        data_index = 8
        episode = np.load(f"/home/jiahe/data/raw_demo/{task_name}/traj/{data_index}.npy", allow_pickle = True)
        inital_pose = [ episode[0]['left_pos'][0:6], episode[0]['right_pos'][0:6] ]
        opening_ceremony(
            self.follower_bot_left,
            self.follower_bot_right,
            inital_pose,
        )

   
        
        self.srv = self.create_service(Trigger, 'reset', self.reset_callback)
        
        self.action_sub = self.create_subscription(
            JointState,
            'action',
            self.action_callback,
            1
        )
        print("end init")


    def reset_callback(self, request, response):
        print("in reset")
        response.success = opening_ceremony(
            self.follower_bot_left,
            self.follower_bot_right,
        )

        return response

    def action_callback(self, joint_msg):
        print("in action_callback")
        goal = np.array(joint_msg.position) 
        left_action = goal[0:6]
        right_action = goal[7:13]
        left_openness = goal[6]
        right_openness = goal[13]
        
        threshold = 0.5
        if(left_openness < threshold):
            left_openness = 0.0
        else:
            left_openness = 1.0

        if(right_openness < threshold):
            right_openness = 0.0
        else:
            right_openness = 1.0

        self.follower_bot_left.arm.set_joint_positions(left_action, blocking=False)
        self.follower_bot_right.arm.set_joint_positions(right_action, blocking=False)

        self.gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            left_openness 
        )
        self.gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
            right_openness
        )

        self.follower_bot_left.gripper.core.pub_single.publish(self.gripper_left_command)
        self.follower_bot_right.gripper.core.pub_single.publish(self.gripper_right_command)
        time.sleep(0.1)
        # get_interbotix_global_node().get_clock().sleep_for(CONTROL_DT_DURATION)

def main() -> None:
    rclpy.init()
    node = AlohaRobot()
    rclpy.spin(node)
    # robot_shutdown(node)


if __name__ == '__main__':
    main()
