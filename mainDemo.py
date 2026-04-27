from robocad.common import CommonRobot
import time

import base_robot_commands as cmds

robot = CommonRobot(False)

def main():
    cmds.lift_up__gripper(robot)
    
    # cmds.move_forward_right(robot, 50)
    # time.sleep(1)
    # cmds.stop_motors(robot)
    
    # cmds.lift_up__gripper(robot)
    # time.sleep(1)

    # cmds.move_forward(robot, 50)
    # time.sleep(1)
    # cmds.move_forward_right(robot, 50)
    # time.sleep(1)
    # cmds.move_right(robot, 50)
    # time.sleep(1)
    # cmds.stop_motors(robot)
        
try:
    main()
finally:
    cmds.stop_motors(robot)
    robot.stop()
