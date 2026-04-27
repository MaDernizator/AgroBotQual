from robocad.common import CommonRobot
import time
import cv2

robot = CommonRobot(False)

def stop_motors():
    robot.motor_speed_0 = 0
    robot.motor_speed_1 = 0
    robot.motor_speed_2 = 0
    robot.motor_speed_3 = 0
    robot.motor_speed_4 = 0
    robot.motor_speed_5 = 0


def turn_right(speed=40):
    robot.motor_speed_0 = speed
    robot.motor_speed_1 = speed
    robot.motor_speed_2 = speed
    robot.motor_speed_3 = speed
    robot.motor_speed_4 = speed
    robot.motor_speed_5 = speed


def turn_left(speed=40):
    robot.motor_speed_0 = -speed
    robot.motor_speed_1 = -speed
    robot.motor_speed_2 = -speed
    robot.motor_speed_3 = -speed
    robot.motor_speed_4 = -speed
    robot.motor_speed_5 = -speed


def open_gripper():
    robot.set_angle_servo(300, 2)
    sleep(1)


def close_gripper():
    robot.set_angle_servo(0, 2)
    
def lift_up__gripper():
    robot.set_angle_servo(300, 1)
    
def lift_down_gripper():
    robot.set_angle_servo(30, 1)


def show_camera(seconds=10):
    start = time.time()

    while time.time() - start < seconds:
        img = robot.camera_image

        if img is not None:
            cv2.imshow("Agrobot camera", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.03)

    cv2.destroyAllWindows()


try:
    show_camera(100)
    

finally:
    stop_motors()