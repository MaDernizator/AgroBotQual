from robocad.common import CommonRobot
import cv2
import numpy as np
import time

robot = CommonRobot(False)

DRIVE_SPEED = 20
TURN_TIME = 1.3
AFTER_TURN_SPEED = 50
LINE_THRESHOLD = 1000

def set_drive(left, right):
    robot.motor_speed_0 = left
    robot.motor_speed_1 = left
    robot.motor_speed_2 = left
    robot.motor_speed_3 = -right
    robot.motor_speed_4 = -right
    robot.motor_speed_5 = -right

def stop():
    for i in range(6):
        setattr(robot, f"motor_speed_{i}", 0)

def get_line_sensors():
    return robot.analog_5, robot.analog_6, robot.analog_7, robot.analog_8

def line_follower_correction(s5, s6, s7, s8):
    """
    Коррекция по датчикам линии.
    Порядок слева направо: 8, 7, 6, 5
    Возвращает error от -1.5 до +1.5 или None если линии нет
    """
    weights   = [-1.5, -0.5,  0.5,  1.5]
    line_vals = [ s8,   s7,   s6,   s5 ]
    on_line = any(s > LINE_THRESHOLD for s in line_vals)
    if not on_line:
        return None
    weighted_sum = sum(w * v for w, v in zip(weights, line_vals))
    total = sum(line_vals)
    return weighted_sum / total  # <0 линия левее, >0 правее


def camera_correction(frame):
   h, w = frame.shape[:2]
   y1 = int(h * 0.40)
   y2 = int(h * 0.70)
   roi = frame[y1:y2, :]

   # Просто яркость — разметка светлее асфальта
   gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

   # Адаптивный порог — работает при любом освещении
   _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

   cv2.rectangle(frame, (0, y1), (w, y2), (255, 255, 0), 1)

   cols = np.where(mask.any(axis=0))[0]
   if len(cols) == 0:
      return None, frame

   center_x = cols.mean()
   error = (center_x - w / 2) / (w / 2)

   cv2.line(frame, (w // 2, y1), (w // 2, y2), (255, 0, 0), 1)
   cv2.line(frame, (int(center_x), y1), (int(center_x), y2), (0, 255, 0), 2)

   return error, frame

# ================================================================
# Фаза 0: поднимаем клешню
# ================================================================
print("Поднимаем клешню...")
robot.set_angle_servo(300, 1)
robot.set_angle_servo(300, 2)
time.sleep(1.5)

# ================================================================
# Фаза 1: едем вперёд, считаем штрихи разметки датчиками
# ================================================================
print("Едем вперёд, ждём вторую линию...")
set_drive(DRIVE_SPEED, DRIVE_SPEED)

line_seen_count = 0
was_on_line = False
start = time.time()

while time.time() - start < 60:
    s5, s6, s7, s8 = get_line_sensors()
    on_line = any(s > LINE_THRESHOLD for s in (s5, s6, s7, s8))

    # Считаем фронты (не было → есть)
    if on_line and not was_on_line:
        line_seen_count += 1
        print(f"Линия #{line_seen_count}!")
    was_on_line = on_line

    if line_seen_count >= 2:
        print("Вторая линия — поворачиваем!")
        break

    # Пока едем — держимся по линии датчиками
    error = line_follower_correction(s5, s6, s7, s8)
    if error is not None:
        correction = error * 20
        set_drive(
            max(-100, min(100, DRIVE_SPEED + correction)),
            max(-100, min(100, DRIVE_SPEED - correction))
        )

    frame = robot.camera_image
    if frame is not None:
        cv2.putText(frame, f"Линия #{line_seen_count} | {'ЛИНИЯ' if on_line else '------'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"8:{s8:.0f} 7:{s7:.0f} 6:{s6:.0f} 5:{s5:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

# ================================================================
# Фаза 2: поворот вправо на 90°
# ================================================================
print("Поворот вправо...")
stop()
time.sleep(0.2)
set_drive(50, -50)
time.sleep(TURN_TIME)
stop()
time.sleep(0.2)

# ================================================================
# Фаза 3: едем вперёд — датчики + камера вместе
# ================================================================
print("Стабилизация по датчикам и камере...")
K_sensor = 20   # коэффициент для датчиков
K_camera = 25   # коэффициент для камеры

start = time.time()
while time.time() - start < 30:
    s5, s6, s7, s8 = get_line_sensors()

    frame = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)

    sensor_error = line_follower_correction(s5, s6, s7, s8)

    # Приоритет: датчики точнее (они прямо под роботом)
    # Камера — когда датчики не видят линию (промежуток разметки)
    if sensor_error is not None:
        correction = sensor_error * K_sensor
        source = "ДАТЧИК"
    elif cam_error is not None:
        correction = cam_error * K_camera
        source = "КАМЕРА"
    else:
        correction = 0
        source = "ПРЯМО"

    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.putText(frame, f"{source} | err:{correction:.1f} L:{left:.0f} R:{right:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"8:{s8:.0f} 7:{s7:.0f} 6:{s6:.0f} 5:{s5:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

stop()
cv2.destroyAllWindows()
print("Готово.")