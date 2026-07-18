from robocad.common import CommonRobot
import cv2
import numpy as np
import time

robot = CommonRobot(False)

DRIVE_SPEED    = 40
TURN_TIME      = 1.3
AFTER_TURN_SPEED = 50
LINE_THRESHOLD = 1000

# Коэффициент стабилизации по направлению ящика
K_BOX    = 30
# Коэффициент стабилизации по белой линии (камера)
K_CAMERA = 25
# Скорость подъезда к ящику
APPROACH_SPEED = 20


# ================================================================
# Вспомогательные функции
# ================================================================

def snap_yaw(yaw):
    """Округляет yaw до ближайшего из 0, 90, 180, -90, -180"""
    steps = [0, 90, 180, -90, -180]
    return min(steps, key=lambda s: abs((yaw - s + 180) % 360 - 180))

def turn_to_yaw(target_yaw, speed=50, timeout=5.0):
    """
    Поворачивает робота до достижения target_yaw (в градусах).
    Направление выбирается автоматически по кратчайшей дуге.
    """
    start = time.time()
    while time.time() - start < timeout:
        current = robot.yaw
        diff = target_yaw - current

        # Нормализуем в диапазон -180..+180
        diff = (diff + 180) % 360 - 180

        if abs(diff) < 3:   # допуск ±3°, можно подтянуть
            break

        # Направление: diff>0 = надо повернуть против часовой, diff<0 = по часовой
        if diff > 0:
            set_drive(-speed, speed)   # влево
        else:
            set_drive(speed, -speed)   # вправо

        time.sleep(0.02)

    stop()
    time.sleep(0.1)

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
    Возвращает error от -1.5 до +1.5 или None если линии нет.
    """
    weights   = [-1.5, -0.5,  0.5,  1.5]
    line_vals = [ s8,   s7,   s6,   s5 ]
    on_line = any(s > LINE_THRESHOLD for s in line_vals)
    if not on_line:
        return None
    weighted_sum = sum(w * v for w, v in zip(weights, line_vals))
    total = sum(line_vals)
    return weighted_sum / total  # <0 линия левее, >0 правее

def camera_correction(frame, wide=False):
    h, w = frame.shape[:2]
    y1 = int(h * (0.15 if wide else 0.40))
    y2 = int(h * 0.70)
    roi = frame[y1:y2, :]

    checker = 130
    b, g, r = cv2.split(roi)
    bright  = (b.astype(int) > checker) & (g.astype(int) > checker) & (r.astype(int) > checker)
    neutral = (np.max(np.stack([b, g, r], axis=2), axis=2).astype(int) -
               np.min(np.stack([b, g, r], axis=2), axis=2).astype(int)) < 40
    mask = (bright & neutral).astype(np.uint8) * 255

    cv2.rectangle(frame, (0, y1), (w, y2), (255, 255, 0), 1)

    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None, frame

    center_x = cols.mean()
    error = (center_x - w / 2) / (w / 2)

    cv2.line(frame, (w // 2, y1), (w // 2, y2), (255, 0, 0), 1)
    cv2.line(frame, (int(center_x), y1), (int(center_x), y2), (0, 255, 0), 2)
    return error, frame

def detect_road(frame):
    error, _ = camera_correction(frame)
    if error is None:
        return False
    return abs(error) < 0.8

def detect_box(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_zone = np.zeros((h, w), np.uint8)
    mask_zone[int(h * 0.12):int(h * 0.65), :] = 255

    mask_frame = cv2.inRange(hsv, np.array([0, 80, 25]), np.array([20, 220, 65]))
    mask_frame = cv2.bitwise_and(mask_frame, mask_zone)

    mask_wood = cv2.inRange(hsv, np.array([0, 30, 25]), np.array([35, 85, 65]))
    mask_wood = cv2.bitwise_and(mask_wood, mask_zone)
    mask_wood = cv2.subtract(mask_wood, mask_frame)

    combined = cv2.bitwise_or(mask_frame, mask_wood)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = frame.copy()
    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if not (0.35 < cw / max(ch, 1) < 2.0):
            continue
        if cv2.countNonZero(mask_frame[y:y+ch, x:x+cw]) < 50:
            continue

        cx     = x + cw / 2
        bottom = y + ch

        # Насколько ящик по центру горизонтально (0 = идеально, 1 = край)
        center_penalty = abs(cx - w / 2) / (w / 2)
        # Насколько ящик низко в кадре (выше = ближе)
        bottom_score   = bottom / h

        # Итоговый score: близость к низу важнее, центр — уточняющий
        score = bottom_score * 2.0 - center_penalty * 1.0

        cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 180, 80), 1)
        cv2.putText(debug, f"{score:.2f}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 0), 1)

        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        cv2.putText(debug, "BOX: none", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, debug

    x, y, cw, ch = best
    cx     = x + cw / 2
    bottom = y + ch
    direction = (cx - w / 2) / (w / 2)

    if bottom > h * 0.60:
        distance = 'very_near'
    elif bottom > h * 0.42:
        distance = 'near'
    else:
        distance = 'far'

    cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
    cv2.circle(debug, (int(cx), y + ch // 2), 6, (0, 255, 255), -1)
    cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.putText(debug, f"BOX: {distance} dir:{direction:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return distance, direction, debug

def detect_obstacle_ahead(frame):
    """
    Возвращает True если впереди забор/столб, False если дорога свободна.
    Смотрит на центральную полосу верхней части кадра.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Центральная полоса, верхние 60% кадра (не земля)
    strip = hsv[int(h * 0.05):int(h * 0.60),
                int(w * 0.25):int(w * 0.75)]

    # Оранжево-красная решётка забора: H=5-20, S>80
    orange_mask = cv2.inRange(strip,
                              np.array([5,  80, 25]),
                              np.array([20, 255, 130]))

    ratio = cv2.countNonZero(orange_mask) / (strip.shape[0] * strip.shape[1])
    return ratio > 0.05   # порог с хорошим запасом (забор даёт 0.09+, дорога 0.002)

def detect_lever(frame, save_debug=False):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Яркий оранжевый: рычаг намного насыщеннее забора
    mask = cv2.inRange(hsv, np.array([5, 150, 100]), np.array([25, 255, 255]))

    # Убираем верх (небо) и самый низ (земля под роботом)
    mask[:int(h * 0.10), :] = 0
    mask[int(h * 0.85):, :] = 0

    debug = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)

        # Рычаг — горизонтальный прямоугольник (aspect > 2.0)
        # Ромбы забора — квадратные (aspect ≈ 1.0)
        if aspect < 2.0:
            continue

        cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 180, 80), 1)
        if area > best_area:
            best_area = area
            best = (x, y, cw, ch)

    if best is None:
        cv2.putText(debug, "LEVER: none", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if save_debug:
            cv2.imwrite("lever_debug.png", debug)
            cv2.imwrite("lever_mask.png", mask)
        return None, debug

    x, y, cw, ch = best
    cx = x + cw / 2
    direction = (cx - w / 2) / (w / 2)

    cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
    cv2.circle(debug, (int(cx), y + ch // 2), 6, (0, 255, 255), -1)
    cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.putText(debug, f"LEVER: dir={direction:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if save_debug:
        cv2.imwrite("lever_debug.png", debug)
        cv2.imwrite("lever_mask.png", mask)

    return direction, debug

def drive_forward_yaw(target_yaw, speed=40, timeout=30.0):
    """
    Едет вперёд удерживая target_yaw.
    stop_condition — функция без аргументов, возвращает True когда надо остановиться.
    """
    start = time.time()
    while time.time() - start < timeout and robot.analog_2 < 800 and robot.analog_1 < 800:
        current = robot.yaw
        diff = (target_yaw - current + 180) % 360 - 180

        correction = diff * 1.5   # P-коэффициент, подобрать при необходимости

        left  = max(-100, min(100, speed + correction))
        right = max(-100, min(100, speed - correction))
        set_drive(left, right)

    stop()

# ================================================================
# Фаза 0: поднимаем лифт и открываем клешню
# ================================================================
print("[Фаза 0] Поднимаем клешню...")
robot.set_angle_servo(300, 1)   # лифт вверх
robot.set_angle_servo(300, 2)   # клешня открыта (300 = открыто)
time.sleep(3)

# ================================================================
# Фаза 1: Едем вперёд до забора
# ================================================================
print("[Фаза 1] Едем вперёд до забора...")
set_drive(DRIVE_SPEED, DRIVE_SPEED)
start = time.time()
while time.time() - start < 60:
    if (robot.analog_1 + robot.analog_2) / 2 > 800:
        break

# ================================================================
# Фаза 2: Поворот вправо на 90°
# ================================================================
print("[Фаза 2] Поворот вправо...")
stop()
time.sleep(0.2)
set_drive(-50, 50)
time.sleep(TURN_TIME)
stop()
time.sleep(0.2)

# ================================================================
# Фаза 3: Едем по дороге (камера), ищем нужный поворот к ящикам
# ================================================================
print("[Фаза 3] Движение по дороге, поиск поворота к ящикам...")

CHECK_INTERVAL = 2.5
FIRST_CHECK    = True
last_check     = time.time()
start          = time.time()

while time.time() - start < 60:
    if time.time() - last_check > CHECK_INTERVAL:
        if FIRST_CHECK:
            FIRST_CHECK    = False
            CHECK_INTERVAL = 4.0
        stop()
        print("Проверяем поворот налево...")
        set_drive(-50, 50)
        time.sleep(TURN_TIME)
        stop()
        frame = robot.camera_image
        last_check = time.time()
        while frame is None:
            frame = robot.camera_image
        if detect_road(frame):
            break
        else:
            set_drive(50, -50)
            time.sleep(TURN_TIME)
            stop()
    print(robot.yaw)
    frame      = robot.camera_image
    cam_error  = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

# ================================================================
# Фаза 4: Подъезд к ящику со стабилизацией по direction
# ================================================================
print("[Фаза 4] Подъезжаем и центрируем ящик...")

start = time.time()
while time.time() - start < 30:
    frame = robot.camera_image
    if frame is None:
        time.sleep(0.05)
        continue

    distance, direction, debug_frame = detect_box(frame)

    cv2.imshow("АГРОБОТ", debug_frame)
    cv2.waitKey(1)

    if distance is None:
        # Ящик потерян — едем медленно вперёд, ищем
        set_drive(APPROACH_SPEED, APPROACH_SPEED)
        time.sleep(0.05)
        continue

    if distance == 'very_near':
        # Ящик достаточно близко — переходим к захвату
        stop()
        print(f"  Ящик очень близко, direction={direction:.2f} — захватываем!")
        break

    # Стабилизация: корректируем курс по горизонтальному положению ящика
    correction = direction * K_BOX
    left  = max(-100, min(100, APPROACH_SPEED + correction))
    right = max(-100, min(100, APPROACH_SPEED - correction))
    set_drive(left, right)

    time.sleep(0.05)

# ================================================================
# Фаза 5: Захват
# ================================================================
print("[Фаза 5] Захват ящика...")
stop()
time.sleep(0.3)

# Опускаем лифт
print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

print("Подъехали к ящику")
set_drive(50, 50)
time.sleep(2.0)
stop()

# Закрываем клешню
print("  Закрываем клешню...")
robot.set_angle_servo(0, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

# Поднимаем лифт с ящиком
print("  Поднимаем ящик...")
robot.set_angle_servo(300, 1)   # лифт вверх
time.sleep(2.0)

set_drive(-50,-50)
time.sleep(2.0)
stop()
print("  Ящик захвачен!")

# ================================================================
# Фаза 6: Разворот на 180°
# ================================================================
print("[Фаза 6] Разворот на 180°...")
stop()
time.sleep(0.2)
set_drive(-50, 50)
time.sleep(TURN_TIME * 2)       # два раза по TURN_TIME ≈ 180°
stop()
time.sleep(0.3)

while robot.analog_1 < 800 and robot.analog_2 < 800:
    set_drive(40, 40)

stop()

turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
set_drive(50, -50)


# ================================================================
# Фаза 7: Едем по линии вперёд до стены (analog_1/2 > 800)
# ================================================================
print("[Фаза 7] Едем по линии обратно до стены...")

start = time.time()
while time.time() - start < 60 and (robot.analog_1 + robot.analog_2) / 2 < 800:
    frame      = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)  # сначала обычный режим
        if cam_error is None:
            cam_error, frame = camera_correction(frame, wide=True)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

# ================================================================
# Фаза 8: Поворот вправо после доставки
# ================================================================
print("[Фаза 8] Поворот вправо после доставки...")
stop()
time.sleep(0.2)
set_drive(50, -50)
time.sleep(TURN_TIME)
stop()
time.sleep(0.2)

print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

# Закрываем клешню
print("  Открываем клешню...")
robot.set_angle_servo(300, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

robot.set_angle_servo(300, 1)
time.sleep(2.0)

time.sleep(0.2)
turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
stop()
time.sleep(0.2)
# ================================================================
# Фаза 9: Поиск коробки 2
# ================================================================

CHECK_INTERVAL = 3.2
FIRST_CHECK    = True
FIRST_PASS = True
last_check     = time.time()
start          = time.time()

while time.time() - start < 60:
    if time.time() - last_check > CHECK_INTERVAL:
        if FIRST_CHECK:
            FIRST_CHECK    = False
            CHECK_INTERVAL = 4.0
        stop()
        print("Проверяем поворот налево...")
        set_drive(-50, 50)
        time.sleep(TURN_TIME)
        stop()
        frame = robot.camera_image
        last_check = time.time()
        if detect_road(frame) and FIRST_PASS:
            FIRST_PASS = False
        elif detect_road(frame):
            break
        set_drive(50, -50)
        time.sleep(TURN_TIME)
        stop()


    frame      = robot.camera_image
    cam_error  = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)


# ================================================================
# Фаза 10: Подъезд к ящику со стабилизацией по direction
# ================================================================
print("[Фаза 10] Подъезжаем и центрируем ящик...")

start = time.time()
while time.time() - start < 30:
    frame = robot.camera_image
    if frame is None:
        time.sleep(0.05)
        continue

    distance, direction, debug_frame = detect_box(frame)

    cv2.imshow("АГРОБОТ", debug_frame)
    cv2.waitKey(1)

    if distance is None:
        # Ящик потерян — едем медленно вперёд, ищем
        set_drive(APPROACH_SPEED, APPROACH_SPEED)
        time.sleep(0.05)
        continue

    if distance == 'very_near':
        # Ящик достаточно близко — переходим к захвату
        stop()
        print(f"  Ящик очень близко, direction={direction:.2f} — захватываем!")
        break

    # Стабилизация: корректируем курс по горизонтальному положению ящика
    correction = direction * K_BOX
    left  = max(-100, min(100, APPROACH_SPEED + correction))
    right = max(-100, min(100, APPROACH_SPEED - correction))
    set_drive(left, right)

    time.sleep(0.05)

# ================================================================
# Фаза 11: Захват
# ================================================================
print("[Фаза 11] Захват ящика...")
stop()
time.sleep(0.3)

# Опускаем лифт
print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

print("Подъехали к ящику")
set_drive(50, 50)
time.sleep(2.0)
stop()

# Закрываем клешню
print("  Закрываем клешню...")
robot.set_angle_servo(0, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

# Поднимаем лифт с ящиком
print("  Поднимаем ящик...")
robot.set_angle_servo(300, 1)   # лифт вверх
time.sleep(2.0)

set_drive(-50,-50)
time.sleep(2.0)
stop()
print("  Ящик захвачен!")

# ================================================================
# Фаза 12: Разворот на 180°
# ================================================================
print("[Фаза 12] Разворот на 180°...")
stop()
time.sleep(0.2)
set_drive(-50, 50)
time.sleep(TURN_TIME * 2)       # два раза по TURN_TIME ≈ 180°
stop()
time.sleep(0.3)

while robot.analog_1 < 800 and robot.analog_2 < 800:
    set_drive(40, 40)

stop()

turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
set_drive(50, -50)


# ================================================================
# Фаза 13: Едем по линии вперёд до стены (analog_1/2 > 800)
# ================================================================
print("[Фаза 13] Едем по линии обратно до стены...")

start = time.time()
while time.time() - start < 60 and (robot.analog_1 + robot.analog_2) / 2 < 800:
    frame      = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)  # сначала обычный режим
        if cam_error is None:
            cam_error, frame = camera_correction(frame, wide=True)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

# ================================================================
# Фаза 14: Поворот вправо после доставки
# ================================================================
print("[Фаза 14] Поворот вправо после доставки...")
stop()
time.sleep(0.2)
set_drive(50, -50)
time.sleep(TURN_TIME)
stop()
time.sleep(0.2)

print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

# Закрываем клешню
print("  Открываем клешню...")
robot.set_angle_servo(300, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

robot.set_angle_servo(300, 1)
time.sleep(2.0)

time.sleep(0.2)
turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
stop()
time.sleep(0.2)

# ================================================================
# Фаза 15: Поиск коробки 2
# ================================================================

CHECK_INTERVAL = 3
FIRST_CHECK    = True
FIRST_PASS = True
last_check     = time.time()
start          = time.time()

while time.time() - start < 60:
    if time.time() - last_check > CHECK_INTERVAL:
        if FIRST_CHECK:
            FIRST_CHECK    = False
            CHECK_INTERVAL = 4.0
        stop()
        print("Проверяем поворот налево...")
        set_drive(-50, 50)
        time.sleep(TURN_TIME)
        stop()
        frame = robot.camera_image
        last_check = time.time()
        if detect_road(frame) and FIRST_PASS:
            FIRST_PASS = False
        elif detect_road(frame):
            break
        set_drive(50, -50)
        time.sleep(TURN_TIME)
        stop()


    frame      = robot.camera_image
    cam_error  = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)


# ================================================================
# Фаза 16: Подъезд к ящику со стабилизацией по direction
# ================================================================
print("[Фаза 16] Подъезжаем и центрируем ящик...")

start = time.time()
while time.time() - start < 30:
    frame = robot.camera_image
    if frame is None:
        time.sleep(0.05)
        continue

    distance, direction, debug_frame = detect_box(frame)

    cv2.imshow("АГРОБОТ", debug_frame)
    cv2.waitKey(1)

    if distance is None:
        # Ящик потерян — едем медленно вперёд, ищем
        set_drive(APPROACH_SPEED, APPROACH_SPEED)
        time.sleep(0.05)
        continue

    if distance == 'very_near':
        # Ящик достаточно близко — переходим к захвату
        stop()
        print(f"  Ящик очень близко, direction={direction:.2f} — захватываем!")
        break

    # Стабилизация: корректируем курс по горизонтальному положению ящика
    correction = direction * K_BOX
    left  = max(-100, min(100, APPROACH_SPEED + correction))
    right = max(-100, min(100, APPROACH_SPEED - correction))
    set_drive(left, right)

    time.sleep(0.05)

# ================================================================
# Фаза 17: Захват
# ================================================================
print("[Фаза 17] Захват ящика...")
stop()
time.sleep(0.3)

# Опускаем лифт
print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

print("Подъехали к ящику")
set_drive(50, 50)
time.sleep(2.0)
stop()

# Закрываем клешню
print("  Закрываем клешню...")
robot.set_angle_servo(0, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

# Поднимаем лифт с ящиком
print("  Поднимаем ящик...")
robot.set_angle_servo(300, 1)   # лифт вверх
time.sleep(2.0)

set_drive(-50,-50)
time.sleep(2.0)
stop()
print("  Ящик захвачен!")

# ================================================================
# Фаза 18: Разворот на 180°
# ================================================================
print("[Фаза 18] Разворот на 180°...")
stop()
time.sleep(0.2)
set_drive(-50, 50)
time.sleep(TURN_TIME * 2 + 0.05)       # два раза по TURN_TIME ≈ 180°
stop()
time.sleep(0.3)

while robot.analog_1 < 800 and robot.analog_2 < 800:
    set_drive(40, 40)

stop()

turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
set_drive(50, -50)


# ================================================================
# Фаза 19: Едем по линии вперёд до стены (analog_1/2 > 800)
# ================================================================
print("[Фаза 19] Едем по линии обратно до стены...")

start = time.time()
while time.time() - start < 60 and (robot.analog_1 + robot.analog_2) / 2 < 800:
    frame      = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)  # сначала обычный режим
        if cam_error is None:
            cam_error, frame = camera_correction(frame, wide=True)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

# ================================================================
# Фаза 20: Поворот вправо после доставки
# ================================================================
print("[Фаза 20] Поворот вправо после доставки...")
stop()
time.sleep(0.2)
set_drive(50, -50)
time.sleep(TURN_TIME)
stop()
time.sleep(0.2)

print("  Опускаем лифт...")
robot.set_angle_servo(0, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

# Закрываем клешню
print("  Открываем клешню...")
robot.set_angle_servo(300, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

robot.set_angle_servo(300, 1)
time.sleep(2.0)

time.sleep(0.2)
turn_to_yaw(snap_yaw(robot.yaw - 90), timeout=3)
stop()
time.sleep(0.2)

# ================================================================
# Фаза 21: Поиск и наезд на рычаг
# ================================================================
print("[Фаза 21] Ищем рычаг...")

CHECK_INTERVAL = 2.5
FIRST_CHECK    = True
last_check     = time.time()

start = time.time()
while time.time() - start < 60:
    if time.time() - last_check > CHECK_INTERVAL:
        if FIRST_CHECK:
            FIRST_CHECK    = False
            CHECK_INTERVAL = 4.0
        stop()
        print("Проверяем поворот налево...")
        set_drive(-50, 50)
        time.sleep(TURN_TIME)
        stop()
        frame = robot.camera_image
        last_check = time.time()
        while frame is None:
            frame = robot.camera_image
        direction, debug_frame = detect_lever(frame)
        if direction is not None:
            break
        else:
            set_drive(50, -50)
            time.sleep(TURN_TIME)
            stop()
    frame = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, AFTER_TURN_SPEED + correction))
    right = max(-100, min(100, AFTER_TURN_SPEED - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)

stop()

print("  Опускаем лифт...")
robot.set_angle_servo(30, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)

# Закрываем клешню
print("  Открываем клешню...")
robot.set_angle_servo(0, 2)     # клешня закрыта (0 = закрыто)
time.sleep(1.5)

print("Рычаг найден, наезжаем...")
start = time.time()
while time.time() - start < 10:
    frame = robot.camera_image
    if frame is None:
        time.sleep(0.05)
        continue

    direction, debug_frame = detect_lever(frame)

    cv2.imshow("АГРОБОТ", debug_frame)
    cv2.waitKey(1)

    if direction is None:
        # Рычаг пропал — скорее всего уже под роботом
        print("Рычаг под роботом — стоп!")
        set_drive(50, 50)
        time.sleep(2)
        stop()
        break

    # Едем на рычаг: коррекция по direction как у ящика
    correction = direction * K_BOX
    left  = max(-100, min(100, APPROACH_SPEED + 20 + correction))
    right = max(-100, min(100, APPROACH_SPEED + 20 - correction))
    set_drive(left, right)
    time.sleep(0.05)

stop()


set_drive(-50, -50)
time.sleep(3)
stop()
robot.set_angle_servo(300, 1)     # лифт вниз (0 = низ)
time.sleep(2.0)
print("[Фаза 15] Рычаг активирован.")

turn_to_yaw(snap_yaw(robot.yaw + 90))
turn_to_yaw(snap_yaw(robot.yaw + 90))

start = time.time()
while time.time() - start < 60:
    frame      = robot.camera_image
    cam_error = None
    if frame is not None:
        cam_error, frame = camera_correction(frame)  # сначала обычный режим
        if cam_error is None:
            cam_error, frame = camera_correction(frame, wide=True)
            if cam_error is None:
                break

    correction = cam_error * K_CAMERA if cam_error is not None else 0
    left  = max(-100, min(100, 40 + correction))
    right = max(-100, min(100, 40 - correction))
    set_drive(left, right)

    if frame is not None:
        cv2.imshow("АГРОБОТ", frame)
        cv2.waitKey(1)

    time.sleep(0.05)
set_drive(50, 50)
time.sleep(2)
stop()
turn_to_yaw(snap_yaw(robot.yaw + 90))
drive_forward_yaw(snap_yaw(robot.yaw))
turn_to_yaw(snap_yaw(robot.yaw - 90))
drive_forward_yaw(snap_yaw(robot.yaw), 50, 60)

# ================================================================
# Финал
# ================================================================
stop()
cv2.destroyAllWindows()
print("Готово. ")