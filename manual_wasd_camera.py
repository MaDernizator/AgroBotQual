from pathlib import Path
from datetime import datetime
import threading
import time
from typing import Optional

import cv2
import numpy as np
from pynput import keyboard
from robocad.common import CommonRobot

import base_robot_commands as cmds


# ===== НАСТРОЙКИ =====
BASE_SPEED = 50                 # обычная скорость
BOOST_MULTIPLIER = 2            # Shift умножает скорость на 2
MAX_SPEED = 100                 # по документации robocad скорости -100..100
COMMAND_PERIOD = 0.05           # как часто повторно отправлять команду моторам
VIDEO_FPS = 20.0                # fps файла записи
WINDOW_NAME = "Agrobot camera"


robot = CommonRobot(False)

# Состояние клавиш хранится отдельно от cv2.waitKey,
# чтобы нормально видеть отпускание клавиш и комбинации WA/WD/SA/SD.
state_lock = threading.Lock()
pressed = set()                 # {'w', 'a', 's', 'd', 'shift', 'p', 'l', 'k'}
record_toggle_requested = False
lift_toggle_requested = False
gripper_toggle_requested = False
exit_requested = False

# Состояние механизмов захвата. Клавиши K/L работают как переключатели.
lift_is_up = False
gripper_is_closed = False


def clamp_speed(value: int) -> int:
    return max(-MAX_SPEED, min(MAX_SPEED, value))


def current_speed() -> int:
    with state_lock:
        shift_pressed = "shift" in pressed
    speed = BASE_SPEED * (BOOST_MULTIPLIER if shift_pressed else 1)
    return clamp_speed(speed)


def get_movement_name() -> str:
    """Возвращает имя текущего движения по нажатым WASD.

    Противоположные клавиши взаимно гасятся:
    W+S = нет движения вперед/назад, A+D = нет движения влево/вправо.
    """
    with state_lock:
        w = "w" in pressed
        a = "a" in pressed
        s = "s" in pressed
        d = "d" in pressed

    forward = w and not s
    backward = s and not w
    left = a and not d
    right = d and not a

    if forward and left:
        return "forward_left"
    if forward and right:
        return "forward_right"
    if backward and left:
        return "backward_left"
    if backward and right:
        return "backward_right"
    if forward:
        return "forward"
    if backward:
        return "backward"
    if left:
        return "left"
    if right:
        return "right"
    return "stop"


def apply_movement(robot_obj: CommonRobot, movement: str, speed: int) -> None:
    """Отправляет в робота нужный пресет скоростей из base_robot_commands.py."""
    if movement == "forward":
        cmds.move_forward(robot_obj, speed)
    elif movement == "backward":
        cmds.move_backward(robot_obj, speed)
    elif movement == "left":
        cmds.move_left(robot_obj, speed)
    elif movement == "right":
        cmds.move_right(robot_obj, speed)
    elif movement == "forward_left":
        cmds.move_forward_left(robot_obj, speed)
    elif movement == "forward_right":
        cmds.move_forward_right(robot_obj, speed)
    elif movement == "backward_left":
        cmds.move_backward_left(robot_obj, speed)
    elif movement == "backward_right":
        cmds.move_backward_right(robot_obj, speed)
    else:
        cmds.stop_motors(robot_obj)


def toggle_lift(robot_obj: CommonRobot) -> str:
    """L: поднять/опустить лифт захвата."""
    global lift_is_up

    if lift_is_up:
        cmds.lift_down_gripper(robot_obj)
        lift_is_up = False
        return "down"

    cmds.lift_up__gripper(robot_obj)
    lift_is_up = True
    return "up"


def toggle_gripper(robot_obj: CommonRobot) -> str:
    """K: схватить/отпустить клешню."""
    global gripper_is_closed

    if gripper_is_closed:
        cmds.open_gripper(robot_obj)
        gripper_is_closed = False
        return "open"

    cmds.close_gripper(robot_obj)
    gripper_is_closed = True
    return "closed"


def normalize_key(key):
    """Преобразует клавишу pynput в строку, которая нужна программе."""
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        return "shift"

    try:
        ch = key.char.lower()
    except AttributeError:
        return None

    if ch in ("w", "a", "s", "d", "p", "q", "l", "k"):
        return ch
    return None


def on_press(key):
    global record_toggle_requested, lift_toggle_requested, gripper_toggle_requested, exit_requested

    k = normalize_key(key)
    if k is None:
        return

    with state_lock:
        # Защита от автоповтора: переключатели срабатывают только один раз на одно нажатие.
        was_already_pressed = k in pressed
        pressed.add(k)

        if k == "p" and not was_already_pressed:
            record_toggle_requested = True

        if k == "l" and not was_already_pressed:
            lift_toggle_requested = True

        if k == "k" and not was_already_pressed:
            gripper_toggle_requested = True

        if k == "q":
            exit_requested = True


def on_release(key):
    k = normalize_key(key)
    if k is None:
        return

    with state_lock:
        pressed.discard(k)


def pop_record_toggle_request() -> bool:
    global record_toggle_requested
    with state_lock:
        result = record_toggle_requested
        record_toggle_requested = False
    return result


def pop_lift_toggle_request() -> bool:
    global lift_toggle_requested
    with state_lock:
        result = lift_toggle_requested
        lift_toggle_requested = False
    return result


def pop_gripper_toggle_request() -> bool:
    global gripper_toggle_requested
    with state_lock:
        result = gripper_toggle_requested
        gripper_toggle_requested = False
    return result


def should_exit() -> bool:
    with state_lock:
        return exit_requested


def make_video_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    media_dir = script_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return media_dir / f"agrobot_record_{stamp}.avi"


def open_video_writer(frame: np.ndarray):
    height, width = frame.shape[:2]
    path = make_video_path()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, (width, height))

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Не удалось открыть файл записи: {path}")

    print(f"[REC] Запись началась: {path}")
    return writer, path


def make_placeholder_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "No camera frame yet", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def draw_overlay(frame: np.ndarray, is_recording: bool, movement: str, speed: int, video_path: Optional[Path]) -> np.ndarray:
    display = frame.copy()

    # Подсказки прямо в окне камеры.
    cv2.putText(display, "WASD move | Shift boost | P rec | L lift | K grab | Q/Esc exit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f"move: {movement}   speed: {speed}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    lift_text = "up" if lift_is_up else "down"
    gripper_text = "closed" if gripper_is_closed else "open"
    cv2.putText(display, f"lift: {lift_text}   gripper: {gripper_text}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if is_recording:
        # Значок записи в окне. В файл пишется чистое изображение без этой плашки.
        cv2.circle(display, (25, 110), 9, (0, 0, 255), -1)
        cv2.putText(display, "REC", (42, 117),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if video_path is not None:
            cv2.putText(display, video_path.name, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return display


def main():
    global exit_requested

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    writer = None
    video_path = None
    want_recording = False

    last_movement = None
    last_speed = None
    next_command_time = 0.0

    try:
        print("Управление: WASD — движение, Shift — скорость x2, P — старт/стоп записи, L — поднять/опустить захват, K — схватить/отпустить, Q или Esc — выход")

        while not should_exit():
            now = time.time()

            movement = get_movement_name()
            speed = current_speed()

            # Немедленно отправляем новый пресет при изменении WASD/Shift,
            # и дополнительно повторяем команду периодически.
            if movement != last_movement or speed != last_speed or now >= next_command_time:
                apply_movement(robot, movement, speed)
                last_movement = movement
                last_speed = speed
                next_command_time = now + COMMAND_PERIOD

            if pop_record_toggle_request():
                want_recording = not want_recording
                if not want_recording and writer is not None:
                    writer.release()
                    writer = None
                    print(f"[REC] Запись остановлена: {video_path}")
                    video_path = None

            if pop_lift_toggle_request():
                lift_state = toggle_lift(robot)
                print(f"[LIFT] {lift_state}")

            if pop_gripper_toggle_request():
                gripper_state = toggle_gripper(robot)
                print(f"[GRIPPER] {gripper_state}")

            frame = robot.camera_image
            if frame is None:
                frame_for_display = make_placeholder_frame()
            else:
                # На случай, если камера отдаст серый кадр.
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame_for_display = frame

                if want_recording and writer is None:
                    try:
                        writer, video_path = open_video_writer(frame)
                    except RuntimeError as exc:
                        print(exc)
                        want_recording = False

                if writer is not None:
                    # В видео пишем кадр без интерфейсных надписей.
                    writer.write(frame)

            display = draw_overlay(frame_for_display, writer is not None, movement, speed, video_path)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc или q в окне OpenCV
                with state_lock:
                    exit_requested = True

            time.sleep(0.005)

    finally:
        cmds.stop_motors(robot)
        robot.stop()

        if writer is not None:
            writer.release()
            print(f"[REC] Запись остановлена: {video_path}")

        listener.stop()
        cv2.destroyAllWindows()
        print("Робот остановлен, окно камеры закрыто.")


if __name__ == "__main__":
    main()
