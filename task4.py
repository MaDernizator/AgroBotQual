from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
from robocad.common import CommonRobot


Direction = Literal["left", "right"]


@dataclass
class Config:
    lift_servo_port: int = 1
    claw_servo_port: int = 2
    lift_up_angle: int = 300
    lift_down_angle: int = 30
    lift_claw_at_start: bool = True
    lift_claw_at_start_pause_s: float = 0.35

    follow_speed: float = 50
    turn_speed: float = 38
    max_speed: float = 70

    yaw_positive_is_right: bool = True
    yaw_tolerance_deg: float = 3.0
    turn_timeout_s: float = 4.5
    k_yaw: float = 0.75

    line_threshold: float = 1500.0
    k_sensor: float = 18.0
    k_camera: float = 52.0
    k_camera_d: float = 8.0
    max_correction: float = 32.0

    left_turn_min_drive_s: float = 3.0
    left_turn_lost_time_s: float = 1.10
    left_turn_camera_error: float = -0.67
    left_turn_confirm_frames: int = 8

    reacquire_timeout_s: float = 4.0
    reacquire_speed: float = 25

    blue_wall_roi_x1: float = 0.58
    blue_wall_roi_x2: float = 0.98
    blue_wall_roi_y1: float = 0.12
    blue_wall_roi_y2: float = 0.88
    blue_wall_min_drive_s: float = 1.0
    blue_wall_ratio_threshold: float = 0.035
    blue_wall_min_contour_area: float = 450.0
    blue_wall_confirm_frames: int = 6
    blue_wall_candidate_frames_to_slow: int = 1
    blue_wall_candidate_speed: float = 5
    blue_wall_after_detect_back_time: float = 1.0
    blue_wall_after_detect_back_speed: float = 18
    blue_wall_approach_time: float = 0.0
    blue_wall_approach_speed: float = 35

    after_blue_turn_direction: Direction = "right"
    claw_down_pause_s: float = 0.8
    push_forward_time: float = 20.0
    push_forward_speed: float = 50

    show_debug: bool = True
    debug_window_name: str = "AGROBOT blue wall lever task"


class BlueWallLeverRobot:
    def __init__(self, robot: CommonRobot, cfg: Config = Config()):
        self.robot = robot
        self.cfg = cfg
        self.prev_cam_err: Optional[float] = None

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @property
    def yaw(self) -> float:
        try:
            return float(self.robot.yaw)
        except Exception:
            return 0.0

    @staticmethod
    def angle_error(target: float, current: float) -> float:
        return (target - current + 180.0) % 360.0 - 180.0

    def set_drive(self, left: float, right: float) -> None:
        left = self.clamp(left, -self.cfg.max_speed, self.cfg.max_speed)
        right = self.clamp(right, -self.cfg.max_speed, self.cfg.max_speed)

        self.robot.motor_speed_0 = left
        self.robot.motor_speed_1 = left
        self.robot.motor_speed_2 = left
        self.robot.motor_speed_3 = -right
        self.robot.motor_speed_4 = -right
        self.robot.motor_speed_5 = -right

    def stop(self, pause: float = 0.15) -> None:
        for i in range(6):
            setattr(self.robot, f"motor_speed_{i}", 0)
        if pause > 0:
            time.sleep(pause)

    def spin(self, direction: Direction, speed: Optional[float] = None) -> None:
        s = self.cfg.turn_speed if speed is None else speed
        if direction == "right":
            self.set_drive(s, -s)
        else:
            self.set_drive(-s, s)

    def turn_by_yaw(self, direction: Direction, degrees: float = 90.0) -> None:
        print(f"Поворот {direction} на {degrees:.0f} градусов")

        sign = 1.0 if direction == "right" else -1.0
        if not self.cfg.yaw_positive_is_right:
            sign *= -1.0

        target = self.yaw + sign * degrees
        start = time.time()

        while time.time() - start < self.cfg.turn_timeout_s:
            err = self.angle_error(target, self.yaw)
            if abs(err) <= self.cfg.yaw_tolerance_deg:
                self.stop(0.2)
                return

            need_right = err > 0 if self.cfg.yaw_positive_is_right else err < 0
            self.spin("right" if need_right else "left")
            self.show_debug(None, f"turn {direction}: yaw_err={err:+.1f}", None, None, None)
            time.sleep(0.025)

        self.stop(0.2)
        print("WARNING: поворот завершился по таймауту, проверьте yaw_positive_is_right/turn_speed")

    def drive_forward_for(self, seconds: float, speed: float, label: str) -> None:
        target_yaw = self.yaw
        start = time.time()

        while time.time() - start < seconds:
            yaw_err = self.angle_error(target_yaw, self.yaw)
            correction = self.clamp(yaw_err * self.cfg.k_yaw, -18, 18)
            self.set_drive(speed + correction, speed - correction)

            left = seconds - (time.time() - start)
            self.show_debug(None, f"{label}: {left:.1f}s", None, None, None)
            time.sleep(0.03)

        self.stop(0.1)

    def get_line_sensors(self) -> tuple[float, float, float, float]:
        return (
            float(self.robot.analog_5),
            float(self.robot.analog_6),
            float(self.robot.analog_7),
            float(self.robot.analog_8),
        )

    def sensor_line_error(self) -> Optional[float]:
        s5, s6, s7, s8 = self.get_line_sensors()
        vals = [s8, s7, s6, s5]
        weights = [-1.5, -0.5, 0.5, 1.5]

        active = [v if v > self.cfg.line_threshold else 0.0 for v in vals]
        total = sum(active)
        if total <= 0:
            return None

        return sum(w * v for w, v in zip(weights, active)) / total

    @staticmethod
    def white_mask(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 95, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return mask

    @staticmethod
    def blue_mask(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([88, 55, 35]), np.array([135, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mask

    def camera_line_error(self, frame: np.ndarray) -> Optional[float]:
        h, w = frame.shape[:2]

        x1, x2 = int(w * 0.16), int(w * 0.84)
        y1, y2 = int(h * 0.38), int(h * 0.84)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        mask = self.white_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xs: list[float] = []
        weights: list[float] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:
                continue

            m = cv2.moments(c)
            if m["m00"] == 0:
                continue

            xs.append(float(m["m10"] / m["m00"]))
            weights.append(area)

        if not xs:
            return None

        cx_roi = float(np.average(xs, weights=weights))
        cx_full = cx_roi + x1
        return (cx_full - w / 2) / (w / 2)

    def blue_wall_right_info(self, frame: np.ndarray) -> tuple[bool, dict[str, float], np.ndarray]:
        h, w = frame.shape[:2]
        x1 = int(w * self.cfg.blue_wall_roi_x1)
        x2 = int(w * self.cfg.blue_wall_roi_x2)
        y1 = int(h * self.cfg.blue_wall_roi_y1)
        y2 = int(h * self.cfg.blue_wall_roi_y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, {"blue_ratio": 0.0, "max_area": 0.0}, frame

        mask = self.blue_mask(roi)
        blue_ratio = float(np.mean(mask > 0)) if mask.size else 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max((cv2.contourArea(c) for c in contours), default=0.0)

        detected = (
            blue_ratio >= self.cfg.blue_wall_ratio_threshold
            or max_area >= self.cfg.blue_wall_min_contour_area
        )

        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

        info = {"blue_ratio": blue_ratio, "max_area": float(max_area)}
        return detected, info, debug

    def drive_by_line_or_yaw(
        self,
        target_yaw: float,
        speed: Optional[float] = None,
    ) -> tuple[bool, str, Optional[float], Optional[float]]:
        frame = self.robot.camera_image
        cam_err = self.camera_line_error(frame) if frame is not None else None
        sensor_err = self.sensor_line_error()

        visible = cam_err is not None or sensor_err is not None

        if cam_err is not None:
            d = 0.0 if self.prev_cam_err is None else cam_err - self.prev_cam_err
            correction = cam_err * self.cfg.k_camera + d * self.cfg.k_camera_d
            self.prev_cam_err = cam_err
            source = "camera"
        elif sensor_err is not None:
            correction = sensor_err * self.cfg.k_sensor
            source = "sensor"
        else:
            yaw_err = self.angle_error(target_yaw, self.yaw)
            correction = yaw_err * self.cfg.k_yaw
            source = "yaw/no_line"

        correction = self.clamp(correction, -self.cfg.max_correction, self.cfg.max_correction)
        base_speed = self.cfg.follow_speed if speed is None else speed
        self.set_drive(base_speed + correction, base_speed - correction)

        return visible, source, cam_err, sensor_err

    def follow_until_left_turn(self) -> None:
        print("Еду по дороге до поворота налево...")

        target_yaw = self.yaw
        start = time.time()
        last_visible_time = time.time()
        left_edge_frames = 0

        while True:
            now = time.time()
            elapsed = now - start
            visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(target_yaw)

            if visible:
                last_visible_time = now

            lost_for = now - last_visible_time

            if (
                elapsed > self.cfg.left_turn_min_drive_s
                and cam_err is not None
                and cam_err <= self.cfg.left_turn_camera_error
            ):
                left_edge_frames += 1
            else:
                left_edge_frames = 0

            self.show_debug(
                None,
                f"to left turn src={source} lost={lost_for:.1f}s left={left_edge_frames}",
                cam_err,
                sensor_err,
                None,
            )

            if elapsed > self.cfg.left_turn_min_drive_s:
                if lost_for >= self.cfg.left_turn_lost_time_s or left_edge_frames >= self.cfg.left_turn_confirm_frames:
                    self.stop(0.15)
                    print("Поворот налево обнаружен")
                    return

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

            time.sleep(0.035)

    def reacquire_line_after_turn(self) -> None:
        print("Ищу линию после поворота...")

        target_yaw = self.yaw
        start = time.time()
        seen_frames = 0

        while time.time() - start < self.cfg.reacquire_timeout_s:
            visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(
                target_yaw,
                speed=self.cfg.reacquire_speed,
            )

            if visible:
                seen_frames += 1
            else:
                seen_frames = 0

            self.show_debug(None, f"reacquire {seen_frames}/5 src={source}", cam_err, sensor_err, None)

            if seen_frames >= 5:
                self.stop(0.1)
                print("Линия найдена")
                return

            time.sleep(0.035)

        print("WARNING: линия после поворота не найдена, продолжаю по yaw")

    def follow_until_blue_wall_right(self) -> None:
        print("Еду до появления синей стены справа в кадре...")

        target_yaw = self.yaw
        start = time.time()
        blue_frames = 0
        blue_detected_at: Optional[float] = None

        while True:
            now = time.time()
            elapsed = now - start

            blue_info: Optional[dict[str, float]] = None
            debug_frame: Optional[np.ndarray] = None

            if blue_detected_at is None and elapsed > self.cfg.blue_wall_min_drive_s:
                frame = self.robot.camera_image
                if frame is not None:
                    blue_found, blue_info, debug_frame = self.blue_wall_right_info(frame)
                    if blue_found:
                        blue_frames += 1
                    else:
                        blue_frames = 0

            if blue_detected_at is None and blue_frames >= self.cfg.blue_wall_confirm_frames:
                blue_detected_at = now
                print("Синяя стена справа обнаружена")
                self.show_debug(debug_frame, "blue wall confirmed", None, None, blue_info)
                self.stop(0.05)

                if self.cfg.blue_wall_after_detect_back_time > 0:
                    self.drive_forward_for(
                        self.cfg.blue_wall_after_detect_back_time,
                        -abs(self.cfg.blue_wall_after_detect_back_speed),
                        "blue wall back correction",
                    )

                if self.cfg.blue_wall_approach_time <= 0:
                    self.stop(0.2)
                    return

            approach_mode = blue_detected_at is not None

            if approach_mode:
                left = max(0.0, self.cfg.blue_wall_approach_time - (now - blue_detected_at))
                if left <= 0:
                    self.stop(0.2)
                    return

                speed = self.cfg.blue_wall_approach_speed
                visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(target_yaw, speed=speed)
                status = f"approach blue {left:.2f}s src={source}"
            else:
                if blue_frames >= self.cfg.blue_wall_candidate_frames_to_slow:
                    speed = self.cfg.blue_wall_candidate_speed
                else:
                    speed = self.cfg.follow_speed

                visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(target_yaw, speed=speed)
                status = f"to blue wall src={source} blue={blue_frames} speed={speed:.0f}"
                if blue_info is not None:
                    status += f" ratio={blue_info['blue_ratio']:.3f} area={blue_info['max_area']:.0f}"

            self.show_debug(debug_frame, status, cam_err, sensor_err, blue_info)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

            time.sleep(0.035)

    def show_debug(
        self,
        frame: Optional[np.ndarray],
        text: str,
        cam_err: Optional[float],
        sensor_err: Optional[float],
        extra_info: Optional[dict[str, float]],
    ) -> None:
        if not self.cfg.show_debug:
            return

        if frame is None:
            frame = self.robot.camera_image
        if frame is None:
            return

        out = frame.copy()
        s5, s6, s7, s8 = self.get_line_sensors()

        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
        cv2.putText(out, f"yaw={self.yaw:.1f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"cam={None if cam_err is None else round(cam_err, 2)} "
            f"sensor={None if sensor_err is None else round(sensor_err, 2)}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"a8={s8:.0f} a7={s7:.0f} a6={s6:.0f} a5={s5:.0f}",
            (10, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
        )

        if extra_info is not None and "blue_ratio" in extra_info:
            cv2.putText(
                out,
                f"blue ratio={extra_info['blue_ratio']:.3f} area={extra_info['max_area']:.0f}",
                (10, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 0),
                2,
            )

        h, w = out.shape[:2]
        cv2.rectangle(
            out,
            (int(w * 0.16), int(h * 0.38)),
            (int(w * 0.84), int(h * 0.84)),
            (255, 255, 0),
            1,
        )
        cv2.rectangle(
            out,
            (int(w * self.cfg.blue_wall_roi_x1), int(h * self.cfg.blue_wall_roi_y1)),
            (int(w * self.cfg.blue_wall_roi_x2), int(h * self.cfg.blue_wall_roi_y2)),
            (255, 0, 0),
            1,
        )

        cv2.imshow(self.cfg.debug_window_name, out)
        cv2.waitKey(1)

    def run(self) -> None:
        if self.cfg.lift_claw_at_start:
            print("Поднимаю клешню перед движением")
            self.robot.set_angle_servo(self.cfg.lift_up_angle, self.cfg.lift_servo_port)
            time.sleep(self.cfg.lift_claw_at_start_pause_s)

        self.follow_until_left_turn()
        self.turn_by_yaw("left", 90.0)
        self.reacquire_line_after_turn()

        self.follow_until_blue_wall_right()
        self.turn_by_yaw(self.cfg.after_blue_turn_direction, 90.0)

        self.stop(0.2)
        print("Опускаю клешню")
        self.robot.set_angle_servo(self.cfg.lift_down_angle, self.cfg.lift_servo_port)
        time.sleep(self.cfg.claw_down_pause_s)
        self.stop(0.1)

        print(f"Толкаю рычаг вперёд {self.cfg.push_forward_time:.2f} c")
        self.drive_forward_for(
            self.cfg.push_forward_time,
            self.cfg.push_forward_speed,
            "push lever",
        )


def main() -> None:
    robot = CommonRobot(False)
    runner = BlueWallLeverRobot(robot, Config())

    try:
        runner.run()
        print("Готово")
    finally:
        runner.stop(0)
        try:
            robot.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
