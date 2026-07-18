from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
from robocad.common import CommonRobot


Direction = Literal["left", "right"]
BendDirection = Literal["auto", "left", "right"]


@dataclass
class Config:
    lift_servo_port: int = 1
    claw_servo_port: int = 2
    lift_up_angle: int = 300
    lift_down_angle: int = 30
    start_forward_speed: float = 25
    start_forward_time: float = 4

    follow_speed: float = 50
    reacquire_speed: float = 20
    turn_speed: float = 38
    max_speed: float = 70

    yaw_positive_is_right: bool = True
    yaw_tolerance_deg: float = 3.0
    turn_timeout_s: float = 4.5

    line_threshold: float = 1500.0

    k_sensor: float = 18.0
    k_camera: float = 52.0
    k_camera_d: float = 8.0
    k_yaw: float = 0.75
    max_correction: float = 32.0

    bend_turn_direction: BendDirection = "auto"
    min_follow_before_bend_s: float = 3.0
    min_follow_after_bend_s: float = 1.7
    lost_time_for_bend_s: float = 1.10
    lost_time_for_finish_s: float = 1.70
    corner_camera_error: float = 0.67
    corner_confirm_frames: int = 8
    reacquire_timeout_s: float = 4.0

    road_end_enabled: bool = True
    road_end_min_drive_s: float = 1.2
    road_end_roi_x1: float = 0.28
    road_end_roi_x2: float = 0.72
    road_end_roi_y1: float = 0.22
    road_end_roi_y2: float = 0.48
    road_light_v_min: int = 82
    road_light_ratio_threshold: float = 0.38
    road_mean_v_dark_threshold: float = 76.0
    road_end_confirm_frames: int = 6
    road_end_approach_time: float = 3.2
    road_end_approach_speed: float = 50
    road_end_extra_forward_time: float = 0.0
    road_end_extra_forward_speed: float = 14

    after_finish_turn_direction: Direction = "left"
    after_finish_claw_down_pause_s: float = 2
    after_finish_forward_speed: float = 23
    after_finish_forward_time: float = 8.0

    show_debug: bool = True
    debug_window_name: str = "AGROBOT line route"


class LineRouteRobot:
    def __init__(self, robot: CommonRobot, cfg: Config = Config()):
        self.robot = robot
        self.cfg = cfg
        self.prev_cam_err: Optional[float] = None
        self.last_cam_err: Optional[float] = None
        self.last_visible_direction: Optional[Direction] = None

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
        err = (cx_full - w / 2) / (w / 2)

        self.last_cam_err = err
        if err > 0.20:
            self.last_visible_direction = "right"
        elif err < -0.20:
            self.last_visible_direction = "left"

        return err

    def road_end_info(self, frame: np.ndarray) -> tuple[bool, dict[str, float], np.ndarray]:
        h, w = frame.shape[:2]
        x1 = int(w * self.cfg.road_end_roi_x1)
        x2 = int(w * self.cfg.road_end_roi_x2)
        y1 = int(h * self.cfg.road_end_roi_y1)
        y2 = int(h * self.cfg.road_end_roi_y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, {"light_ratio": 1.0, "mean_v": 255.0}, frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        light_road = (v_ch >= self.cfg.road_light_v_min) & (s_ch <= 105)
        white_marking = (v_ch >= 135) & (s_ch <= 95)
        light_road = light_road & (~white_marking)

        light_ratio = float(np.mean(light_road)) if light_road.size else 1.0
        mean_v = float(np.mean(v_ch)) if v_ch.size else 255.0

        is_dark_end = (
            light_ratio < self.cfg.road_light_ratio_threshold
            and mean_v < self.cfg.road_mean_v_dark_threshold
        )

        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 180, 255), 2)

        info = {"light_ratio": light_ratio, "mean_v": mean_v}
        return is_dark_end, info, debug

    def detect_side_with_camera(self) -> Optional[Direction]:
        frame = self.robot.camera_image
        if frame is None:
            return None

        h, w = frame.shape[:2]
        mask = self.white_mask(frame)

        y1, y2 = int(h * 0.28), int(h * 0.78)
        left_roi = mask[y1:y2, int(w * 0.02):int(w * 0.35)]
        right_roi = mask[y1:y2, int(w * 0.65):int(w * 0.98)]

        left_ratio = float(np.mean(left_roi > 0)) if left_roi.size else 0.0
        right_ratio = float(np.mean(right_roi > 0)) if right_roi.size else 0.0

        if max(left_ratio, right_ratio) < 0.003:
            return None

        return "left" if left_ratio > right_ratio else "right"

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

    def choose_bend_direction(self) -> Direction:
        if self.cfg.bend_turn_direction in ("left", "right"):
            return self.cfg.bend_turn_direction

        side = self.detect_side_with_camera()
        if side is not None:
            return side

        if self.last_visible_direction is not None:
            return self.last_visible_direction

        return "right"

    def reacquire_line_after_turn(self) -> None:
        print("Ищу линию после поворота...")

        target_yaw = self.yaw
        start = time.time()
        seen_frames = 0

        while time.time() - start < self.cfg.reacquire_timeout_s:
            visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(target_yaw)

            if visible:
                seen_frames += 1
            else:
                seen_frames = 0

            self.show_debug(
                None,
                f"reacquire {seen_frames}/5 src={source}",
                cam_err,
                sensor_err,
                None,
            )

            if seen_frames >= 5:
                self.stop(0.1)
                print("Линия найдена")
                return

            time.sleep(0.035)

        print("WARNING: линия после поворота не найдена, продолжаю по yaw")

    def follow_until_bend(self) -> Direction:
        print("Еду по линии до 90-градусного поворота дороги...")

        target_yaw = self.yaw
        start = time.time()
        last_visible_time = time.time()
        edge_frames = 0

        while True:
            now = time.time()
            elapsed = now - start
            visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(target_yaw)

            if visible:
                last_visible_time = now

            lost_for = now - last_visible_time

            if (
                elapsed > self.cfg.min_follow_before_bend_s
                and cam_err is not None
                and abs(cam_err) > self.cfg.corner_camera_error
            ):
                edge_frames += 1
            else:
                edge_frames = 0

            self.show_debug(
                None,
                f"to bend src={source} lost={lost_for:.1f}s edge={edge_frames}",
                cam_err,
                sensor_err,
                None,
            )

            if elapsed > self.cfg.min_follow_before_bend_s:
                if lost_for >= self.cfg.lost_time_for_bend_s or edge_frames >= self.cfg.corner_confirm_frames:
                    self.stop(0.15)
                    direction = self.choose_bend_direction()
                    print(f"Поворот дороги обнаружен, направление: {direction}")
                    return direction

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

            time.sleep(0.035)

    def follow_until_road_end(self) -> None:
        print("Еду по линии до конца светлой дороги...")

        target_yaw = self.yaw
        start = time.time()
        last_visible_time = time.time()
        line_was_seen = False
        road_end_frames = 0
        road_end_detected_at: Optional[float] = None

        while True:
            now = time.time()
            elapsed = now - start
            approach_mode = road_end_detected_at is not None
            drive_speed = self.cfg.road_end_approach_speed if approach_mode else self.cfg.follow_speed

            visible, source, cam_err, sensor_err = self.drive_by_line_or_yaw(
                target_yaw,
                speed=drive_speed,
            )

            if visible:
                line_was_seen = True
                last_visible_time = now

            lost_for = now - last_visible_time
            road_info: Optional[dict[str, float]] = None
            debug_frame: Optional[np.ndarray] = None

            if (
                not approach_mode
                and self.cfg.road_end_enabled
                and elapsed > self.cfg.road_end_min_drive_s
            ):
                frame = self.robot.camera_image
                if frame is not None:
                    road_end, road_info, debug_frame = self.road_end_info(frame)
                    if road_end:
                        road_end_frames += 1
                    else:
                        road_end_frames = 0

            if road_end_detected_at is None:
                status = f"to road end src={source} lost={lost_for:.1f}s dark={road_end_frames}"
                if road_info is not None:
                    status += f" light={road_info['light_ratio']:.2f} V={road_info['mean_v']:.0f}"
            else:
                left = max(0.0, self.cfg.road_end_approach_time - (now - road_end_detected_at))
                status = f"approach edge {left:.2f}s src={source} lost={lost_for:.1f}s"

            self.show_debug(debug_frame, status, cam_err, sensor_err, road_info)

            if road_end_detected_at is None and road_end_frames >= self.cfg.road_end_confirm_frames:
                road_end_detected_at = now
                print(
                    "Конец светлой дороги обнаружен по камере — "
                    f"доезжаю ещё {self.cfg.road_end_approach_time:.2f} c"
                )

            if (
                road_end_detected_at is not None
                and now - road_end_detected_at >= self.cfg.road_end_approach_time
            ):
                print("Доехал до края дороги")
                self.stop(0.1)

                if self.cfg.road_end_extra_forward_time > 0:
                    self.drive_forward_for(
                        self.cfg.road_end_extra_forward_time,
                        self.cfg.road_end_extra_forward_speed,
                        "extra to edge",
                    )
                return

            if (
                road_end_detected_at is None
                and elapsed > self.cfg.min_follow_after_bend_s
                and line_was_seen
                and lost_for >= self.cfg.lost_time_for_finish_s
            ):
                print("Линия закончилась — запасная остановка")
                self.stop(0.2)
                return

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

            time.sleep(0.035)

    def show_debug(
        self,
        frame: Optional[np.ndarray],
        text: str,
        cam_err: Optional[float],
        sensor_err: Optional[float],
        road_info: Optional[dict[str, float]],
    ) -> None:
        if not self.cfg.show_debug:
            return

        if frame is None:
            frame = self.robot.camera_image
        if frame is None:
            return

        out = frame.copy()
        s5, s6, s7, s8 = self.get_line_sensors()

        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
        cv2.putText(out, f"yaw={self.yaw:.1f}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"cam={None if cam_err is None else round(cam_err, 2)} "
            f"sensor={None if sensor_err is None else round(sensor_err, 2)}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            out,
            f"a8={s8:.0f} a7={s7:.0f} a6={s6:.0f} a5={s5:.0f}",
            (10, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
        )

        if road_info is not None:
            cv2.putText(
                out,
                f"road light={road_info['light_ratio']:.2f} meanV={road_info['mean_v']:.0f}",
                (10, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 180, 255),
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

        if self.cfg.road_end_enabled:
            cv2.rectangle(
                out,
                (int(w * self.cfg.road_end_roi_x1), int(h * self.cfg.road_end_roi_y1)),
                (int(w * self.cfg.road_end_roi_x2), int(h * self.cfg.road_end_roi_y2)),
                (0, 180, 255),
                1,
            )

        cv2.imshow(self.cfg.debug_window_name, out)
        cv2.waitKey(1)

    def run(self) -> None:
        print("Поднимаю клешню")
        self.robot.set_angle_servo(self.cfg.lift_up_angle, self.cfg.lift_servo_port)
        time.sleep(0.8)

        print("Немного еду вперёд")
        self.drive_forward_for(self.cfg.start_forward_time, self.cfg.start_forward_speed, "start forward")

        self.turn_by_yaw("left", 90.0)

        bend_direction = self.follow_until_bend()
        self.turn_by_yaw(bend_direction, 90.0)
        self.reacquire_line_after_turn()
        self.follow_until_road_end()

        print("На конце дороги поворачиваю налево")
        self.turn_by_yaw(self.cfg.after_finish_turn_direction, 90.0)
        self.stop(0.2)

        print("Опускаю клешню")
        self.robot.set_angle_servo(self.cfg.lift_down_angle, self.cfg.lift_servo_port)
        time.sleep(self.cfg.after_finish_claw_down_pause_s)
        self.stop(0.1)

        print(f"Еду прямо после опускания клешни {self.cfg.after_finish_forward_time:.2f} c")
        self.drive_forward_for(
            self.cfg.after_finish_forward_time,
            self.cfg.after_finish_forward_speed,
            "after finish forward",
        )


def main() -> None:
    robot = CommonRobot(False)
    runner = LineRouteRobot(robot, Config())

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
