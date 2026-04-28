#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autonomous_barrel_route_v005.py

Автономный сценарий для Агробота:
1) поднять захват;
2) найти бочку спереди через YOLO;
3) подъехать, раскрыть захват, опустить, схватить, поднять;
4) повернуть налево и ехать по белой разметке;
5) найти правое ответвление, повернуть направо;
6) ехать до следующего правого поворота;
7) ехать прямо по дороге;
8) найти правое ответвление / дырку в заборе и повернуть;
9) найти бочки;
10) подъехать, опустить и отпустить бочку;
11) развернуться и ехать на выезд.

ВАЖНО:
- Файл best.pt должен лежать рядом со скриптом.
- base_robot_commands.py должен лежать рядом со скриптом.
- Пока захват опускается/поднимается/открывается/закрывается, детекция НЕ запускается.
- Если кадра нет, он не передаётся в YOLO, чтобы детекция не ломалась.

Установка:
    pip install ultralytics opencv-python numpy

Запуск:
    python autonomous_barrel_route_v005.py

Отладка с окном камеры:
    python autonomous_barrel_route_v005.py --show

Если робот едет слишком долго/коротко, меняй константы в CONFIG.
Поиск объектов и ответвлений идёт до фактического обнаружения, без аварийного выхода по времени.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from robocad.common import CommonRobot
from ultralytics import YOLO

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

import base_robot_commands as cmds


# ============================================================
# НАСТРОЙКИ, КОТОРЫЕ СКОРЕЕ ВСЕГО ПРИДЁТСЯ ПОДКРУЧИВАТЬ
# ============================================================

@dataclass
class Config:
    model_path: Path = Path(__file__).resolve().parent / "best.pt"

    # Скорости движения
    forward_speed: int = 42
    slow_forward_speed: int = 28
    turn_speed: int = 35
    search_turn_speed: int = 25
    line_speed: int = 40
    correction_speed: int = 34
    reverse_speed: int = 35

    # YOLO
    imgsz: int = 640
    conf: float = 0.25

    # Подход к бочке/месту сброса.
    # Робот считает, что объект достаточно близко, если bbox занимает
    # достаточную часть высоты/площади кадра.
    # Для захвата подъезжаем ближе: теперь стоп только когда bbox бочки крупнее.
    barrel_close_height_ratio: float = 0.46
    barrel_close_area_ratio: float = 0.145
    drop_close_height_ratio: float = 0.44
    drop_close_area_ratio: float = 0.13

    # Дополнительный короткий подполз после того, как бочка уже считается близкой.
    # Помогает реально заехать захватом под объект перед закрытием.
    pickup_extra_forward_sec: float = 0.03
    drop_extra_forward_sec: float = 0.05

    # Центрирование объекта
    center_dead_zone: float = 0.14

    # При подходе к нескольким бочкам не переключаемся на дальнюю:
    # после первого выбора держим lock по центру bbox и выбираем ближайшую к lock бочку.
    target_lock_max_dist: float = 0.42
    close_stop_center_zone: float = 0.46

    # Белая разметка
    line_dead_zone: float = 0.18
    right_branch_x_ratio: float = 0.66
    right_branch_min_area_ratio: float = 0.018
    right_branch_confirm_frames: int = 4

    # Тайминги механизма захвата.
    # Во время этих пауз YOLO НЕ используется. На --show показывается только сырой кадр камеры.
    # Сделаны специально большими, чтобы сервоприводы успевали открыть/опустить/закрыть/поднять.
    lift_time: float = 3.20
    lower_time: float = 3.40
    gripper_time: float = 2.50

    # Тайминги манёвров.
    left_yaw_turn_deg: float = 90.0
    left_turn_fallback_sec: float = 1.25
    right_turn_sec: float = 1.05
    u_turn_sec: float = 2.35
    reverse_after_drop_sec: float = 1.00

    # Тайминги маршрута.
    # После взятия бочки: строго повернуть налево по yaw на 90° и ехать прямо 10 секунд,
    # только потом начинать искать правый поворот.
    after_left_forward_sec: float = 10.0
    # Правые ответвления ищутся БЕЗ ограничения по времени: едем по разметке, пока реально не увидим ответвление.
    straight_by_fence_sec: float = 5.5
    exit_line_sec: float = 8.0


    # Главный цикл
    loop_sleep: float = 0.04


CONFIG = Config()


# Алиасы классов. Названия берутся из best.pt, поэтому оставлены русские и запасные варианты.
BARREL_ALIASES = {"бочка", "barrel", "bochka", "бак", "баки", "barrels"}
WHITE_LINE_ALIASES = {
    "белые полосы дорожной разметки",
    "белая разметка",
    "разметка",
    "white line",
    "white_lines",
    "road marking",
}


def get_cyrillic_font(size: int = 18):
    """Шрифт с кириллицей для подписей YOLO в окне --show."""
    if not PIL_AVAILABLE:
        return None

    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_utf8_text(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    size: int = 18,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Рисует русский текст. cv2.putText кириллицу не поддерживает, поэтому используем PIL."""
    if not PIL_AVAILABLE:
        safe = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(img, safe, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return

    font = get_cyrillic_font(size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    rgb = (color[2], color[1], color[0])

    if bg is not None:
        bbox = draw.textbbox(xy, text, font=font)
        pad = 3
        bg_rgb = (bg[2], bg[1], bg[0])
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=bg_rgb,
        )

    draw.text(xy, text, font=font, fill=rgb)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@dataclass
class Detection:
    class_id: int
    name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    cx: float
    cy: float
    w: float
    h: float
    area: float


class Vision:
    def __init__(self, model_path: Path, imgsz: int, conf: float):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели: {model_path}\n"
                "Положи обученный best.pt рядом с autonomous_barrel_route_v005.py"
            )

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.names = self._load_names()

        print("[YOLO] model:", model_path)
        print("[YOLO] names:", self.names)

    def _load_names(self) -> Dict[int, str]:
        raw = getattr(self.model, "names", {})
        result: Dict[int, str] = {}

        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    result[int(k)] = str(v)
                except Exception:
                    pass
        elif isinstance(raw, list):
            for i, v in enumerate(raw):
                result[i] = str(v)

        return result

    def get_frame(self, robot: CommonRobot) -> Optional[np.ndarray]:
        """Безопасно берёт кадр. Если кадра нет — возвращает None."""
        frame = robot.camera_image

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            return None

        if frame.size == 0:
            return None

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if len(frame.shape) != 3:
            return None

        return frame

    def detect(self, frame: Optional[np.ndarray]) -> List[Detection]:
        """Не вызывает YOLO на None/битом кадре."""
        if frame is None:
            return []

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )
        except Exception as exc:
            print(f"[YOLO] detect error, frame skipped: {exc}")
            return []

        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: List[Detection] = []
        boxes = result.boxes

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
            cls = int(boxes.cls[i].detach().cpu().item())
            conf = float(boxes.conf[i].detach().cpu().item())
            x1, y1, x2, y2 = xyxy.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            detections.append(
                Detection(
                    class_id=cls,
                    name=self.names.get(cls, str(cls)),
                    conf=conf,
                    xyxy=(x1, y1, x2, y2),
                    cx=(x1 + x2) / 2.0,
                    cy=(y1 + y2) / 2.0,
                    w=w,
                    h=h,
                    area=w * h,
                )
            )

        return detections

    @staticmethod
    def filter_by_aliases(detections: List[Detection], aliases: set[str]) -> List[Detection]:
        aliases_lower = {a.lower() for a in aliases}
        return [d for d in detections if d.name.lower() in aliases_lower]

    @staticmethod
    def largest(detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    @staticmethod
    def draw(frame: np.ndarray, detections: List[Detection], state_name: str) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text = f"{d.name} {d.conf:.2f}"
            draw_utf8_text(out, text, (x1, max(4, y1 - 24)), size=18, color=(0, 255, 255), bg=(0, 0, 0))

        if state_name:
            draw_utf8_text(out, state_name, (10, 10), size=20, color=(255, 255, 255), bg=(0, 0, 0))
        return out


class AutonomousBarrelRoute:
    def __init__(self, show: bool = False):
        self.robot = CommonRobot(False)
        self.vision = Vision(CONFIG.model_path, CONFIG.imgsz, CONFIG.conf)
        self.show = show
        self.last_debug_frame: Optional[np.ndarray] = None

    # -------------------------
    # Низкоуровневое управление
    # -------------------------

    def stop(self, delay: float = 0.05) -> None:
        cmds.stop_motors(self.robot)
        time.sleep(delay)

    def forward(self, speed: Optional[int] = None) -> None:
        cmds.move_forward(self.robot, speed if speed is not None else CONFIG.forward_speed)

    def backward(self, speed: Optional[int] = None) -> None:
        cmds.move_backward(self.robot, speed if speed is not None else CONFIG.reverse_speed)

    def turn_left(self, speed: Optional[int] = None) -> None:
        cmds.move_left(self.robot, speed if speed is not None else CONFIG.turn_speed)

    def turn_right(self, speed: Optional[int] = None) -> None:
        cmds.move_right(self.robot, speed if speed is not None else CONFIG.turn_speed)

    def steer_forward_to_error(self, error: float, speed: int) -> None:
        """error < 0 значит объект/линия левее центра, error > 0 правее."""
        if abs(error) <= CONFIG.center_dead_zone:
            cmds.move_forward(self.robot, speed)
        elif error < 0:
            cmds.move_forward_left(self.robot, speed)
        else:
            cmds.move_forward_right(self.robot, speed)

    def run_for(self, action, seconds: float, speed: Optional[int] = None, state: str = "") -> None:
        """Таймерное движение без YOLO. На --show показывается сырой кадр камеры."""
        start = time.time()
        while time.time() - start < seconds:
            if speed is None:
                action()
            else:
                action(speed)
            self.debug_show_raw(state)
            time.sleep(CONFIG.loop_sleep)
        self.stop()

    # -------------------------
    # Захват: во время движения серво камера не используется
    # -------------------------

    def lift_up_no_vision(self) -> None:
        self.stop()
        print(f"[GRIPPER] lift up, vision paused for {CONFIG.lift_time:.1f}s")
        cmds.lift_up__gripper(self.robot)
        self.sleep_no_vision(CONFIG.lift_time, "RAW CAMERA: lift up, YOLO paused")

    def lift_down_no_vision(self) -> None:
        self.stop()
        print(f"[GRIPPER] lift down, vision paused for {CONFIG.lower_time:.1f}s")
        cmds.lift_down_gripper(self.robot)
        self.sleep_no_vision(CONFIG.lower_time, "RAW CAMERA: lift down, YOLO paused")

    def open_no_vision(self) -> None:
        self.stop()
        print(f"[GRIPPER] open, vision paused for {CONFIG.gripper_time:.1f}s")
        cmds.open_gripper(self.robot)
        self.sleep_no_vision(CONFIG.gripper_time, "RAW CAMERA: gripper open, YOLO paused")

    def close_no_vision(self) -> None:
        self.stop()
        print(f"[GRIPPER] close, vision paused for {CONFIG.gripper_time:.1f}s")
        cmds.close_gripper(self.robot)
        self.sleep_no_vision(CONFIG.gripper_time, "RAW CAMERA: gripper close, YOLO paused")

    def grab_barrel_sequence(self) -> None:
        print("[ACTION] grab barrel")
        self.open_no_vision()
        self.lift_down_no_vision()
        self.close_no_vision()
        self.lift_up_no_vision()

    def drop_barrel_sequence(self) -> None:
        print("[ACTION] drop barrel")
        self.lift_down_no_vision()
        self.open_no_vision()
        self.lift_up_no_vision()

    # -------------------------
    # Зрение и отладка
    # -------------------------

    def sense(self, state_name: str = "") -> Tuple[Optional[np.ndarray], List[Detection]]:
        frame = self.vision.get_frame(self.robot)
        if frame is None:
            print(f"[VISION] no frame in state={state_name}, skip")
            return None, []

        detections = self.vision.detect(frame)
        self.last_debug_frame = self.vision.draw(frame, detections, state_name)
        self.debug_show(state_name)
        return frame, detections

    def debug_show(self, state_name: str = "") -> None:
        """Показывает последний кадр с YOLO-боксами. Используется только в режимах зрения."""
        if not self.show:
            return

        frame = self.last_debug_frame
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "no frame", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("autonomous route debug", frame)
        cv2.waitKey(1)

    def debug_show_raw(self, state_name: str = "") -> None:
        """Показывает именно сырой кадр камеры: без YOLO, bbox и любых надписей."""
        if not self.show:
            return

        frame = self.vision.get_frame(self.robot)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame = frame.copy()

        cv2.imshow("autonomous route debug", frame)
        cv2.waitKey(1)

    def sleep_no_vision(self, seconds: float, state: str) -> None:
        """Пауза без YOLO; в --show обновляет только сырой кадр камеры."""
        start = time.time()
        while time.time() - start < seconds:
            self.debug_show_raw(state)
            time.sleep(CONFIG.loop_sleep)

    # -------------------------
    # Движение по объектам
    # -------------------------

    def find_target_by_rotation(self, aliases: set[str], state: str) -> bool:
        """Поворачивается на месте, пока не увидит объект нужного класса."""
        print(f"[STATE] {state}")

        while True:
            frame, detections = self.sense(state)
            targets = self.vision.filter_by_aliases(detections, aliases)
            target = self.vision.largest(targets)

            if frame is not None and target is not None:
                h, w = frame.shape[:2]
                error = (target.cx - w / 2) / (w / 2)
                if abs(error) < 0.28:
                    self.stop()
                    print(f"[FOUND] {target.name} conf={target.conf:.2f}")
                    return True

                # Доворачиваем к центру объекта.
                if error < 0:
                    self.turn_left(CONFIG.search_turn_speed)
                else:
                    self.turn_right(CONFIG.search_turn_speed)
            else:
                # Если кадра нет или объекта нет, медленно сканируем.
                self.turn_left(CONFIG.search_turn_speed)

            time.sleep(CONFIG.loop_sleep)

    @staticmethod
    def target_score_for_approach(target: Detection, frame_w: int, frame_h: int) -> float:
        """Оценка бочки для старта подхода.

        Нам нужна не просто самая большая bbox, а ближайшая/фронтальная бочка:
        - крупная;
        - ниже в кадре;
        - ближе к центру;
        - не дальняя боковая бочка из группы.
        """
        area_ratio = target.area / float(frame_w * frame_h)
        height_ratio = target.h / float(frame_h)
        bottom_ratio = min(1.0, target.cy / float(frame_h))
        center_error = abs((target.cx - frame_w / 2) / (frame_w / 2))
        center_score = max(0.0, 1.0 - center_error)
        return height_ratio * 3.0 + area_ratio * 4.0 + bottom_ratio * 0.8 + center_score * 0.9

    def choose_target_for_approach(
        self,
        targets: List[Detection],
        frame_w: int,
        frame_h: int,
        locked_center: Optional[Tuple[float, float]],
    ) -> Optional[Detection]:
        """Выбирает бочку для подхода и не даёт перескочить с ближней на дальнюю."""
        if not targets:
            return None

        if locked_center is not None:
            lx, ly = locked_center

            def lock_distance(t: Detection) -> float:
                dx = (t.cx - lx) / max(1.0, frame_w)
                dy = (t.cy - ly) / max(1.0, frame_h)
                return (dx * dx + dy * dy) ** 0.5

            near_lock = [t for t in targets if lock_distance(t) <= CONFIG.target_lock_max_dist]
            if near_lock:
                # Среди близких к lock выбираем не дальнюю боковую, а самую фронтальную/крупную.
                return max(near_lock, key=lambda t: self.target_score_for_approach(t, frame_w, frame_h) - lock_distance(t) * 2.0)

        return max(targets, key=lambda t: self.target_score_for_approach(t, frame_w, frame_h))


    def approach_target(
        self,
        aliases: set[str],
        close_height_ratio: float,
        close_area_ratio: float,
        state: str,
    ) -> bool:
        """Подъезжает к крупнейшему объекту заданного класса и центрируется по нему."""
        print(f"[STATE] {state}")
        last_seen = time.time()
        locked_center: Optional[Tuple[float, float]] = None

        while True:
            frame, detections = self.sense(state)
            if frame is None:
                self.stop(0.12)
                continue

            h, w = frame.shape[:2]
            targets = self.vision.filter_by_aliases(detections, aliases)

            # Если любая фронтальная бочка уже достаточно близко, останавливаемся.
            # Это не даёт роботу проехать мимо ближней бочки и начать ехать к дальней из группы.
            close_front = []
            for t in targets:
                area_ratio_t = t.area / float(w * h)
                height_ratio_t = t.h / float(h)
                error_t = (t.cx - w / 2) / (w / 2)
                if (height_ratio_t >= close_height_ratio or area_ratio_t >= close_area_ratio) and abs(error_t) <= CONFIG.close_stop_center_zone:
                    close_front.append(t)

            if close_front:
                target = self.choose_target_for_approach(close_front, w, h, locked_center)
                self.stop()
                print(f"[TARGET] close front barrel, stop: {target.name if target else 'target'}")
                return True

            target = self.choose_target_for_approach(targets, w, h, locked_center)

            if target is None:
                # Если только что видели объект — немного подождать/ползти,
                # иначе начать поиск поворотом.
                if time.time() - last_seen < 0.8:
                    self.forward(CONFIG.slow_forward_speed)
                else:
                    self.turn_left(CONFIG.search_turn_speed)
                time.sleep(CONFIG.loop_sleep)
                continue

            last_seen = time.time()

            # Lock обновляем плавно, чтобы не перескакивать на дальнюю соседнюю бочку.
            if locked_center is None:
                locked_center = (target.cx, target.cy)
                print(f"[TARGET] locked: {target.name} cx={target.cx:.0f} cy={target.cy:.0f}")
            else:
                lx, ly = locked_center
                locked_center = (lx * 0.72 + target.cx * 0.28, ly * 0.72 + target.cy * 0.28)

            area_ratio = target.area / float(w * h)
            height_ratio = target.h / float(h)
            error = (target.cx - w / 2) / (w / 2)

            print(
                f"[TARGET] {target.name} conf={target.conf:.2f} "
                f"err={error:.2f} h={height_ratio:.2f} area={area_ratio:.2f}"
            )

            if height_ratio >= close_height_ratio or area_ratio >= close_area_ratio:
                self.stop()
                print("[TARGET] close enough")
                return True

            if abs(error) > CONFIG.center_dead_zone:
                if error < 0:
                    cmds.move_forward_left(self.robot, CONFIG.slow_forward_speed)
                else:
                    cmds.move_forward_right(self.robot, CONFIG.slow_forward_speed)
            else:
                # В зоне бочек лучше ехать медленно, чтобы не проскочить точку захвата.
                self.forward(CONFIG.slow_forward_speed)

            time.sleep(CONFIG.loop_sleep)

    # -------------------------
    # Езда по белой разметке / ответвления
    # -------------------------

    def get_line_error_and_branch(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[Optional[float], bool]:
        h, w = frame.shape[:2]
        lines = self.vision.filter_by_aliases(detections, WHITE_LINE_ALIASES)

        if not lines:
            return None, False

        # Для следования берём крупнейшую разметку в нижних 70% кадра.
        lower_lines = [d for d in lines if d.cy > h * 0.30]
        line = self.vision.largest(lower_lines or lines)

        if line is None:
            return None, False

        error = (line.cx - w / 2) / (w / 2)

        # Ответвление направо: белая разметка/полоса появляется далеко справа
        # и занимает достаточную площадь.
        right_candidates = [
            d for d in lines
            if d.cx > w * CONFIG.right_branch_x_ratio
            and d.area / float(w * h) > CONFIG.right_branch_min_area_ratio
        ]
        right_branch = bool(right_candidates)

        return error, right_branch

    def follow_line_for(self, seconds: float, state: str) -> None:
        print(f"[STATE] {state} for {seconds:.1f}s")
        start = time.time()

        while time.time() - start < seconds:
            frame, detections = self.sense(state)
            if frame is None:
                self.stop(0.10)
                continue

            line_error, _right_branch = self.get_line_error_and_branch(frame, detections)

            if line_error is None:
                # Если разметка потеряна — не паникуем, едем медленно вперёд.
                self.forward(CONFIG.slow_forward_speed)
            elif abs(line_error) <= CONFIG.line_dead_zone:
                self.forward(CONFIG.line_speed)
            elif line_error < 0:
                cmds.move_forward_left(self.robot, CONFIG.correction_speed)
            else:
                cmds.move_forward_right(self.robot, CONFIG.correction_speed)

            time.sleep(CONFIG.loop_sleep)

        self.stop()

    def follow_until_right_branch(self, state: str) -> bool:
        print(f"[STATE] {state}: searching right branch without ограничения по времени")
        branch_frames = 0

        while True:
            frame, detections = self.sense(state)
            if frame is None:
                self.stop(0.10)
                branch_frames = 0
                continue

            line_error, right_branch = self.get_line_error_and_branch(frame, detections)

            if right_branch:
                branch_frames += 1
            else:
                branch_frames = max(0, branch_frames - 1)

            if branch_frames >= CONFIG.right_branch_confirm_frames:
                self.stop()
                print("[BRANCH] right branch confirmed")
                return True

            if line_error is None:
                self.forward(CONFIG.slow_forward_speed)
            elif abs(line_error) <= CONFIG.line_dead_zone:
                self.forward(CONFIG.line_speed)
            elif line_error < 0:
                cmds.move_forward_left(self.robot, CONFIG.correction_speed)
            else:
                cmds.move_forward_right(self.robot, CONFIG.correction_speed)

            time.sleep(CONFIG.loop_sleep)

        # Сейчас поиск ответвления без ограничения по времени, поэтому сюда не попадаем.
        self.stop()
        return False

    @staticmethod
    def yaw_delta_deg(current: float, previous: float) -> float:
        """Короткая разница углов с учётом перехода через -180/180 или 0/360."""
        return (current - previous + 180.0) % 360.0 - 180.0

    def read_yaw(self) -> Optional[float]:
        try:
            yaw = float(self.robot.yaw)
        except Exception as exc:
            print(f"[YAW] cannot read yaw: {exc}")
            return None
        return yaw

    def turn_left_by_yaw(self, degrees: float, state: str) -> bool:
        """Поворачивает налево, пока модуль накопленного изменения yaw не достигнет degrees.

        Знак yaw в симуляторе может зависеть от модели/координат, поэтому скрипт не
        предполагает +90 или -90. Он командует именно левый поворот и считает модуль
        накопленного изменения yaw.
        """
        print(f"[ACTION] left yaw turn: {degrees:.1f} deg, state={state}")

        prev_yaw = self.read_yaw()
        while prev_yaw is None:
            self.stop(0.05)
            self.debug_show_raw(f"{state}: waiting yaw")
            time.sleep(CONFIG.loop_sleep)
            prev_yaw = self.read_yaw()

        accumulated = 0.0
        while abs(accumulated) < degrees:
            self.turn_left(CONFIG.turn_speed)
            self.debug_show_raw(f"{state}: yaw {abs(accumulated):.1f}/{degrees:.1f}")
            time.sleep(CONFIG.loop_sleep)

            yaw = self.read_yaw()
            if yaw is None:
                continue

            delta = self.yaw_delta_deg(yaw, prev_yaw)
            # Отбрасываем невозможные скачки датчика, чтобы не остановиться случайно.
            if abs(delta) < 45.0:
                accumulated += delta
            prev_yaw = yaw

        self.stop()
        print(f"[YAW] left turn done, accumulated={accumulated:.1f} deg")
        return True

    def turn_right_timed(self, state: str) -> None:
        print(f"[ACTION] right turn: {state}")
        self.run_for(self.turn_right, CONFIG.right_turn_sec, CONFIG.turn_speed, state)

    def turn_left_timed(self, state: str) -> None:
        print(f"[ACTION] left turn fallback: {state}")
        self.run_for(self.turn_left, CONFIG.left_turn_fallback_sec, CONFIG.turn_speed, state)

    def u_turn(self) -> None:
        print("[ACTION] u-turn")
        self.run_for(self.turn_left, CONFIG.u_turn_sec, CONFIG.turn_speed, "u-turn")

    # -------------------------
    # Главный сценарий
    # -------------------------

    def run(self) -> None:
        try:
            print("[START] autonomous barrel route")
            self.stop()

            # 1. Поднять захват. Камеру не используем во время движения серво.
            self.lift_up_no_vision()

            # 2. Увидеть бочку спереди.
            if not self.find_target_by_rotation(BARREL_ALIASES, "find first barrel"):
                return

            # 3. Подъехать, раскрыть, опустить, схватить, поднять.
            if not self.approach_target(
                BARREL_ALIASES,
                CONFIG.barrel_close_height_ratio,
                CONFIG.barrel_close_area_ratio,
                "approach first barrel",
            ):
                return

            # Подъехать ещё немного ближе, чтобы бочка точно оказалась внутри захвата.
            self.run_for(self.forward, CONFIG.pickup_extra_forward_sec, CONFIG.slow_forward_speed, "final creep to barrel")
            self.grab_barrel_sequence()

            # 4. Повернуться налево строго по yaw на 90°, затем 10 секунд ехать прямо.
            # В эти 10 секунд правый поворот НЕ ищем — начинаем искать только после проезда.
            self.turn_left_by_yaw(CONFIG.left_yaw_turn_deg, "turn left 90 by yaw after pickup")
            self.run_for(self.forward, CONFIG.after_left_forward_sec, CONFIG.line_speed, "forward 10s after yaw left")

            # 5. Только теперь ищем правое ответвление: повернуть и ехать до следующего правого поворота.
            self.follow_until_right_branch("find first right branch")
            self.turn_right_timed("first right branch")
            self.follow_until_right_branch("find second right branch")
            self.turn_right_timed("second right branch")

            # 6. Ехать прямо по дороге, справа вдоль дороги будет забор, но его не видно.
            self.follow_line_for(CONFIG.straight_by_fence_sec, "straight by invisible fence")

            # 7-8. Справа ответвление / дырка в заборе: проехать в ответвление.
            self.follow_until_right_branch("find fence hole/right branch")
            self.turn_right_timed("turn into fence hole")

            # 9. Найти бочки.
            if not self.find_target_by_rotation(BARREL_ALIASES, "find destination barrels"):
                return

            # 10. Подъехать туда и опустить/отпустить бочку.
            if not self.approach_target(
                BARREL_ALIASES,
                CONFIG.drop_close_height_ratio,
                CONFIG.drop_close_area_ratio,
                "approach destination barrels",
            ):
                return

            self.run_for(self.forward, CONFIG.drop_extra_forward_sec, CONFIG.slow_forward_speed, "final creep to drop zone")
            self.drop_barrel_sequence()

            # 11. Развернуться и поехать на выезд.
            self.run_for(self.backward, CONFIG.reverse_after_drop_sec, CONFIG.reverse_speed, "reverse after drop")
            self.u_turn()
            self.follow_line_for(CONFIG.exit_line_sec, "exit by white line")

            print("[DONE] scenario completed")

        finally:
            self.stop()
            self.robot.stop()
            if self.show:
                cv2.destroyAllWindows()
            print("[STOP] robot stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Автономный сценарий Агробота с YOLO-распознаванием.")
    parser.add_argument("--show", action="store_true", help="Показывать окно камеры с bbox для отладки.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route = AutonomousBarrelRoute(show=args.show)
    route.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
