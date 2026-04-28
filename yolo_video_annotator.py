#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter-разметчик видео для YOLO.

Что делает:
- по умолчанию берёт видео из папки ./media;
- открывает одно видео или папку с видео;
- показывает кадры в окне Tkinter;
- позволяет выбрать класс мышкой справа;
- позволяет мышкой выделить bounding box на кадре;
- сохраняет картинки и txt-разметку в формате YOLO;
- позволяет скипнуть кадр без сохранения.

Зависимости:
    pip install opencv-python pillow

Если Tkinter не установлен в Linux:
    sudo apt install python3-tk

Запуск без аргументов:
    python yolo_tk_annotator.py

Будет использована папка:
    ./media

Запуск с другой папкой/видео:
    python yolo_tk_annotator.py ./my_videos --out dataset_yolo

Горячие клавиши:
    1..8       выбрать класс
    Enter/N    сохранить кадр и перейти дальше
    Space/S    скипнуть кадр без сохранения
    U          удалить последний bbox
    C          очистить bbox на текущем кадре
    Q/Esc      выход
"""

from __future__ import annotations

import argparse
import re
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import List, Optional, Sequence, Tuple

import cv2
from PIL import Image, ImageTk


# Номер строки = id класса в YOLO.
CLASSES = [
    "коробка",
    "дерево",
    "забор",
    "белые полосы дорожной разметки",
    "заграждение",
    "контейнер",
    "бочка",
    "рычаг",
]

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_SOURCE = "./media"
DEFAULT_OUT_DIR = "dataset_yolo"

# Цвета только для отображения в интерфейсе.
BOX_COLORS = ["#ff4040", "#35b779", "#4aa3ff", "#ffd43b", "#b197fc", "#ff922b", "#20c997", "#f783ac"]


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def clipped(self, width: int, height: int) -> "Box":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return Box(
            class_id=self.class_id,
            x1=max(0, min(width - 1, x1)),
            y1=max(0, min(height - 1, y1)),
            x2=max(0, min(width - 1, x2)),
            y2=max(0, min(height - 1, y2)),
        )

    def to_yolo_line(self, width: int, height: int) -> str:
        box = self.clipped(width, height)
        bw = max(0, box.x2 - box.x1)
        bh = max(0, box.y2 - box.y1)
        xc = box.x1 + bw / 2
        yc = box.y1 + bh / 2
        return (
            f"{box.class_id} "
            f"{xc / width:.6f} "
            f"{yc / height:.6f} "
            f"{bw / width:.6f} "
            f"{bh / height:.6f}"
        )


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ_.-]+", "_", text)
    text = text.strip("._-")
    return text or "video"


def collect_videos(source: Path) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Файл не похож на видео: {source}")
        return [source]

    if not source.exists():
        source.mkdir(parents=True, exist_ok=True)
        return []

    return [
        p for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def write_dataset_files(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    (out_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    names_yaml = "\n".join(f"  {i}: {name!r}" for i, name in enumerate(CLASSES))
    data_yaml = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


class YoloTkAnnotator:
    def __init__(
        self,
        root: tk.Tk,
        videos: Sequence[Path],
        out_dir: Path,
        step: int,
        max_w: int,
        max_h: int,
        source_text: str,
    ):
        self.root = root
        self.videos = list(videos)
        self.out_dir = out_dir
        self.step = max(1, step)
        self.max_w = max_w
        self.max_h = max_h
        self.source_text = source_text

        self.video_index = -1
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_video: Optional[Path] = None
        self.current_frame = None
        self.current_frame_index = 0
        self.saved_count = 0
        self.skipped_count = 0

        self.boxes: List[Box] = []
        self.class_var = tk.IntVar(value=0)

        self.display_scale = 1.0
        self.display_w = 1
        self.display_h = 1
        self.photo = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.temp_rect_id: Optional[int] = None

        self.root.title("YOLO video annotator — Tkinter")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self._build_ui()
        self._bind_keys()
        write_dataset_files(self.out_dir)

        if not self.videos:
            messagebox.showinfo(
                "Видео не найдены",
                f"Видео не найдены в:\n{self.source_text}\n\n"
                "Если запускаешь без аргументов, положи .avi/.mp4/.mov/.mkv/.webm/.m4v в папку ./media."
            )
            self.set_status(f"Видео не найдены. Папка источника: {self.source_text}")
            self._set_controls_state(tk.DISABLED)
            return

        self.open_next_video()
        self.load_next_frame(first_frame=True)

    def _build_ui(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main = tk.Frame(self.root)
        main.grid(row=0, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        left = tk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=800, height=600, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        right = tk.Frame(main, width=280)
        right.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        right.grid_propagate(False)

        tk.Label(right, text="Классы", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 6))
        self.class_buttons = []
        for i, name in enumerate(CLASSES):
            rb = tk.Radiobutton(
                right,
                text=f"{i + 1}. {name}",
                variable=self.class_var,
                value=i,
                anchor="w",
                justify="left",
                wraplength=250,
                font=("Arial", 11),
            )
            rb.pack(fill="x", anchor="w", pady=2)
            self.class_buttons.append(rb)

        tk.Frame(right, height=10).pack()

        self.btn_save = tk.Button(right, text="Сохранить кадр  Enter / N", command=self.save_and_next)
        self.btn_save.pack(fill="x", pady=3)

        self.btn_skip = tk.Button(right, text="Скипнуть кадр  Space / S", command=self.skip_frame)
        self.btn_skip.pack(fill="x", pady=3)

        self.btn_undo = tk.Button(right, text="Удалить последний bbox  U", command=self.undo_box)
        self.btn_undo.pack(fill="x", pady=3)

        self.btn_clear = tk.Button(right, text="Очистить кадр  C", command=self.clear_boxes)
        self.btn_clear.pack(fill="x", pady=3)

        self.btn_quit = tk.Button(right, text="Выход  Q / Esc", command=self.quit_app)
        self.btn_quit.pack(fill="x", pady=(12, 3))

        tk.Frame(right, height=10).pack()

        help_text = (
            "Как размечать:\n"
            "1) Выбери класс справа.\n"
            "2) Зажми ЛКМ на кадре.\n"
            "3) Протяни прямоугольник.\n"
            "4) Сохрани или скипни кадр.\n\n"
            "Save с пустым bbox создаёт пустой label — это можно использовать как негативный пример."
        )
        tk.Label(right, text=help_text, justify="left", wraplength=250).pack(anchor="w", pady=(8, 0))

        self.status_var = tk.StringVar(value="Готово")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken")
        status.grid(row=1, column=0, sticky="ew")

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _event: self.save_and_next())
        self.root.bind("n", lambda _event: self.save_and_next())
        self.root.bind("N", lambda _event: self.save_and_next())
        self.root.bind("<space>", lambda _event: self.skip_frame())
        self.root.bind("s", lambda _event: self.skip_frame())
        self.root.bind("S", lambda _event: self.skip_frame())
        self.root.bind("u", lambda _event: self.undo_box())
        self.root.bind("U", lambda _event: self.undo_box())
        self.root.bind("c", lambda _event: self.clear_boxes())
        self.root.bind("C", lambda _event: self.clear_boxes())
        self.root.bind("q", lambda _event: self.quit_app())
        self.root.bind("Q", lambda _event: self.quit_app())
        self.root.bind("<Escape>", lambda _event: self.quit_app())
        for i in range(len(CLASSES)):
            self.root.bind(str(i + 1), lambda _event, cid=i: self.class_var.set(cid))

    def _set_controls_state(self, state: str) -> None:
        for widget in [self.btn_save, self.btn_skip, self.btn_undo, self.btn_clear]:
            widget.configure(state=state)
        for rb in self.class_buttons:
            rb.configure(state=state)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def open_next_video(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_index += 1
        if self.video_index >= len(self.videos):
            self.current_video = None
            return False

        self.current_video = self.videos[self.video_index]
        self.cap = cv2.VideoCapture(str(self.current_video))
        if not self.cap.isOpened():
            messagebox.showwarning("Ошибка видео", f"Не удалось открыть видео:\n{self.current_video}")
            return self.open_next_video()

        self.set_status(f"Открыто видео {self.video_index + 1}/{len(self.videos)}: {self.current_video.name}")
        return True

    def load_next_frame(self, first_frame: bool = False) -> None:
        if self.cap is None:
            self.finish_all()
            return

        if not first_frame:
            for _ in range(self.step - 1):
                if self.cap is not None:
                    self.cap.grab()

        while True:
            if self.cap is None:
                self.finish_all()
                return

            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.current_frame = frame
                self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.boxes.clear()
                self.drag_start = None
                self.temp_rect_id = None
                self.show_current_frame()
                return

            if not self.open_next_video():
                self.finish_all()
                return

    def show_current_frame(self) -> None:
        if self.current_frame is None:
            return

        frame_h, frame_w = self.current_frame.shape[:2]
        self.display_scale = min(self.max_w / frame_w, self.max_h / frame_h, 1.0)
        self.display_w = max(1, int(frame_w * self.display_scale))
        self.display_h = max(1, int(frame_h * self.display_scale))

        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if self.display_w != frame_w or self.display_h != frame_h:
            image = image.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.canvas.configure(width=self.display_w, height=self.display_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.redraw_boxes()

        video_name = self.current_video.name if self.current_video else "-"
        self.set_status(
            f"Видео: {video_name} | кадр: {self.current_frame_index} | "
            f"bbox: {len(self.boxes)} | saved: {self.saved_count} | skipped: {self.skipped_count} | "
            f"out: {self.out_dir}"
        )

    def redraw_boxes(self) -> None:
        self.canvas.delete("bbox")
        for idx, box in enumerate(self.boxes):
            color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
            x1 = int(box.x1 * self.display_scale)
            y1 = int(box.y1 * self.display_scale)
            x2 = int(box.x2 * self.display_scale)
            y2 = int(box.y2 * self.display_scale)

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="bbox")
            label = f"{idx + 1}: {CLASSES[box.class_id]}"
            text_id = self.canvas.create_text(x1 + 4, y1 + 4, text=label, anchor="nw", fill="white", tags="bbox")
            bbox = self.canvas.bbox(text_id)
            if bbox:
                pad = 2
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - pad,
                    bbox[1] - pad,
                    bbox[2] + pad,
                    bbox[3] + pad,
                    fill=color,
                    outline=color,
                    tags="bbox",
                )
                self.canvas.tag_lower(bg_id, text_id)

    def display_to_frame_xy(self, x: int, y: int) -> Tuple[int, int]:
        if self.current_frame is None:
            return 0, 0

        frame_h, frame_w = self.current_frame.shape[:2]
        fx = int(max(0, min(frame_w - 1, x / self.display_scale)))
        fy = int(max(0, min(frame_h - 1, y / self.display_scale)))
        return fx, fy

    def on_mouse_down(self, event) -> None:
        if self.current_frame is None:
            return

        x = max(0, min(self.display_w - 1, event.x))
        y = max(0, min(self.display_h - 1, event.y))
        self.drag_start = (x, y)
        color = BOX_COLORS[self.class_var.get() % len(BOX_COLORS)]
        self.temp_rect_id = self.canvas.create_rectangle(x, y, x, y, outline=color, width=2, dash=(4, 2))

    def on_mouse_move(self, event) -> None:
        if self.drag_start is None or self.temp_rect_id is None:
            return

        x0, y0 = self.drag_start
        x = max(0, min(self.display_w - 1, event.x))
        y = max(0, min(self.display_h - 1, event.y))
        self.canvas.coords(self.temp_rect_id, x0, y0, x, y)

    def on_mouse_up(self, event) -> None:
        if self.current_frame is None or self.drag_start is None:
            return

        x0, y0 = self.drag_start
        x1 = max(0, min(self.display_w - 1, event.x))
        y1 = max(0, min(self.display_h - 1, event.y))

        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        self.drag_start = None

        # Слишком маленький прямоугольник считаем случайным кликом.
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        fx0, fy0 = self.display_to_frame_xy(x0, y0)
        fx1, fy1 = self.display_to_frame_xy(x1, y1)
        self.boxes.append(Box(self.class_var.get(), fx0, fy0, fx1, fy1))
        self.show_current_frame()

    def image_label_paths(self) -> Tuple[Path, Path]:
        assert self.current_video is not None
        video_stem = safe_name(self.current_video.stem)
        base = f"{video_stem}_f{self.current_frame_index:06d}"
        img_path = self.out_dir / "images" / "train" / f"{base}.jpg"
        label_path = self.out_dir / "labels" / "train" / f"{base}.txt"

        # Защита от перезаписи, если разные видео имеют одинаковые имена.
        counter = 1
        while img_path.exists() or label_path.exists():
            img_path = self.out_dir / "images" / "train" / f"{base}_{counter}.jpg"
            label_path = self.out_dir / "labels" / "train" / f"{base}_{counter}.txt"
            counter += 1

        return img_path, label_path

    def save_and_next(self) -> None:
        if self.current_frame is None:
            return

        frame_h, frame_w = self.current_frame.shape[:2]
        img_path, label_path = self.image_label_paths()

        ok = cv2.imwrite(str(img_path), self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить изображение:\n{img_path}")
            return

        lines = []
        for box in self.boxes:
            clipped = box.clipped(frame_w, frame_h)
            if clipped.x2 - clipped.x1 >= 2 and clipped.y2 - clipped.y1 >= 2:
                lines.append(clipped.to_yolo_line(frame_w, frame_h))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.saved_count += 1
        self.load_next_frame()

    def skip_frame(self) -> None:
        if self.current_frame is None:
            return
        self.skipped_count += 1
        self.load_next_frame()

    def undo_box(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self.show_current_frame()

    def clear_boxes(self) -> None:
        self.boxes.clear()
        self.show_current_frame()

    def finish_all(self) -> None:
        self.current_frame = None
        self.canvas.delete("all")
        self.canvas.create_text(
            400,
            300,
            text="Все видео закончились",
            fill="white",
            font=("Arial", 20, "bold"),
        )
        self._set_controls_state(tk.DISABLED)
        self.set_status(f"Готово. Сохранено кадров: {self.saved_count}, скипнуто: {self.skipped_count}. Dataset: {self.out_dir}")
        messagebox.showinfo("Готово", f"Разметка завершена.\n\nDataset сохранён в:\n{self.out_dir}")

    def quit_app(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter-разметчик видео для YOLO")
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help="Видео или папка с видео. По умолчанию ./media",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help="Папка для YOLO dataset. По умолчанию dataset_yolo",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Брать каждый N-й кадр. Например --step 5. По умолчанию 1",
    )
    parser.add_argument("--max-width", type=int, default=1000, help="Максимальная ширина показа кадра")
    parser.add_argument("--max-height", type=int, default=720, help="Максимальная высота показа кадра")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source = Path(args.source)
    out_dir = Path(args.out)

    try:
        videos = collect_videos(source)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    root = tk.Tk()
    root.minsize(900, 650)
    YoloTkAnnotator(root, videos, out_dir, args.step, args.max_width, args.max_height, str(source))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
