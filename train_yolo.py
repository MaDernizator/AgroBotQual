#!/usr/bin/env python3
# train_yolo.py
#
# Скрипт обучения YOLO на датасете, созданном аннотатором:
# dataset_yolo/
#   images/train/
#   labels/train/
#   classes.txt
#   data.yaml
#
# Новые классы добавлены в конец, поэтому старая разметка не ломается:
# 0 коробка
# 1 дерево
# 2 забор
# 3 белые полосы дорожной разметки
# 4 заграждение
# 5 контейнер
# 6 бочка
# 7 рычаг
#
# Установка:
#   pip install ultralytics
#   pip install torch==2.10.0+cu128 torchaudio==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128   
#
# Запуск:
#   python train_yolo.py --device 0 --model yolov8s.pt

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CLASSES = [
    "коробка",
    "дерево",
    "забор",
    "белые полосы дорожной разметки",
    "заграждение",
    "контейнер",
    "бочка",
    "рычаг",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение Ultralytics YOLO на dataset_yolo."
    )

    parser.add_argument(
        "--dataset",
        default=str(SCRIPT_DIR / "dataset_yolo"),
        help="Папка датасета YOLO. По умолчанию: dataset_yolo рядом со скриптом",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Путь к data.yaml. Если не указан, используется <dataset>/data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Стартовая модель/чекпоинт. "
            "Для слабого ПК: yolov8n.pt. "
            "Точнее, но тяжелее: yolov8s.pt/yolov8m.pt."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Количество эпох обучения. По умолчанию: 80",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер изображения для обучения. По умолчанию: 640",
    )
    parser.add_argument(
        "--batch",
        default="0",
        help="Размер batch. Можно число, например 8, или auto. По умолчанию: auto",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство: 0 для первой NVIDIA GPU, cpu для процессора. Если не указано, Ultralytics выберет сам.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Количество worker-процессов загрузки данных. На Windows часто лучше 0 или 2.",
    )
    parser.add_argument(
        "--project",
        default=str((SCRIPT_DIR / "runs" / "agrobot_yolo").resolve()),
        help="Папка для результатов. По умолчанию: runs/agrobot_yolo рядом со скриптом",
    )
    parser.add_argument(
        "--name",
        default="train",
        help="Имя запуска внутри project. По умолчанию: train",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience. По умолчанию: 30",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с чекпоинта, указанного в --model.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="После обучения запустить validation.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="После обучения экспортировать best.pt в ONNX.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Не кешировать изображения при обучении.",
    )

    return parser.parse_args()


def read_classes(classes_path: Path) -> List[str]:
    if not classes_path.exists():
        return DEFAULT_CLASSES.copy()

    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return classes or DEFAULT_CLASSES.copy()


def merge_classes(existing: List[str]) -> Tuple[List[str], bool]:
    """Сохраняет старые id классов и дописывает недостающие классы в конец."""
    merged = list(existing)
    changed = False
    for name in DEFAULT_CLASSES:
        if name not in merged:
            merged.append(name)
            changed = True
    return merged, changed


def write_classes_file(classes_path: Path, classes: List[str]) -> None:
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def yaml_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def ensure_data_yaml(dataset_dir: Path, data_yaml: Path) -> List[str]:
    """Обновляет classes.txt и data.yaml, не удаляя images/labels.

    Это именно дополняет существующий dataset_yolo: старые id 0..5 остаются на местах,
    новые классы становятся id 6 и 7.
    """
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    classes_path = dataset_dir / "classes.txt"

    if not images_train.exists():
        raise FileNotFoundError(f"Не найдена папка с картинками: {images_train}")
    if not labels_train.exists():
        raise FileNotFoundError(f"Не найдена папка с label-файлами: {labels_train}")

    existing_classes = read_classes(classes_path)
    classes, changed = merge_classes(existing_classes)
    write_classes_file(classes_path, classes)

    if changed:
        print(f"[OK] classes.txt дополнен новыми классами: {classes_path}")
    else:
        print(f"[OK] classes.txt синхронизирован: {classes_path}")

    names_block = "\n".join(f"  {i}: {yaml_quote(name)}" for i, name in enumerate(classes))
    content = f"""# Автоматически обновлено train_yolo.py
path: {yaml_quote(str(dataset_dir.resolve()))}
train: images/train
val: images/train
nc: {len(classes)}

names:
{names_block}
"""
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(content, encoding="utf-8")
    print(f"[OK] data.yaml обновлён под текущий dataset_yolo: {data_yaml}")

    return classes


def count_files(folder: Path, suffixes: tuple[str, ...]) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def check_dataset(dataset_dir: Path, classes: List[str]) -> None:
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    image_count = count_files(images_dir, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    label_count = count_files(labels_dir, (".txt",))

    print(f"[DATASET] images/train: {image_count} изображений")
    print(f"[DATASET] labels/train: {label_count} файлов разметки")
    print(f"[DATASET] classes: {len(classes)} ({', '.join(classes)})")

    if image_count == 0:
        raise RuntimeError("В датасете нет изображений. Сначала разметь кадры аннотатором.")
    if label_count == 0:
        raise RuntimeError("В датасете нет label-файлов. Проверь, что кадры сохранялись с bbox-разметкой.")

    if label_count < image_count:
        print(
            "[WARN] label-файлов меньше, чем изображений. "
            "Это нормально только если часть кадров сохранена без объектов."
        )

    max_class_id = -1
    bad_lines = 0
    for label_file in labels_dir.rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                class_id = int(float(parts[0]))
            except Exception:
                bad_lines += 1
                continue
            max_class_id = max(max_class_id, class_id)

    if bad_lines:
        print(f"[WARN] Строк разметки с нераспознанным class_id: {bad_lines}")
    if max_class_id >= len(classes):
        raise RuntimeError(
            f"В labels есть class_id={max_class_id}, но в classes.txt/data.yaml только {len(classes)} классов."
        )


def normalize_batch(batch_arg: str):
    text = str(batch_arg).strip().lower()
    if text == "auto":
        return "auto"

    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("--batch должен быть числом или auto") from exc


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset).resolve()
    data_yaml = Path(args.data).resolve() if args.data else dataset_dir / "data.yaml"

    classes = ensure_data_yaml(dataset_dir, data_yaml)
    check_dataset(dataset_dir, classes)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Не установлен пакет ultralytics.\n"
            "Установи его командой:\n\n"
            "    pip install ultralytics\n",
            file=sys.stderr,
        )
        return 1

    batch = normalize_batch(args.batch)

    print("[TRAIN] Старт обучения")
    print(f"[TRAIN] model   = {args.model}")
    print(f"[TRAIN] data    = {data_yaml}")
    print(f"[TRAIN] epochs  = {args.epochs}")
    print(f"[TRAIN] imgsz   = {args.imgsz}")
    print(f"[TRAIN] batch   = {batch}")
    print(f"[TRAIN] project = {args.project}")
    print(f"[TRAIN] name    = {args.name}")

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=not args.no_cache,
        exist_ok=True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.resume:
        train_kwargs["resume"] = True

    train_results = model.train(**train_kwargs)

    run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("\n[DONE] Обучение завершено.")
    if best_pt.exists():
        print(f"[DONE] Лучшие веса: {best_pt}")
    if last_pt.exists():
        print(f"[DONE] Последние веса: {last_pt}")

    if args.validate:
        print("\n[VAL] Запускаю проверку модели...")
        val_model = YOLO(str(best_pt)) if best_pt.exists() else model
        val_model.val(data=str(data_yaml), imgsz=args.imgsz, batch=batch)

    if args.export_onnx:
        if not best_pt.exists():
            print("[EXPORT] best.pt не найден, экспорт пропущен.")
        else:
            print("\n[EXPORT] Экспортирую best.pt в ONNX...")
            export_model = YOLO(str(best_pt))
            export_model.export(format="onnx", imgsz=args.imgsz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
