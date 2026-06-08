import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("utils/garbage_yolov8m_best.pt")

CLASS_INFO: Dict[str, Dict[str, str]] = {
    "BIODEGRADABLE": {
        "cn_name": "可降解垃圾 / 厨余垃圾",
        "garbage_type": "厨余垃圾",
        "suggestion": "建议投放至厨余垃圾桶。投放前尽量沥干水分，避免汤水外溢。"
    },
    "CARDBOARD": {
        "cn_name": "纸板 / 纸箱",
        "garbage_type": "可回收物",
        "suggestion": "建议保持干燥、压扁整理后投放至可回收物桶。"
    },
    "GLASS": {
        "cn_name": "玻璃",
        "garbage_type": "可回收物",
        "suggestion": "建议小心投放至可回收物桶。若为破碎玻璃，应包裹后再投放，避免划伤。"
    },
    "METAL": {
        "cn_name": "金属",
        "garbage_type": "可回收物",
        "suggestion": "建议清空内容物后投放至可回收物桶。易拉罐等可压扁后投放。"
    },
    "PAPER": {
        "cn_name": "纸张",
        "garbage_type": "可回收物",
        "suggestion": "建议保持干燥，避免油污污染后投放至可回收物桶。"
    },
    "PLASTIC": {
        "cn_name": "塑料",
        "garbage_type": "可回收物",
        "suggestion": "建议清空内容物后投放至可回收物桶。塑料瓶可压扁后投放。"
    },
}

_model = None


def load_model() -> YOLO:
    """加载YOLO模型。"""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"找不到模型文件：{MODEL_PATH}\n"
                f"请把 best.pt 放到 utils 文件夹，并改名为 garbage_yolov8n_best.pt"
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


def predict_garbage(
    image: Image.Image,
    conf_threshold: float = 0.25
) -> Tuple[Image.Image, List[Dict[str, object]]]:
    """对上传图片进行YOLO检测。"""
    model = load_model()

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_rgb)

    detections: List[Dict[str, object]] = []

    if result.boxes is None or len(result.boxes) == 0:
        return annotated_image, detections

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()

        class_name = names.get(cls_id, str(cls_id))
        class_name_upper = class_name.upper()

        info = CLASS_INFO.get(
            class_name_upper,
            {
                "cn_name": class_name_upper,
                "garbage_type": "未知类别",
                "suggestion": "建议结合本地垃圾分类规则进一步判断。"
            }
        )

        detections.append(
            {
                "class_name": class_name_upper,
                "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in xyxy],
                "cn_name": info["cn_name"],
                "garbage_type": info["garbage_type"],
                "suggestion": info["suggestion"],
            }
        )

    return annotated_image, detections


def summarize_detections(detections: List[Dict[str, object]]) -> str:
    """把检测结果整理成文本，方便 main.py 展示或传给 RAG。"""
    if not detections:
        return "未检测到明显垃圾目标。"

    lines = []
    for i, det in enumerate(detections, start=1):
        lines.append(
            f"{i}. 检测类别：{det['class_name']}；"
            f"中文名称：{det['cn_name']}；"
            f"垃圾大类：{det['garbage_type']}；"
            f"置信度：{det['confidence']}；"
            f"投放建议：{det['suggestion']}"
        )

    return "\n".join(lines)
