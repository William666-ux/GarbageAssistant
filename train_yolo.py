from ultralytics import YOLO
from pathlib import Path


DATA_YAML = Path(
    r"D:\pycharm\PycharmProjects\Garbage assistant\GarbageImageDataset\dataset\data.yaml"
)

PROJECT_DIR = Path(
    r"D:\pycharm\PycharmProjects\Garbage assistant\utils\yolo_runs"
)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到 data.yaml：{DATA_YAML}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        project=str(PROJECT_DIR),
        name="garbage_yolov8n",
        patience=10,
        pretrained=True
    )

    print("训练完成！")
    print("模型保存位置：")
    print(PROJECT_DIR / "garbage_yolov8n" / "weights" / "best.pt")


if __name__ == "__main__":
    main()