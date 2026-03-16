from ultralytics import YOLO


def main():
    mode = YOLO(r"yolov8m.pt")
    mode.train(
        data="Knife.yaml", workers=2, epochs=150, batch=4, copy_paste=0.5, mosaic=1.0, mixup=0.2, name="yolov8m_150_05"
    )


if __name__ == "__main__":
    main()
