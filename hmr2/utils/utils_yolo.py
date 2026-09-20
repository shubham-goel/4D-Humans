from typing import Optional
import torch

class YOLOPredictor:
    """YOLOv8-based human detector compatible with Detectron2's output interface for HMR2.

    Requires `ultralytics` package: pip install ultralytics
    """
    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None, conf: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLO detector requires the 'ultralytics' package. "
                "Please install it using: pip install ultralytics"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.model = YOLO(model_name)
        if device == "cuda":
            self.model.to("cuda")

    def __call__(self, img_cv2):
        """Run YOLO human detector on an image.

        Args:
            img_cv2 (np.ndarray): Image in BGR format (OpenCV).

        Returns:
            dict: Dictionary with 'instances' conforming to Detectron2 interface:
                - pred_boxes.tensor: (N, 4) tensor of bounding boxes [x1, y1, x2, y2]
                - pred_classes: (N,) tensor with class index (0 for person)
                - scores: (N,) tensor with detection confidence scores
        """
        results = self.model(img_cv2, classes=[0], conf=self.conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()
        scores = results[0].boxes.conf.cpu()
        pred_classes = torch.zeros(len(boxes), dtype=torch.long)

        class Boxes:
            def __init__(self, tensor):
                self.tensor = tensor

        class Instances:
            def __init__(self, pred_boxes, pred_classes, scores):
                self.pred_boxes = pred_boxes
                self.pred_classes = pred_classes
                self.scores = scores

        return {
            'instances': Instances(
                pred_boxes=Boxes(boxes),
                pred_classes=pred_classes,
                scores=scores,
            )
        }
