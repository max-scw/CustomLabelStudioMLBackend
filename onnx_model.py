import onnxruntime as ort
import numpy as np

from PIL import Image, ImageOps

from typing import Tuple, Literal

from utils import xyxy2xywh, xywh2xyxy


def letterbox(
        img: Image.Image,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scale_up: bool = True,
        stride: int = 32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.size  # current shape [width, height] in PIL
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = (new_shape[0] / shape[0], new_shape[1] / shape[1]) # height, width ratio
    if not scale_up:  # only scale down, do not scale up
        ratio = (min(el, 1.0) for el in ratio)

    # Compute padding
    new_unpad = (int(round(shape[1] * ratio[1])), int(round(shape[0] * ratio[0])))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape  # overwrite with new_shape
        ratio = (new_shape[0] / shape[0], new_shape[1] / shape[1])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = img.resize(new_unpad, Image.Resampling.BILINEAR)

    # Add border (pad the image)
    padding = (int(round(dw - 0.1)), int(round(dh - 0.1)), int(round(dw + 0.1)), int(round(dh + 0.1)))
    img = ImageOps.expand(img, border=padding, fill=color)  # use ImageOps.expand for padding

    return img, ratio, (dw, dh)


def precision_to_type(precision: Literal["fp64", "fp32", "fp16", "int8"]) -> type:
    if precision.lower() == "fp64":
        return np.float64
    elif precision.lower() == "fp32":
        return np.float32
    elif precision.lower() == "fp16":
        return np.float16
    elif precision.lower() == "int8":
        return np.int8
    else:
        raise ValueError(f"Unknown precision: {precision}")


class ONNXModel:
    def __init__(
            self,
            model_path,
            precision: Literal["fp64", "fp32", "fp16", "int8"] = "fp32",
            input_shape: Tuple[int, int] = (640, 640),
            bbox_format: Literal["xywh", "xyxy"] = "xyxy",
            th_score: float = 0.5
    ):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.precision = precision_to_type(precision)
        self.input_shape = input_shape
        self.bbox_format = bbox_format
        self.th_score = th_score

    def __repr__(self) -> str:
        return f"ONNXModel({self.model_path}, precision={self.precision}, input_shape={self.input_shape}, bbox_format={self.bbox_format})"

    def predict(self, image):
        # Preprocess the image (resize, normalize, etc.)
        input_tensor = self.preprocess_image(image)

        # Make the prediction
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess to get bounding boxes, scores, etc.
        results = self.postprocess(outputs)
        return results

    def preprocess_image(self, image):
        # Example preprocessing: assuming image is PIL Image
        # Resize, convert to numpy array, normalize, etc.

        # resize
        image = letterbox(image, new_shape=self.input_shape, stride=32)[0]
        # convert image to floats
        image = np.array(image).astype(np.float32)
        # normalize image
        image /= 255.0
        # to precision
        image = image.astype(self.precision)
        image = np.transpose(image, (2, 0, 1))  # Change to CxHxW
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        return image

    def postprocess(self, outputs):
        # Convert ONNX output into Label Studio's object detection format
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0]

        if output.shape[1] == 7:
            # YOLOv7
            idx_bbx = 1
            idx_cls = 5
            idx_scr = 6
        elif output.shape[1] == 6:
            # YOLOv10
            idx_bbx = 0
            idx_cls = 5
            idx_scr = 4
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        scores = output[:, idx_scr]
        lg = scores > self.th_score
        scores = scores[lg]

        bboxs, class_ids = output[lg, idx_bbx:idx_bbx + 4], output[lg, idx_cls].astype(int)

        # make relative
        if (bboxs > 1).any():
            bboxs = (bboxs / (self.input_shape[::-1] * 2))

        # convert to corner-point-format
        if self.bbox_format == "xywh":
            bboxs = xywh2xyxy(bboxs)

        predictions = [(cls, scr, bbx.tolist()) for cls, scr, bbx in zip(class_ids, scores, bboxs)]
        return predictions


if __name__ == "__main__":
    mdl = ONNXModel(
        model_path="./models/20241026_CRU1_2_YOLOv7tiny.onnx",
        # model_path="./models/20241107_CRU1_2_YOLOv10n.onnx",
        precision="fp32",
        input_shape=(640, 640),
    )

    image = Image.open("../BaslerCameraAdapter/test_images/20240813_120110.jpg")

    out = mdl.predict(image)
