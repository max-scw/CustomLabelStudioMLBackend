from label_studio_ml.model import LabelStudioMLBase
from onnx_model import ONNXModel
from PIL import Image
import base64
import io

import requests


class ObjectDetectionModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # Initialize the ONNX model
        super(ObjectDetectionModel, self).__init__(**kwargs)

        self.model = ONNXModel(  # TODO: make configurable via env variables
            "20240905_CRUplus_crop_YOLOv7tiny.onnx",
            precision="fp32",
            input_shape=(544, 640),
        )

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            # Load the image from the task's input
            image_data = task['data']['image']  # Image URL or base64-encoded image
            image = self._load_image(image_data)

            # Predict bounding boxes using the ONNX model
            results = self.model.predict(image)

            # Convert model predictions to Label Studio format
            prediction = {
                "result": [{
                    "from_name": "label",  # Object detection label name in Label Studio
                    "to_name": "image",    # Input field in Label Studio
                    "type": "rectanglelabels",
                    "value": {
                        "x": bbox[0] * 100,  # Normalized coordinates (percentage of image size)
                        "y": bbox[1] * 100,
                        "width": (bbox[2] - bbox[0]) * 100,
                        "height": (bbox[3] - bbox[1]) * 100
                    },
                    "score": score,  # Model confidence score
                    "label": [label]  # Detected object class
                } for bbox, score, label in results]
            }

            predictions.append(prediction)

        return predictions

    @staticmethod
    def _load_image(image_data):
        # Check if image is in base64 or URL
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            # For URL: handle downloading the image
            response = requests.get(image_data)
            image = Image.open(io.BytesIO(response.content))
        return image
