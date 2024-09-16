from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from PIL import Image

from utils import get_env_variable, setup_logging
from onnx_model import ONNXModel


logger = setup_logging(__name__)


class NewModel(LabelStudioMLBase):
    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")

        model_path = get_env_variable("MODEL_PATH", "model.onnx")
        precision = get_env_variable("MODEL_PRECISION", "fp32")
        input_shape = get_env_variable("MODEL_INPUT_SHAPE", (640, 640))
        logger.debug(f"model_path={model_path}, precision={precision}, input_shape={input_shape}")

        # setup ONNX session
        self.model = ONNXModel(
            model_path=model_path,
            precision=precision,
            input_shape=input_shape,
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.debug(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')


        predictions = []
        for task in tasks:
            # Load the image from the task's input
            # image_data = task['data']['image']  # Image URL or base64-encoded image

            # Resource downloading from Label Studio instance requires the
            # env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY being set
            path = self.get_local_path(task["data"]["image"], task_id=task["id"])
            logger.debug(f"self.get_local_path({task['data']['image']}, task_id={task['id']}) -> {path}")

            # image = load_image(path)
            image = Image.open(path).convert('RGB')
            logger.debug(f"Image: {image}")

            # Predict bounding boxes using the ONNX model
            results = self.model.predict(image)
            logger.debug(f"Model results: {results}")

            # Convert model predictions to Label Studio format
            prediction = {
                "result": [{
                    "from_name": "label",  # Object detection label name in Label Studio
                    "to_name": "image",    # Input field in Label Studio
                    "type": "rectanglelabels",
                    "value": {
                        "x": float(bbox[0] * 100),  # Normalized coordinates (percentage of image size)
                        "y": float(bbox[1] * 100),
                        "width": float((bbox[2] - bbox[0]) * 100),
                        "height": float((bbox[3] - bbox[1]) * 100),
                        # "rotation": 0,
                        "rectanglelabels": [self.parsed_label_config["label"]["labels"][class_id]]
                    },
                    "score": float(score),  # Model confidence score
                    # "label": [class_id],  # Detected object class
                } for class_id, score, bbox in results]
            }

            predictions.append(prediction)

        logger.debug(f"Response: predictions = {predictions}")
        return ModelResponse(predictions=predictions)
    
    # def fit(self, event, data, **kwargs):
    #     """
    #     This method is called each time an annotation is created or updated
    #     You can run your logic here to update the model and persist it to the cache
    #     It is not recommended to perform long-running operations here, as it will block the main thread
    #     Instead, consider running a separate process or a thread (like RQ worker) to perform the training
    #     :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
    #     :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
    #     """
    #
    #     # use cache to retrieve the data from the previous fit() runs
    #     old_data = self.get('my_data')
    #     old_model_version = self.get('model_version')
    #     print(f'Old data: {old_data}')
    #     print(f'Old model version: {old_model_version}')
    #
    #     # store new data to the cache
    #     self.set('my_data', 'my_new_data_value')
    #     self.set('model_version', 'my_new_model_version')
    #     print(f'New data: {self.get("my_data")}')
    #     print(f'New model version: {self.get("model_version")}')
    #
    #     print('fit() completed successfully.')

