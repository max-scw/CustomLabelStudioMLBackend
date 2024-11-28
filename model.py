from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from PIL import Image
from pathlib import Path

from typing import Union, Dict, List, Optional, Any

from utils import get_env_variable, setup_logging
from onnx_model import ONNXModel


logger = setup_logging(__name__)


class NewModel(LabelStudioMLBase):
    model: Union[ONNXModel, Dict[int, ONNXModel]]

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("MODEL_VERSION", "0.0.1")

        model_name = get_env_variable("MODEL_NAME", "model.onnx")
        model_dir = Path(get_env_variable("MODEL_DIR", "/models"))
        precision = get_env_variable("MODEL_PRECISION", "fp32")
        input_shape = get_env_variable("MODEL_INPUT_SHAPE", (640, 640))
        th_score = get_env_variable("MODEL_TH_SCORE", 0.5)

        logger.debug(f"model_dir={model_dir}, model_name={model_name}, precision={precision}, input_shape={input_shape}")

        # setup ONNX session
        if isinstance(model_name, str):
            self.model = ONNXModel(
                model_path=model_dir / model_name,
                precision=precision,
                input_shape=input_shape,
                th_score=th_score
            )

        elif isinstance(model_name, dict):

            self.model = dict()
            for ky, vl in model_name.items():
                _model_path = model_dir / vl
                _precision = self._get_variable(precision, ky, "model precision")
                _input_shape = self._get_variable(input_shape, ky, "input shape")
                _th_score = self._get_variable(th_score, ky, "threshold score")

                logger.debug(f"Initial ONNX session for model_path={_model_path}, precision={_precision}, input_shape={_input_shape} with key={ky} (type {type(ky)})")
                self.model[ky] = ONNXModel(
                    model_path=_model_path,
                    precision=_precision,
                    input_shape=_input_shape,
                    th_score=_th_score
                )
        else:
            raise TypeError(f"Expecting path to model file or dictionary to multiple model files as input but MODEL_NAME was {model_name} (type={type(model_name)})")

        logger.info(f"Session(s) {self.model} created.")

    @staticmethod
    def _get_variable(
            variable: Union[Any, Dict[int, Any]],
            project_id: int,
            description: str = None,
    ):
        logger.debug(f"get_variable(variable={variable}, project_id={project_id}, description={description})")
        # cast to integer
        project_id = int(project_id)
        # get variable
        if isinstance(variable, dict):
            if project_id in variable:
                return variable[project_id]
            else:
                raise Exception(f"No {description} for project {project_id} (type: {type(project_id)}).")
        elif isinstance(variable, (str, int, float, tuple, ONNXModel)):
             return variable
        else:
            raise Exception(f"No {description} for project {project_id}. Input was {variable} (type {type(variable)})")


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

        # get model
        model = self._get_variable(self.model, self.project_id, "Model")

        predictions = []
        for task in tasks:
            # Resource downloading from Label Studio instance requires the
            # env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY being set
            path = self.get_local_path(task["data"]["image"], task_id=task["id"])
            logger.debug(f"self.get_local_path({task['data']['image']}, task_id={task['id']}) -> {path}")

            # image = load_image(path)
            image = Image.open(path).convert('RGB')
            logger.debug(f"Image: {image}")

            # Predict bounding boxes using the ONNX model
            results = model.predict(image)
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

