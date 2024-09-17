# Custom Label Studio ML Backend

The open-source labeling software [Label Studio](https://labelstud.io/) offers to include a machine-learning backend in order to support the labeling process. This project provides an example how to customize such a backend for bounding box prediction based to ONNX converted models. The example is designed for YOLOv7.

The official repository from HumanSignal / Label Studio can be accessed under [HumanSignal/label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend).

## Quick Start

Use docker or the docker compose plugin with [docker-compose.yml](docker-compose.yml).
The container uses [gunicorn](https://gunicorn.org/), Python web server for unix, to spinn up a [Flask](https://flask.palletsprojects.com/en/3.0.x/) server that provides a REST api. Label Studio in turn queries this api to get predictions for tasks.

Docker container are released on [DockerHub/maxscw/labelstudio-ml-backend](https://hub.docker.com/r/maxscw/labelstudio-ml-backend).

## Project Structure

The customized code can be found in [model.py](model.py) (and the corresponding files: [onnx_model.py](onnx_model.py), [utils](utils)).
The file [_wsgi.py](_wsgi.py) provides the configured REST api.

````
MinimalImageInferenceService
+-- utils  # helper functions
    |-- __init__.py
    |-- bbox.py
    |-- env_vars.py
    |-- general.py
|-- _wsgi.py  # Falsk-based REST api
|-- docker-compose.yml  # example to spin up a Label Studio instance + ml-backend
|-- Dockerfile
|-- LICENSE
|-- model.py  # This is where the customized code goes
|-- onnx_model.py
|-- README.md
|-- requirements.txt  # custom requirements
|-- requirements-base.txt  # default requirements for the base Flask server
|-- requirements-test.txt  # requirements for test_api.py
|-- test_api.py  # if you want to test the code (see the original fork for this)
````

## Connect Backend to Label Studio

Go to a project on your Label Studio instance. Go to *Settings* > *Model*.

![LabelStudioMLBackend_Settings_Model_new.jpg](docs%2FLabelStudioMLBackend_Settings_Model_new.jpg)

Click on *Connect Model* and type in the url to your backend. If you are using the [docker-compose.yml](docker-compose.yml) file, the address is http://ml-backend:9090

![LabelStudioMLBackend_Model_Connect_Model.jpg](docs%2FLabelStudioMLBackend_Model_Connect_Model.jpg)

Click on *Validate and Save*. You'll see a green light and a new page if this was successful.

![LabelStudioMLBackend_Model_Connected.jpg](docs%2FLabelStudioMLBackend_Model_Connected.jpg)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author

 - max-scw
