# Face Recognition App

Live webcam app that detects faces and overlays age, gender, and emotion labels in real time. Built with OpenCV and Python 3.12

## Models Required

Download these files and place them in the project root before running:

| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| `age_googlenet.caffemodel` | [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel) |
| `age_googlenet.prototxt` | [GilLevi/AgeGenderDeepLearning](https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt) |
| `gender_googlenet.caffemodel` | [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel) |
| `gender_googlenet.prototxt` | [GilLevi/AgeGenderDeepLearning](https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt) |
| `emotion-ferplus-8.onnx` | [ONNX Model Zoo](https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy
```

## Run

```bash
python3 main.py
```

Press `q` to quit.
