import cv2


YUNET_PATH = "face_detection_yunet_2023mar.onnx"
AGE_PROTO = "age_googlenet.prototxt"
AGE_MODEL = "age_googlenet.caffemodel"
GENDER_PROTO = "gender_googlenet.prototxt"
GENDER_MODEL = "gender_googlenet.caffemodel"
FERPLUS_PATH = "emotion-ferplus-8.onnx"

print(f"OpenCV version: {cv2.__version__}")

try:
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320))
    print("✓ YuNet loaded")
except Exception as e:
    print(f"✗ YuNet failed: {e}")

try:
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    print("✓ Age model loaded")
except Exception as e:
    print(f"✗ Age model failed: {e}")

try:
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("✓ Gender model loaded")
except Exception as e:
    print(f"✗ Gender model failed: {e}")

try:
    emotion_net = cv2.dnn.readNetFromONNX(FERPLUS_PATH)
    print("✓ FER+ loaded")
except Exception as e:
    print(f"✗ FER+ failed: {e}")

print("\nDone. Fix any ✗ before proceeding.")