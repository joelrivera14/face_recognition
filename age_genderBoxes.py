import cv2
import numpy as np

YUNET_PATH = "face_detection_yunet_2023mar.onnx"
AGE_PROTO = "age_googlenet.prototxt"
AGE_MODEL = "age_googlenet.caffemodel"
GENDER_PROTO = "gender_googlenet.prototxt"
GENDER_MODEL = "gender_googlenet.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (width, height))
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

while True:
    success, frame = cap.read()
    if not success:
        break

    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            # Clamp to frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()