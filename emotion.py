import cv2
import numpy as np

YUNET_PATH = "face_detection_yunet_2023mar.onnx"
AGE_PROTO = "age_googlenet.prototxt"
AGE_MODEL = "age_googlenet.caffemodel"
GENDER_PROTO = "gender_googlenet.prototxt"
GENDER_MODEL = "gender_googlenet.caffemodel"
FERPLUS_PATH = "emotion-ferplus-8.onnx"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'contempt', 'fear']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (width, height))
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
emotion_net = cv2.dnn.readNetFromONNX(FERPLUS_PATH)

frame_count = 0
last_labels = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    _, faces = detector.detect(frame)

    if faces is not None:
        for i, face in enumerate(faces):
            x, y, w, h = face[:4].astype(int)
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if frame_count % 5 == 0:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
                gender_net.setInput(blob)
                gender = GENDER_LIST[gender_net.forward()[0].argmax()]

                age_net.setInput(blob)
                age = AGE_BUCKETS[age_net.forward()[0].argmax()]

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                emotion_blob = resized.astype('float32').reshape(1, 1, 64, 64)
                emotion_net.setInput(emotion_blob)
                emotion = EMOTIONS[emotion_net.forward()[0].argmax()]

                last_labels[i] = (gender, age, emotion)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if i in last_labels:
                gender, age, emotion = last_labels[i]
                for j, text in enumerate([gender, age, emotion]):
                    cv2.putText(frame, text, (x, y - 10 - (j * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()