import cv2
import dlib
import winsound
from scipy.spatial import distance
from imutils import face_utils

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[42:48]
        rightEye = shape[36:42]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            count += 1
            if count >= FRAME_THRESHOLD:
                cv2.putText(frame,
                            "DROWSINESS DETECTED",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)
                # Play a beep sound: frequency=1000Hz, duration=500ms
                winsound.Beep(1000, 500)
        else:
            count = 0

    cv2.imshow("Driver Drowsiness System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()