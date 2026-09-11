# Driver Drowsiness Monitoring System
## Setup Guide & Demo Notes

---

## Part 1: Software Setup (One-Time Installation)

### Step 1: Install Python 3.10
1. Download Python 3.10.x (64-bit) from https://www.python.org/downloads/windows/
2. Run the installer.
3. **Check "Add python.exe to PATH"** on the first screen — this is critical.
4. Click **Install Now**, wait, then **Close**.
5. Verify: open Command Prompt and run:
   ```
   python --version
   ```
   Expect: `Python 3.10.x`

### Step 2: Install CMake
1. Download from https://cmake.org/download/ → Windows x64 Installer.
2. Run the installer, select **"Add CMake to the system PATH"**, finish install.
3. Verify:
   ```
   cmake --version
   ```

### Step 3: Install Visual Studio Build Tools
1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. In the installer, check **"Desktop development with C++"**.
3. Click **Install** (takes 15–40 minutes, several GB).
4. **Restart the laptop** after install completes.

### Step 4: Install Required Python Libraries
Open Command Prompt and run each line one at a time:
```
python -m pip install --upgrade pip
pip install opencv-python
pip install dlib
pip install scipy
pip install imutils
```
> Note: `dlib` takes 5–15 minutes to compile — this is normal.

Verify all installed correctly:
```
python -c "import cv2, dlib, scipy, imutils; print('All libraries loaded successfully')"
```

### Step 5: Download the Facial Landmark Model
1. Download `shape_predictor_68_face_landmarks.dat.bz2` from:
   https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
2. Install **7-Zip** (https://www.7-zip.org/download.html) if not already installed.
3. Right-click the downloaded file → **7-Zip → Extract Here** to get the `.dat` file.
4. Create a project folder, e.g. `Desktop\drowsiness_project`.
5. Move `shape_predictor_68_face_landmarks.dat` into that folder.

### Step 6: Create the Script File
1. Inside `drowsiness_project`, create a new file named exactly `drowsiness_detection.py`
   (enable **File name extensions** in File Explorer's View tab if needed, so `.txt` doesn't get silently appended).
2. Paste in the final script (see **Part 2** below).
3. Save the file.

**Final folder contents should be exactly:**
```
drowsiness_project/
├── drowsiness_detection.py
└── shape_predictor_68_face_landmarks.dat
```

---

## Part 2: Final Script (with sound alert)

```python
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
                winsound.Beep(1000, 500)
        else:
            count = 0

    cv2.imshow("Driver Drowsiness System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Part 3: How to Run It

1. Open the `drowsiness_project` folder in File Explorer.
2. Click the address bar, type `cmd`, press Enter (opens Command Prompt in that folder).
3. Run:
   ```
   python drowsiness_detection.py
   ```
4. A window titled **"Driver Drowsiness System"** opens showing the live webcam feed.
5. Press **`q`** at any time to close the program.

---

## Part 4: Demo Guide (How to Show It Live)

Use this short script when demonstrating the project to an examiner, supervisor, or evaluator.

### Before the demo
- Sit in a well-lit room, facing the camera directly (poor lighting is a known limitation of this system).
- Make sure your face is 25–80 cm from the webcam — too close or too far reduces detection accuracy.
- Close any other apps that might be using the webcam.
- Have the Command Prompt window and script ready to launch with one command.

### Demo flow (2–3 minutes)

1. **Introduce the system (15 sec):**
   "This is a real-time driver drowsiness detection system built in Python. It uses a webcam to track eye movements and calculates something called the Eye Aspect Ratio, or EAR, to determine if the driver's eyes are closing for an unusually long time."

2. **Launch it (10 sec):**
   Run `python drowsiness_detection.py` and let the webcam window open.
   "You can see it's now tracking my face in real time using facial landmark detection — 68 points across the eyes, nose, and mouth."

3. **Show the normal/awake state (15 sec):**
   Look directly at the camera with eyes open.
   "Right now, since my eyes are open and the EAR value is above the threshold, no alert is triggered."

4. **Trigger the drowsy alert (20 sec):**
   Close your eyes and hold for 2–3 seconds.
   "Now I'll close my eyes to simulate drowsiness... and you can see the system displays 'DROWSINESS DETECTED' in red text, along with an audible beep alert — this is exactly how it would warn a driver in a real vehicle."

5. **Explain the significance (20 sec):**
   "This kind of early-warning system could be integrated into vehicle dashboards to reduce accidents caused by driver fatigue, which is a major contributor to road accidents globally."

6. **Close out (10 sec):**
   Press `q` to close the program.
   "That concludes the live demonstration. Happy to answer any questions about the detection algorithm or the system design."

### Anticipated questions & short answers
- **"How does it detect drowsiness?"** → It calculates the Eye Aspect Ratio (EAR) from 6 points around each eye; when EAR drops below a threshold (0.25) for a sustained number of frames (20), it's classified as drowsy.
- **"What happens in poor lighting?"** → Accuracy drops, since the facial landmark detector relies on visible contrast; this is a documented limitation of camera-based systems.
- **"Does it work with glasses?"** → Detection can be less reliable with glasses, especially with glare — also a known limitation of this type of system.
- **"How is this different from the YOLOv8 approach mentioned in the report?"** → YOLOv8 is a newer, deep-learning object-detection approach trained specifically to classify "awake" vs "drowsy" states directly from images, whereas this script uses classical facial-landmark geometry (EAR) — both are valid approaches to the same problem.

---

*Keep this guide alongside your project files for future reference or if setting up on another machine.*
