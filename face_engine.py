import cv2
import os


# =====================================================
# FACE DETECTOR
# =====================================================

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_detector = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)


# =====================================================
# FIND FACE
# =====================================================

def get_face(image):

    if image is None:
        return None

    # Check whether detector loaded correctly
    if face_detector.empty():
        print("ERROR: Face detector could not be loaded.")
        print("Cascade path:", CASCADE_PATH)
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Use the largest detected face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = gray[
        y:y + h,
        x:x + w
    ]

    return face


# =====================================================
# TRAIN MODEL
# =====================================================

def train_model():

    faces_folder = "faces"

    if not os.path.exists(faces_folder):
        print("ERROR: faces folder does not exist.")
        return None, {}

    training_faces = []
    training_labels = []

    label_to_student = {}

    label = 0

    for filename in os.listdir(faces_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        filepath = os.path.join(
            faces_folder,
            filename
        )

        image = cv2.imread(filepath)

        if image is None:
            print("Could not read:", filepath)
            continue

        face = get_face(image)

        if face is None:
            print(
                "No face found in:",
                filename
            )
            continue

        training_faces.append(face)

        training_labels.append(label)

        # 001.jpg -> 001
        student_id = os.path.splitext(
            filename
        )[0]

        label_to_student[label] = student_id

        label += 1


    if len(training_faces) == 0:

        print(
            "ERROR: No usable face images found."
        )

        return None, {}


    # Check LBPH availability
    if not hasattr(cv2, "face"):

        print(
            "ERROR: cv2.face is not available."
        )

        return None, {}


    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        training_faces,
        __import__("numpy").array(
            training_labels,
            dtype="int32"
        )
    )

    return recognizer, label_to_student


# =====================================================
# RECOGNIZE STUDENT
# =====================================================

def recognize_student(image):

    recognizer, label_to_student = train_model()

    if recognizer is None:

        print("Could not train face recognizer.")

        return None


    face = get_face(image)

    if face is None:

        print("No face detected in camera image.")

        return None


    label, confidence = recognizer.predict(face)

    print(
        "Predicted label:",
        label
    )

    print(
        "Confidence/distance:",
        confidence
    )


    # LBPH uses distance:
    # smaller value = better match
    #
    # 100 is deliberately used as a
    # reasonable starting threshold.

    if confidence > 100:

        print("Face does not match.")

        return None


    student_id = label_to_student.get(label)

    return student_id