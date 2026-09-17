import cv2
import os
import sqlite3
import numpy as np


# =====================================================
# FACE DETECTOR
# =====================================================

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACES_DIR = os.path.join(BASE_DIR, "faces")

face_detector = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

# All faces (both at training time and at scan time) are resized to this
# exact size before anything else happens. Previously, faces kept
# whatever size the cascade happened to crop, which is different for
# every photo (a phone-uploaded registration photo can be many times
# larger than a webcam frame). LBPH does not require identical sizes,
# but comparing wildly different sizes/qualities makes the distance
# scores unreliable, which was part of why one student in particular
# kept "winning" the match regardless of who was actually scanned.
STANDARD_FACE_SIZE = (200, 200)

# How much worse the SECOND best match must be, compared to the best
# match, before we trust the best match. This is the key fix: a plain
# "nearest neighbour wins" search (what the code did before) always
# returns SOME student, even when the real answer is "nobody matches
# well" - it has no concept of "I'm not sure". With very few training
# photos (often just one per student), that meant almost any face
# ended up being pulled toward whichever registered student's photo
# happened to sit closest to it overall, so scans kept coming back as
# the same one or two students no matter who stood in front of the
# camera. Requiring the top match to be clearly better than the
# runner-up (from a DIFFERENT student) stops that from happening.
MATCH_MARGIN_RATIO = 1.15

# LBPH "confidence" is really a distance: lower = more similar.
# This is intentionally on the stricter side since faces are now
# equalized/normalized. If real registrations start getting rejected
# too often, raise this a little (e.g. 85-95) - the console log below
# prints the actual distances seen for every scan so it can be tuned
# to your camera/lighting instead of guessing blindly.
MATCH_DISTANCE_THRESHOLD = 80


def get_registered_students():
    """
    Returns {student_id: name} for every student that actually exists
    in the database right now. Used to skip any leftover/orphaned
    image in faces/ that doesn't belong to a real registration.
    """

    db_path = os.path.join(BASE_DIR, "attendance.db")

    if not os.path.exists(db_path):
        return {}

    connection = sqlite3.connect(db_path)

    try:
        rows = connection.execute(
            "SELECT student_id, name FROM students"
        ).fetchall()

        return {row[0]: row[1] for row in rows}

    except sqlite3.Error:
        return {}

    finally:
        connection.close()


# =====================================================
# FIND + NORMALIZE FACE
# =====================================================

def get_face(image):
    """
    Detects the largest face in `image` and returns it as a
    standardized, lighting-normalized grayscale crop ready for
    training or matching. Returns None if no face is found.
    """

    if image is None:
        return None

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

    return normalize_face(face)


def normalize_face(face):
    """
    Puts every face (whether coming from a registration photo or a
    live webcam frame) through the exact same size + lighting
    normalization, so comparisons between them are meaningful instead
    of being skewed by one photo being bigger/brighter than another.
    """

    resized = cv2.resize(
        face,
        STANDARD_FACE_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    # Histogram equalization makes matching much more robust to
    # different lighting between the registration photo and the
    # webcam feed (a very common source of bad matches).
    equalized = cv2.equalizeHist(resized)

    return equalized


def augment_face(face):
    """
    A single registration photo only gives the recognizer one example
    of a student's face, which is a very weak, easily-confused amount
    of training data. This generates a handful of realistic variations
    (mirrored, slightly brighter/darker, slightly rotated) from that
    one photo so the model has more to compare against and is less
    likely to be thrown off by minor differences in the live scan.
    """

    variants = [face, cv2.flip(face, 1)]

    variants.append(cv2.convertScaleAbs(face, alpha=1.15, beta=10))
    variants.append(cv2.convertScaleAbs(face, alpha=0.85, beta=-10))

    h, w = face.shape
    for angle in (-8, 8):
        rotation_matrix = cv2.getRotationMatrix2D(
            (w / 2, h / 2), angle, 1.0
        )
        rotated = cv2.warpAffine(
            face,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        variants.append(rotated)

    return variants


# =====================================================
# TRAIN MODEL
# =====================================================

def train_model():

    if not os.path.exists(FACES_DIR):
        print("ERROR: faces folder does not exist.")
        return None, {}

    training_faces = []
    training_labels = []

    label_to_student = {}

    label = 0

    # Only train on faces that belong to a student who is actually
    # registered right now, and always process files in the same
    # (sorted) order so label numbers are assigned consistently on
    # every run.
    valid_student_ids = get_registered_students()

    for filename in sorted(os.listdir(FACES_DIR)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        # 001.jpg -> 001
        student_id = os.path.splitext(
            filename
        )[0]

        if student_id not in valid_student_ids:
            print(
                "Skipping orphaned face file "
                f"(no matching student record): {filename}"
            )
            continue

        filepath = os.path.join(
            FACES_DIR,
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

        # Expand the single registration photo into several variants
        # so this student is represented by more than one sample.
        for variant in augment_face(face):
            training_faces.append(variant)
            training_labels.append(label)

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
        np.array(
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


    # Collect the distance to every training sample (not just the
    # single closest one overall) so we can compare the best-matching
    # student against the next-best-matching student, instead of
    # blindly trusting whichever sample happened to be nearest.
    collector = cv2.face.StandardCollector_create()
    recognizer.predict_collect(face, collector)

    results = collector.getResults(sorted=True)

    if not results:
        print("No prediction produced.")
        return None

    best_distance_per_label = {}

    for candidate_label, distance in results:
        if (
            candidate_label not in best_distance_per_label
            or distance < best_distance_per_label[candidate_label]
        ):
            best_distance_per_label[candidate_label] = distance

    ranked = sorted(
        best_distance_per_label.items(),
        key=lambda item: item[1]
    )

    best_label, best_distance = ranked[0]
    second_best_distance = (
        ranked[1][1] if len(ranked) > 1 else float("inf")
    )

    best_student = label_to_student.get(best_label)
    second_best_student = (
        label_to_student.get(ranked[1][0]) if len(ranked) > 1 else None
    )

    print(
        f"Best match: {best_student} (distance {best_distance:.1f}) | "
        f"Runner-up: {second_best_student} "
        f"(distance {second_best_distance:.1f})"
    )

    # Reject if even the best match is too far away to trust.
    if best_distance > MATCH_DISTANCE_THRESHOLD:
        print("Face does not clearly match any registered student.")
        return None

    # Reject if the best match isn't meaningfully better than the
    # runner-up - this is what stops the scanner from confidently
    # (and wrongly) assigning attendance to one particular student
    # whenever the real match is ambiguous.
    if second_best_distance < best_distance * MATCH_MARGIN_RATIO:
        print(
            "Best match is too close to the runner-up to be "
            "confident - treating as not recognized."
        )
        return None

    return best_student
