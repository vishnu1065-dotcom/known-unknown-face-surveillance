import os
import cv2
import numpy as np
import face_recognition

# ============================================================
# CONFIG
# ============================================================

CAMERA = "/dev/video0"

YOLO_CFG = "models/yolov4-tiny.cfg"
YOLO_WEIGHTS = "models/yolov4-tiny.weights"

KNOWN_DIR = "known_faces/vishnu"

PERSON_CONFIDENCE = 0.40
NMS_THRESHOLD = 0.40

FACE_TOLERANCE = 0.50

# Tracking settings
MAX_DISAPPEARED = 20
MAX_DISTANCE = 100

# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO...")

net = cv2.dnn.readNetFromDarknet(
    YOLO_CFG,
    YOLO_WEIGHTS
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers().flatten()
]

print("YOLO loaded successfully")

# ============================================================
# LOAD KNOWN FACES
# ============================================================

known_encodings = []
known_names = []

print("Loading known faces...")

if not os.path.exists(KNOWN_DIR):
    print("ERROR: Folder not found:", KNOWN_DIR)
    exit()

for filename in os.listdir(KNOWN_DIR):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    path = os.path.join(KNOWN_DIR, filename)

    try:

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)

        if encodings:

            known_encodings.append(encodings[0])
            known_names.append("VISHNU")

            print("Loaded:", filename)

        else:

            print("No face:", filename)

    except Exception as e:

        print("Error:", filename, e)

print("Known face samples:", len(known_encodings))

if not known_encodings:
    print("ERROR: No known faces loaded")
    exit()

# ============================================================
# TRACKER DATA
# ============================================================

next_track_id = 1

tracks = {}

# Each track:
#
# {
#   "centroid": (x,y),
#   "box": (x,y,w,h),
#   "name": "UNKNOWN",
#   "missed": 0
# }

# ============================================================
# FACE RECOGNITION FUNCTION
# ============================================================

def recognize_person(person_crop):

    if person_crop is None or person_crop.size == 0:
        return "UNKNOWN"

    try:

        rgb = cv2.cvtColor(
            person_crop,
            cv2.COLOR_BGR2RGB
        )

        locations = face_recognition.face_locations(
            rgb,
            model="hog"
        )

        if not locations:
            return "UNKNOWN"

        encodings = face_recognition.face_encodings(
            rgb,
            locations
        )

        for encoding in encodings:

            distances = face_recognition.face_distance(
                known_encodings,
                encoding
            )

            if len(distances) == 0:
                continue

            best_index = np.argmin(distances)

            if distances[best_index] < FACE_TOLERANCE:

                return known_names[best_index]

    except Exception as e:

        print("Face recognition error:", e)

    return "UNKNOWN"


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(CAMERA)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print("ERROR: Camera could not be opened")
    exit()

print("Camera started")
print("Tracking started")
print("Press Q to quit")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame failed")
        break

    height, width = frame.shape[:2]

    # ========================================================
    # YOLO
    # ========================================================

    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = int(np.argmax(scores))

            confidence = float(scores[class_id])

            # Person class
            if class_id == 0 and confidence > PERSON_CONFIDENCE:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                box_w = int(detection[2] * width)
                box_h = int(detection[3] * height)

                x = int(center_x - box_w / 2)
                y = int(center_y - box_h / 2)

                boxes.append(
                    [x, y, box_w, box_h]
                )

                confidences.append(confidence)

    # ========================================================
    # NMS
    # ========================================================

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        PERSON_CONFIDENCE,
        NMS_THRESHOLD
    )

    detections = []

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append(
                {
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "centroid": (cx, cy)
                }
            )

    # ========================================================
    # MARK ALL TRACKS AS MISSED
    # ========================================================

    for track_id in tracks:

        tracks[track_id]["missed"] += 1

    # ========================================================
    # MATCH DETECTIONS TO EXISTING TRACKS
    # ========================================================

    used_tracks = set()

    for detection in detections:

        cx, cy = detection["centroid"]

        best_track = None
        best_distance = MAX_DISTANCE

        for track_id, track in tracks.items():

            if track_id in used_tracks:
                continue

            old_cx, old_cy = track["centroid"]

            distance = np.sqrt(
                (cx - old_cx) ** 2 +
                (cy - old_cy) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_track = track_id

        # ====================================================
        # EXISTING PERSON
        # ====================================================

        if best_track is not None:

            track_id = best_track

            used_tracks.add(track_id)

            tracks[track_id]["centroid"] = (
                cx,
                cy
            )

            tracks[track_id]["box"] = detection["box"]

            tracks[track_id]["missed"] = 0

        # ====================================================
        # NEW PERSON
        # ====================================================

        else:

            track_id = next_track_id

            next_track_id += 1

            tracks[track_id] = {

                "centroid": (cx, cy),

                "box": detection["box"],

                "name": "UNKNOWN",

                "missed": 0
            }

            used_tracks.add(track_id)

    # ========================================================
    # REMOVE OLD TRACKS
    # ========================================================

    remove_ids = []

    for track_id, track in tracks.items():

        if track["missed"] > MAX_DISAPPEARED:

            remove_ids.append(track_id)

    for track_id in remove_ids:

        del tracks[track_id]

    # ========================================================
    # FACE RECOGNITION + DRAWING
    # ========================================================

    known_count = 0
    unknown_count = 0

    for track_id, track in tracks.items():

        # Don't draw people that disappeared
        if track["missed"] > 0:
            continue

        x, y, w, h = track["box"]

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        person_crop = frame[y1:y2, x1:x2]

        # ----------------------------------------------------
        # IMPORTANT:
        # Try recognition if currently UNKNOWN.
        #
        # This means a new person can become VISHNU when
        # the face becomes visible.
        # ----------------------------------------------------

        if track["name"] == "UNKNOWN":

            identity = recognize_person(person_crop)

            if identity != "UNKNOWN":

                track["name"] = identity

        # ----------------------------------------------------
        # KNOWN
        # ----------------------------------------------------

        if track["name"] == "VISHNU":

            color = (0, 255, 0)

            label = "VISHNU"

            known_count += 1

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        else:

            color = (0, 0, 255)

            label = "UNKNOWN"

            unknown_count += 1

        # ----------------------------------------------------
        # DRAW BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        text = f"{label} | ID {track_id}"

        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ========================================================
    # PEOPLE COUNT
    # ========================================================

    total_people = known_count + unknown_count

    # ========================================================
    # INFORMATION
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (430, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Total People: {total_people}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Known: {known_count}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Unknown: {unknown_count}",
        (190, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Surveillance Rover",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Camera stopped")
print("Tracking stopped")
