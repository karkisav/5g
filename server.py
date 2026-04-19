"""
server.py — Flask web server for the Face Recognition Dashboard
Run: python server.py
Visit: http://localhost:5050
"""

import cv2
import numpy as np
import json
import os
import threading
import time
from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import insightface

# ── InsightFace model ──────────────────────────────────────────────────────────
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "embeddings.npy"
NAMES_FILE      = "names.json"
ENROLL_SAMPLES  = 10
ENROLL_DELAY_S  = 0.8
THRESHOLD       = 0.45

# ── Shared state ───────────────────────────────────────────────────────────────
db_lock = threading.Lock()
known_embeddings: dict = {}

cam_lock = threading.Lock()
_cap = None

state = {
    "stream_active":   False,
    "enroll_active":   False,
    "enroll_name":     "",
    "enroll_total":    ENROLL_SAMPLES,
    "enroll_collected": [],
    "enroll_status":   "",
    "enroll_done":     False,
    "frame_jpg":       None,
    "faces":           [],
}


# ── Database ───────────────────────────────────────────────────────────────────

def db_load():
    global known_embeddings
    if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE)):
        known_embeddings = {}
        return
    vectors = np.load(EMBEDDINGS_FILE, allow_pickle=False)
    with open(NAMES_FILE) as f:
        names = json.load(f)
    if len(names) != len(vectors):
        known_embeddings = {}
        return
    with db_lock:
        known_embeddings = dict(zip(names, vectors))
    print(f"[db] Loaded {len(known_embeddings)} person(s)")


def db_save():
    with db_lock:
        if not known_embeddings:
            return
        names   = list(known_embeddings.keys())
        vectors = np.array(list(known_embeddings.values()))
    np.save(EMBEDDINGS_FILE, vectors)
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f)


def identify(embedding):
    qn = np.linalg.norm(embedding)
    if qn == 0:
        return "Unknown", -1.0
    q = embedding / qn
    best_name, best_score = "Unknown", -1.0
    with db_lock:
        items = list(known_embeddings.items())
    for name, stored in items:
        sn = np.linalg.norm(stored)
        if sn == 0:
            continue
        score = float(np.dot(q, stored / sn))
        if score > best_score:
            best_name, best_score = name, score
    return (best_name, best_score) if best_score >= THRESHOLD else ("Unknown", best_score)


# ── Camera thread ──────────────────────────────────────────────────────────────

def _open_camera():
    global _cap
    with cam_lock:
        if _cap is None or not _cap.isOpened():
            _cap = cv2.VideoCapture(0)
        return _cap is not None and _cap.isOpened()


def camera_loop():
    last_enroll_t = 0.0
    while True:
        if not state["stream_active"] and not state["enroll_active"]:
            time.sleep(0.05)
            continue
        if not _open_camera():
            time.sleep(0.5)
            continue

        with cam_lock:
            ret, frame = (_cap.read() if _cap else (False, None))
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        faces = face_app.get(frame)

        # Enrollment mode
        if state["enroll_active"]:
            now       = time.time()
            collected = state["enroll_collected"]
            total     = state["enroll_total"]

            if len(faces) == 1 and (now - last_enroll_t) >= ENROLL_DELAY_S:
                collected.append(faces[0].embedding)
                last_enroll_t = now
                state["enroll_status"] = f"Captured {len(collected)}/{total}"
                if len(collected) >= total:
                    name = state["enroll_name"]
                    with db_lock:
                        known_embeddings[name] = np.mean(collected, axis=0)
                    db_save()
                    state["enroll_status"] = f"✓ '{name}' enrolled successfully!"
                    state["enroll_active"] = False
                    state["enroll_done"]   = True
            elif len(faces) > 1:
                state["enroll_status"] = "Multiple faces — one person only"
            else:
                state["enroll_status"] = f"No face detected ({len(collected)}/{total})"

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 150), 2)
            pct   = int(len(collected) / total * frame.shape[1]) if total else 0
            cv2.rectangle(frame, (0, frame.shape[0]-6), (pct, frame.shape[0]), (0, 230, 150), -1)
            cv2.putText(frame, state["enroll_status"], (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 150), 2)

        # Recognition mode
        else:
            current = []
            for face in faces:
                name, score = identify(face.embedding)
                x1, y1, x2, y2 = face.bbox.astype(int)
                color = (0, 230, 120) if name != "Unknown" else (40, 40, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name}  {score:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw+8, y1), color, -1)
                cv2.putText(frame, label, (x1+4, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                current.append({"name": name, "score": round(score, 3),
                                 "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            state["faces"] = current

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        state["frame_jpg"] = buf.tobytes()
        time.sleep(0.030)


threading.Thread(target=camera_loop, daemon=True).start()


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
db_load()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpg = state.get("frame_jpg")
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.030)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    def gen():
        while True:
            payload = {
                "faces":           state["faces"],
                "enroll_active":   state["enroll_active"],
                "enroll_status":   state["enroll_status"],
                "enroll_progress": len(state["enroll_collected"]),
                "enroll_total":    state["enroll_total"],
                "enroll_done":     state["enroll_done"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.25)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stream/start",  methods=["POST"])
def stream_start():
    state["stream_active"] = True
    return jsonify(ok=True)


@app.route("/api/stream/stop",   methods=["POST"])
def stream_stop():
    state["stream_active"] = False
    state["faces"] = []
    return jsonify(ok=True)


@app.route("/api/enroll/start",  methods=["POST"])
def enroll_start():
    data    = request.get_json(force=True) or {}
    name    = str(data.get("name", "")).strip()
    samples = int(data.get("samples", ENROLL_SAMPLES))
    if not name:
        return jsonify(ok=False, error="Name is required"), 400
    state.update({
        "enroll_active":    True,
        "enroll_name":      name,
        "enroll_total":     max(3, min(40, samples)),
        "enroll_collected": [],
        "enroll_status":    f"Starting — look at the camera…",
        "enroll_done":      False,
        "stream_active":    True,
    })
    return jsonify(ok=True)


@app.route("/api/enroll/cancel", methods=["POST"])
def enroll_cancel():
    state["enroll_active"]    = False
    state["enroll_collected"] = []
    state["enroll_status"]    = "Cancelled"
    return jsonify(ok=True)


@app.route("/api/people")
def people_list():
    with db_lock:
        names = list(known_embeddings.keys())
    return jsonify(people=names)


@app.route("/api/people/<path:name>", methods=["DELETE"])
def people_delete(name):
    with db_lock:
        if name not in known_embeddings:
            return jsonify(ok=False, error="Not found"), 404
        del known_embeddings[name]
    db_save()
    return jsonify(ok=True)


if __name__ == "__main__":
    print("\n  FaceOS server → http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)