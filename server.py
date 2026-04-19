import atexit
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

import detect_face


flask_app = Flask(__name__, static_folder=".", static_url_path="")
port = 5050


class FaceRuntime:
	def __init__(self, camera_index: int = 0):
		self.camera_index = camera_index
		self.cap = None
		self.lock = threading.Lock()
		self.stop_event = threading.Event()
		self.thread = None

		self.mode = "recognition"
		self.message = "Ready"
		self.latest_frame = None

		self.known_embeddings = detect_face.load_database()
		self.visible_names = []

		self.enroll_name = None
		self.enroll_target = detect_face.ENROLL_SAMPLES
		self.enroll_embeddings = []
		self.last_enroll_capture_at = 0.0

	def start(self) -> None:
		if self.thread and self.thread.is_alive():
			return
		self.thread = threading.Thread(target=self._camera_loop, daemon=True)
		self.thread.start()

	def shutdown(self) -> None:
		self.stop_event.set()
		if self.thread and self.thread.is_alive():
			self.thread.join(timeout=2)
		if self.cap is not None:
			self.cap.release()

	def _open_camera(self) -> bool:
		if self.cap is not None and self.cap.isOpened():
			return True
		self.cap = cv2.VideoCapture(self.camera_index)
		return self.cap.isOpened()

	def _camera_loop(self) -> None:
		while not self.stop_event.is_set():
			if not self._open_camera():
				with self.lock:
					self.message = "Camera unavailable"
				time.sleep(1.0)
				continue

			ret, frame = self.cap.read()
			if not ret or frame is None:
				with self.lock:
					self.message = "Failed to read frame"
				time.sleep(0.05)
				continue

			faces = detect_face.app.get(frame)
			with self.lock:
				mode = self.mode

			if mode == "recognition":
				self._process_recognition(frame, faces)
			elif mode == "enroll":
				self._process_enrollment(frame, faces)
			else:
				self.visible_names = []
				self._draw_header(frame, "Idle mode")

			ok, encoded = cv2.imencode(".jpg", frame)
			if ok:
				with self.lock:
					self.latest_frame = encoded.tobytes()

	def _draw_header(self, frame, text: str) -> None:
		cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (25, 25, 25), -1)
		cv2.putText(
			frame,
			text,
			(12, 28),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(240, 240, 240),
			2,
		)

	def _process_recognition(self, frame, faces) -> None:
		names = []
		for face in faces:
			name, score = detect_face.identify(self.known_embeddings, face.embedding)
			names.append(name)

			x1, y1, x2, y2 = face.bbox.astype(int)
			color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
			label = f"{name} ({score:.2f})"

			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

		unique_names = sorted(set(names))
		with self.lock:
			self.visible_names = unique_names
			self.message = "Recognition running"

		known_count = sum(1 for n in names if n != "Unknown")
		self._draw_header(frame, f"Recognition | Faces: {len(faces)} | Known: {known_count}")

	def _process_enrollment(self, frame, faces) -> None:
		now = time.time()
		with self.lock:
			enroll_name = self.enroll_name
			enroll_target = self.enroll_target

		self._draw_header(frame, f"Enrolling {enroll_name} ({len(self.enroll_embeddings)}/{enroll_target})")

		if len(faces) > 1:
			cv2.putText(
				frame,
				"Only one face please",
				(12, 72),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 0, 255),
				2,
			)
			return

		if len(faces) == 0:
			cv2.putText(
				frame,
				"No face detected",
				(12, 72),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 0, 255),
				2,
			)
			return

		face = faces[0]
		x1, y1, x2, y2 = face.bbox.astype(int)
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

		if now - self.last_enroll_capture_at >= (detect_face.ENROLL_CAPTURE_DELAY_MS / 1000.0):
			self.enroll_embeddings.append(face.embedding)
			self.last_enroll_capture_at = now

			with self.lock:
				self.message = f"Captured {len(self.enroll_embeddings)}/{enroll_target}"

		if len(self.enroll_embeddings) >= enroll_target:
			averaged = np.mean(self.enroll_embeddings, axis=0)
			with self.lock:
				self.known_embeddings[enroll_name] = averaged
				detect_face.save_database(self.known_embeddings)
				self.mode = "recognition"
				self.visible_names = []
				self.message = f"Enrollment completed for {enroll_name}"
				self.enroll_name = None
				self.enroll_embeddings = []

	def set_mode(self, mode: str) -> None:
		with self.lock:
			self.mode = mode
			if mode != "enroll":
				self.enroll_name = None
				self.enroll_embeddings = []
			self.message = f"Mode changed to {mode}"

	def start_enrollment(self, name: str, samples: int) -> tuple[bool, str]:
		if not name:
			return False, "Name is required"
		if samples <= 0:
			return False, "Samples must be positive"

		with self.lock:
			self.mode = "enroll"
			self.enroll_name = name
			self.enroll_target = samples
			self.enroll_embeddings = []
			self.last_enroll_capture_at = 0.0
			self.message = f"Starting enrollment for {name}"

		return True, "Enrollment started"

	def cancel_enrollment(self) -> None:
		with self.lock:
			self.mode = "recognition"
			self.enroll_name = None
			self.enroll_embeddings = []
			self.message = "Enrollment cancelled"

	def delete_person(self, name: str) -> tuple[bool, str]:
		with self.lock:
			deleted = detect_face.delete_person(self.known_embeddings, name)
		if deleted:
			return True, f"Deleted {name}"
		return False, f"{name} not found"

	def get_status(self) -> dict:
		with self.lock:
			return {
				"mode": self.mode,
				"message": self.message,
				"enrolled_people": sorted(list(self.known_embeddings.keys())),
				"visible_names": self.visible_names,
				"enroll": {
					"name": self.enroll_name,
					"target": self.enroll_target,
					"collected": len(self.enroll_embeddings),
				},
			}

	def get_frame(self):
		with self.lock:
			return self.latest_frame


runtime = FaceRuntime()
runtime.start()
atexit.register(runtime.shutdown)


def generate_mjpeg():
	while True:
		frame = runtime.get_frame()
		if frame is None:
			time.sleep(0.03)
			continue

		yield (
			b"--frame\r\n"
			b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
		)


@flask_app.route("/")
def index():
	return send_from_directory(".", "index.html")


@flask_app.get("/video_feed")
def video_feed():
	return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.get("/api/status")
def api_status():
	return jsonify(runtime.get_status())


@flask_app.post("/api/recognition/start")
def api_recognition_start():
	runtime.set_mode("recognition")
	return jsonify({"ok": True, "message": "Recognition started"})


@flask_app.post("/api/recognition/stop")
def api_recognition_stop():
	runtime.set_mode("idle")
	return jsonify({"ok": True, "message": "Recognition stopped"})


@flask_app.post("/api/enroll")
def api_enroll():
	data = request.get_json(silent=True) or {}
	name = str(data.get("name", "")).strip()

	samples_raw = data.get("samples", detect_face.ENROLL_SAMPLES)
	try:
		samples = int(samples_raw)
	except (TypeError, ValueError):
		return jsonify({"ok": False, "message": "Samples must be an integer"}), 400

	ok, message = runtime.start_enrollment(name=name, samples=samples)
	status = 200 if ok else 400
	return jsonify({"ok": ok, "message": message}), status


@flask_app.post("/api/enroll/cancel")
def api_enroll_cancel():
	runtime.cancel_enrollment()
	return jsonify({"ok": True, "message": "Enrollment cancelled"})


@flask_app.post("/api/person/delete")
def api_delete_person():
	data = request.get_json(silent=True) or {}
	name = str(data.get("name", "")).strip()
	if not name:
		return jsonify({"ok": False, "message": "Name is required"}), 400

	ok, message = runtime.delete_person(name)
	status = 200 if ok else 404
	return jsonify({"ok": ok, "message": message}), status


if __name__ == "__main__":
	flask_app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
