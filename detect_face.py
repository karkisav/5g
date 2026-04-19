import insightface
import numpy as np
import cv2
import os
import json

EMBEDDINGS_FILE = "embeddings.npy"
NAMES_FILE = "names.json"
ENROLL_SAMPLES = 10
ENROLL_CAPTURE_DELAY_MS = 800   # ms between captures so poses actually vary
IDENTIFY_THRESHOLD = 0.45       # lowered from 0.6; tune up if false positives appear


# Model

app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database

def save_database(known_embeddings: dict) -> None:
    if not known_embeddings:
        return
    names = list(known_embeddings.keys())
    vectors = np.array(list(known_embeddings.values()))  # shape (N, 512)
    np.save(EMBEDDINGS_FILE, vectors)
    with open(NAMES_FILE, "w") as f:
        json.dump(names, f)


def load_database() -> dict:
    if not (os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE)):
        return {}
    vectors = np.load(EMBEDDINGS_FILE, allow_pickle=False)
    with open(NAMES_FILE, "r") as f:
        names = json.load(f)
    if len(names) != len(vectors):
        print("[warn] Database files are inconsistent. Starting fresh.")
        return {}
    return dict(zip(names, vectors))


def delete_person(known_embeddings: dict, name: str) -> bool:
    if name not in known_embeddings:
        print(f"'{name}' not found in database.")
        return False
    del known_embeddings[name]
    save_database(known_embeddings)
    print(f"'{name}' removed from database.")
    return True

# Enrollment

def enroll(known_embeddings: dict, name: str, n_samples: int = ENROLL_SAMPLES) -> bool:
    embeddings = []
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Could not open camera for enrollment.")
        return False

    print(f"\nEnrolling '{name}'. Look at the camera.")
    print("Vary your pose, angle, and expression between captures.")
    print("Press 'q' to cancel.\n")

    while len(embeddings) < n_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[error] Failed to read frame.")
            break

        faces = app.get(frame)

        # Overlay status
        status = f"Captured {len(embeddings)}/{n_samples}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if len(faces) > 1:
            cv2.putText(frame, "Multiple faces! Only one person please.",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        elif len(faces) == 1:
            embeddings.append(faces[0].embedding)
            remaining = n_samples - len(embeddings)
            if remaining > 0:
                print(f"  [{len(embeddings)}/{n_samples}] Got it — now change your pose or expression.")
            # Draw box so the user can see they were detected
            x1, y1, x2, y2 = faces[0].bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Enrollment", frame)
            cv2.waitKey(ENROLL_CAPTURE_DELAY_MS)  # pause so next frame is different
            continue

        else:
            cv2.putText(frame, "No face detected.", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Enrollment", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < n_samples:
        print(f"[warn] Only captured {len(embeddings)}/{n_samples} samples. Enrollment incomplete.")
        return False

    known_embeddings[name] = np.mean(embeddings, axis=0)
    save_database(known_embeddings)
    print(f"'{name}' enrolled successfully.\n")
    return True

# Identification

def identify(known_embeddings: dict, embedding: np.ndarray,
             threshold: float = IDENTIFY_THRESHOLD):
    query_norm = np.linalg.norm(embedding)
    if query_norm == 0 or not known_embeddings:
        return "Unknown", -1.0

    query = embedding / query_norm
    best_match, best_score = "Unknown", -1.0

    for name, stored in known_embeddings.items():
        stored_norm_val = np.linalg.norm(stored)
        if stored_norm_val == 0:
            continue
        score = float(np.dot(query, stored / stored_norm_val))
        if score > best_score:
            best_match, best_score = name, score

    if best_score >= threshold:
        return best_match, best_score
    return "Unknown", best_score

# Recognition loop

def run_recognition(known_embeddings: dict) -> None:
    print("\nStarting recognition. Press 'q' to return to menu.\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[error] Failed to read frame. Exiting recognition.")
            break

        faces = app.get(frame)
        for face in faces:
            name, score = identify(known_embeddings, face.embedding)
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# CLI menu

def print_menu(known_embeddings: dict) -> None:
    enrolled = list(known_embeddings.keys()) or ["(none)"]
    print("\n" + "=" * 40)
    print("  Face Recognition System")
    print("=" * 40)
    print(f"  Enrolled: {', '.join(enrolled)}")
    print("-" * 40)
    print("  [1] Enroll new person")
    print("  [2] Delete person")
    print("  [3] Run recognition")
    print("  [4] List enrolled people")
    print("  [q] Quit")
    print("=" * 40)


def main() -> None:
    known_embeddings = load_database()
    print(f"Loaded {len(known_embeddings)} person(s) from database.")

    while True:
        print_menu(known_embeddings)
        cmd = input("Choice: ").strip().lower()

        if cmd == "1":
            name = input("Enter name to enroll: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            if name in known_embeddings:
                overwrite = input(f"'{name}' already exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != "y":
                    continue
            enroll(known_embeddings, name)

        elif cmd == "2":
            name = input("Enter name to delete: ").strip()
            delete_person(known_embeddings, name)

        elif cmd == "3":
            if not known_embeddings:
                print("[warn] No faces enrolled yet. Please enroll someone first.")
            else:
                run_recognition(known_embeddings)

        elif cmd == "4":
            if known_embeddings:
                print("\nEnrolled people:")
                for i, name in enumerate(known_embeddings, 1):
                    print(f"  {i}. {name}")
            else:
                print("No one enrolled yet.")

        elif cmd == "q":
            print("Goodbye.")
            break

        else:
            print("Invalid option. Please choose 1, 2, 3, 4, or q.")


if __name__ == "__main__":
    main()