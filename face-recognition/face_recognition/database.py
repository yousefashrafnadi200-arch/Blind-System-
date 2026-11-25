import os
import cv2
import numpy as np

from .detector import FaceDetector


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Guards against division by zero when an embedding has zero length.
EPSILON = 1e-12


class FaceDatabase:
    """Face embeddings loaded from ``Faces/<person>/<image>``.

    Embeddings are normalized once at load time and kept in a single matrix,
    so recognition is one matrix-vector product instead of a Python loop.
    """

    def __init__(
        self,
        faces_folder,
        detector: FaceDetector,
        threshold=0.5,
        margin=0.05,
        cache_file=None,
    ):
        self.faces_folder = faces_folder
        self.detector = detector
        self.threshold = threshold
        self.margin = margin
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(os.path.abspath(faces_folder)),
            "embeddings.npz",
        )

        self.embeddings = []
        self.names = []
        self.matrix = np.empty((0, 0), dtype=np.float32)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):

        print("Loading face database...")

        if not os.path.isdir(self.faces_folder):
            print(f"Faces folder not found: {self.faces_folder}")
            return

        signature = self._signature()

        if self._load_cache(signature):
            return

        self.embeddings = []
        self.names = []

        for person_name in sorted(os.listdir(self.faces_folder)):

            person_folder = os.path.join(
                self.faces_folder,
                person_name
            )

            if not os.path.isdir(person_folder):
                continue

            print(f"Loading: {person_name}")

            for image_name in sorted(os.listdir(person_folder)):

                if not image_name.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                image_path = os.path.join(
                    person_folder,
                    image_name
                )

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read: {image_path}")
                    continue

                try:
                    faces = self.detector.detect(image)
                except Exception as exc:
                    print(f"Detection failed for {image_path}: {exc}")
                    continue

                if len(faces) == 0:
                    print(f"No face found: {image_path}")
                    continue

                # Use the largest face in the picture.
                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )

                embedding = getattr(face, "embedding", None)

                if embedding is None:
                    print(f"No embedding for: {image_path}")
                    continue

                self.embeddings.append(
                    np.asarray(embedding, dtype=np.float32)
                )
                self.names.append(person_name)

                print(f"Loaded: {person_name} -> {image_name}")

        self._build_matrix()

        print(f"Loaded {len(self.names)} face embeddings.")

        self._save_cache(signature)

    def _signature(self):
        """Cheap fingerprint of the image files, used to validate the cache."""

        entries = []

        for person_name in sorted(os.listdir(self.faces_folder)):

            person_folder = os.path.join(
                self.faces_folder,
                person_name
            )

            if not os.path.isdir(person_folder):
                continue

            for image_name in sorted(os.listdir(person_folder)):

                if not image_name.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                image_path = os.path.join(
                    person_folder,
                    image_name
                )

                try:
                    stat = os.stat(image_path)
                except OSError:
                    continue

                entries.append(
                    f"{person_name}/{image_name}:"
                    f"{stat.st_mtime_ns}:{stat.st_size}"
                )

        return entries

    def _build_matrix(self):

        if not self.embeddings:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            return

        matrix = np.vstack(self.embeddings).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        self.matrix = matrix / np.clip(norms, EPSILON, None)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _load_cache(self, signature):

        if not os.path.isfile(self.cache_file):
            return False

        try:
            data = np.load(self.cache_file, allow_pickle=False)

            cached_signature = [str(item) for item in data["signature"].tolist()]

            if cached_signature != signature:
                return False

            names = [str(item) for item in data["names"].tolist()]
            embeddings = data["embeddings"]

            if len(names) != len(embeddings):
                return False

            self.names = names
            self.embeddings = [row for row in embeddings]
            self._build_matrix()

            print(f"Loaded {len(self.names)} cached face embeddings.")

            return True

        except Exception as exc:
            print(f"Could not read embedding cache: {exc}")
            return False

    def _save_cache(self, signature):

        try:
            embeddings = (
                np.vstack(self.embeddings).astype(np.float32)
                if self.embeddings
                else np.empty((0, 0), dtype=np.float32)
            )

            np.savez_compressed(
                self.cache_file,
                names=np.array(self.names, dtype=str),
                embeddings=embeddings,
                signature=np.array(signature, dtype=str),
            )

        except Exception as exc:
            print(f"Could not save embedding cache: {exc}")

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(self, face, threshold=None, margin=None):
        """Match a face (or raw embedding) against the database.

        Returns ``(name, similarity)``; ``name`` is ``"Unknown"`` when the
        best score is below ``threshold`` or too close to the runner-up.
        """

        embedding = getattr(face, "embedding", face)

        if self.matrix.size == 0 or embedding is None:
            return "Unknown", 0.0

        threshold = self.threshold if threshold is None else threshold
        margin = self.margin if margin is None else margin

        query = np.asarray(embedding, dtype=np.float32).ravel()
        query = query / max(float(np.linalg.norm(query)), EPSILON)

        similarities = self.matrix @ query

        # Best score per person, so several photos of the same person do not
        # count as separate candidates.
        names = np.asarray(self.names)
        unique_names, inverse = np.unique(names, return_inverse=True)

        best_per_name = np.full(unique_names.shape, -1.0, dtype=np.float32)
        np.maximum.at(best_per_name, inverse, similarities)

        order = np.argsort(best_per_name)[::-1]

        best_index = int(order[0])
        best_similarity = float(best_per_name[best_index])

        second_similarity = (
            float(best_per_name[int(order[1])])
            if len(order) > 1
            else -1.0
        )

        if best_similarity < threshold:
            return "Unknown", best_similarity

        if best_similarity - second_similarity < margin:
            return "Unknown", best_similarity

        return str(unique_names[best_index]), best_similarity
