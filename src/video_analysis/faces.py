"""Local face detection (InsightFace) and anonymous grouping by embedding similarity."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schema import Face


class FaceGrouper:
    """Detect faces in sampled frames and cluster them into anonymous groups.

    Every face gets a 512-d ArcFace embedding. It joins the existing group whose
    mean embedding is most similar, provided the cosine similarity reaches
    ``similarity_threshold``; otherwise it starts a new group. Groups are
    therefore similarity clusters, not verified identities: one person can be
    split across groups (pose, lighting, occlusion) and look-alikes can merge.

    Usage: call ``reset()`` before each video, then ``add_frame()`` per sample.
    """

    def __init__(self, similarity_threshold: float = 0.48, min_face_size: int = 40,
                 min_detection_score: float = 0.7, app: Any | None = None) -> None:
        """
        Args:
            similarity_threshold: Minimum cosine similarity to a group's mean
                embedding for a face to join that group.
            min_face_size: Faces whose box is narrower or shorter than this
                many pixels are skipped; tiny faces give unreliable embeddings.
            min_detection_score: Minimum detector confidence for a face.
            app: Pre-built InsightFace ``FaceAnalysis``; loaded lazily if omitted.
        """
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.min_detection_score = min_detection_score
        self._app = app
        self.reset()

    def load_model(self) -> None:
        """Load InsightFace's ``buffalo_l`` detector and recognizer on the CPU.

        The weights are downloaded to ``~/.insightface`` on first use.
        """
        if self._app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app

    def reset(self) -> None:
        """Forget all groups so the next video starts again from ``group_001``."""
        self._group_sums: list[np.ndarray] = []
        self._face_count = 0

    def add_frame(self, shot_id: str, timestamp_s: float, image: np.ndarray,
                  artifact_root: Path) -> list[Face]:
        """Detect, group and save a crop of every qualifying face in one BGR frame."""
        self.load_model()
        height, width = image.shape[:2]
        faces = []
        for detection in self._app.get(image):
            left, top, right, bottom = np.clip(detection.bbox, 0, [width, height, width, height])
            too_small = min(right - left, bottom - top) < self.min_face_size
            if too_small or detection.det_score < self.min_detection_score:
                continue
            embedding = np.asarray(detection.embedding, dtype=np.float64)
            group_id = self._assign_group(embedding / np.linalg.norm(embedding))

            self._face_count += 1
            face_id = f"face_{self._face_count:06d}"
            crop_path = Path("faces") / f"{face_id}.jpg"
            crop = image[int(top):int(np.ceil(bottom)), int(left):int(np.ceil(right))]
            destination = artifact_root / crop_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), crop):
                raise OSError(f"Could not write {destination}")
            faces.append(Face(
                face_id=face_id, shot_id=shot_id, group_id=group_id,
                timestamp_s=float(timestamp_s),
                bbox_xyxy=[float(left), float(top), float(right), float(bottom)],
                crop_path=crop_path.as_posix(), detection_score=float(detection.det_score),
            ))
        return faces

    def _assign_group(self, embedding: np.ndarray) -> str:
        """Return the group for a unit-length embedding, updating that group's mean."""
        if self._group_sums:
            sums = np.stack(self._group_sums)
            similarities = sums @ embedding / np.linalg.norm(sums, axis=1)
            best = int(np.argmax(similarities))
            if similarities[best] >= self.similarity_threshold:
                self._group_sums[best] += embedding
                return f"group_{best + 1:03d}"
        self._group_sums.append(embedding.copy())
        return f"group_{len(self._group_sums):03d}"
