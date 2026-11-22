import sys
from types import SimpleNamespace

import cv2
import numpy as np

from video_analysis.faces import FaceGrouper


class FakeApp:
    """Stands in for InsightFace: returns detections keyed by the frame's first pixel value."""

    def __init__(self, responses):
        self.responses = responses

    def get(self, image):
        return self.responses.get(int(image[0, 0, 0]), [])


def detection(embedding, bbox=(10, 10, 70, 70), score=0.95):
    return SimpleNamespace(embedding=np.array(embedding, dtype=float),
                           bbox=np.array(bbox, dtype=float), det_score=score)


def frame(marker):
    return np.full((100, 120, 3), marker, dtype=np.uint8)


def test_similar_faces_share_a_group_across_shots(tmp_path):
    app = FakeApp({1: [detection([1, 0, 0])], 2: [detection([0.98, 0.2, 0])], 3: [detection([0, 1, 0])]})
    grouper = FaceGrouper(similarity_threshold=0.8, app=app)
    faces = [
        *grouper.add_frame("shot_0001", 0.5, frame(1), tmp_path),
        *grouper.add_frame("shot_0001", 1.0, frame(2), tmp_path),
        *grouper.add_frame("shot_0002", 2.5, frame(3), tmp_path),
        *grouper.add_frame("shot_0002", 3.0, frame(1), tmp_path),
    ]
    assert [face.group_id for face in faces] == ["group_001", "group_001", "group_002", "group_001"]
    assert [face.face_id for face in faces] == [f"face_{n:06d}" for n in range(1, 5)]
    assert all((tmp_path / face.crop_path).is_file() for face in faces)


def test_low_confidence_and_small_faces_are_skipped(tmp_path):
    app = FakeApp({1: [detection([1, 0], score=0.5), detection([1, 0], bbox=(0, 0, 20, 80))]})
    assert FaceGrouper(app=app).add_frame("shot_0001", 0.0, frame(1), tmp_path) == []


def test_box_is_clipped_to_the_frame_and_crop_saved(tmp_path):
    app = FakeApp({1: [detection([1, 0], bbox=(-4.5, 20.25, 140, 101))]})
    face, = FaceGrouper(app=app).add_frame("shot_0001", 0.5, frame(1), tmp_path)
    assert face.bbox_xyxy == [0.0, 20.25, 120.0, 100.0]
    assert cv2.imread(str(tmp_path / face.crop_path)).shape == (80, 120, 3)


def test_reset_starts_numbering_again(tmp_path):
    grouper = FaceGrouper(app=FakeApp({1: [detection([1, 0])], 2: [detection([0, 1])]}))
    grouper.add_frame("shot_0001", 0.0, frame(1), tmp_path)
    grouper.reset()
    face, = grouper.add_frame("shot_0001", 0.0, frame(2), tmp_path)
    assert (face.face_id, face.group_id) == ("face_000001", "group_001")


def test_insightface_is_loaded_lazily_once_on_cpu(monkeypatch):
    calls = []

    class FaceAnalysis:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def prepare(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "insightface", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=FaceAnalysis))
    grouper = FaceGrouper()
    assert calls == []
    grouper.load_model()
    grouper.load_model()
    assert calls == [
        {"name": "buffalo_l", "allowed_modules": ["detection", "recognition"],
         "providers": ["CPUExecutionProvider"]},
        {"ctx_id": -1, "det_size": (640, 640)},
    ]
