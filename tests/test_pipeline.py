from pathlib import Path

import numpy as np
import pytest

from video_analysis import pipeline
from video_analysis.schema import Claim, Face, Scene, Shot, TranscriptSegment, Video

URL = "https://www.youtube.com/watch?v=test"


class FakeGrouper:
    similarity_threshold, min_face_size, min_detection_score = 0.5, 40, 0.7

    def reset(self):
        pass

    def add_frame(self, shot_id, timestamp_s, image, artifact_root):
        crop = Path("faces") / f"{shot_id}.jpg"
        (artifact_root / crop).parent.mkdir(parents=True, exist_ok=True)
        (artifact_root / crop).write_bytes(b"jpeg")
        return [Face(face_id=f"face_{shot_id}", shot_id=shot_id, group_id="group_001", timestamp_s=timestamp_s,
                     bbox_xyxy=[1.0, 1.0, 20.0, 20.0], crop_path=crop.as_posix(), detection_score=0.9)]


class FakeAnalyzer:
    model, requests_made, cache_hits = "fake-model", 0, 0

    def analyze(self, shots, transcript, frames, artifact_root):
        assert all((artifact_root / frame.path).is_file() for frame in frames)
        return [Scene(scene_id="scene_0001", start_s=0.0, end_s=2.0, shot_ids=[shot.shot_id for shot in shots],
                      kind="story", setting=None, description="Two people talk.",
                      claims=[Claim(text="Someone says hello.", evidence_ids=[transcript[0].evidence_id])],
                      face_group_ids=["group_001"], boundary_reason="Start", needs_review=False)]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Replace media decoding with stubs and count how often each stage really runs."""
    calls = {"shots": 0, "transcript": 0, "samples": 0}

    def detect_shots(*args):
        calls["shots"] += 1
        return [Shot(shot_id="shot_0001", start_s=0.0, end_s=1.0), Shot(shot_id="shot_0002", start_s=1.0, end_s=2.0)]

    def transcribe(*args):
        calls["transcript"] += 1
        return [TranscriptSegment(evidence_id="text_0001", start_s=0.2, end_s=0.8, text="Hello")]

    def sample_shots(path, shots, fps):
        calls["samples"] += 1
        for shot in shots:
            yield shot.shot_id, shot.start_s + 0.5, np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline, "probe", lambda path, start, end: Video(
        filename="clip.mp4", sha256="abc", duration_s=2.0, analyzed_start_s=0.0, analyzed_end_s=2.0,
        width=32, height=32, timing_tolerance_s=0.001))
    monkeypatch.setattr(pipeline, "detect_shots", detect_shots)
    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline, "sample_shots", sample_shots)

    def run(**options):
        grouper = FakeGrouper()
        grouper.__dict__.update(options)
        return pipeline.run_analysis(tmp_path / "clip.mp4", URL, tmp_path / "outputs", tmp_path / "cache",
                                     grouper, FakeAnalyzer())

    return run, calls


def test_pipeline_links_stages_and_writes_outputs(stubbed, tmp_path):
    run, _ = stubbed
    analysis = run()
    assert [shot.scene_id for shot in analysis.shots] == ["scene_0001", "scene_0001"]
    assert [shot.face_group_ids for shot in analysis.shots] == [["group_001"], ["group_001"]]
    assert analysis.settings["llm_model"] == "fake-model"
    for name in ("analysis.json", "pipeline_result.json", "extracted_faces/scene_000/group_001"):
        assert (tmp_path / "outputs" / name).exists()


def test_stages_are_cached_by_the_settings_they_depend_on(stubbed):
    run, calls = stubbed
    run()
    run()
    assert calls == {"shots": 1, "transcript": 1, "samples": 1}
    run(similarity_threshold=0.6)
    assert calls == {"shots": 1, "transcript": 1, "samples": 2}


@pytest.mark.parametrize("url", ["http://www.youtube.com/watch?v=x", "https://example.com/video"])
def test_non_youtube_urls_are_rejected_before_processing(tmp_path, url):
    with pytest.raises(ValueError, match="YouTube"):
        pipeline.run_analysis(tmp_path / "clip.mp4", url, tmp_path, tmp_path, FakeGrouper(), FakeAnalyzer())
