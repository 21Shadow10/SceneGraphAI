import pytest

from video_analysis.schema import Analysis, Claim, Face, Frame, Scene, Shot, TranscriptSegment, Video
from video_analysis.validation import validate_analysis


def make_analysis(root):
    for path in ("frames/frame_000001.jpg", "frames/frame_000002.jpg", "faces/face_000001.jpg"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"jpeg")
    return Analysis(
        video=Video(filename="clip.mp4", sha256="abc", duration_s=4.0, analyzed_start_s=0.0,
                    analyzed_end_s=4.0, width=100, height=100, timing_tolerance_s=0.001),
        video_url="https://www.youtube.com/watch?v=test",
        settings={},
        shots=[
            Shot(shot_id="shot_0001", start_s=0.0, end_s=2.0, scene_id="scene_0001", face_group_ids=["group_001"]),
            Shot(shot_id="shot_0002", start_s=2.0, end_s=4.0, scene_id="scene_0002"),
        ],
        scenes=[
            Scene(scene_id="scene_0001", start_s=0.0, end_s=2.0, shot_ids=["shot_0001"], kind="story",
                  setting="Kitchen", description="Someone says hello.",
                  claims=[Claim(text="Someone says hello.", evidence_ids=["text_0001", "frame_000001"])],
                  face_group_ids=["group_001"], boundary_reason="Start", needs_review=False),
            Scene(scene_id="scene_0002", start_s=2.0, end_s=4.0, shot_ids=["shot_0002"], kind="credits",
                  setting=None, description="Credits roll.",
                  claims=[Claim(text="Credits roll.", evidence_ids=["frame_000002"])],
                  face_group_ids=[], boundary_reason="Cut to credits", needs_review=True),
        ],
        faces=[Face(face_id="face_000001", shot_id="shot_0001", group_id="group_001", timestamp_s=0.5,
                    bbox_xyxy=[10.0, 10.0, 50.0, 50.0], crop_path="faces/face_000001.jpg", detection_score=0.9)],
        transcript=[TranscriptSegment(evidence_id="text_0001", start_s=0.2, end_s=1.0, text="Hello")],
        frames=[
            Frame(evidence_id="frame_000001", shot_id="shot_0001", timestamp_s=0.5, path="frames/frame_000001.jpg"),
            Frame(evidence_id="frame_000002", shot_id="shot_0002", timestamp_s=2.5, path="frames/frame_000002.jpg"),
        ],
    )


def test_consistent_result_passes(tmp_path):
    validate_analysis(make_analysis(tmp_path), tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda a: setattr(a.shots[1], "start_s", 2.5),
    lambda a: a.scenes[0].shot_ids.append("shot_0002"),
    lambda a: setattr(a.scenes[1], "description", "  "),
    lambda a: a.scenes[0].claims[0].evidence_ids.append("frame_000002"),
    lambda a: a.scenes[0].claims[0].evidence_ids.append("text_9999"),
    lambda a: setattr(a.faces[0], "timestamp_s", 3.0),
    lambda a: setattr(a.faces[0], "bbox_xyxy", [10.0, 10.0, 150.0, 50.0]),
    lambda a: setattr(a.faces[0], "crop_path", "../outside.jpg"),
    lambda a: setattr(a.shots[0], "face_group_ids", []),
    lambda a: a.faces.append(a.faces[0].model_copy()),
], ids=["shot_gap", "shot_in_two_scenes", "empty_description", "frame_from_other_scene",
        "unknown_evidence", "face_outside_shot", "box_outside_frame", "unsafe_artifact_path",
        "stale_shot_groups", "duplicate_face"])
def test_inconsistent_results_are_rejected(tmp_path, mutate):
    analysis = make_analysis(tmp_path)
    mutate(analysis)
    with pytest.raises(ValueError):
        validate_analysis(analysis, tmp_path)


def test_missing_artifact_is_rejected(tmp_path):
    analysis = make_analysis(tmp_path)
    (tmp_path / "faces" / "face_000001.jpg").unlink()
    with pytest.raises(ValueError, match="Missing artifact"):
        validate_analysis(analysis, tmp_path)
