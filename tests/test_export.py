import json

from video_analysis.export import export_assignment, timecode
from video_analysis.schema import Analysis, Claim, Face, Scene, Shot, Video


def face(face_id, shot_id, group_id, timestamp_s):
    return Face(face_id=face_id, shot_id=shot_id, group_id=group_id, timestamp_s=timestamp_s,
                bbox_xyxy=[1.0, 1.0, 10.0, 10.0], crop_path=f"faces/{face_id}.jpg", detection_score=0.9)


def scene(scene_id, start_s, end_s, shot_ids, description):
    return Scene(scene_id=scene_id, start_s=start_s, end_s=end_s, shot_ids=shot_ids, kind="story", setting=None,
                 description=description, claims=[Claim(text=description, evidence_ids=["frame_1"])],
                 face_group_ids=[], boundary_reason="Cut", needs_review=False)


def make_analysis(artifact_root):
    faces = [face("face_1", "shot_1", "group_001", 0.5), face("face_2", "shot_1", "group_002", 0.5),
             face("face_3", "shot_1", "group_001", 1.25), face("face_4", "shot_2", "group_001", 3.0)]
    for item in faces:
        (artifact_root / item.crop_path).parent.mkdir(parents=True, exist_ok=True)
        (artifact_root / item.crop_path).write_bytes(item.face_id.encode())
    return Analysis(
        video=Video(filename="clip.mp4", sha256="abc", duration_s=4.0, analyzed_start_s=0.0, analyzed_end_s=4.0,
                    width=20, height=20, timing_tolerance_s=0.001),
        video_url="https://youtu.be/example", settings={},
        shots=[Shot(shot_id="shot_1", start_s=0.0, end_s=2.0, scene_id="scene_1"),
               Shot(shot_id="shot_2", start_s=2.0, end_s=4.0, scene_id="scene_2")],
        scenes=[scene("scene_1", 0.0, 2.0, ["shot_1"], "Two people meet."),
                scene("scene_2", 2.0, 4.0, ["shot_2"], "One of them leaves.")],
        faces=faces, transcript=[], frames=[],
    )


def test_timecode_format():
    assert timecode(0) == "00:00:00.000"
    assert timecode(3725.0426) == "01:02:05.043"


def test_writes_assignment_json_and_one_folder_per_group_per_scene(tmp_path):
    output = tmp_path / "outputs"
    stale = output / "extracted_faces" / "scene_009"
    stale.mkdir(parents=True)

    submission = export_assignment(make_analysis(tmp_path / "artifacts"), tmp_path / "artifacts", output)

    assert submission == json.loads((output / "analysis.json").read_text())
    assert submission["video_url"] == "https://youtu.be/example"
    assert submission["shots"] == [{"id": 0, "start_seconds": 0.0, "end_seconds": 2.0},
                                   {"id": 1, "start_seconds": 2.0, "end_seconds": 4.0}]
    assert submission["scenes"][0] == {
        "id": 0, "start_seconds": 0.0, "end_seconds": 2.0, "description": "Two people meet.",
        "faces": {"group_001": ["00:00:00.500", "00:00:01.250"], "group_002": ["00:00:00.500"]},
    }
    assert submission["scenes"][1]["faces"] == {"group_001": ["00:00:03.000"]}

    folders = sorted(path.relative_to(output / "extracted_faces").as_posix()
                     for path in (output / "extracted_faces").glob("scene_*/group_*"))
    assert folders == ["scene_000/group_001", "scene_000/group_002", "scene_001/group_001"]
    crops = sorted((output / "extracted_faces" / "scene_000" / "group_001").iterdir())
    assert [path.name for path in crops] == ["face_000_00-00-00.500.jpg", "face_001_00-00-01.250.jpg"]
    assert crops[1].read_bytes() == b"face_3"
    assert not stale.exists()
