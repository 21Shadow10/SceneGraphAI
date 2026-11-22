"""Write the assignment deliverables: ``analysis.json`` and per-scene face folders."""

import shutil
from collections import defaultdict
from pathlib import Path

from .schema import Analysis, Face, atomic_json


def timecode(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def export_assignment(analysis: Analysis, artifact_root: Path, output_dir: Path) -> dict:
    """Write ``analysis.json`` and copy face crops into ``extracted_faces/``.

    Shot and scene IDs in the JSON are zero-based positions, and the face
    folders use the same numbers: ``extracted_faces/scene_003/group_002/``
    holds every crop of face group 2 seen during scene 3. The folder tree is
    rebuilt from scratch so crops from an earlier run never leak in.

    Returns the JSON document that was written.
    """
    scene_number = {scene.scene_id: number for number, scene in enumerate(analysis.scenes)}
    scene_of_shot = {shot.shot_id: scene_number[shot.scene_id] for shot in analysis.shots}
    faces_by_scene: dict[int, dict[str, list[Face]]] = defaultdict(lambda: defaultdict(list))
    for face in sorted(analysis.faces, key=lambda item: (item.timestamp_s, item.face_id)):
        faces_by_scene[scene_of_shot[face.shot_id]][face.group_id].append(face)

    face_root = output_dir / "extracted_faces"
    shutil.rmtree(face_root, ignore_errors=True)
    for number, groups in faces_by_scene.items():
        for group_id, faces in groups.items():
            folder = face_root / f"scene_{number:03d}" / group_id
            folder.mkdir(parents=True)
            for index, face in enumerate(faces):
                name = f"face_{index:03d}_{timecode(face.timestamp_s).replace(':', '-')}.jpg"
                shutil.copy2(artifact_root / face.crop_path, folder / name)

    submission = {
        "video_url": analysis.video_url,
        "shots": [
            {"id": number, "start_seconds": round(shot.start_s, 3), "end_seconds": round(shot.end_s, 3)}
            for number, shot in enumerate(analysis.shots)
        ],
        "scenes": [
            {
                "id": number,
                "start_seconds": round(scene.start_s, 3),
                "end_seconds": round(scene.end_s, 3),
                "description": scene.description,
                "faces": {
                    group_id: sorted({timecode(face.timestamp_s) for face in faces})
                    for group_id, faces in sorted(faces_by_scene.get(number, {}).items())
                },
            }
            for number, scene in enumerate(analysis.scenes)
        ],
    }
    atomic_json(output_dir / "analysis.json", submission)
    return submission
