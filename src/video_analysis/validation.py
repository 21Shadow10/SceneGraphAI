"""Integrity checks run on the final result before it is exported."""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .schema import Analysis


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_unique(ids: Iterable[str], label: str) -> None:
    ids = list(ids)
    _require(len(ids) == len(set(ids)), f"Duplicate {label} IDs")


def validate_analysis(analysis: Analysis, artifact_root: Path) -> None:
    """Raise ``ValueError`` if the result is internally inconsistent.

    Checks that shots and scenes each tile the analysed interval with no gaps
    or overlaps, that every ID reference resolves, that every timestamp and
    face box lies inside its owning shot and frame, and that every referenced
    image exists under ``artifact_root``. This proves integrity, not accuracy.
    """
    Analysis.model_validate(analysis.model_dump())  # re-check field types after in-place edits
    video = analysis.video
    tolerance = video.timing_tolerance_s
    artifact_root = artifact_root.resolve()

    def require_partition(items: list, label: str) -> None:
        _require(bool(items), f"No {label}")
        cursor = video.analyzed_start_s
        for item in items:
            _require(item.end_s > item.start_s, f"A {label[:-1]} has a non-positive duration")
            _require(abs(item.start_s - cursor) <= tolerance, f"{label} have a gap or overlap")
            cursor = item.end_s
        _require(abs(cursor - video.analyzed_end_s) <= tolerance, f"{label} do not reach the clip end")

    def require_artifact(relative: str) -> None:
        path = (artifact_root / relative).resolve()
        _require(path.is_relative_to(artifact_root) and path.is_file(), f"Missing artifact {relative}")

    _require_unique((shot.shot_id for shot in analysis.shots), "shot")
    _require_unique((scene.scene_id for scene in analysis.scenes), "scene")
    _require_unique((face.face_id for face in analysis.faces), "face")
    _require_unique([item.evidence_id for item in [*analysis.transcript, *analysis.frames]], "evidence")
    require_partition(analysis.shots, "shots")
    require_partition(analysis.scenes, "scenes")

    shots = {shot.shot_id: shot for shot in analysis.shots}
    frames = {frame.evidence_id: frame for frame in analysis.frames}
    segments = {segment.evidence_id: segment for segment in analysis.transcript}

    _require([shot_id for scene in analysis.scenes for shot_id in scene.shot_ids] == list(shots),
             "Scenes must contain every shot exactly once, in order")
    for scene in analysis.scenes:
        members = [shots[shot_id] for shot_id in scene.shot_ids]
        _require(all(shot.scene_id == scene.scene_id for shot in members), f"{scene.scene_id}: shot links differ")
        _require(abs(scene.start_s - members[0].start_s) <= tolerance
                 and abs(scene.end_s - members[-1].end_s) <= tolerance, f"{scene.scene_id}: bounds differ from shots")
        _require(scene.face_group_ids == sorted({group for shot in members for group in shot.face_group_ids}),
                 f"{scene.scene_id}: face groups differ from its shots")
        _require(bool(scene.description.strip()), f"{scene.scene_id}: empty description")
        for evidence_id in (evidence_id for claim in scene.claims for evidence_id in claim.evidence_ids):
            if evidence_id in frames:
                _require(frames[evidence_id].shot_id in scene.shot_ids, f"{scene.scene_id} cites a frame outside it")
            else:
                _require(evidence_id in segments, f"{scene.scene_id} cites unknown evidence {evidence_id}")
                segment = segments[evidence_id]
                _require(segment.start_s < scene.end_s and segment.end_s > scene.start_s,
                         f"{scene.scene_id} cites dialogue outside it")

    for segment in analysis.transcript:
        _require(video.analyzed_start_s <= segment.start_s < segment.end_s <= video.analyzed_end_s + tolerance,
                 f"{segment.evidence_id} lies outside the clip")
    for frame in analysis.frames:
        shot = shots.get(frame.shot_id)
        _require(shot is not None and shot.start_s <= frame.timestamp_s < shot.end_s,
                 f"{frame.evidence_id} lies outside its shot")
        require_artifact(frame.path)

    groups_by_shot: dict[str, set[str]] = defaultdict(set)
    for face in analysis.faces:
        shot = shots.get(face.shot_id)
        _require(shot is not None and shot.start_s <= face.timestamp_s < shot.end_s,
                 f"{face.face_id} lies outside its shot")
        left, top, right, bottom = face.bbox_xyxy
        _require(0 <= left < right <= video.width and 0 <= top < bottom <= video.height,
                 f"{face.face_id} box lies outside the frame")
        require_artifact(face.crop_path)
        groups_by_shot[face.shot_id].add(face.group_id)
    for shot in analysis.shots:
        _require(shot.face_group_ids == sorted(groups_by_shot[shot.shot_id]),
                 f"{shot.shot_id}: face groups differ from its faces")
