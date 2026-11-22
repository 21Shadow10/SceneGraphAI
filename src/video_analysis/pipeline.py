"""End-to-end orchestration: probe -> shots -> transcript -> faces -> scenes -> export."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2

from .export import export_assignment
from .faces import FaceGrouper
from .media import detect_shots, probe, sample_shots, transcribe
from .scenes import SemanticAnalyzer
from .schema import Analysis, Face, Frame, Shot, TranscriptSegment, atomic_json
from .validation import validate_analysis

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _settings_key(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def _load_or_build(path: Path, build: Callable[[], Any]) -> Any:
    """Return the JSON cached at ``path``, or run ``build`` and cache its result."""
    if path.is_file():
        return json.loads(path.read_text())
    result = build()
    atomic_json(path, result)
    return result


def _sample_and_group_faces(video_path: Path, shots: list[Shot], sample_fps: float,
                            grouper: FaceGrouper, artifact_root: Path) -> dict:
    """Save sampled frames as evidence and detect and group the faces in each one."""
    grouper.reset()
    frames: list[Frame] = []
    faces: list[Face] = []
    for shot_id, timestamp, image in sample_shots(video_path, shots, sample_fps):
        evidence_id = f"frame_{len(frames) + 1:06d}"
        path = Path("frames") / f"{evidence_id}.jpg"
        (artifact_root / path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(artifact_root / path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"Could not write {artifact_root / path}")
        frames.append(Frame(evidence_id=evidence_id, shot_id=shot_id, timestamp_s=timestamp, path=path.as_posix()))
        faces.extend(grouper.add_frame(shot_id, timestamp, image, artifact_root))
    return {"frames": [frame.model_dump() for frame in frames], "faces": [face.model_dump() for face in faces]}


def run_analysis(video_path: Path, video_url: str, output_dir: Path, cache_dir: Path,
                 face_grouper: FaceGrouper, semantic_analyzer: SemanticAnalyzer, *,
                 start_s: float = 0.0, end_s: float | None = None, sample_fps: float = 2.0,
                 whisper_model: str = "small", cpu_threads: int = 4,
                 shot_threshold: float = 3.0, min_shot_frames: int = 3) -> Analysis:
    """Run every stage and write the assignment outputs.

    Writes ``analysis.json``, ``pipeline_result.json`` and ``extracted_faces/``
    to ``output_dir``. Each local stage is cached in ``cache_dir`` under a key
    built from the video hash and only the settings that affect that stage,
    so changing, say, the face threshold reuses the shots and transcript.
    Model answers are cached separately by ``SemanticAnalyzer``.

    Args:
        video_path: Local copy of the video.
        video_url: The video's YouTube URL, recorded in the submission.
        start_s, end_s: Interval to analyse; ``end_s=None`` means the end of the file.
        sample_fps: Frames sampled per second of each shot for faces and the model.
        whisper_model: faster-whisper model size, e.g. ``"small"``.
        cpu_threads: CPU threads for Whisper.
        shot_threshold: ``AdaptiveDetector`` threshold; lower finds more cuts.
        min_shot_frames: Shortest allowed shot, in frames.
    """
    url = urlsplit(video_url)
    if url.scheme != "https" or url.hostname not in YOUTUBE_HOSTS:
        raise ValueError(f"Expected an https YouTube URL, got {video_url!r}")

    video = probe(video_path, start_s, end_s)
    clip = {"sha256": video.sha256, "start_s": video.analyzed_start_s, "end_s": video.analyzed_end_s}
    shot_settings = {**clip, "shot_threshold": shot_threshold, "min_shot_frames": min_shot_frames}
    transcript_settings = {**clip, "whisper_model": whisper_model}
    face_settings = {
        **shot_settings, "sample_fps": sample_fps,
        "similarity_threshold": face_grouper.similarity_threshold,
        "min_face_size": face_grouper.min_face_size,
        "min_detection_score": face_grouper.min_detection_score,
    }

    shots = [Shot.model_validate(item) for item in _load_or_build(
        cache_dir / f"shots-{_settings_key(shot_settings)}.json",
        lambda: [shot.model_dump() for shot in detect_shots(video_path, video, shot_threshold, min_shot_frames)],
    )]
    transcript = [TranscriptSegment.model_validate(item) for item in _load_or_build(
        cache_dir / f"transcript-{_settings_key(transcript_settings)}.json",
        lambda: [segment.model_dump() for segment in transcribe(
            video_path, video, cache_dir / "audio", whisper_model, cpu_threads)],
    )]
    # Frames and face crops live next to their index, so the cache is self-contained.
    artifact_root = cache_dir / f"faces-{_settings_key(face_settings)}"
    sampled = _load_or_build(
        artifact_root / "index.json",
        lambda: _sample_and_group_faces(video_path, shots, sample_fps, face_grouper, artifact_root),
    )
    frames = [Frame.model_validate(item) for item in sampled["frames"]]
    faces = [Face.model_validate(item) for item in sampled["faces"]]

    groups_by_shot: dict[str, set[str]] = defaultdict(set)
    for face in faces:
        groups_by_shot[face.shot_id].add(face.group_id)
    for shot in shots:
        shot.face_group_ids = sorted(groups_by_shot[shot.shot_id])

    scenes = semantic_analyzer.analyze(shots, transcript, frames, artifact_root)
    scene_of_shot = {shot_id: scene.scene_id for scene in scenes for shot_id in scene.shot_ids}
    for shot in shots:
        shot.scene_id = scene_of_shot[shot.shot_id]

    analysis = Analysis(
        video=video, video_url=video_url,
        settings={**face_settings, "whisper_model": whisper_model, "llm_model": semantic_analyzer.model,
                  "llm_requests": semantic_analyzer.requests_made, "llm_cache_hits": semantic_analyzer.cache_hits},
        shots=shots, scenes=scenes, faces=faces, transcript=transcript, frames=frames,
    )
    validate_analysis(analysis, artifact_root)
    atomic_json(output_dir / "pipeline_result.json", analysis.model_dump(mode="json"))
    export_assignment(analysis, artifact_root, output_dir)
    return analysis
