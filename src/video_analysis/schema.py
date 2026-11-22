"""Typed records shared by every pipeline stage, plus strict JSON helpers.

All times are seconds on the source video's presentation timeline, so shots,
transcript segments, frames, faces and scenes can be joined without conversion.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, type coercion and NaN/inf values."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Video(StrictModel):
    """Source file metadata and the interval selected for analysis."""

    filename: str
    sha256: str
    duration_s: float = Field(gt=0)
    analyzed_start_s: float = Field(ge=0)
    analyzed_end_s: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    timing_tolerance_s: float = Field(gt=0)


class TranscriptSegment(StrictModel):
    """One speech segment recognised by Whisper; citable as evidence."""

    evidence_id: str
    start_s: float
    end_s: float
    text: str


class Frame(StrictModel):
    """One sampled video frame saved to disk; citable as evidence."""

    evidence_id: str
    shot_id: str
    timestamp_s: float
    path: str


class Face(StrictModel):
    """One detected face in one sampled frame, assigned to a similarity group."""

    face_id: str
    shot_id: str
    group_id: str
    timestamp_s: float
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    crop_path: str
    detection_score: float = Field(ge=0, le=1)


class Shot(StrictModel):
    """A continuous camera take between two detected cuts."""

    shot_id: str
    start_s: float
    end_s: float
    scene_id: str | None = None
    face_group_ids: list[str] = Field(default_factory=list)


class Claim(StrictModel):
    """A statement about a scene and the evidence IDs that support it."""

    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Scene(StrictModel):
    """A run of consecutive shots sharing one time, place and action."""

    scene_id: str
    start_s: float
    end_s: float
    shot_ids: list[str] = Field(min_length=1)
    kind: Literal["story", "opening", "credits", "montage", "other"]
    setting: str | None
    description: str
    claims: list[Claim]
    face_group_ids: list[str]
    boundary_reason: str
    needs_review: bool


class Analysis(StrictModel):
    """The complete pipeline result, saved as ``pipeline_result.json``."""

    video: Video
    video_url: str
    settings: dict[str, Any]
    shots: list[Shot]
    scenes: list[Scene]
    faces: list[Face]
    transcript: list[TranscriptSegment]
    frames: list[Frame]


def strict_json_loads(text: str) -> Any:
    """Parse JSON, rejecting duplicate keys and NaN/Infinity, which ``json.loads`` allows."""

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def atomic_json(path: Path, payload: Any) -> None:
    """Write strict JSON via a temporary file so readers never see a partial file."""
    content = json.dumps(payload, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
