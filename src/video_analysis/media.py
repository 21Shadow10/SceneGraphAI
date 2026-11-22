"""Video decoding, shot detection, frame sampling and speech transcription."""

import bisect
import hashlib
import subprocess
import warnings
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
from scenedetect.detectors import AdaptiveDetector

from .schema import Shot, TranscriptSegment, Video


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path, start_s: float = 0.0, end_s: float | None = None) -> Video:
    """Read stream metadata and validate the interval to analyse.

    ``end_s=None`` analyses to the end of the file. The SHA-256 of the file is
    recorded so cached results are tied to the exact input.
    """
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name} has no video stream")
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            # WebM/MKV files often only store the duration at container level.
            duration = container.duration / av.time_base
        else:
            raise ValueError(f"{path.name} has no duration; remux it with FFmpeg first")
        end = duration if end_s is None else min(end_s, duration)
        if not 0 <= start_s < end:
            raise ValueError(f"Invalid analysis interval {start_s}-{end} s")
        return Video(
            filename=path.name, sha256=_sha256(path), duration_s=duration,
            analyzed_start_s=float(start_s), analyzed_end_s=float(end),
            width=stream.width, height=stream.height,
            timing_tolerance_s=max(float(stream.time_base), 1e-6),
        )


def decoded_frames(path: Path, start_s: float, end_s: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, bgr_image)`` for every frame in ``[start_s, end_s)``.

    Timestamps come from each frame's presentation time rather than
    ``index / fps``, so variable-frame-rate files stay accurate.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp >= end_s:
                break
            if timestamp >= start_s:
                yield timestamp, frame.to_ndarray(format="bgr24")


def detect_shots(path: Path, video: Video, threshold: float = 3.0, min_shot_frames: int = 3) -> list[Shot]:
    """Split the analysed interval into shots at detected camera cuts.

    PySceneDetect's ``AdaptiveDetector`` compares each frame's content change
    with a rolling average of its neighbours, which tolerates camera motion
    better than a fixed threshold. The returned shots tile the interval exactly.
    """
    detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_shot_frames)
    timestamps: list[float] = []
    cut_indices: list[int] = []
    for index, (timestamp, image) in enumerate(
        decoded_frames(path, video.analyzed_start_s, video.analyzed_end_s)
    ):
        timestamps.append(timestamp)
        height, width = image.shape[:2]
        if width > 640:  # detection quality is unchanged at this size and it is much faster
            image = cv2.resize(image, (640, round(height * 640 / width)))
        cut_indices.extend(detector.process_frame(index, image))
    if not timestamps:
        raise ValueError("The analysis interval contains no decodable frames")
    cut_indices.extend(detector.post_process(len(timestamps)))

    cuts = sorted({timestamps[index] for index in cut_indices if 0 < index < len(timestamps)})
    boundaries = [video.analyzed_start_s, *cuts, video.analyzed_end_s]
    return [
        Shot(shot_id=f"shot_{number:04d}", start_s=start, end_s=end)
        for number, (start, end) in enumerate(pairwise(boundaries), start=1)
    ]


def sample_shots(path: Path, shots: list[Shot], sample_fps: float) -> Iterator[tuple[str, float, np.ndarray]]:
    """Yield ``(shot_id, timestamp_s, bgr_image)`` at about ``sample_fps`` per shot.

    Sampling restarts at every cut, half an interval into the shot, so samples
    avoid transition frames. Shots too short to reach their first sample point
    still yield their first frame, guaranteeing at least one image per shot.
    """
    if not sample_fps > 0:
        raise ValueError("sample_fps must be positive")
    starts = [shot.start_s for shot in shots]
    current: Shot | None = None
    first_frame: tuple[float, np.ndarray] | None = None
    sampled = False
    next_sample_s = 0.0
    for timestamp, image in decoded_frames(path, shots[0].start_s, shots[-1].end_s):
        shot = shots[bisect.bisect_right(starts, timestamp) - 1]
        if shot is not current:
            if current is not None and not sampled:
                yield current.shot_id, *first_frame
            current, first_frame, sampled = shot, (timestamp, image), False
            next_sample_s = shot.start_s + min(0.5 / sample_fps, (shot.end_s - shot.start_s) / 2)
        if timestamp >= next_sample_s:
            yield shot.shot_id, timestamp, image
            sampled = True
            next_sample_s = timestamp + 1 / sample_fps
    if current is not None and not sampled:
        yield current.shot_id, *first_frame


def transcribe(path: Path, video: Video, work_dir: Path, model_name: str = "small",
               cpu_threads: int = 4) -> list[TranscriptSegment]:
    """Transcribe the analysed interval locally with faster-whisper on the CPU.

    Returns an empty transcript, with a warning, when the file has no audio
    track (common for video-only YouTube downloads). Whisper weights are
    downloaded from Hugging Face on first use.
    """
    with av.open(str(path)) as container:
        has_audio = bool(container.streams.audio)
    if not has_audio:
        warnings.warn(f"{path.name} has no audio track; continuing without a transcript", stacklevel=2)
        return []

    from faster_whisper import WhisperModel  # slow import, only needed for this stage

    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(path),
        "-ss", str(video.analyzed_start_s), "-t", str(video.analyzed_end_s - video.analyzed_start_s),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg could not extract audio: {completed.stderr.strip()}")

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    segments, _ = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
    transcript = []
    for segment in segments:
        # Extracted audio starts at 0, so shift back onto the source timeline.
        start = video.analyzed_start_s + float(segment.start)
        end = min(video.analyzed_end_s, video.analyzed_start_s + float(segment.end))
        text = segment.text.strip()
        if end > start and text:
            transcript.append(TranscriptSegment(
                evidence_id=f"text_{len(transcript) + 1:04d}", start_s=start, end_s=end, text=text,
            ))
    return transcript
