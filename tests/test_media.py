from pathlib import Path

import av
import numpy as np
import pytest

from video_analysis.media import detect_shots, probe, sample_shots, transcribe


def make_video(path: Path) -> None:
    """Write a silent 3 s, 24 fps clip: 1.5 s of red then 1.5 s of green."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for index in range(72):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:] = (0, 0, 200) if index < 36 else (0, 200, 0)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    return path


def test_probe_reads_metadata_and_rejects_bad_interval(video_path):
    video = probe(video_path)
    assert (video.width, video.height) == (160, 120)
    assert video.duration_s == pytest.approx(3.0)
    assert video.analyzed_end_s == video.duration_s
    with pytest.raises(ValueError, match="interval"):
        probe(video_path, 5.0)


def test_detects_the_cut_and_samples_every_shot(video_path):
    shots = detect_shots(video_path, probe(video_path))
    assert len(shots) == 2
    assert shots[0].end_s == pytest.approx(1.5, abs=1 / 24)
    assert shots[0].end_s == shots[1].start_s
    assert shots[-1].end_s == pytest.approx(3.0)
    samples = list(sample_shots(video_path, shots, 2.0))
    assert {shot_id for shot_id, _, _ in samples} == {shot.shot_id for shot in shots}
    assert len(samples) == 6


def test_clip_offset_without_a_cut(video_path):
    shots = detect_shots(video_path, probe(video_path, 1.75, 2.75))
    assert [(shot.start_s, shot.end_s) for shot in shots] == [(1.75, 2.75)]
    assert all(1.75 <= timestamp < 2.75 for _, timestamp, _ in sample_shots(video_path, shots, 2.0))


def test_shot_shorter_than_the_sampling_interval_still_gets_a_frame(video_path):
    shots = detect_shots(video_path, probe(video_path, 1.0, 1.02))
    assert len(list(sample_shots(video_path, shots, 2.0))) == 1


def test_video_without_audio_gives_an_empty_transcript(video_path, tmp_path):
    with pytest.warns(UserWarning, match="no audio"):
        assert transcribe(video_path, probe(video_path), tmp_path / "audio") == []
