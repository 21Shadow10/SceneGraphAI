import pytest
from pydantic import ValidationError

from video_analysis.schema import Video, atomic_json, strict_json_loads


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers(text):
    with pytest.raises(ValueError):
        strict_json_loads(text)


def test_models_reject_type_coercion_and_unknown_fields():
    fields = dict(filename="v.mp4", sha256="x", duration_s=1.0, analyzed_start_s=0.0,
                  analyzed_end_s=1.0, width=640, height=360, timing_tolerance_s=0.001)
    Video(**fields)
    with pytest.raises(ValidationError):
        Video(**{**fields, "width": "640"})
    with pytest.raises(ValidationError):
        Video(**fields, fps=25)


def test_atomic_json_round_trips_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "nested" / "result.json"
    atomic_json(path, {"values": [1, 2.5]})
    assert strict_json_loads(path.read_text()) == {"values": [1, 2.5]}
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert strict_json_loads(path.read_text()) == {"values": [1, 2.5]}
    assert not list(path.parent.glob("*.tmp"))
