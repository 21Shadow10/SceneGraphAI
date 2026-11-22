import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from video_analysis.scenes import (
    MAX_ATTEMPTS,
    MAX_RATE_LIMIT_RETRIES,
    RequestBudgetError,
    SemanticAnalyzer,
    SemanticError,
)
from video_analysis.schema import Frame, Shot, TranscriptSegment


class RateLimited(Exception):
    status_code = 429


class MockChatModel:
    """Mimics ``ChatOpenAI.with_structured_output(...).invoke`` and records each request.

    ``respond(request)`` returns a dict (valid JSON), a str (unparseable output)
    or an exception to raise.
    """

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def with_structured_output(self, schema, **options):
        assert "title" in schema
        assert options == {"method": "json_schema", "strict": True, "include_raw": True}
        return SimpleNamespace(invoke=lambda messages, config: self._invoke(schema, messages, config))

    def _invoke(self, schema, messages, config):
        content = messages[1].content
        request = {
            "schema": schema,
            "config": config,
            "evidence": json.loads(content[0]["text"])["evidence"],
            "images": [item["image_url"]["url"] for item in content if item["type"] == "image_url"],
        }
        self.requests.append(request)
        answer = self.respond(request)
        if isinstance(answer, Exception):
            raise answer
        raw = SimpleNamespace(id="resp_1", usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20})
        if isinstance(answer, str):
            return {"parsed": None, "parsing_error": ValueError(answer), "raw": raw}
        return {"parsed": answer, "parsing_error": None, "raw": raw}


def make_evidence(tmp_path, count):
    Image.new("RGB", (32, 24), "green").save(tmp_path / "frame.jpg")
    shots = [Shot(shot_id=f"shot_{n}", start_s=float(n), end_s=float(n + 1)) for n in range(count)]
    frames = [Frame(evidence_id=f"frame_{n}", shot_id=f"shot_{n}", timestamp_s=n + 0.5, path="frame.jpg")
              for n in range(count)]
    return shots, frames


def answer(request):
    """A valid answer: no scene boundaries, and a summary citing the first frame."""
    evidence = request["evidence"]
    if "allowed_boundary_ids" in evidence:
        return {"decisions": [{"after_shot_id": shot_id, "is_boundary": False, "reason": "Same room",
                               "needs_review": False} for shot_id in evidence["allowed_boundary_ids"]]}
    return {"kind": "story", "setting": None, "description": "A green wall is shown.",
            "claims": [{"text": "The wall is green.", "evidence_ids": [evidence["frames"][0]["evidence_id"]]}],
            "needs_review": False}


def analyzer(tmp_path, respond=answer, **options):
    model = MockChatModel(respond)
    return SemanticAnalyzer("test-model", tmp_path / "cache", chat_model=model,
                            min_interval_s=0, retry_delay_s=0, **options), model


def is_boundary_request(request):
    return "allowed_boundary_ids" in request["evidence"]


def test_boundary_windows_decide_every_cut_once_and_answers_are_cached(tmp_path):
    shots, frames = make_evidence(tmp_path, 17)
    semantic, model = analyzer(tmp_path)
    scenes = semantic.analyze(shots, [], frames, tmp_path)
    decided = [shot_id for request in model.requests if is_boundary_request(request)
               for shot_id in request["evidence"]["allowed_boundary_ids"]]
    assert decided == [shot.shot_id for shot in shots[:-1]]
    assert [scene.shot_ids for scene in scenes] == [[shot.shot_id for shot in shots]]
    assert semantic.requests_made == 4  # three boundary windows and one summary

    assert semantic.analyze(shots, [], frames, tmp_path) == scenes
    assert (semantic.requests_made, semantic.cache_hits) == (4, 4)


def test_multiple_scenes_carry_reasons_kinds_groups_and_dialogue(tmp_path):
    shots, frames = make_evidence(tmp_path, 10)
    shots[0].face_group_ids = ["group_002"]
    shots[2].face_group_ids = ["group_001", "group_002"]
    transcript = [TranscriptSegment(evidence_id=f"text_{n}", start_s=n * 0.5, end_s=n * 0.5 + 0.5, text="Line")
                  for n in range(20)]

    def respond(request):
        result = answer(request)
        if is_boundary_request(request):
            for decision in result["decisions"]:
                decision["is_boundary"] = decision["after_shot_id"] in {"shot_2", "shot_7"}
                decision["reason"] = "Time jump" if decision["is_boundary"] else "Same room"
                decision["needs_review"] = decision["after_shot_id"] == "shot_7"
        elif request["evidence"]["shots"][0]["shot_id"] == "shot_8":
            result["kind"] = "credits"
        return result

    semantic, model = analyzer(tmp_path, respond, max_images=4)
    scenes = semantic.analyze(shots, transcript, frames, tmp_path)
    assert [scene.shot_ids for scene in scenes] == [
        ["shot_0", "shot_1", "shot_2"], [f"shot_{n}" for n in range(3, 8)], ["shot_8", "shot_9"]]
    assert [(scene.start_s, scene.end_s) for scene in scenes] == [(0.0, 3.0), (3.0, 8.0), (8.0, 10.0)]
    assert scenes[0].face_group_ids == ["group_001", "group_002"]
    assert [scene.boundary_reason for scene in scenes] == ["Start of the analysed clip", "Time jump", "Time jump"]
    assert [scene.kind for scene in scenes] == ["story", "story", "credits"]
    assert [scene.needs_review for scene in scenes] == [False, False, True]

    summaries = [request["evidence"] for request in model.requests if not is_boundary_request(request)]
    assert [segment["evidence_id"] for summary in summaries for segment in summary["transcript"]] == \
        [segment.evidence_id for segment in transcript]
    assert [frame["evidence_id"] for frame in summaries[1]["frames"]] == ["frame_3", "frame_4", "frame_5", "frame_7"]


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "invented_id", "wrong_type", "extra_field", "blank_reason",
])
def test_invalid_boundary_answers_are_retried_and_never_cached(tmp_path, mutation):
    shots, frames = make_evidence(tmp_path, 3)

    def respond(request):
        result = answer(request)
        if is_boundary_request(request):
            decisions = result["decisions"]
            if mutation == "missing":
                decisions.pop()
            elif mutation == "duplicate":
                decisions[1] = decisions[0]
            elif mutation == "invented_id":
                decisions[0]["after_shot_id"] = "shot_99"
            elif mutation == "wrong_type":
                decisions[0]["is_boundary"] = "no"
            elif mutation == "extra_field":
                decisions[0]["confidence"] = 0.9
            else:
                decisions[0]["reason"] = " "
        return result

    semantic, model = analyzer(tmp_path, respond)
    with pytest.raises(SemanticError, match=f"after {MAX_ATTEMPTS} attempts"):
        semantic.analyze(shots, [], frames, tmp_path)
    assert semantic.requests_made == len(model.requests) == MAX_ATTEMPTS
    assert not list((tmp_path / "cache").glob("*.json"))


@pytest.mark.parametrize("mutation", [
    "blank_description", "no_claims", "blank_claim", "unknown_id", "other_scene_frame",
])
def test_summaries_must_cite_evidence_from_their_own_scene(tmp_path, mutation):
    shots, frames = make_evidence(tmp_path, 2)

    def respond(request):
        result = answer(request)
        if is_boundary_request(request):
            result["decisions"][0]["is_boundary"] = True
        elif mutation == "blank_description":
            result["description"] = " "
        elif mutation == "no_claims":
            result["claims"] = []
        elif mutation == "blank_claim":
            result["claims"][0]["text"] = ""
        else:
            result["claims"][0]["evidence_ids"] = ["frame_99" if mutation == "unknown_id" else "frame_1"]
        return result

    semantic, _ = analyzer(tmp_path, respond)
    with pytest.raises(SemanticError):
        semantic.analyze(shots, [], frames, tmp_path)


def test_a_retry_can_recover(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    answers = iter(["truncated", RuntimeError("HTTP 429"), None])
    semantic, _ = analyzer(tmp_path, lambda request: next(answers) or answer(request))
    assert semantic.analyze(shots, [], frames, tmp_path)
    assert semantic.requests_made == 3


def test_request_budget_includes_retries(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    semantic, model = analyzer(tmp_path, lambda request: "not json", max_requests=2)
    with pytest.raises(RequestBudgetError):
        semantic.analyze(shots, [], frames, tmp_path)
    assert len(model.requests) == 2


def test_rate_limits_are_waited_out_without_using_attempts_or_budget(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    answers = iter([RateLimited(), RateLimited(), "bad json", RateLimited(), "bad json", None])
    semantic, model = analyzer(tmp_path, lambda request: next(answers) or answer(request), max_requests=3)
    assert semantic.analyze(shots, [], frames, tmp_path)
    assert (len(model.requests), semantic.requests_made) == (6, 3)


def test_persistent_rate_limit_gives_a_clear_error(tmp_path):
    semantic, model = analyzer(tmp_path, lambda request: RateLimited())
    with pytest.raises(SemanticError, match="rate-limited"):
        semantic.preflight()
    assert len(model.requests) == MAX_RATE_LIMIT_RETRIES + 1
    assert semantic.requests_made == 0


def test_oversized_requests_fail_before_calling_the_model(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    transcript = [TranscriptSegment(evidence_id="long", start_s=0.0, end_s=1.0, text="x" * 60_001)]
    semantic, model = analyzer(tmp_path)
    with pytest.raises(SemanticError, match="too large"):
        semantic.analyze(shots, transcript, frames, tmp_path)
    assert model.requests == []


def test_every_shot_needs_a_frame(tmp_path):
    shots, frames = make_evidence(tmp_path, 2)
    with pytest.raises(ValueError, match="Every shot"):
        analyzer(tmp_path)[0].analyze(shots, [], frames[:1], tmp_path)


def test_images_are_capped_and_downscaled(tmp_path):
    shots, frames = make_evidence(tmp_path, 3)
    Image.new("RGB", (1800, 1200), "blue").save(tmp_path / "frame.jpg")
    semantic, model = analyzer(tmp_path, max_images=2)
    semantic.analyze(shots, [], frames, tmp_path)
    for request in model.requests:
        assert 1 <= len(request["images"]) <= 2
        for url in request["images"]:
            with Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))) as image:
                assert image.format == "JPEG" and max(image.size) == 768


def test_cache_key_covers_images_and_model(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    model = MockChatModel(answer)

    def run(model_id):
        semantic = SemanticAnalyzer(model_id, tmp_path / "cache", chat_model=model)
        semantic.analyze(shots, [], frames, tmp_path)
        return semantic

    assert run("model-a").requests_made == 1
    assert run("model-a").cache_hits == 1
    Image.new("RGB", (32, 24), "red").save(tmp_path / "frame.jpg")
    assert run("model-a").requests_made == 1
    assert run("model-b").requests_made == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 3


def test_preflight_sends_a_green_jpeg_and_is_never_cached(tmp_path):
    semantic, model = analyzer(tmp_path, lambda request: {"color": "green"})
    semantic.preflight()
    semantic.preflight()
    assert semantic.requests_made == 2
    with Image.open(io.BytesIO(base64.b64decode(model.requests[0]["images"][0].split(",")[1]))) as image:
        red, green, blue = image.getpixel((8, 8))
    assert green > red + 100 and green > blue + 100


def test_preflight_rejects_a_wrong_answer(tmp_path):
    semantic, _ = analyzer(tmp_path, lambda request: {"color": "blue"})
    with pytest.raises(SemanticError):
        semantic.preflight()


def test_openrouter_client_is_created_lazily_with_strict_routing(tmp_path, monkeypatch):
    import video_analysis.scenes as scenes

    created = []

    class FakeChatOpenAI(MockChatModel):
        def __init__(self, **options):
            created.append(options)
            super().__init__(lambda request: {"color": "green"})

    monkeypatch.setattr(scenes, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    semantic = SemanticAnalyzer("test-model", tmp_path / "cache")
    assert created == []
    semantic.preflight()
    assert created == [{
        "model": "test-model", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-test",
        "use_responses_api": True, "max_completion_tokens": 3000, "max_retries": 0, "timeout": 120,
        "extra_body": {"provider": {"require_parameters": True, "allow_fallbacks": False}},
    }]


def test_missing_api_key_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SemanticError, match="OPENROUTER_API_KEY"):
        SemanticAnalyzer("test-model", tmp_path / "cache").preflight()


def test_error_messages_and_traces_never_contain_secrets_or_content(tmp_path):
    shots, frames = make_evidence(tmp_path, 1)
    transcript = [TranscriptSegment(evidence_id="text_1", start_s=0.1, end_s=0.9, text="private dialogue")]
    semantic, _ = analyzer(tmp_path)
    semantic.analyze(shots, transcript, frames, tmp_path)

    failing, _ = analyzer(tmp_path, lambda request: RuntimeError("Bearer sk-secret"))
    with pytest.raises(SemanticError) as caught:
        failing.preflight()
    assert "sk-secret" not in str(caught.value)

    records = [json.loads(line) for line in semantic.trace_path.read_text().splitlines()]
    success = next(record for record in records if record["status"] == "success")
    assert success["stage"] == "scene_summary"
    assert success["evidence_ids"] == ["text_1", "frame_0"]
    assert success["usage"]["total_tokens"] == 20
    assert {record["status"] for record in records} == {"success", "error"}
    log = semantic.trace_path.read_text()
    assert "private dialogue" not in log and "sk-secret" not in log and "base64" not in log


@pytest.mark.parametrize("options", [{"max_requests": -1}, {"max_images": 1}, {"max_completion_tokens": 0}])
def test_invalid_limits_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        SemanticAnalyzer("test-model", tmp_path, **options)
