import pytest

from video_analysis.evaluation import boundary_metrics, grouping_metrics


def test_boundaries_are_matched_one_to_one_within_tolerance():
    result = boundary_metrics([1.0, 1.04, 5.0], [1.02, 3.0], tolerance_s=0.05)
    assert (result["matched"], result["precision"], result["recall"]) == (1, 1 / 3, 0.5)
    assert result["mean_abs_error_s"] == pytest.approx(0.02)


def test_boundary_edge_cases():
    assert boundary_metrics([], [], 0)["f1"] == 1
    assert boundary_metrics([], [1.0], 0)["f1"] == 0
    with pytest.raises(ValueError):
        boundary_metrics([], [], -1)


def test_grouping_metrics_penalise_merges_and_splits():
    people = ["ann", "ann", "ann", "bob", "bob", "cat"]
    groups = ["g1", "g1", "g2", "g3", "g3", "g3"]  # ann is split; bob and cat are merged
    result = grouping_metrics(people, groups)
    assert result["purity"] == pytest.approx(5 / 6)
    assert result["inverse_purity"] == pytest.approx(5 / 6)
    assert (result["faces"], result["people"], result["groups"]) == (6, 3, 3)
    assert grouping_metrics(["ann", "bob"], ["g1", "g2"])["purity"] == 1


def test_grouping_metrics_need_aligned_labels():
    with pytest.raises(ValueError):
        grouping_metrics(["ann"], [])
