"""Checks for entity-level decision rules before the cloud OOF run."""

from scripts.aws.evaluate_set_decision_200k import decide, f05


def test_f05_singletons_and_precision() -> None:
    assert f05(set(), set()) == 1
    assert f05(set(), {"S2-X"}) == 0
    assert f05({"S2-X"}, {"S2-X", "S3-Y"}) == 1.25 / 2.25


def test_source_caps_and_margins() -> None:
    rows = [("S2-X", .96), ("S2-Y", .92), ("S3-Z", .89), ("S3-W", .84)]
    assert decide("base", rows, .83) == {x for x, _ in rows}
    assert decide("cap:1", rows, .83) == {"S2-X", "S3-Z"}
    assert decide("margin:0.03", rows, .83) == {"S2-X", "S3-Z"}
    assert decide("salvage:0.5", rows, .99) == {"S2-X"}
    assert decide("two_source:0.85", rows, .99) == {"S2-X", "S3-Z"}
