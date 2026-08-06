import pytest

from ordo.event import Event


def test_starts_untriggered():
    ev = Event()
    assert ev.triggered is False


def test_succeed_sets_triggered_and_value():
    ev = Event()
    ev.succeed(42)
    assert ev.triggered is True
    assert ev.ok is True
    assert ev.value == 42


def test_succeed_twice_raises():
    ev = Event()
    ev.succeed(1)
    with pytest.raises(RuntimeError):
        ev.succeed(2)


def test_fail_sets_triggered_and_exception():
    ev = Event()
    exc = ValueError("boom")
    ev.fail(exc)
    assert ev.triggered is True
    assert ev.ok is False
    assert ev.exception is exc


def test_fail_twice_raises():
    ev = Event()
    ev.fail(ValueError("first"))
    with pytest.raises(RuntimeError):
        ev.fail(ValueError("second"))
