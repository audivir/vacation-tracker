"""Test the vacation-tracker module."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import holidays
import msgspec
import pytest

from vacation_tracker import Config, Vacation, add_or_show, cli, new, parse_args, show

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def by_holidays() -> holidays.HolidayBase:
    return holidays.country_holidays("DE", "BY", categories={"public", "catholic"})


@pytest.fixture
def christmas_vacation(by_holidays: holidays.HolidayBase) -> Vacation:
    vac = Vacation("christmas", date(2022, 12, 25), date(2023, 1, 1))
    vac.holidays = by_holidays
    return vac


@pytest.fixture
def assumption_vacation(by_holidays: holidays.HolidayBase) -> Vacation:
    vac = Vacation("assumption", date(2022, 8, 15))
    vac.holidays = by_holidays
    return vac


@pytest.fixture
def config(christmas_vacation: Vacation, assumption_vacation: Vacation) -> Config:
    return Config(2, (2022, 1), (2022, 12), [assumption_vacation, christmas_vacation])


def test_days(christmas_vacation: Vacation, assumption_vacation: Vacation) -> None:
    assert christmas_vacation.days == 4
    assert str(christmas_vacation) == "25.12-01.01 (4): christmas"
    assert assumption_vacation.days == 0
    assert str(assumption_vacation) == "15.08 (0): assumption"


def test_last_date_fail() -> None:
    with pytest.raises(ValueError, match="Last date before first date"):
        Vacation("test", date(2022, 1, 1), date(2021, 12, 31))


@pytest.mark.parametrize("last", [None, "2022-01-02", "2022-01-01"])
def test_last_date(last: str | None) -> None:
    last_date = date.fromisoformat(last) if isinstance(last, str) else None
    assert isinstance(Vacation("test", date(2022, 1, 1), last_date), Vacation)


@pytest.mark.parametrize(
    ("last", "error"), [((2022, 2), False), ((2022, 1), True), ((2021, 12), True)]
)
def test_last_year(last: tuple[int, int], error: bool) -> None:
    if error:
        with pytest.raises(
            ValueError, match="Last year is not greater than first year"
        ):
            Config(2, (2022, 1), last)
    else:
        assert isinstance(Config(2, (2022, 1), last), Config)


def test_set_holidays(by_holidays: holidays.HolidayBase) -> None:
    vac = Vacation("test", date(2022, 1, 1))
    with pytest.raises(ValueError, match="Set holidays first"):
        vac.days  # noqa: B018
    vac.holidays = by_holidays
    assert vac.days == 0


def test_overlapping(christmas_vacation: Vacation) -> None:
    silvester = Vacation("silvester", date(2022, 12, 31))
    conf = Config(2, (2022, 1), (2022, 12), [christmas_vacation, silvester])
    with pytest.raises(ValueError, match="Overlapping vacation days"):
        conf.verify()


def test_outside_tracking(christmas_vacation: Vacation) -> None:
    conf = Config(2, (2023, 1), (2023, 12), [christmas_vacation])
    with pytest.raises(ValueError, match="First vacation day before tracking start"):
        conf.verify()
    conf = Config(2, (2021, 1), (2021, 12), [christmas_vacation])
    with pytest.raises(ValueError, match="Last vacation day after tracking end"):
        conf.verify()


def test_show(config: Config, capsys: pytest.CaptureFixture) -> None:
    show(config)
    outerr = capsys.readouterr()
    summary = " Year  Months  Entitlement  Utilisation  Remaining\n 2022      12           24            4         20\n"  # noqa: E501
    assert outerr.out == summary
    detailed = " ├─ 15.08 (0): assumption\n └─ 25.12-31.12 (4): christmas\n"
    show(config, detailed=True)
    outerr = capsys.readouterr()
    assert outerr.out == summary + detailed


def test_parse_args() -> None:
    with pytest.raises(ValueError, match="Please provide path to a toml file"):
        parse_args(["show", "-f", "test.tom"])


@pytest.fixture
def test_toml(tmp_path: Path) -> Path:
    return tmp_path / "test.toml"


@pytest.mark.parametrize("use_cli", [True, False])
def test_new_fail(use_cli: bool, test_toml: Path) -> None:
    test_toml.touch()

    def func() -> None:
        if use_cli:
            cli(["new", "2", "2022", "1", "-l", "2022", "12", "-f", str(test_toml)])
        else:
            new(test_toml, 2, (2022, 1), (2022, 12))

    with pytest.raises(
        FileExistsError, match="Vacation days toml file exists. Edit file if necessary."
    ):
        func()


@pytest.mark.parametrize("use_cli", [True, False])
def test_new(use_cli: bool, test_toml: Path) -> None:
    if use_cli:
        cli(["new", "2", "2022", "1", "-l", "2022", "12", "-f", str(test_toml)])
    else:
        new(test_toml, 2, (2022, 1), (2022, 12))
    assert test_toml.is_file()
    content = msgspec.toml.decode(test_toml.read_bytes())
    assert content == {
        "days-per-month": 2,
        "first-year": [2022, 1],
        "last-year": [2022, 12],
        "categories": ["public", "catholic"],
        "country": ["DE", "BY"],
        "vacation-periods": [],
    }


@pytest.mark.parametrize("use_cli", [True, False])
def test_add_or_show_fail(use_cli: bool, test_toml: Path) -> None:
    def func() -> None:
        if use_cli:
            cli(["show", "-f", str(test_toml)])
        else:
            add_or_show("show", test_toml)

    with pytest.raises(
        FileNotFoundError,
        match="Vacation days toml file doesnt exists. Call `vacation-tracker new` first.",  # noqa: E501
    ):
        func()


@pytest.mark.parametrize("use_cli", [True, False])
def test_add_or_show_show(
    use_cli: bool, test_toml: Path, capsys: pytest.CaptureFixture
) -> None:
    new(test_toml, 2, (2022, 1), (2022, 12))
    if use_cli:
        cli(["show", "-f", str(test_toml)])
    else:
        add_or_show("show", test_toml, detailed=False)
    summary = " Year  Months  Entitlement  Utilisation  Remaining\n 2022      12           24            0         24\n"  # noqa: E501
    outerr = capsys.readouterr()
    assert outerr.out == summary


@pytest.mark.parametrize("use_cli", [True, False])
def test_add_or_show_add(use_cli: bool, test_toml: Path) -> None:
    new(test_toml, 2, (2022, 1), (2022, 12))
    if use_cli:
        cli(["add", "test", "2022-01-01", "-l", "2022-01-02", "-f", str(test_toml)])
    else:
        add_or_show(
            "add", test_toml, name="test", first="2022-01-01", last="2022-01-02"
        )
    content = msgspec.toml.decode(test_toml.read_bytes())
    assert content == {
        "days-per-month": 2,
        "first-year": [2022, 1],
        "last-year": [2022, 12],
        "categories": ["public", "catholic"],
        "country": ["DE", "BY"],
        "vacation-periods": [
            {"name": "test", "first": date(2022, 1, 1), "last": date(2022, 1, 2)}
        ],
    }
