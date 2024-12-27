"""Tests for the vacation-tracker module.

This module contains comprehensive tests for the vacation tracking functionality,
including configuration management, vacation period calculations, and CLI operations.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import holidays
import msgspec
import pytest

from vacation_tracker import (
    Config,
    Vacation,
    _track_single_period,
    add_or_show,
    cli,
    new,
    parse_args,
    show,
    track_periods,
)

if TYPE_CHECKING:
    from pathlib import Path


# Fixtures
@pytest.fixture
def bavaria_holidays() -> holidays.HolidayBase:
    """Provide German/Bavarian holiday calendar."""
    return holidays.country_holidays("DE", "BY", categories={"public", "catholic"})


@pytest.fixture
def christmas_holiday(bavaria_holidays: holidays.HolidayBase) -> Vacation:
    """Provide a Christmas holiday vacation period."""
    vacation = Vacation("christmas", date(2022, 12, 25), date(2023, 1, 1))
    vacation.holidays = bavaria_holidays
    return vacation


@pytest.fixture
def assumption_holiday(bavaria_holidays: holidays.HolidayBase) -> Vacation:
    """Provide an Assumption Day holiday vacation period."""
    vacation = Vacation("assumption", date(2022, 8, 15))
    vacation.holidays = bavaria_holidays
    return vacation


@pytest.fixture
def sample_config(christmas_holiday: Vacation, assumption_holiday: Vacation) -> Config:
    """Provide a sample configuration with two vacation periods."""
    return Config(2, (2022, 1), (2023, 12), [assumption_holiday, christmas_holiday])


@pytest.fixture
def expected_summary() -> str:
    """Provide expected summary output format."""
    return """\
 Year  Months  Entitlement  Real Util  Adj Util    Remaining
 2022      12           24          4         4 20 (expired)
 2023      12           24          0         0 24 (expired)
"""


@pytest.fixture
def expected_detailed() -> str:
    """Provide expected detailed output format."""
    return """\
 Year  Months  Entitlement  Real Util  Adj Util    Remaining
 2022      12           24          4         4 20 (expired)
 ├─ 15.08 (0): assumption
 └─ 25.12-31.12 (4): christmas
 2023      12           24          0         0 24 (expired)
 └─ 01.01 (0): christmas
"""


# Test Classes
class TestVacationPeriod:
    """Tests for the Vacation class functionality."""

    def test_vacation_day_counting(
        self, christmas_holiday: Vacation, assumption_holiday: Vacation
    ) -> None:
        """Verify correct counting of vacation days."""
        assert christmas_holiday.days == 4
        assert str(christmas_holiday) == "25.12-01.01 (4): christmas"
        assert assumption_holiday.days == 0
        assert str(assumption_holiday) == "15.08 (0): assumption"

    def test_invalid_date_range(self) -> None:
        """Verify rejection of invalid date ranges."""
        with pytest.raises(ValueError, match="Last date cannot be before first date"):
            Vacation("test", date(2022, 1, 1), date(2021, 12, 31))

    @pytest.mark.parametrize("last", [None, "2022-01-02", "2022-01-01"])
    def test_valid_date_ranges(self, last: str | None) -> None:
        """Verify acceptance of valid date ranges."""
        last_date = date.fromisoformat(last) if isinstance(last, str) else None
        vacation = Vacation("test", date(2022, 1, 1), last_date)
        assert isinstance(vacation, Vacation)

    def test_missing_holidays_calendar(self) -> None:
        """Verify error when holidays calendar is not set."""
        vacation = Vacation("test", date(2022, 1, 1))
        with pytest.raises(
            ValueError, match="Holidays calendar must be set before counting days"
        ):
            _ = vacation.days

    def test_period_verification(self, christmas_holiday: Vacation) -> None:
        """Test vacation period verification rules."""
        with pytest.raises(
            ValueError, match="Vacation periods must not cross year boundaries"
        ):
            christmas_holiday.verify()

        cross_september = Vacation("test", date(2022, 9, 27), date(2022, 10, 27))
        with pytest.raises(
            ValueError, match="Vacation periods must not cross September 30th"
        ):
            cross_september.verify()

        # Test valid period
        first_part, *_ = christmas_holiday.split(date(2022, 12, 31))
        year, expiration_date = first_part.verify()
        assert year == 2022
        assert expiration_date == date(2022, 9, 30)


class TestConfiguration:
    """Tests for the Config class functionality."""

    @pytest.mark.parametrize(
        ("last_date", "should_error"),
        [((2022, 2), False), ((2022, 1), True), ((2021, 12), True)],
    )
    def test_date_range_validation(
        self, last_date: tuple[int, int], should_error: bool
    ) -> None:
        """Test configuration date range validation."""
        if should_error:
            with pytest.raises(ValueError, match="Last year must be after first year"):
                Config(2, (2022, 1), last_date)
        else:
            config = Config(2, (2022, 1), last_date)
            assert isinstance(config, Config)

    def test_overlapping_periods(self, christmas_holiday: Vacation) -> None:
        """Verify rejection of overlapping vacation periods."""
        silvester = Vacation("silvester", date(2022, 12, 31))
        config = Config(2, (2022, 1), (2022, 12), [christmas_holiday, silvester])
        with pytest.raises(ValueError, match="Vacation periods cannot overlap"):
            config.verify()

    def test_period_within_tracking_range(self, christmas_holiday: Vacation) -> None:
        """Verify vacation periods stay within tracking range."""
        # Test period before tracking start
        early_config = Config(2, (2023, 1), (2023, 12), [christmas_holiday])
        with pytest.raises(
            ValueError, match="First vacation day must be after tracking start"
        ):
            early_config.verify()

        # Test period after tracking end
        late_config = Config(2, (2021, 1), (2021, 12), [christmas_holiday])
        with pytest.raises(
            ValueError, match="Last vacation day must be before tracking end"
        ):
            late_config.verify()


class TestVacationTracking:
    """Tests for vacation day tracking functionality."""

    def test_single_period_tracking(self) -> None:
        """Test tracking of single vacation periods."""
        available_days = {2022: 2.5, 2023: 3.5}

        # Test partial use of days
        remaining = _track_single_period(2022, 2, available_days)
        assert remaining == 0
        assert available_days == {2022: 0.5, 2023: 3.5}

        # Test overflow
        remaining = _track_single_period(2022, 2, available_days)
        assert remaining == 1.5
        assert available_days == {2022: 0, 2023: 3.5}

    def test_multiple_period_tracking(self, christmas_holiday: Vacation) -> None:
        """Test tracking of multiple vacation periods."""
        available_days = {2022: 2.5, 2023: 3.5}

        # Track single day
        track_periods(
            [Vacation("test", date(2022, 12, 23), None, christmas_holiday.holidays)],
            available_days,
        )
        assert available_days == {2022: 1.5, 2023: 3.5}

        # Track split period
        splits = christmas_holiday.split(date(2022, 12, 31))
        track_periods(splits, available_days)
        assert available_days == {2022: 0, 2023: 1}

        # Test insufficient days
        insufficient_days = {2022: 1, 2023: 1}
        with pytest.raises(ValueError, match="Insufficient vacation days available"):
            track_periods(splits, insufficient_days)  # type: ignore[arg-type]


class TestDisplay:
    """Tests for vacation tracking display functionality."""

    @pytest.mark.parametrize(
        ("detailed", "expected_fixture"),
        [(True, "expected_detailed"), (False, "expected_summary")],
    )
    def test_display_output(
        self,
        detailed: bool,
        expected_fixture: str,
        sample_config: Config,
        capsys: pytest.CaptureFixture,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test vacation tracking display output."""
        show(sample_config, detailed=detailed)
        output = capsys.readouterr()
        expected = request.getfixturevalue(expected_fixture)
        assert output.out == expected


class TestFileOperations:
    """Tests for file-based operations."""

    @pytest.fixture
    def test_file(self, tmp_path: Path) -> Path:
        """Provide test file path."""
        return tmp_path / "test.toml"

    def test_file_extension_validation(self) -> None:
        """Test validation of configuration file extension."""
        with pytest.raises(
            ValueError, match="Configuration file must have .toml extension"
        ):
            parse_args(["show", "-f", "test.txt"])

    @pytest.mark.parametrize("use_cli", [True, False])
    def test_new_file_creation(self, use_cli: bool, test_file: Path) -> None:
        """Test creation of new configuration file."""
        # Test file already exists
        test_file.touch()
        with pytest.raises(  # noqa: PT012
            FileExistsError,
            match=r"Configuration file .+\.toml already exists\. Edit manually if needed\.",  # noqa: E501
        ):
            if use_cli:
                cli(["new", "2", "2022", "1", "-l", "2022", "12", "-f", str(test_file)])
            else:
                new(test_file, 2, (2022, 1), (2022, 12))

        # Test successful creation
        test_file.unlink()
        if use_cli:
            cli(["new", "2", "2022", "1", "-l", "2022", "12", "-f", str(test_file)])
        else:
            new(test_file, 2, (2022, 1), (2022, 12))

        assert test_file.is_file()
        content = msgspec.toml.decode(test_file.read_bytes())
        assert content == {
            "days-per-month": 2,
            "first-year": [2022, 1],
            "last-year": [2022, 12],
            "categories": ["public", "catholic"],
            "country": ["DE", "BY"],
            "vacation-periods": [],
        }

    @pytest.mark.parametrize("use_cli", [True, False])
    def test_file_operations(
        self, use_cli: bool, test_file: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Test file operations (add/show)."""
        # Test missing file
        with pytest.raises(  # noqa: PT012
            FileNotFoundError,
            match=r"Configuration file .+\.toml not found\. Use 'vacation-tracker new' to create\.",  # noqa: E501
        ):
            if use_cli:
                cli(["show", "-f", str(test_file)])
            else:
                add_or_show("show", test_file)

        # Test show operation
        new(test_file, 2, (2022, 1), (2022, 12))
        if use_cli:
            cli(["show", "-f", str(test_file)])
        else:
            add_or_show("show", test_file, detailed=False)

        expected_output = """\
 Year  Months  Entitlement  Real Util  Adj Util    Remaining
 2022      12           24          0         0 24 (expired)
"""
        output = capsys.readouterr()
        assert output.out == expected_output

        # Test add operation
        if use_cli:
            cli(["add", "test", "2022-01-01", "-l", "2022-01-02", "-f", str(test_file)])
        else:
            add_or_show(
                "add", test_file, name="test", first="2022-01-01", last="2022-01-02"
            )

        content = msgspec.toml.decode(test_file.read_bytes())
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


@pytest.mark.parametrize(
    ("split_date", "expected_dates"),
    [
        (
            date(2022, 12, 31),
            [
                (date(2022, 12, 25), date(2022, 12, 31)),
                (date(2023, 1, 1), date(2023, 1, 1)),
            ],
        ),
        (date(2023, 1, 1), [(date(2022, 12, 25), date(2023, 1, 1))]),
    ],
)
def test_period_splitting(
    split_date: date,
    expected_dates: list[tuple[date, date]],
    christmas_holiday: Vacation,
) -> None:
    """Test splitting of vacation periods at specific dates."""
    split_periods = christmas_holiday.split(split_date)
    actual_dates = [(period.first, period.real_last) for period in split_periods]
    assert actual_dates == expected_dates
