"""Track and manage vacation periods with support for holidays and date ranges.

This module provides functionality to track vacation periods, calculate available days,
and handle holiday schedules across different countries. It supports command-line
interface operations for creating new tracking periods, adding vacation entries,
and displaying vacation summaries.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import holidays
import msgspec
import pandas as pd
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

logger = logging.getLogger(__name__)

# Constants
TREE_MARKER = " ├─"
TREE_LAST = " └─"
WEEKEND_DAYS = (5, 6)  # Saturday and Sunday

MonthSpecifier: TypeAlias = Annotated[int, msgspec.Meta(ge=1, le=12)]
"""Integer between 1 and 12 (including)."""


class Vacation(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Represents a single vacation period with start and end dates.

    This class handles vacation period validation, holiday checking, and day counting.

    Raises:
        ValueError: If last date is before first date
    """

    name: str
    """Name or description of the vacation period"""
    first: date
    """First day of the vacation period"""
    last: date | None = None
    """Last day of the vacation period (optional)"""
    holidays: holidays.HolidayBase | None = None
    """Holiday calendar for day counting"""

    def __post_init__(self) -> None:
        """Validate vacation period dates."""
        if self.last and self.last < self.first:
            raise ValueError("Last date cannot be before first date")

    @property
    def days(self) -> int:
        """Calculate the number of working days in the vacation period.

        Returns:
            Number of working days (excluding weekends and holidays)

        Raises:
            ValueError: If holidays calendar is not set
        """
        if self.holidays is None:
            raise ValueError("Holidays calendar must be set before counting days")

        delta = self.real_last - self.first
        counter = 0

        for add in range(delta.days + 1):
            current_date = self.first + timedelta(days=add)

            if current_date.weekday() in WEEKEND_DAYS:
                logger.debug("%s skipped: weekend", current_date.strftime("%d.%m.%Y"))
                continue

            if current_date in self.holidays:
                logger.debug(
                    "%s skipped: %s",
                    current_date.strftime("%d.%m.%Y"),
                    self.holidays[current_date],
                )
                continue

            counter += 1

        return counter

    @property
    def real_last(self) -> date:
        """Get the actual end date of the vacation period.

        Returns:
            The last date if set, otherwise the first date
        """
        return self.first if self.last is None else self.last

    @override
    def __str__(self) -> str:
        """Format vacation period as a string with dates and days count."""
        first_day = self.first.strftime("%d.%m")
        last_day = self.real_last.strftime("%d.%m")

        if self.first == self.real_last:
            return f"{first_day} ({self.days}): {self.name}"

        return f"{first_day}-{last_day} ({self.days}): {self.name}"

    def split(self, on: date) -> tuple[Vacation] | tuple[Vacation, Vacation]:
        """Split the vacation period at the specified date.

        Args:
            on: Date to split the vacation period

        Returns:
            Tuple containing either the original vacation (if no split needed)
            or two new vacation periods
        """
        if on < self.first or on >= self.real_last:
            return (self,)

        before = Vacation(self.name, self.first, on, self.holidays)
        after = Vacation(
            self.name, on + timedelta(days=1), self.real_last, self.holidays
        )
        return before, after

    def verify(self) -> tuple[int, date]:
        """Verify vacation period constraints.

        Checks that the period:
        - Stays within the same year
        - Doesn't cross September 30th

        Returns:
            Tuple of (year, expiration_date)

        Raises:
            ValueError: If period crosses year boundary or September 30th
        """
        year = self.first.year
        if year != self.real_last.year:
            raise ValueError("Vacation periods must not cross year boundaries")

        expiration_date = date(year, 9, 30)
        if self.first <= expiration_date and self.real_last > expiration_date:
            raise ValueError("Vacation periods must not cross September 30th")

        return year, expiration_date


class Config(
    msgspec.Struct,
    forbid_unknown_fields=True,
    rename={
        "days_per_month": "days-per-month",
        "first_year": "first-year",
        "last_year": "last-year",
        "vacation_periods": "vacation-periods",
    },
):
    """Configuration for vacation tracking."""

    days_per_month: float | int
    """Vacation days earned per month"""
    first_year: tuple[int, MonthSpecifier]
    """(year, month) tuple for start of tracking"""
    last_year: tuple[int, MonthSpecifier]
    """(year, month) tuple for end of tracking"""
    vacation_periods: list[Vacation] = []
    """List of vacation periods to track"""
    country: str | tuple[str, str | None] = ("DE", "BY")
    """Country code(s) for holiday calendar"""
    categories: tuple[str, ...] = ("public", "catholic")
    """Categories of holidays to include"""

    @property
    def holidays(self) -> holidays.HolidayBase:
        """Get holiday calendar for the configured country."""
        if isinstance(self.country, str):  # pragma: no cover
            self.country = (self.country, None)
        return holidays.country_holidays(*self.country, categories=self.categories)

    def __post_init__(self) -> None:
        """Validate configuration dates."""
        if self.last_year <= self.first_year:
            raise ValueError("Last year must be after first year")

    def verify(self) -> None:
        """Verify vacation periods for overlaps and date range constraints."""
        self.vacation_periods.sort(key=lambda x: x.first)

        prev_last = date.min
        for period in self.vacation_periods:
            if period.first <= prev_last:
                raise ValueError("Vacation periods cannot overlap")
            prev_last = period.real_last

        if not self.vacation_periods:
            return

        first_period = self.vacation_periods[0]
        last_period = self.vacation_periods[-1]

        if first_period.first < date(*self.first_year, 1):
            raise ValueError("First vacation day must be after tracking start")

        last_tracking_date = (
            date(self.last_year[0] + 1, 1, 1)
            if self.last_year[1] == 12  # noqa: PLR2004
            else date(self.last_year[0], self.last_year[1] + 1, 1)
        )

        if last_period.real_last >= last_tracking_date:
            raise ValueError("Last vacation day must be before tracking end")

    def _get_entitlement(self) -> dict[int, tuple[float | int, int]]:
        """Calculate vacation day entitlements per year.

        Returns:
            Dictionary mapping years to (entitlement, months) tuples
        """
        first_year, first_month = self.first_year
        last_year, last_month = self.last_year
        entitlements: dict[int, tuple[float | int, int]] = {}

        for year in range(first_year, last_year + 1):
            curr_first_month = first_month if year == first_year else 1
            curr_last_month = last_month if year == last_year else 12
            months = 1 + curr_last_month - curr_first_month

            entitlement = months * self.days_per_month
            if (
                isinstance(entitlement, float) and entitlement.is_integer()
            ):  # pragma: no cover
                entitlement = int(entitlement)

            entitlements[year] = (entitlement, months)

        return entitlements

    def _order_periods(self) -> tuple[list[Vacation], dict[int, list[Vacation]]]:
        """Order and split vacation periods by year.

        Returns:
            Tuple of (all_periods, periods_by_year)
        """
        self.verify()

        periods_by_year: dict[int, list[Vacation]] = defaultdict(list)
        for period in self.vacation_periods:
            period.holidays = self.holidays
            current: Vacation | None = period

            for year in range(period.first.year, period.real_last.year + 1):
                if current is None:  # pragma: no cover
                    raise RuntimeError("Unexpected None period during splitting")

                year_splits = current.split(date(year, 12, 31))
                if not 1 <= len(year_splits) <= 2:  # pragma: no cover # noqa: PLR2004
                    raise RuntimeError("Invalid number of period splits")

                this_year, current = (
                    year_splits if len(year_splits) == 2 else (year_splits[0], None)  # noqa: PLR2004
                )
                periods_by_year[year].extend(this_year.split(date(year, 9, 30)))

        all_periods = [
            period
            for year_periods in periods_by_year.values()
            for period in year_periods
        ]
        return all_periods, periods_by_year


def _track_single_period(
    year: int, days: float, track_dict: MutableMapping[int, float | int]
) -> float | int:
    """Track vacation days for a single year.

    Args:
        year: Year to track days for
        days: Number of days to track
        track_dict: Dictionary of available days by year

    Returns:
        Remaining days to track
    """
    if (available := track_dict.get(year, 0)) > 0:
        if available < days:
            used_days = available
            days -= available
        else:
            used_days = days
            days = 0
        track_dict[year] -= used_days

    return days


def track_periods(
    periods: Sequence[Vacation], track_dict: MutableMapping[int, float | int]
) -> None:
    """Track multiple vacation periods against available days.

    Args:
        periods: Sequence of vacation periods to track
        track_dict: Dictionary of available days by year

    Raises:
        ValueError: If there are insufficient vacation days
    """
    for period in periods:
        year, expiration_date = period.verify()
        first_year = year - 1 if period.real_last <= expiration_date else year

        remaining_days: float | int = period.days
        for tracking_year in range(first_year, max(track_dict) + 1):
            remaining_days = _track_single_period(
                tracking_year, remaining_days, track_dict
            )
            if remaining_days == 0:
                break
        else:
            raise ValueError("Insufficient vacation days available")


def show(config: Config, detailed: bool = False) -> None:
    """Display vacation period summary and details.

    Args:
        config: Vacation tracking configuration
        detailed: Whether to show individual vacation periods
    """
    periods, periods_by_year = config._order_periods()  # noqa: SLF001
    entitlements = config._get_entitlement()  # noqa: SLF001

    remaining = {year: entitlement for year, (entitlement, _) in entitlements.items()}

    track_periods(periods, remaining)

    rows: list[
        tuple[int, int, float | int, float | int, float | int, float | int | str]
    ] = []

    now = datetime.now(tz=timezone.utc)
    threshold_year = now.year - (1 if now.date() > date(now.year, 9, 30) else 2)

    for year, (entitlement, months) in entitlements.items():
        remaining_days = remaining[year]
        utilized = sum(day.days for day in periods_by_year[year])
        adjusted = entitlement - remaining_days

        remaining_str = (
            f"{remaining_days} (expired)"
            if year <= threshold_year and remaining_days > 0
            else remaining_days
        )

        rows.append((year, months, entitlement, utilized, adjusted, remaining_str))

    columns = ("Year", "Months", "Entitlement", "Real Util", "Adj Util", "Remaining")
    show_df = pd.DataFrame(rows, columns=columns)

    lines = show_df.to_string(index=False).splitlines()
    print(lines[0])  # noqa: T201

    for row, line in zip(show_df.itertuples(index=False), lines[1:], strict=True):
        print(line)  # noqa: T201
        if detailed:
            year_periods = periods_by_year[row.Year]
            for idx, period in enumerate(year_periods):
                marker = TREE_MARKER if idx < len(year_periods) - 1 else TREE_LAST
                print(marker, period)  # noqa: T201


class _TypedNamespace(Namespace):
    """Type hints for command line arguments."""

    cmd: Literal["new", "add", "show"]
    days: float
    first_year: int
    first_month: int
    last_year: tuple[int, int]
    name: str
    first: str
    last: str | None
    detailed: bool
    file: Path
    verbose: bool


def parse_args(argv: Sequence[str] | None = None) -> _TypedNamespace:
    """Parse command line arguments.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace

    Raises:
        ValueError: If specified file doesn't have .toml extension
    """
    parser = ArgumentParser(description="Track and manage vacation periods")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # New command
    new_parser = subparsers.add_parser(
        "new", help="Create new vacation tracking configuration"
    )
    new_parser.add_argument("days", type=float, help="Vacation days earned per month")
    new_parser.add_argument(
        "first_year", type=int, help="First year to track vacation days"
    )
    new_parser.add_argument(
        "first_month", type=int, help="First month to track vacation days"
    )
    new_parser.add_argument(
        "-l",
        "--last-year",
        nargs=2,
        type=int,
        metavar=("year", "month"),
        default=(datetime.now(tz=timezone.utc).year, 12),
        help="Last year and month to track vacation days (default: end of current year)",  # noqa: E501
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new vacation period")
    add_parser.add_argument(
        "name", type=str, help="Name or description of the vacation period"
    )
    add_parser.add_argument(
        "first", type=str, help="First day of the vacation period (ISO format)"
    )
    add_parser.add_argument(
        "-l",
        "--last",
        type=str,
        default=None,
        help="Last day of the vacation period (ISO format, default: same as first day)",
    )

    # Show command
    show_parser = subparsers.add_parser(
        "show", help="Display vacation tracking summary"
    )
    show_parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        default=False,
        help="Show individual vacation periods (default: summary only)",
    )

    # Common arguments
    for subparser in (new_parser, add_parser, show_parser):
        subparser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path("vacation-periods.toml"),
            help="Path to vacation periods storage file (default: vacation-periods.toml)",  # noqa: E501
        )
        subparser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="Enable debug logging output",
        )

    args: _TypedNamespace = parser.parse_args(argv)  # type: ignore[assignment]

    if args.file.suffix != ".toml":
        raise ValueError("Configuration file must have .toml extension")

    return args


def new(file: Path, days: float, first: tuple[int, int], last: tuple[int, int]) -> None:
    """Create a new vacation tracking configuration file.

    Args:
        file: Path to the configuration file
        days: Number of vacation days earned per month
        first: Tuple of (year, month) for start of tracking
        last: Tuple of (year, month) for end of tracking

    Raises:
        FileExistsError: If configuration file already exists
    """
    if file.exists():
        raise FileExistsError(
            f"Configuration file {file} already exists. Edit manually if needed."
        )

    config = msgspec.convert(
        {"days-per-month": days, "first-year": first, "last-year": last}, Config
    )
    file.write_bytes(msgspec.toml.encode(config))


def add_or_show(cmd: Literal["add", "show"], file: Path, **kwargs: Any) -> None:
    """Add a vacation period or display tracking summary.

    Args:
        cmd: Command to execute ("add" or "show")
        file: Path to configuration file
        **kwargs: Additional arguments for the specific command

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If vacation period validation fails
    """
    if not file.exists():
        raise FileNotFoundError(
            f"Configuration file {file} not found. Use 'vacation-tracker new' to create."  # noqa: E501
        )

    config_content = msgspec.toml.decode(file.read_bytes())
    config = msgspec.convert(config_content, Config)

    if cmd == "show":
        show(config, kwargs["detailed"])
        return

    vacation = msgspec.convert(kwargs, Vacation)
    config.vacation_periods.append(vacation)
    config.verify()
    file.write_bytes(msgspec.toml.encode(config))


def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line interface for vacation tracking.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success)

    Raises:
        ValueError: If unknown command specified
    """
    args = parse_args(argv)

    if args.verbose:  # pragma: no cover
        logging.basicConfig(level=logging.DEBUG)

    if args.cmd == "new":
        new(args.file, args.days, (args.first_year, args.first_month), args.last_year)
    elif args.cmd in {"add", "show"}:
        kwargs = dict(args._get_kwargs())  # noqa: SLF001
        kwargs.pop("verbose")
        add_or_show(**kwargs)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
