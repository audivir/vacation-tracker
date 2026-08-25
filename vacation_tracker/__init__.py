"""Tracks vacation periods, available days, and country-specific holiday schedules."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import doctyper
import holidays
import msgspec
import polars as pl
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

logger = logging.getLogger(__name__)

__version__ = "0.1.5"

# Constants
TREE_MARKER = " ├─"
TREE_LAST = " └─"
WEEKEND_DAYS = (5, 6)  # Saturday and Sunday


class Vacation(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Stores a single vacation period with start and end dates."""

    name: str
    """Name or description of the vacation period."""
    first: date
    """First day of the vacation period."""
    last: date | None = None
    """Last day of the vacation period, or None if unspecified."""
    holidays: holidays.HolidayBase | None = None
    """Holiday calendar used for day counting."""
    special_days: set[tuple[int, int]] = set()
    """Set of month-day tuples to exclude from day counting."""

    def __post_init__(self) -> None:
        """Validates vacation period dates."""
        if not self.name.strip():
            raise ValueError("Vacation name cannot be empty")
        if self.last and self.last < self.first:
            raise ValueError("Last date cannot be before first date")

    @property
    def days(self) -> int:
        """Calculates the number of working days in the vacation period.

        Raises:
            ValueError: If the holidays calendar is not set.
        """
        if self.holidays is None:
            raise ValueError("Holidays calendar must be set before counting days")

        delta = self.real_last - self.first
        counter = 0

        for add in range(delta.days + 1):
            current_date = self.first + timedelta(days=add)

            if (current_date.month, current_date.day) in self.special_days:
                logger.debug("%s skipped: special day", current_date.strftime("%d.%m.%Y"))
                continue

            if current_date.weekday() in WEEKEND_DAYS:
                logger.debug("%s skipped: weekend", current_date.strftime("%d.%m.%Y"))
                continue

            if current_date in self.holidays:
                logger.debug(
                    "%s skipped: %s", current_date.strftime("%d.%m.%Y"), self.holidays[current_date]
                )
                continue

            counter += 1

        return counter

    @property
    def real_last(self) -> date:
        """Returns the actual end date of the vacation period."""
        return self.first if self.last is None else self.last

    @override
    def __str__(self) -> str:
        """Formats the vacation period as a string with dates and day count."""
        first_day = self.first.strftime("%d.%m")
        last_day = self.real_last.strftime("%d.%m")

        if self.first == self.real_last:
            return f"{first_day} ({self.days}): {self.name}"

        return f"{first_day}-{last_day} ({self.days}): {self.name}"

    def split(self, on: date) -> tuple[Vacation] | tuple[Vacation, Vacation]:
        """Splits the vacation period at the specified date.

        Args:
            on: Date at which to split the vacation period.

        Returns:
            The original vacation if no split is needed, or two new vacation
            periods.
        """
        if on < self.first or on >= self.real_last:
            return (self,)

        before = Vacation(self.name, self.first, on, self.holidays, self.special_days)
        after = Vacation(
            self.name, on + timedelta(days=1), self.real_last, self.holidays, self.special_days
        )
        return before, after

    def verify(self) -> tuple[int, date]:
        """Verifies that the vacation period stays within the same year and does not cross Sep 30.

        Returns:
            Tuple of (year, expiration_date).

        Raises:
            ValueError: If the period crosses a year boundary or September 30th.
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
        "special_days": "special-days",
        "pre_entitlement": "pre-entitlement",
    },
):
    """Stores configuration for vacation tracking."""

    days_per_month: float | int
    """Vacation days earned per month."""
    first_year: str
    """Year-month string marking the start of tracking."""
    last_year: str
    """Year-month string marking the end of the tracking period."""
    vacation_periods: list[Vacation] = []
    """List of vacation periods to track."""
    country: str | tuple[str, str | None] = ("DE", "BY")
    """Country code(s) for the holiday calendar."""
    categories: tuple[str, ...] = ("public", "catholic")
    """Categories of holidays to include in the calendar."""
    special_days: list[str] = []
    """List of "month-day" strings for vacation days that are not holidays."""
    # pre_entitlement: int = 0 # noqa: ERA001
    # """Number of days to include before the start of tracking"""

    @property
    def holidays(self) -> holidays.HolidayBase:
        """Returns the holiday calendar for the configured country."""
        if isinstance(self.country, str):  # pragma: no cover
            self.country = (self.country, None)
        return holidays.country_holidays(*self.country, categories=self.categories)

    def _parse_year(self, attr: Literal["first_year", "last_year"]) -> tuple[int, int]:
        """Parses a year-month string into a year and month tuple."""
        date_str: str = getattr(self, attr)
        try:
            full_date = date.fromisoformat(f"{date_str}-01")
        except ValueError as e:
            raise ValueError(f"Invalid {attr.replace('_', ' ')}: {date_str}") from e
        return (full_date.year, full_date.month)

    @property
    def parsed_first_year(self) -> tuple[int, int]:
        """Returns the parsed first year from configuration."""
        return self._parse_year("first_year")

    @property
    def parsed_last_year(self) -> tuple[int, int]:
        """Returns the parsed last year from configuration."""
        return self._parse_year("last_year")

    @staticmethod
    def _parse_special_day(day: str) -> date:
        """Parses a single "month-day" string into a date."""
        try:
            return date.fromisoformat(f"1970-{day}")
        except ValueError as e:
            raise ValueError(f"Invalid special day: {day}") from e

    @property
    def parsed_special_days(self) -> set[tuple[int, int]]:
        """Parses special days into a set of (month, day) tuples."""
        dates = [self._parse_special_day(elem) for elem in self.special_days]
        return {(date.month, date.day) for date in dates}

    def __post_init__(self) -> None:
        """Validates configuration dates and raises `ValueError` on invalid values."""
        if not 0 <= self.days_per_month <= 28:  # noqa: PLR2004
            raise ValueError("Days per month must be between 0 and 28")
        if self.parsed_last_year <= self.parsed_first_year:
            raise ValueError("Last year must be after first year")
        self.parsed_special_days  # Validate special days # noqa: B018

    def verify(self) -> None:
        """Verifies vacation periods for overlaps and date range constraints."""
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

        if first_period.first < date(*self.parsed_first_year, 1):
            raise ValueError("First vacation day must be after tracking start")

        last_tracking_date = (
            date(self.parsed_last_year[0] + 1, 1, 1)
            if self.parsed_last_year[1] == 12  # noqa: PLR2004
            else date(self.parsed_last_year[0], self.parsed_last_year[1] + 1, 1)
        )

        if last_period.real_last >= last_tracking_date:
            raise ValueError("Last vacation day must be before tracking end")

    def _get_entitlement(self) -> dict[int, tuple[float | int, int]]:
        """Calculates vacation day entitlements per year."""
        first_year, first_month = self.parsed_first_year
        last_year, last_month = self.parsed_last_year
        entitlements: dict[int, tuple[float | int, int]] = {}

        for year in range(first_year, last_year + 1):
            curr_first_month = first_month if year == first_year else 1
            curr_last_month = last_month if year == last_year else 12
            months = 1 + curr_last_month - curr_first_month

            entitlement = months * self.days_per_month
            # if year == first_year:
            #     entitlement += self.pre_entitlement # noqa: ERA001
            if isinstance(entitlement, float) and entitlement.is_integer():  # pragma: no cover
                entitlement = int(entitlement)

            entitlements[year] = (entitlement, months)

        return entitlements

    def _order_periods(self) -> tuple[list[Vacation], dict[int, list[Vacation]]]:
        """Orders and splits vacation periods by year."""
        self.verify()

        periods_by_year: dict[int, list[Vacation]] = defaultdict(list)
        for period in self.vacation_periods:
            period.holidays = self.holidays
            period.special_days = self.parsed_special_days
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
            period for year_periods in periods_by_year.values() for period in year_periods
        ]
        return all_periods, periods_by_year


def _track_single_period(
    year: int, days: float, track_dict: MutableMapping[int, float | int]
) -> float | int:
    """Tracks vacation days for a single year and returns the remaining days."""
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
    """Tracks multiple vacation periods against available days.

    Args:
        periods: Sequence of vacation periods to track.
        track_dict: Mapping of available days by year.

    Raises:
        ValueError: If there are insufficient vacation days.
    """
    for period in periods:
        year, expiration_date = period.verify()
        first_year = year - 1 if period.real_last <= expiration_date else year

        remaining_days: float | int = period.days
        for tracking_year in range(first_year, max(track_dict) + 1):
            remaining_days = _track_single_period(tracking_year, remaining_days, track_dict)
            if remaining_days == 0:
                break
        else:
            raise ValueError("Insufficient vacation days available")


def _verify_config(file: Path) -> Config:
    """Verifies that the configuration file exists and is valid."""
    if not file.exists():
        raise FileNotFoundError(
            f"Configuration file {file} not found. Use 'vacation-tracker new' to create."
        )
    content = msgspec.toml.decode(file.read_bytes())

    # backwards compatibility with old config format
    changed = False
    if "special-days" not in content:
        changed = True
    for key in ("first-year", "last-year"):
        value = content.get(key)
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):  # noqa: PLR2004
            changed = True
            content[key] = f"{value[0]:02d}-{value[1]:02d}"

    config = msgspec.convert(content, Config)

    if changed:
        file.write_bytes(msgspec.toml.encode(config))
        print("Updated configuration file to new format.")  # noqa: T201

    return config


def show(detailed: bool = False, config_file: Path = Path("vacation-periods.toml")) -> None:
    """Displays a vacation period summary, with optional per-period details."""
    config = _verify_config(config_file)
    periods, periods_by_year = config._order_periods()  # noqa: SLF001
    entitlements = config._get_entitlement()  # noqa: SLF001

    remaining = {year: entitlement for year, (entitlement, _) in entitlements.items()}

    track_periods(periods, remaining)

    rows: list[tuple[int, int, float | int, float | int, float | int, float | int | str]] = []

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
    show_df = pl.DataFrame(rows, schema=columns, orient="row")

    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("NOTHING")
    pl.Config.set_tbl_hide_column_data_types(True)

    lines = str(show_df).splitlines()
    print(lines[0].rstrip("\n "))  # noqa: T201

    for row, line in zip(show_df.iter_rows(named=True), lines[1:], strict=True):
        print(line.rstrip("\n "))  # noqa: T201
        if detailed:
            year_periods = periods_by_year[row["Year"]]
            for idx, period in enumerate(year_periods):
                marker = TREE_MARKER if idx < len(year_periods) - 1 else TREE_LAST
                print(marker, period)  # noqa: T201


def new(
    days: float,
    first_year: int = datetime.now(tz=timezone.utc).year,
    first_month: int = 1,
    last_year: int = datetime.now(tz=timezone.utc).year,
    last_month: int = 12,
    special_days: list[str] | None = None,
    # pre_entitlement: int = 0, # noqa: ERA001
    # pre_entitlement: Number of vacation days earned before tracking starts
    config_file: Path = Path("vacation-periods.toml"),
) -> None:
    """Creates a new vacation tracking configuration file.

    Args:
        days: Number of vacation days earned per month.
        first_year: First year to track vacation days.
        first_month: First month to track vacation days.
        last_year: Last year to track vacation days.
        last_month: Last month to track vacation days.
        special_days: List of days to exclude from vacation days as
            isoformat-like month-day strings (e.g. "01-01").
        config_file: Path to the vacation tracking configuration file.

    Raises:
        FileExistsError: If configuration file already exists.
    """
    if config_file.exists():
        raise FileExistsError(
            f"Configuration file {config_file} already exists. Edit manually if needed."
        )
    if config_file.suffix != ".toml":
        raise ValueError("Configuration file must have .toml extension")

    config = msgspec.convert(
        {
            "days-per-month": days,
            "first-year": f"{first_year:02d}-{first_month:02d}",
            "last-year": f"{last_year:02d}-{last_month:02d}",
            "special-days": special_days or [],
            # "pre-entitlement": pre_entitlement,
        },
        Config,
    )
    config_file.write_bytes(msgspec.toml.encode(config))


def add(
    name: str,
    first: str,
    last: Annotated[str | None, doctyper.Option(show_default="Single Day")] = None,
    config_file: Path = Path("vacation-periods.toml"),
) -> None:
    """Adds a vacation period to the configuration file.

    Args:
        name: Name or description of the vacation period.
        first: First day of the vacation period in ISO format.
        last: Last day of the vacation period in ISO format.
        config_file: Path to the vacation tracker configuration file.
    """
    config = _verify_config(config_file)

    # Validate date formats
    try:
        first_date = date.fromisoformat(first)
        last_date = date.fromisoformat(last) if last else None
    except ValueError as e:
        raise ValueError("Dates must be in ISO format (YYYY-MM-DD)") from e

    vacation = msgspec.convert({"name": name, "first": first_date, "last": last_date}, Vacation)
    config.vacation_periods.append(vacation)
    config.verify()
    config_file.write_bytes(msgspec.toml.encode(config))


def cli() -> None:  # pragma: no cover
    """CLI entry point for the doctyper application."""
    app = doctyper.SlimTyper()
    app.command("new")(new)
    app.command("add")(add)
    app.command("show")(show)
    app()


if __name__ == "__main__":
    cli()
