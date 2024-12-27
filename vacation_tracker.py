"""Track vacation periods."""

# ruff: noqa: T201
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
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

MARKER = " ├─"
LAST = " └─"

MonthSpecifier: TypeAlias = Annotated[int, msgspec.Meta(ge=1, le=12)]


class Vacation(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """A single vacation period.

    Args:
        name (str): Name of the vacation period.
        first (date): First day of the vacation period.
        last (date | None): Last day of the vacation period. If none, only 1 day.
    """

    name: str
    first: date
    last: date | None = None
    holidays: holidays.HolidayBase | None = None

    def __post_init__(self) -> None:
        if self.last and self.last < self.first:
            raise ValueError("Last date before first date")

    @property
    def days(self) -> int:
        """Number of vacation days used during the period."""
        if self.holidays is None:
            raise ValueError("Set holidays first.")

        delta = self.real_last - self.first
        counter = 0

        # iterate through all days in the period and count work days
        for add in range(delta.days + 1):
            new_date = self.first + timedelta(days=add)
            if new_date.weekday() > 4:  # only monday to friday # noqa: PLR2004
                logger.debug("%s skipped: weekend.", new_date.strftime("%d.%m.%Y"))
                continue
            if new_date in self.holidays:
                logger.debug(
                    "%s skipped: %s",
                    new_date.strftime("%d.%m.%Y"),
                    self.holidays[new_date],
                )
                continue
            counter += 1
        return counter

    @property
    def real_last(self) -> date:
        """Last date or the first date if last is None."""
        return self.first if self.last is None else self.last

    @override
    def __str__(self) -> str:
        first_day = self.first.strftime("%d.%m")
        last_day = self.real_last.strftime("%d.%m")
        if self.first == self.real_last:
            return f"{first_day} ({self.days}): {self.name}"
        return f"{first_day}-{last_day} ({self.days}): {self.name}"

    def split(self, on: date) -> tuple[Vacation] | tuple[Vacation, Vacation]:
        """Split date on the provided date."""
        # split date outside this period, last date is included.
        if on < self.first or on >= self.real_last:
            return (self,)

        before = Vacation(self.name, self.first, on, self.holidays)
        after = Vacation(
            self.name, on + timedelta(days=1), self.real_last, self.holidays
        )
        return before, after

    def verify(self) -> tuple[int, date]:
        """Verify splits on year ends and exdate."""
        year = self.first.year
        if year != self.real_last.year:
            raise ValueError("Split periods on Silvester")
        exdate = date(year, 9, 30)
        if self.first <= exdate and self.real_last > exdate:
            raise ValueError("Split periods on 30th of September")
        return year, exdate


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
    """Configuration including the tracking range and the vacation days.

    Args:
        days_per_month (float): Vacation days per month.
        first_year (tuple[int, MonthSpecifier]):
            First year and month to track the vacation days for.
        last_year (tuple[int, MonthSpecifier]):
            Last year and month to track the vacation days for.
        vacation_periods (list[Vacation]): List of vacation periods.
        country (str | tuple[str, str | None]): Country to lookup legal holidays.
            Defaults to ("DE", "BY").
        categories (tuple[str, ...]): Categories to lookup holidays for.
            Defaults to {"public", "catholic"}).
    """

    days_per_month: float | int
    first_year: tuple[int, MonthSpecifier]
    last_year: tuple[int, MonthSpecifier]
    vacation_periods: list[Vacation] = []
    country: str | tuple[str, str | None] = "DE", "BY"
    categories: tuple[str, ...] = ("public", "catholic")

    @property
    def holidays(self) -> holidays.HolidayBase:
        """Holidays for the configured country."""
        if isinstance(self.country, str):  # pragma: no cover
            self.country = self.country, None
        return holidays.country_holidays(*self.country, categories=self.categories)

    def __post_init__(self) -> None:
        if self.last_year <= self.first_year:
            raise ValueError("Last year is not greater than first year")

    def verify(self) -> None:
        """Sorts days, assert no overlaps and not outside tracking range."""
        self.vacation_periods.sort(key=lambda x: x.first)

        prev_last = date(1, 1, 1)
        for day in self.vacation_periods:
            if day.first <= prev_last:
                raise ValueError("Overlapping vacation days.")
            prev_last = day.real_last

        if not self.vacation_periods:  # pragma: no cover
            return

        first_day, last_day = self.vacation_periods[0], self.vacation_periods[-1]
        if first_day.first < date(*self.first_year, 1):
            raise ValueError("First vacation day before tracking start")
        last_year = (
            date(self.last_year[0] + 1, 1, 1)
            if self.last_year[1] == 12  # noqa: PLR2004
            else date(self.last_year[0], self.last_year[1] + 1, 1)
        )
        if last_day.real_last >= last_year:
            raise ValueError("Last vacation day after tracking end")

    def _get_entitlement(self) -> dict[int, tuple[float | int, int]]:
        first_year, first_month = self.first_year
        last_year, last_month = self.last_year

        entitlement_dict: dict[int, tuple[float | int, int]] = {}

        for year in range(first_year, last_year + 1):
            curr_first_month = first_month if year == first_year else 1
            curr_last_month = last_month if year == last_year else 12
            months = 1 + curr_last_month - curr_first_month  # + 1 to include last month
            entitlement = months * self.days_per_month
            # just for visuals
            if (
                isinstance(entitlement, float) and entitlement.is_integer()
            ):  # pragma: no cover
                entitlement = int(entitlement)
            entitlement_dict[year] = entitlement, months

        return entitlement_dict

    def _order_periods(self) -> tuple[list[Vacation], dict[int, list[Vacation]]]:
        self.verify()  # sorts vacation days

        vacation_periods_dict: dict[int, list[Vacation]] = defaultdict(list)
        for day in self.vacation_periods:
            day.holidays = self.holidays
            curr_day: Vacation | None = day
            for year in range(day.first.year, day.real_last.year + 1):
                if curr_day is None:  # pragma: no cover
                    raise RuntimeError("should not happen")
                splits = curr_day.split(date(year, 12, 31))
                if not 1 <= len(splits) <= 2:  # pragma: no cover # noqa: PLR2004
                    raise RuntimeError("should not happen")
                this_year, curr_day = splits if len(splits) == 2 else (splits[0], None)  # noqa: PLR2004
                vacation_periods_dict[year].extend(this_year.split(date(year, 9, 30)))
        vacation_periods = [
            day
            for year_periods in vacation_periods_dict.values()
            for day in year_periods
        ]
        return vacation_periods, vacation_periods_dict


def _track_single_period(
    year: int, days: float, track_dict: dict[int, float | int]
) -> float | int:
    """Try to remove `days` from `year`'s entry in `track_dict`."""
    if (avail := track_dict.get(year, 0)) > 0:
        if avail < days:
            prev_days = avail
            days -= avail
        else:
            prev_days = days
            days = 0
        track_dict[year] -= prev_days

    return days


def track_periods(
    periods: Sequence[Vacation], track_dict: dict[int, float | int]
) -> None:
    """Track vacation `periods` using available days from `track_dict`."""
    for period in periods:
        year, exdate = period.verify()
        first_year = year - 1 if period.real_last <= exdate else year

        days: float | int = period.days
        for year in range(first_year, max(track_dict) + 1):
            days = _track_single_period(year, days, track_dict)
            if days == 0:
                break
        else:
            raise ValueError("No more vacation days.")


def show(config: Config, detailed: bool = False) -> None:
    """Show the vacation periods.

    Args:
        config: Configuration with tracking range and vacation periods.
        detailed: Show each vacation period additional to the summary.
            Defaults to False.
    """
    vacation_periods, vacation_periods_dict = config._order_periods()  # noqa: SLF001

    entitlement_dict = config._get_entitlement()  # noqa: SLF001

    remaining_dict = {
        year: entitlement for year, (entitlement, _) in entitlement_dict.items()
    }

    track_periods(vacation_periods, remaining_dict)

    rows: list[
        tuple[int, int, float | int, float | int, float | int, float | int | str]
    ] = []

    now = datetime.now(tz=timezone.utc)
    threshold_year = now.year - (1 if now.date() > date(now.year, 9, 30) else 2)
    for year, (entitlement, months) in entitlement_dict.items():
        remaining = remaining_dict[year]
        real_utilisation = sum(day.days for day in vacation_periods_dict[year])
        adjusted_utilisation = entitlement - remaining

        remaining_str = (
            f"{remaining} (expired)"
            if year <= threshold_year and remaining > 0
            else remaining
        )
        rows.append(
            (
                year,
                months,
                entitlement,
                real_utilisation,
                adjusted_utilisation,
                remaining_str,
            )
        )

    columns = ("Year", "Months", "Entitlement", "Real Util", "Adj Util", "Remaining")
    final_df = pd.DataFrame(rows, columns=columns)

    lines = final_df.to_string(index=False).splitlines()

    print(lines[0])
    for row, line in zip(final_df.itertuples(index=False), lines[1:], strict=True):
        print(line)
        if detailed:
            curr_days = vacation_periods_dict[row.Year]
            for ix, day in enumerate(curr_days):
                marker = MARKER if ix < len(curr_days) - 1 else LAST
                print(marker, day)


class TypedNamespace(Namespace):
    """Type hints for the command line arguments."""

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


def parse_args(argv: Sequence[str] | None = None) -> TypedNamespace:
    """Define and parse the command line arguments."""
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    new_parser = subparsers.add_parser("new")
    add_parser = subparsers.add_parser("add")
    show_parser = subparsers.add_parser("show")

    new_parser.add_argument("days", type=float, help="Vacation days per month.")
    new_parser.add_argument(
        "first_year", type=int, help="First year to track the vacation days for."
    )
    new_parser.add_argument(
        "first_month", type=int, help="First month to track the vacation days for."
    )
    new_parser.add_argument(
        "-l",
        "--last-year",
        nargs=2,
        type=int,
        metavar=("year", "month"),
        default=(datetime.now(tz=timezone.utc).year, 12),
        help="Last year and month to track vacation days for. Defaults to end of current year.",  # noqa: E501
    )

    add_parser.add_argument("name", type=str, help="Name of the vacation period")
    add_parser.add_argument(
        "first", type=str, help="First day of the vacation period (iso format)."
    )
    add_parser.add_argument(
        "-l",
        "--last",
        type=str,
        default=None,
        help="Last day of the vacation period (iso format). If none, only 1 day.",
    )

    show_parser.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        default=False,
        help="Show each vacation period. Defaults to summary only.",
    )
    for subparser in (new_parser, add_parser, show_parser):
        subparser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path("vacation-periods.toml"),
            help="Storage file of the vacation periods. Defaults to vacation-periods.toml.",  # noqa: E501
        )
        subparser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="Turn on debugging messages.",
        )

    args: TypedNamespace = parser.parse_args(argv)  # type: ignore[assignment]

    if args.file.suffix != ".toml":
        raise ValueError("Please provide path to a toml file.")

    return args


def new(file: Path, days: float, first: tuple[int, int], last: tuple[int, int]) -> None:
    """Create a new vacation periods toml file."""
    if file.exists():
        raise FileExistsError("Vacation days toml file exists. Edit file if necessary.")
    config = msgspec.convert(
        {"days-per-month": days, "first-year": first, "last-year": last}, Config
    )
    file.write_bytes(msgspec.toml.encode(config))


def add_or_show(cmd: Literal["add", "show"], file: Path, **kwargs: Any) -> None:
    """Add an entry to an existing toml or show its content."""
    if not file.exists():
        raise FileNotFoundError(
            "Vacation days toml file doesnt exists. Call `vacation-tracker new` first."
        )

    config_content = msgspec.toml.decode(file.read_bytes())
    config = msgspec.convert(config_content, Config)

    if cmd == "show":
        show(config, kwargs["detailed"])
        return

    vd = msgspec.convert(kwargs, Vacation)
    config.vacation_periods.append(vd)
    config.verify()
    file.write_bytes(msgspec.toml.encode(config))


def cli(argv: Sequence[str] | None = None) -> int:
    """Track vacation periods."""
    args = parse_args(argv)

    if args.verbose:  # pragma: no cover
        logging.basicConfig(level=logging.DEBUG)

    if args.cmd == "new":
        new(args.file, args.days, (args.first_year, args.first_month), args.last_year)
    elif args.cmd in {"add", "show"}:
        kwargs = dict(args._get_kwargs())  # noqa: SLF001
        del kwargs["verbose"]
        add_or_show(**kwargs)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
