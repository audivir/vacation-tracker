# Vacation Tracker

A command-line tool to track and manage vacation periods with support for holidays and date ranges. This tool helps you manage your vacation days, taking into account:

- Vacation day entitlements per month
- Public holidays based on country/region
- Weekend exclusions
- Year transitions and expiration dates
- Detailed summaries and reporting

## Installation

Install using pip:

```bash
pip install vacation-tracker
```

## Configuration

First, create a new tracking configuration:

```bash
vacation-tracker new --days 2.5 --first-year 2023 --first-month 1 \
                     --last-year 2024 --last-month 12 \
                     --special-days 12-31 --special-days 01-02
```

This creates a `vacation-periods.toml` file with your vacation tracking settings. The configuration includes:

- Days earned per month
- Tracking period (start/end dates)
- Country/region for holiday calculations
- List of vacation periods

Example configuration:

```toml
days-per-month = 2.5
first-year = "2023-01" # Start tracking from January 2023
last-year = "2024-12"  # Track until December 2024
country = ["DE", "BY"]  # Germany, Bavaria region
categories = ["public", "catholic"]  # Holiday categories to consider
special-days = ["12-31"] # Special days to exclude from vacation calculation
```

## Usage

### Adding Vacation Periods

Add single day vacations:

```bash
vacation-tracker add "Doctor Appointment" --first 2023-05-15 
```

Add multi-day periods:

```bash
vacation-tracker add "Summer Holiday" --first 2023-08-01 --last 2023-08-15
```

### Viewing Vacation Summary

Basic summary:

```bash
vacation-tracker show
```

This displays a table with:
- Yearly entitlements
- Utilized days
- Remaining days
- Expired allocations

Detailed view with individual periods:

```bash
vacation-tracker show --detailed
```

Example output:
```
 Year  Months  Entitlement  Real Util  Adj Util  Remaining
 2023  12      30           25         25        5 (expired)
 ├─ 01.08-15.08 (11): Summer Holiday
 └─ 15.05 (1): Doctor Appointment
 2024  12      30           0          0         30
```

## Features

- Automatically handles:
  - Weekend exclusions
  - Public holidays based on country/region
  - Vacation day expiration
  - Year transitions
  - Period overlaps prevention
  - Cross-year vacation splits
- Configurable holiday categories
- Support for partial day accumulation (e.g., 2.5 days/month)
- Detailed vacation period tracking and reporting
