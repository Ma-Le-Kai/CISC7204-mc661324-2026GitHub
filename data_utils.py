"""Local data definitions and filters; no parent-project imports or downloads."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WEATHER = ['temperature_c', 'precipitation_mm', 'wind_speed_ms']
SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def validate_daily(frame, label, columns):
    expected = pd.date_range('2015-01-01', '2016-12-31')
    if frame['date'].isna().any() or frame['date'].duplicated().any():
        raise ValueError(f'{label}: invalid or duplicate dates')
    if not pd.DatetimeIndex(frame['date']).sort_values().equals(expected):
        raise ValueError(f'{label}: expected exactly 731 consecutive dates in 2015–2016')
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f'{label}: missing or non-finite observations')
    for col in ['daily_hires', 'precipitation_mm', 'wind_speed_ms']:
        if col in columns and (frame[col] < 0).any():
            raise ValueError(f'{label}: negative {col}')
    if 'daily_hires' in columns and not np.equal(frame.daily_hires % 1, 0).all():
        raise ValueError(f'{label}: daily hires must be whole counts')


def filter_data(data, start=None, end=None, day_type='All', seasons=None):
    """Empty season lists select no days, consistently for charts and downloads."""
    mask = pd.Series(True, index=data.index)
    if start:
        mask &= data.date >= pd.Timestamp(start)
    if end:
        mask &= data.date <= pd.Timestamp(end)
    if day_type in ['Weekday', 'Weekend']:
        mask &= data.day_type.eq(day_type)
    if seasons is not None:
        mask &= data.season.isin(seasons)
    return data.loc[mask].copy()
