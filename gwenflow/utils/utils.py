from datetime import datetime


def now_utc() -> datetime:
    from gwenflow.utils.durable import now_utc as _now_utc

    return _now_utc()
