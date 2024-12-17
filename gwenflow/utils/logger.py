import logging

from rich.logging import RichHandler

LOGGER_NAME = "gwenflow"


def get_logger(logger_name: str) -> logging.Logger:
    rich_handler = RichHandler(
        show_time=False,
        rich_tracebacks=False,
        tracebacks_show_locals=False,
    )
    rich_handler.setFormatter(
        logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
    )

    _logger = logging.getLogger(logger_name)
    _logger.addHandler(rich_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    return _logger


logger: logging.Logger = get_logger(LOGGER_NAME)


def set_log_level_to_debug():
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(logging.DEBUG)


def set_log_level_to_info():
    _logger = logging.getLogger(LOGGER_NAME)
    _logger.setLevel(logging.INFO)