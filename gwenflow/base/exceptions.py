from __future__ import annotations

from typing import Optional


class GwenflowException(Exception):
    """A base class for all Gwenflow exceptions."""

    def __init__(self, message: Optional[str] = None) -> None:
        super(GwenflowException, self).__init__(message)

        self.message = message

    def __str__(self) -> str:
        msg = self.message or "<empty message>"
        return msg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={str(self)})"
    

class GwenflowError(GwenflowException):
    """An error from Gwenlake."""

