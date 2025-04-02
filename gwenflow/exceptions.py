

class GwenflowException(Exception):
    """Base class for all exceptions in Gwenflow."""


class MaxTurnsExceeded(GwenflowException):
    """Exception raised when the maximum number of turns is exceeded."""

    message: str

    def __init__(self, message: str):
        self.message = message


class ModelBehaviorError(GwenflowException):
    """Exception raised when the model does something unexpected"""

    message: str

    def __init__(self, message: str):
        self.message = message


class UserError(GwenflowException):
    """Exception raised when the user makes an error."""

    message: str

    def __init__(self, message: str):
        self.message = message