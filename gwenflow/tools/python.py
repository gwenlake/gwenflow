from pydantic import Field
from pydantic.fields import FieldInfo
from typing import Optional, Union
from pathlib import Path

from gwenflow.tools import BaseTool
from gwenflow.logger import logger


DESCRIPTION = """\
This function to runs Python code in the current environment.
If successful, returns the value of `variable_to_return` if provided otherwise returns a success message.
If failed, returns an error message.
Returns the value of `variable_to_return` if successful, otherwise returns an error message.
"""


class PythonCodeTool(BaseTool):

    name: str = "PythonCodeTool"
    description: str = DESCRIPTION
    
    base_dir: Optional[Union[Path, str]] = None
    safe_locals: Optional[dict] = None
    safe_globals: Optional[dict] = None

    def _run(self,
        code: str = Field(description="The code to run."),
        variable_to_return: str = Field(description="The variable to return.", default=None),
    ):
        
        if isinstance(variable_to_return, FieldInfo):
            variable_to_return = variable_to_return.default

        safe_globals = self.safe_globals if self.safe_globals is not None else globals()
        safe_locals = self.safe_locals if self.safe_locals is not None else locals()

        try:
            logger.debug(f"Running code:\n\n{code}\n\n")
            exec(code, safe_globals, safe_locals)
            
            if variable_to_return:
                variable_value = safe_locals.get(variable_to_return)
                if variable_value is None:
                    return f"Variable {variable_to_return} not found"
                logger.debug(f"Variable {variable_to_return} value: {variable_value}")
                return str(variable_value)
            else:
                return "successfully ran python code"
            
        except Exception as e:
            logger.exception("Error running python code")
            return f"Error running python code: {e}"