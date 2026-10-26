import dotenv

from gwenflow import Agent, ChatOpenAI
from gwenflow.types import ToolDenied

dotenv.load_dotenv(override=True)

agent = Agent(
    name="File Assistant",
    instructions="You manage the user's workspace files.",
    llm=ChatOpenAI(model="gpt-4o-mini"),
)


@agent.tool
def list_files() -> str:
    """List the files of the workspace."""
    return "report.pdf, notes.txt, old_draft.docx"


@agent.tool(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file from the workspace.

    Args:
        path: Name of the file to delete.
    """
    return f"{path} deleted."  # demo: nothing is actually deleted


def ask_user(response) -> dict:
    """Ask the user about each tool call the run paused on."""
    approvals = {}
    for request in response.pending_approvals:
        answer = input(f"  Run {request.tool_name}({request.tool_args})? [y/N or a reason] ").strip()
        if answer.lower() == "y":
            approvals[request.tool_call_id] = True
        elif answer and answer.lower() != "n":
            approvals[request.tool_call_id] = ToolDenied(answer)
        else:
            approvals[request.tool_call_id] = False
    return approvals


while True:
    task = input("\nYou: ").strip()
    if not task:
        break
    response = agent.run(task)
    while response.finish_reason == "approval_required":
        if response.content:
            print(f"Agent: {response.content}")
        response = agent.run(approvals=ask_user(response))
    print(f"Agent: {response.content}")
