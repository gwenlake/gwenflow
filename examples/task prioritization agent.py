import dotenv
from gwenflow import ChatOpenAI, Agent, Task

# Load API key from .env file
dotenv.load_dotenv(override=True)

# Define the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the task prioritization agent
task_priority_agent = Agent(
    role="Task Prioritizer",
    instructions=(
        "Evaluate the provided tasks based on urgency and importance. "
        "Assign a priority level: High, Medium, or Low."
    ),
    llm=llm
)

# Define the task prioritization function
def prioritize_tasks(task_list: list[dict]) -> list[dict]:
    """
    Prioritize a list of tasks.
    
    Args:
        task_list: A list of dictionaries, where each dictionary contains
                   'task' (description of the task),
                   'urgency' (e.g., 'High', 'Medium', 'Low'),
                   and 'importance' (e.g., 'High', 'Medium', 'Low').
    
    Returns:
        A list of dictionaries with an added 'priority' field.
    """
    task_descriptions = [
        f"Task: {task['task']}"
        for task in task_list
    ]
    tasks_summary = "\n".join(task_descriptions)
    
    task = Task(
        description=f"Prioritize the following tasks:\n{tasks_summary}",
        expected_output="Return the tasks with an assigned priority: High, Medium, or Low.",
        agent=task_priority_agent
    )
    
    return task.run()

# Example usage
tasks = [
    {"task": "Prepare project proposal"},
    {"task": "Schedule team meeting"},
    {"task": "Update personal LinkedIn profile"},
    {"task": "Organize office supplies"},
]

prioritized_tasks = prioritize_tasks(tasks)
print(prioritized_tasks)
