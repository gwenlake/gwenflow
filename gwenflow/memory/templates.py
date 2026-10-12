"""Default contents of memory files, so the model always sees the structure."""

from __future__ import annotations

from gwenflow.utils.markdown import render_frontmatter

USER_MEMORY_TEMPLATE = """\
# User memory

## Profile
- (role, organisation, language, timezone)

## Preferences
- (answer format, tone, units, recurring constraints)

## Recurring context
- (subjects that come up repeatedly)

## Notes
- (facts the user explicitly asked to remember, dated)
"""

THREAD_NOTES_TEMPLATE = """\
# Thread notes

## Goal
- (what the user is trying to achieve in this conversation)

## Entities in play
- (ids, names, references found via tools)

## Decisions
- (choices made so far)

## Open items
- (questions and tasks still pending)

## Key results
- (tool results still needed later, with the arguments that produced them)

## Candidate memories
- (durable facts about the user noticed in this thread, not yet in the user memory)
"""

FORBIDDEN_CONTENT = (
    "credentials, passwords, API keys or tokens; bank, card or IBAN numbers; "
    "health data; political, religious or union opinions; national identity numbers; "
    "private contact details of a natural person (home address, personal phone or email) "
    "unless they are a business contact; anything about the private life of third parties"
)

CORE_FILE_NAMES = ("MEMORY.md", "NOTES.md")


def topic_type_for_path(path: str) -> str:
    """The directory a topic file sits in (``/memories/user/projects/x.md`` -> ``projects``), else ``topic``."""
    parts = path.strip("/").split("/")
    return parts[-2] if len(parts) >= 4 else "topic"


def topic_name_for_path(path: str) -> str:
    return path.rsplit("/", 1)[-1][: -len(".md")]


def render_topic_file(path: str, description: str, body: str, type_: str | None = None) -> str:
    """A topic file with its frontmatter (``name``, ``description``, ``type``).

    ``type`` defaults to the file's directory, so applications choose their own
    categories by choosing their directories.
    """
    frontmatter = {
        "name": topic_name_for_path(path),
        "description": " ".join(description.split()),
        "type": type_ or topic_type_for_path(path),
    }
    return render_frontmatter(frontmatter, body)


def default_template(path: str) -> str:
    """Default content for a file that does not exist yet."""
    name = path.rsplit("/", 1)[-1]
    if name == "NOTES.md":
        return THREAD_NOTES_TEMPLATE
    if name == "MEMORY.md":
        return USER_MEMORY_TEMPLATE
    return render_topic_file(path, description="", body="")
