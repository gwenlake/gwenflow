PROMPT_KNOWLEDGE = """\
## Use the following references from the knowledge base if it helps:

<references>
{references}
<references>
"""

PROMPT_CONTEXT = """\
## Use the following context if it helps:

<context>
{context}
</context>
"""

PROMPT_PREVIOUS_INTERACTIONS = """\
## Answer the question considering the previous interactions:

<previous_interactions>
{previous_interactions}
</previous_interactions>
"""

PROMPT_JSON_SCHEMA = """\
## Provide your output using the following JSON schema:

<json_schema>
{json_schema}
</json_schema>
"""
