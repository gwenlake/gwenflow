import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from gwenflow.flows.runtime import bounded_timeout, raise_if_stopped, remaining_time, stop_requested

NODE_TYPE_REGISTRY: dict[str, Callable] = {}

Idempotency = bool | Callable[[dict], bool]
PathParameters = Sequence[str] | Callable[[dict], Iterable[str]]


@dataclass(frozen=True)
class NodeTypeOptions:
    idempotent: Idempotency = False
    paths: PathParameters = ()


NODE_TYPE_OPTIONS: dict[str, NodeTypeOptions] = {}


def node_type(type_name: str, *, idempotent: Idempotency = False, paths: PathParameters = ()) -> Callable:
    def decorator(fn: Callable) -> Callable:
        NODE_TYPE_REGISTRY[type_name] = fn
        NODE_TYPE_OPTIONS[type_name] = NodeTypeOptions(idempotent=idempotent, paths=paths)
        return fn

    return decorator


def is_idempotent(type_name: str, parameters: dict) -> bool:
    flag = NODE_TYPE_OPTIONS.get(type_name, NodeTypeOptions()).idempotent
    return bool(flag(parameters)) if callable(flag) else flag


def node_paths(type_name: str, parameters: dict) -> list[str]:
    paths = NODE_TYPE_OPTIONS.get(type_name, NodeTypeOptions()).paths
    if callable(paths):
        return [str(p) for p in paths(parameters) if p]
    return [str(parameters[name]) for name in paths if parameters.get(name)]


SAFE_HTTP_METHODS = {"GET", "HEAD", "OPTIONS"}
READ_ONLY_OPENSEARCH_OPERATIONS = {"search", "count", "get"}
POLL_INTERVAL = 0.2
STDERR_TAIL = 2000


def _python_paths(parameters: dict) -> list[str]:
    directory = parameters.get("directory", ".")
    return [directory, str(Path(directory) / parameters.get("file", ""))]


def _communicate(process: subprocess.Popen, stdin: str | None) -> tuple[str, str]:
    pending = stdin
    while True:
        try:
            return process.communicate(pending, timeout=POLL_INTERVAL)
        except subprocess.TimeoutExpired:
            pending = None
            if stop_requested():
                process.kill()
                process.communicate()
                raise_if_stopped()


@node_type("gwenflow.Python", paths=_python_paths)
def handle_python(parameters: dict, input_data: dict | None) -> str:
    script = parameters["file"]
    args = [str(a) for a in parameters.get("args", [])]
    stdin = json.dumps(input_data) if input_data else None
    process = subprocess.Popen(
        [sys.executable, script, *args],
        cwd=parameters.get("directory", "."),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = _communicate(process, stdin)
    if process.returncode != 0:
        detail = stderr.strip()[-STDERR_TAIL:] or "nothing on stderr"
        raise RuntimeError(f"{script} exited with code {process.returncode}:\n{detail}")
    return stdout.strip()


@node_type("gwenflow.httpRequest", idempotent=lambda p: str(p.get("method", "GET")).upper() in SAFE_HTTP_METHODS)
def handle_http_request(parameters: dict, input_data: dict | None) -> Any:
    raise_if_stopped()
    url = parameters["url"]
    method = parameters.get("method", "GET").upper()
    headers = dict(parameters.get("headers", {}))
    query_params = parameters.get("query_params", {})
    body = parameters.get("body")
    timeout = bounded_timeout(parameters.get("timeout", 30))

    auth = parameters.get("auth", {})
    if auth.get("type") == "bearer":
        headers["Authorization"] = f"Bearer {auth['token']}"
    elif auth.get("type") == "basic":
        import base64

        creds = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
        headers["Authorization"] = f"Basic {creds}"

    response = httpx.request(method, url, headers=headers, params=query_params, json=body, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return response.text


def compose_task(task: str, input_data: dict | None, append_inputs: bool = False) -> str:
    """Fill the `{node_name}` placeholders of `task` with the parent outputs.

    A parent the task does not reference is left out, as in v1.5.1. With
    `append_inputs`, it is appended as context instead, so every input reaches the agent.
    """
    unreferenced = []
    for name, value in (input_data or {}).items():
        serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        placeholder = f"{{{name}}}"
        if placeholder in task:
            task = task.replace(placeholder, serialized)
        else:
            unreferenced.append(f"Output from {name}:\n{serialized}")
    if append_inputs and unreferenced:
        task = f"{task}\n\nContext from previous steps:\n" + "\n\n".join(unreferenced)

    return task


def agent_llm(parameters: dict) -> Any:
    """The LLM client of an agent node.

    Without `provider`, the client is the OpenAI one (so any OpenAI-compatible
    endpoint too). A `model` prefix that names a provider is removed, as in v1.5.1:
    `openai/gpt-4o-mini` and `gpt-4o-mini` are the same model. It never chooses the
    client. Any other prefix is part of the model name and is sent whole: an
    open-weights model keeps its organization (`Qwen/Qwen3-32B`).

    With `provider` (`openai`, `mistral`, `anthropic`, `gwenlake`...), the client is
    that provider's, and `model` is sent as written, whatever its prefix
    (`google/gemma-3-27b-it` on an OpenAI-compatible endpoint needs `provider: openai`).
    """
    from gwenflow.registry import DEFAULT_MODEL, PROVIDERS, make_llm

    provider = parameters.get("provider")
    if provider:
        model = parameters.get("model")
        if not model:
            raise ValueError("an agent node with `provider` needs a `model`")
        return make_llm(f"{provider}/{model}")

    from gwenflow import ChatOpenAI
    from gwenflow.logger import logger

    model = parameters.get("model", DEFAULT_MODEL)
    prefix, _, name = model.partition("/")
    if name and prefix.lower() in PROVIDERS:
        if prefix.lower() != "openai":
            logger.warning(
                f"Agent node model '{model}': the prefix does not choose the client. An OpenAI client "
                f"is used, with the model '{name}'. Add `provider: {prefix.lower()}` to use that "
                f"provider, or `provider: openai` to send '{model}' whole."
            )
        model = name
    return ChatOpenAI(model=model)


@node_type("gwenflow.Agent", paths=("harness",))
def handle_agent(parameters: dict, input_data: dict | None) -> str:
    """Run an agent on the node's task, from a harness or from the parameters written in the flow.

    Parameters: `harness` (a harness directory or a single-file agent, relative to the
    working directory) or `instructions`; `model` and `provider` (see `agent_llm`;
    `model` overrides the harness model), `task`, `tools` (built-in tool names), an
    optional `name`, and `append_inputs`.

    Parent outputs fill the `{node_name}` placeholders of `task`. With
    `append_inputs: true`, those the task does not reference are appended as context.
    """
    from gwenflow import Agent
    from gwenflow.registry import make_llm, make_tools

    task = compose_task(parameters.get("task", ""), input_data, bool(parameters.get("append_inputs", False)))

    harness = parameters.get("harness")
    if harness:
        if "instructions" in parameters:
            raise ValueError("an agent node has either `harness` or `instructions`, not both")
        from gwenflow.harness import Harness

        loaded = Harness.load(harness)
        model, provider = parameters.get("model"), parameters.get("provider")
        if provider:
            # The harness model runs on that provider's client instead of the harness default.
            name = model or loaded.config.model
            if not name:
                raise ValueError("an agent node with `provider` needs a `model` (in the node or in the harness)")
            agent = loaded.build_agent(llm=make_llm(f"{provider}/{name}"))
        else:
            agent = loaded.build_agent(model=model)
        return agent.run(task).content

    instructions = parameters.get("instructions", "")
    if isinstance(instructions, str):
        instructions = [instructions]

    agent = Agent(
        name=parameters.get("name", "Flow Agent"),
        instructions=instructions,
        llm=agent_llm(parameters),
        tools=make_tools(parameters.get("tools")),
    )
    return agent.run(task).content


@node_type("gwenflow.postgresQuery")
def handle_postgres(parameters: dict, input_data: dict | None) -> Any:
    import psycopg
    from psycopg.rows import dict_row

    conn_string = parameters.get("uri")
    if not conn_string:
        host = parameters.get("host", "localhost")
        port = parameters.get("port", 5432)
        dbname = parameters["database"]
        user = parameters["user"]
        password = parameters.get("password", "")
        conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    query = parameters["query"]
    query_params = parameters.get("params") or []

    with psycopg.connect(conn_string, row_factory=dict_row) as conn:
        left = remaining_time()
        if left is not None:
            raise_if_stopped()
            conn.execute(f"SET statement_timeout = {max(1, int(left * 1000))}")
        with conn.cursor() as cur:
            cur.execute(query, query_params)
            if cur.description is None:
                conn.commit()
                return {"affected_rows": cur.rowcount}
            rows = cur.fetchall()
            return [dict(row) for row in rows]


@node_type("gwenflow.pdfReader", idempotent=True, paths=("file",))
def handle_pdf_reader(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    from gwenflow.readers.pdf import PDFReader

    file_path = parameters["file"]
    reader = PDFReader()
    documents = reader.read(Path(file_path))
    return [
        {
            "page": doc.metadata.get("page"),
            "content": doc.content,
            "tables": doc.metadata.get("tables", []),
        }
        for doc in documents
    ]


@node_type("gwenflow.csvReader", idempotent=True, paths=("file",))
def handle_csv_reader(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    from gwenflow.readers.csv import CSVReader

    reader = CSVReader()
    documents = reader.read(
        Path(parameters["file"]),
        sep=parameters.get("sep", ","),
        decimal=parameters.get("decimal", "."),
        max_rows=parameters.get("max_rows"),
    )
    if not documents:
        return []
    doc = documents[0]
    return {"content": doc.content, **doc.metadata}


@node_type("gwenflow.excelReader", idempotent=True, paths=("file",))
def handle_excel_reader(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    from gwenflow.readers.excel import ExcelReader

    reader = ExcelReader()
    documents = reader.read(
        Path(parameters["file"]),
        sheet_name=parameters.get("sheet_name"),
        max_rows=parameters.get("max_rows"),
    )
    return [{"content": doc.content, **doc.metadata} for doc in documents]


@node_type("gwenflow.jsonReader", idempotent=True, paths=("file",))
def handle_json_reader(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    path = Path(parameters["file"])
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@node_type("gwenflow.websiteReader", idempotent=True)
def handle_website_reader(parameters: dict, input_data: dict | None) -> Any:
    from gwenflow.readers.website import WebsiteReader

    reader = WebsiteReader(
        max_depth=parameters.get("max_depth", 1),
        max_links=parameters.get("max_links", 100000),
        delay=parameters.get("delay", False),
    )
    documents = reader.read(parameters["url"])
    return [{"url": doc.metadata.get("url"), "content": doc.content} for doc in documents]


@node_type("gwenflow.csvWriter", idempotent=True, paths=("file",))
def handle_csv_writer(parameters: dict, input_data: dict | None) -> Any:
    import csv
    from pathlib import Path

    file_path = parameters["file"]
    data = (
        input_data.get(parameters["source"])
        if parameters.get("source")
        else next(iter(input_data.values()))
        if input_data
        else []
    )
    if not data or not isinstance(data, list):
        raise ValueError("csv_writer expects a list of dicts from a parent node.")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter=parameters.get("sep", ","))
        writer.writeheader()
        writer.writerows(data)
    return {"file": str(path), "rows_written": len(data)}


@node_type("gwenflow.jsonWriter", idempotent=True, paths=("file",))
def handle_json_writer(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    file_path = parameters["file"]
    data = (
        input_data.get(parameters["source"])
        if parameters.get("source")
        else next(iter(input_data.values()))
        if input_data
        else None
    )
    if data is None:
        raise ValueError("json_writer expects data from a parent node.")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"file": str(path)}


@node_type("gwenflow.textSplitter", idempotent=True)
def handle_text_splitter(parameters: dict, input_data: dict | None) -> Any:
    from gwenflow.parsers.text_splitters import TokenTextSplitter
    from gwenflow.types.document import Document

    splitter = TokenTextSplitter(
        chunk_size=parameters.get("chunk_size", 500),
        chunk_overlap=parameters.get("chunk_overlap", 100),
        encoding_name=parameters.get("encoding_name", "cl100k_base"),
    )

    raw = next(iter(input_data.values())) if input_data else parameters.get("text", "")
    if isinstance(raw, str):
        documents = [Document(id="0", content=raw, metadata={})]
    elif isinstance(raw, list):
        documents = [
            Document(
                id=str(i), content=item.get("content", ""), metadata={k: v for k, v in item.items() if k != "content"}
            )
            for i, item in enumerate(raw)
            if isinstance(item, dict)
        ]
    else:
        raise ValueError("text_splitter expects a string or list of dicts with a 'content' key.")

    chunks = splitter.split_documents(documents)
    return [{"chunk_id": c.metadata.get("chunk_id"), "content": c.content} for c in chunks]


@node_type(
    "gwenflow.opensearchQuery",
    idempotent=lambda p: p.get("operation", "search") in READ_ONLY_OPENSEARCH_OPERATIONS,
    paths=("ca_certs",),
)
def handle_opensearch_query(parameters: dict, input_data: dict | None) -> Any:
    from gwenflow.stores.opensearch import OpenSearchDocumentStore

    store = OpenSearchDocumentStore(
        uri=parameters["uri"],
        index=parameters["index"],
        use_ssl=parameters.get("use_ssl", True),
        verify_certs=parameters.get("verify_certs", False),
        ca_certs=parameters.get("ca_certs"),
        timeout=parameters.get("timeout", 30),
        aws=parameters.get("aws", False),
    )

    operation = parameters.get("operation", "search")

    if operation == "search":
        return store.search(
            query=parameters["query"],
            aggs=parameters.get("aggs"),
            sort=parameters.get("sort"),
            page=parameters.get("page", 1),
            per_page=parameters.get("per_page", 5000),
            fields=parameters.get("fields"),
            all=parameters.get("all", False),
        )
    elif operation == "count":
        return {"count": store.count(parameters["query"])}
    elif operation == "get":
        return store.get(parameters["id"])
    elif operation == "put":
        data = parameters.get("documents") or (next(iter(input_data.values())) if input_data else [])
        return {"success": store.put(data, column_id=parameters.get("column_id", "id"))}
    elif operation == "delete":
        ids = parameters.get("ids")
        return {"success": store.delete(ids)}
    else:
        raise ValueError(f"Unknown operation '{operation}'. Expected: search, count, get, put, delete.")


@node_type("gwenflow.sqliteQuery", paths=("database",))
def handle_sqlite_query(parameters: dict, input_data: dict | None) -> Any:
    import sqlite3

    db_path = parameters["database"]
    query = parameters["query"]
    query_params = parameters.get("params") or []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.set_progress_handler(stop_requested, 10_000)
        try:
            cur = conn.execute(query, query_params)
            if cur.description is None:
                conn.commit()
                return {"affected_rows": cur.rowcount}
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            raise_if_stopped()
            raise
