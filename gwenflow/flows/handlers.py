import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import httpx

NODE_TYPE_REGISTRY: dict[str, Callable] = {}


def node_type(type_name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        NODE_TYPE_REGISTRY[type_name] = fn
        return fn

    return decorator


@node_type("gwenflow.Python")
def handle_python(parameters: dict, input_data: dict | None) -> str:
    args = [str(a) for a in parameters.get("args", [])]
    stdin = json.dumps(input_data) if input_data else None
    result = subprocess.run(
        [sys.executable, parameters["file"], *args],
        cwd=parameters.get("directory", "."),
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@node_type("gwenflow.httpRequest")
def handle_http_request(parameters: dict, input_data: dict | None) -> Any:
    url = parameters["url"]
    method = parameters.get("method", "GET").upper()
    headers = dict(parameters.get("headers", {}))
    query_params = parameters.get("query_params", {})
    body = parameters.get("body")
    timeout = parameters.get("timeout", 30)

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


@node_type("gwenflow.Agent")
def handle_agent(parameters: dict, input_data: dict | None) -> str:
    from gwenflow import Agent, ChatOpenAI

    raw_model = parameters.get("model", "openai/gpt-4o-mini")
    model = raw_model.split("/", 1)[-1]

    instructions = parameters.get("instructions", "")
    if isinstance(instructions, str):
        instructions = [instructions]

    task = parameters.get("task", "")
    if input_data:
        for name, value in input_data.items():
            serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            task = task.replace(f"{{{name}}}", serialized)

    llm = ChatOpenAI(model=model)
    agent = Agent(
        name="Flow Agent",
        instructions=instructions,
        llm=llm,
        tools=[],
    )

    response = agent.run(task)
    return response.content


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
        with conn.cursor() as cur:
            cur.execute(query, query_params)
            if cur.description is None:
                conn.commit()
                return {"affected_rows": cur.rowcount}
            rows = cur.fetchall()
            return [dict(row) for row in rows]


@node_type("gwenflow.pdfReader")
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


@node_type("gwenflow.csvReader")
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


@node_type("gwenflow.excelReader")
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


@node_type("gwenflow.jsonReader")
def handle_json_reader(parameters: dict, input_data: dict | None) -> Any:
    from pathlib import Path

    path = Path(parameters["file"])
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@node_type("gwenflow.websiteReader")
def handle_website_reader(parameters: dict, input_data: dict | None) -> Any:
    from gwenflow.readers.website import WebsiteReader

    reader = WebsiteReader(
        max_depth=parameters.get("max_depth", 1),
        max_links=parameters.get("max_links", 100000),
        delay=parameters.get("delay", False),
    )
    documents = reader.read(parameters["url"])
    return [{"url": doc.metadata.get("url"), "content": doc.content} for doc in documents]


@node_type("gwenflow.csvWriter")
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


@node_type("gwenflow.jsonWriter")
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


@node_type("gwenflow.textSplitter")
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


@node_type("gwenflow.opensearchQuery")
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


@node_type("gwenflow.sqliteQuery")
def handle_sqlite_query(parameters: dict, input_data: dict | None) -> Any:
    import sqlite3

    db_path = parameters["database"]
    query = parameters["query"]
    query_params = parameters.get("params") or []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, query_params)
        if cur.description is None:
            conn.commit()
            return {"affected_rows": cur.rowcount}
        return [dict(row) for row in cur.fetchall()]
