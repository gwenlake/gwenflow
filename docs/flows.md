# Flows

A flow is a graph of nodes described in YAML or JSON. Each node has a `name`, a `type`
and `parameters`; `connections` says which nodes feed which. A node receives the results
of its parents as `input_data`, a dict keyed by parent name.

```yaml
name: News Summary Pipeline

settings:
  parallel: true          # default
  node_timeout: 120       # seconds, default for every node
  node_retries: 0         # default for every node

nodes:
  - name: fetch_news
    type: gwenflow.httpRequest
    parameters:
      url: https://hacker-news.firebaseio.com/v0/topstories.json
    settings:
      timeout: 30
      retries: 2

  - name: summarize
    type: gwenflow.Agent
    parameters:
      model: gpt-5-mini
      instructions: You are a tech journalist who writes concise summaries.
      task: "Top story ids:\n\n{fetch_news}\n\nWhat topics are trending?"

connections:
  fetch_news:
    - - node: summarize
```

```python
from gwenflow import Flow, FlowRunner

results = FlowRunner("flows/news_summary.yaml").run()   # prints the last result, returns them all
results = Flow.from_file("flows/news_summary.yaml").run()        # prints nothing
results = await Flow.from_file("flows/news_summary.yaml").arun() # from async code
```

`run()` is a blocking call. Called from async code (a FastAPI endpoint, a notebook), it
runs the flow in a thread of its own, with the caller's context variables, and waits for
it; prefer `await ...arun()` there, which runs on the current loop.

## Order of execution

A node starts as soon as all its parents are done, so two nodes without a path between
them run at the same time, each in a worker thread. This holds locally and in a Temporal
workflow.

Two such nodes must not depend on each other through a side effect (one writes a file the
other reads, both write the same SQLite database). To find them:

```
$ gwenflow check flows/etl.yaml
Flow 'etl': 5 node(s), valid.
Nodes that may run at the same time:
  orders | clients  <- both declare /data/shop.sqlite
  report | audit
1 pair(s) declare the same path: connect the two nodes, or set `settings.parallel: false`.
```

`Flow.concurrent_pairs()` returns these pairs, and `Flow.concurrent_report()` adds the
paths both nodes declare (`file`, `database`... as given by `node_paths()`). A shared
path is a hint, not a proof: two nodes can also meet in a table or a remote service.

To run the nodes one at a time, as flows did up to v1.5.1:

```yaml
settings:
  parallel: false
```

The order is then `Flow.sequential_order()`: parents first, and otherwise the order
`networkx.topological_sort` gave in v1.5.1. For nodes that are not connected at all, it is
the declaration order.

## Results

Locally, a node receives the results of its parents as they are: any Python object can
go from one node to the next. In a Temporal workflow results are recorded in the
history, so they are converted to JSON data (see [durability.md](durability.md)).

## Timeouts, retries, credentials

- `settings.timeout` (node) or `settings.node_timeout` (flow): seconds allowed for one
  attempt.
- `settings.retries` (node) or `settings.node_retries` (flow): extra attempts after a
  failure. Without it, node types that are safe to repeat are retried up to 2 times:
  the file readers and writers, `websiteReader`, `textSplitter`, `httpRequest` with GET,
  HEAD or OPTIONS, and `opensearchQuery` search, count and get. Set `retries: 0` to switch
  it off. Errors that a retry cannot fix (a missing file, a wrong parameter) are not
  retried.
- `credentials`: parameter name -> name of the environment variable that holds its value,
  read when the node runs. An entry that does not name a variable is ignored, with a
  warning.

## Agent node

| Parameter | |
|---|---|
| `instructions` | the system prompt, a string or a list |
| `harness` | a harness directory or agent file, in place of `instructions` |
| `task` | the prompt; `{node_name}` is replaced by the result of that parent |
| `model` | the model name |
| `provider` | the provider of the client (see below) |
| `tools` | built-in tool names (`wikipedia`, `duckduckgo`, `python`...) |
| `append_inputs` | `true`: parent results the task does not reference are appended as context |
| `name` | the agent's name, `Flow Agent` by default |

Without `provider`, the client is the OpenAI one (so also any OpenAI-compatible endpoint
set by `OPENAI_BASE_URL`). A prefix that names a provider is removed and never chooses
the client: `gpt-5-mini` and `openai/gpt-5-mini` are the same model, and
`mistral/mistral-large-latest` runs on the OpenAI client as `mistral-large-latest`, with
a warning. Any other prefix is part of the model name and is sent whole, so an
open-weights model keeps its organization: `Qwen/Qwen3-32B`,
`meta-llama/Llama-3.3-70B-Instruct`.

With `provider` (`openai`, `anthropic`, `azure`, `google`, `mistral`, `ollama`,
`deepseek`, `gwenlake`), the client is that provider's and `model` is sent as written:

```yaml
parameters:
  provider: mistral
  model: mistral-large-latest
---
parameters:
  provider: openai            # an organization that is also a provider's name:
  model: google/gemma-3-27b-it   # sent whole only with `provider`
```

With a `harness`, `provider` runs the harness on that provider's client (with `model`,
else the harness model) instead of the harness default.

## Customizing a run

`FlowRunner._execute_node(node, input_data)` runs one attempt of one node. Override it
to audit, cache or replace what nodes do:

```python
class Audited(FlowRunner):
    def _execute_node(self, node, input_data):
        log.info("node %s", node.name)
        return super()._execute_node(node, input_data)
```

It is called in the node's worker thread, once per attempt, inside the node's timeout and
retries. It is not called in a Temporal workflow, where nodes are activities.

New node types are registered with `@node_type`:

```python
from gwenflow.flows.handlers import node_type

@node_type("acme.Lookup", idempotent=True, paths=("file",))
def lookup(parameters, input_data):
    ...
```

## Coming from v1.5.1

A flow written for v1.5.1 runs as it did, with these differences:

- two nodes that are not connected run in parallel, not in declaration order
  (`settings.parallel: false` restores it);
- two nodes of the same name: the flow is refused;
- `run()` returns a dict of every node's result instead of `None`;
- an invalid flow raises `FlowError`, a failed node `FlowNodeError`;
- an unknown node type is found before anything runs;
- nodes run in worker threads, not in the main thread;
- node types that are safe to repeat are retried up to 2 times (see above);
- an agent `model` whose prefix is not a provider's name is sent whole (`Qwen/Qwen3-32B`;
  v1.5.1 sent `Qwen3-32B`);
- a failing `gwenflow.Python` script raises a `RuntimeError` carrying its stderr.
