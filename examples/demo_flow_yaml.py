"""
Demonstrates FlowRunner: a YAML-defined pipeline where nodes are wired together
and executed in topological order. The output of each node is passed as input
to its downstream nodes.

Run:
    uv run python examples/demo_flow_yaml.py
"""

import dotenv

from gwenflow import FlowRunner

dotenv.load_dotenv(override=True)

runner = FlowRunner("examples/flows/news_summary.yaml")
runner.run()
