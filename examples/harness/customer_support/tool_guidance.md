<!-- How to use the tools (tools/*.py) and the subagents (subagents/). -->

- `order_status`: call it with the order number the customer gives you, before answering.
- `start_return`: only once the customer has clearly asked for a return and the order is delivered.
- `ask_technician`: for any question about how the bike works or a malfunction. Give it the bike
  model and what the customer observes; you do not need the order for that.
