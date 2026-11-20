"""Tools of the harness: every public function with a docstring and type annotations becomes a tool.

(A module may instead define a module-level `tools` list of `gwenflow.Tool` objects.)
"""

ORDERS = {
    "NB-1042": {"customer": "Lena", "bike": "City One", "status": "delivered", "delivered_on": "2026-10-28"},
    "NB-1057": {"customer": "Lena", "bike": "Cargo Max", "status": "shipped", "tracking": "DK123456789"},
    "NB-1101": {"customer": "Omar", "bike": "City One", "status": "preparing"},
}


def order_status(order_id: str) -> dict:
    """The status of an order.

    Args:
        order_id: The order number, as printed on the confirmation email (e.g. NB-1042).
    """
    return ORDERS.get(order_id.strip().upper(), {"error": f"no order {order_id}"})


def start_return(order_id: str, reason: str) -> dict:
    """Open a return for a delivered order and give the customer a return label number.

    Args:
        order_id: The order number.
        reason: Why the customer returns the bike, in their words.
    """
    order = ORDERS.get(order_id.strip().upper())
    if order is None or order["status"] != "delivered":
        return {"error": "only a delivered order can be returned"}
    return {"return_label": f"RET-{order_id[-4:]}", "reason": reason, "refund": "within 14 days of reception"}
