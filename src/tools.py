
"""Adapters for the official ZentroMart mock tools."""

import importlib.util
import inspect

from src.config import DATASET_DIR

_TOOL_FILE = DATASET_DIR / "tools" / "tools.py"


def _load_dataset_tools():
    if not _TOOL_FILE.is_file():
        raise FileNotFoundError(
            f"ZentroMart tools file not found: {_TOOL_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "zentromart_dataset_tools", _TOOL_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tools from {_TOOL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("lookup_order", "create_ticket"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"Missing supplied tool: {name}()")

    return module


_dataset_tools = _load_dataset_tools()


def lookup_order(order_id, verification_value=None):
    """Use the supplied lookup tool; verification is optional for lookup."""
    function = _dataset_tools.lookup_order
    parameters = inspect.signature(function).parameters

    if "order_id" not in parameters:
        raise TypeError("Supplied lookup_order has no order_id parameter.")

    kwargs = {"order_id": str(order_id)}

    verification_parameter = next(
        (
            name for name in (
                "verification_value",
                "verification",
                "phone_last4",
                "delivery_pincode",
            )
            if name in parameters
        ),
        None,
    )

    if verification_parameter:
        kwargs[verification_parameter] = (
            str(verification_value)
            if verification_value is not None
            else None
        )
    elif verification_value is not None:
        raise TypeError(
            "Supplied lookup_order has no supported verification parameter."
        )

    return function(**kwargs)


def create_ticket(summary, order_id=None, reason=None, intent=None):
    """Create a ticket using the supplied dataset tool."""
    if not str(summary or "").strip():
        raise ValueError("A non-empty ticket summary is required.")

    function = _dataset_tools.create_ticket
    parameters = inspect.signature(function).parameters

    kwargs = {"summary": str(summary).strip()}

    optional_values = {
        "order_id": order_id,
        "reason": reason,
        "intent": intent,
    }

    for name, value in optional_values.items():
        if value is not None and name in parameters:
            kwargs[name] = str(value)

    return function(**kwargs)
