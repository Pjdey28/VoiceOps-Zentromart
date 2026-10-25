
from pprint import pprint

from src.tools import _dataset_tools


def main():
    print("=== lookup_order: nonexistent test ID ===")

    result = _dataset_tools.lookup_order(
        order_id="INVALID_TEST_ORDER_ID",
        verification_value="0000",
    )

    print("Return type:", type(result).__name__)
    print("Return value:")
    pprint(result, sort_dicts=False)

    print("\n=== create_ticket signature ===")
    print("Ticket creation supports optional order_id, reason, and intent.")
    print("No ticket was created by this test.")


if __name__ == "__main__":
    main()
