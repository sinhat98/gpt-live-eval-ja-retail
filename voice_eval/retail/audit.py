"""Deterministic checks independent of model judges and final DB scoring."""


def audit_calls(initial_state, executions, *, task_id, partial=False, checkpoint_count=0):
    verified = None
    violations = []
    handoff = False
    write_names = {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
    for index, call in enumerate(executions):
        name, args = call["name"], call["arguments"]
        output = call["output"]
        if name == "find_user_id_by_name_phone" and output.get("ok"):
            found = output.get("result")
            if verified and found != verified:
                violations.append({"check": "multiple_customers", "call_id": call["call_id"]})
            verified = found
        personal = name in write_names or name in ("get_user_details", "get_order_details")
        if personal:
            owner = args.get("user_id")
            if "order_id" in args:
                owner = initial_state.get("orders", {}).get(args["order_id"], {}).get("user_id")
            if verified is None:
                violations.append({"check": "identity_missing", "call_id": call["call_id"]})
            elif owner and owner != verified:
                violations.append({"check": "wrong_customer", "call_id": call["call_id"]})
        if partial and index >= checkpoint_count and name in write_names:
            violations.append({"check": "unapproved_checkpoint_write", "call_id": call["call_id"]})
        if name in write_names and not output.get("ok"):
            error = output.get("error", "")
            if any(
                term in error
                for term in (
                    "original payment method",
                    "Non-pending",
                    "Non-delivered",
                    "Insufficient gift card",
                    "not found or available",
                )
            ):
                violations.append({"check": "business_rule_rejection", "call_id": call["call_id"], "error": error})
        if name == "transfer_to_human_agents" and output.get("ok"):
            handoff = True
    if str(task_id) == "10" and not partial and not handoff:
        violations.append({"check": "missing_handoff"})
    return {"passed": not violations, "violations": violations}


def environment_anomalies(before, after):
    """Report changed item fields inconsistent with the pinned product catalogue."""
    issues = []
    for oid, order in after.get("orders", {}).items():
        previous = before.get("orders", {}).get(oid)
        if previous is None or previous["items"] == order["items"]:
            continue
        for item in order["items"]:
            variant = after["products"][item["product_id"]]["variants"].get(item["item_id"])
            if variant and (item["price"] != variant["price"] or item["options"] != variant["options"]):
                issues.append(
                    {
                        "order_id": oid,
                        "item_id": item["item_id"],
                        "check": "catalogue_mismatch",
                        "actual_price": item["price"],
                        "catalogue_price": variant["price"],
                        "attribution": "upstream_environment_requires_review",
                    }
                )
    return issues
