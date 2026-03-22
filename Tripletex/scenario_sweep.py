import importlib
import json
import os
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv("TRIPLETEX_BASE_URL", "").rstrip("/")
session_token = os.getenv("TRIPLETEX_SESSION_TOKEN", "").strip()

if not base_url or not session_token:
    raise SystemExit("Missing TRIPLETEX_BASE_URL or TRIPLETEX_SESSION_TOKEN in environment/.env")

os.environ["TRIPLETEX_DRY_RUN"] = "false"
os.environ["TRIPLETEX_DEBUG_RESPONSE"] = "true"

import main
importlib.reload(main)
from fastapi.testclient import TestClient

client = TestClient(main.app)

state = {
    "customer_id": None,
    "invoice_id": None,
    "employee_id": None,
    "order_id": None,
}

scenario_prompts = [
    ("customer", "Opprett kunde Scenario Kunde AS med e-post scenario.kunde@example.no"),
    ("employee", "Opprett en ansatt med navn Kari Nordmann, kari.nordmann@example.org. Hun skal være kontoadministrator."),
    ("project", "Opprett prosjekt for kunde Scenario Kunde AS"),
    ("invoice", "Opprett faktura for kunde Scenario Kunde AS med produkt Scenario Produkt og beløp 1250"),
    ("travelExpense", "Registrer reiseutgift på 799 NOK i dag"),
    ("department", "Opprett avdeling Scenario Department"),
]

results = []

for name, prompt in scenario_prompts:
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": base_url,
            "session_token": session_token,
        },
    }
    response = client.post("/solve", json=payload)
    body = response.json()
    details = body.get("details", {})
    ok = (
        response.status_code == 200
        and body.get("status") == "completed"
        and details.get("status") == "completed"
    )

    if name == "customer":
        customer = details.get("customer", {})
        state["customer_id"] = (customer.get("value", {}) or {}).get("id") or customer.get("id")
    if name == "employee":
        employee = details.get("employee", {})
        state["employee_id"] = (employee.get("value", {}) or {}).get("id") or employee.get("id")
    if name == "invoice":
        invoice = details.get("invoice", {})
        order = details.get("order", {})
        state["invoice_id"] = (invoice.get("value", {}) or {}).get("id")
        state["order_id"] = (order.get("value", {}) or {}).get("id")

    results.append(
        {
            "scenario": name,
            "http": response.status_code,
            "solve_status": body.get("status"),
            "details_status": details.get("status"),
            "ok": ok,
            "error": details.get("error"),
            "details": details,
            "metrics": body.get("metrics", {}),
        }
    )

if state["invoice_id"]:
    prompt = f"Registrer betaling på 1250 for invoice_id {state['invoice_id']}"
    payload = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": base_url,
            "session_token": session_token,
        },
    }
    response = client.post("/solve", json=payload)
    body = response.json()
    details = body.get("details", {})
    ok = (
        response.status_code == 200
        and body.get("status") == "completed"
        and details.get("status") == "completed"
    )
    results.append(
        {
            "scenario": "payment",
            "http": response.status_code,
            "solve_status": body.get("status"),
            "details_status": details.get("status"),
            "ok": ok,
            "error": details.get("error"),
            "details": details,
            "metrics": body.get("metrics", {}),
        }
    )
else:
    results.append(
        {
            "scenario": "payment",
            "http": None,
            "solve_status": "skipped",
            "details_status": "skipped",
            "ok": False,
            "error": "Skipped because invoice was not created",
            "details": {},
            "metrics": {},
        }
    )

correction_prompt = None
if state["order_id"]:
    correction_prompt = f"Slett order_id {state['order_id']}"
elif state["invoice_id"]:
    correction_prompt = f"Slett invoice_id {state['invoice_id']}"
elif state["employee_id"]:
    correction_prompt = f"Slett employee_id {state['employee_id']}"

if correction_prompt:
    payload = {
        "prompt": correction_prompt,
        "files": [],
        "tripletex_credentials": {
            "base_url": base_url,
            "session_token": session_token,
        },
    }
    response = client.post("/solve", json=payload)
    body = response.json()
    details = body.get("details", {})
    ok = (
        response.status_code == 200
        and body.get("status") == "completed"
        and details.get("status") == "completed"
    )
    results.append(
        {
            "scenario": "correction",
            "http": response.status_code,
            "solve_status": body.get("status"),
            "details_status": details.get("status"),
            "ok": ok,
            "error": details.get("error"),
            "details": details,
            "metrics": body.get("metrics", {}),
        }
    )
else:
    results.append(
        {
            "scenario": "correction",
            "http": None,
            "solve_status": "skipped",
            "details_status": "skipped",
            "ok": False,
            "error": "No created entities to delete",
            "details": {},
            "metrics": {},
        }
    )

summary = {
    "results": results,
    "state": state,
    "pass_count": sum(1 for item in results if item["ok"]),
    "total": len(results),
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
