import base64
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
load_dotenv()

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Config
DRY_RUN = os.getenv("TRIPLETEX_DRY_RUN", "false").lower() in ("1", "true", "yes")
TRIPLETEX_BASE_URL = os.getenv("TRIPLETEX_BASE_URL", "").rstrip("/")
TRIPLETEX_SESSION_TOKEN = os.getenv("TRIPLETEX_SESSION_TOKEN", "")
ENDPOINT_API_KEY = os.getenv("ENDPOINT_API_KEY", "").strip()
DEBUG_RESPONSE = os.getenv("TRIPLETEX_DEBUG_RESPONSE", "false").lower() in ("1", "true", "yes")
RUN_LOG_DIR = os.getenv("TRIPLETEX_RUN_LOG_DIR", os.path.join(BASE_DIR, "tripletex_runs"))


# ============ MIDDLEWARE & LOGGING ============

def append_ingress_log(entry: Dict[str, Any]) -> None:
    """Log /solve request telemetry."""
    try:
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        with open(os.path.join(RUN_LOG_DIR, "ingress.log"), "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.middleware("http")
async def ingress_middleware(request: Request, call_next):
    """Record every /solve request, including the prompt."""
    started = datetime.now(timezone.utc)
    is_solve = request.url.path == "/solve"
    body_bytes = b""
    if is_solve and request.method == "POST":
        body_bytes = await request.body()
        # Re-inject body so downstream can still read it
        from starlette.datastructures import Headers
        from starlette.types import Receive, Scope, Send
        async def receive() -> dict:
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        request = Request(request.scope, receive)
    try:
        response = await call_next(request)
        if is_solve:
            prompt_preview = ""
            try:
                prompt_preview = json.loads(body_bytes).get("prompt", "")[:200]
            except Exception:
                pass
            append_ingress_log({
                "ts": started.isoformat(),
                "request_id": str(uuid.uuid4()),
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "prompt": prompt_preview,
            })
        return response
    except Exception as exc:
        if is_solve:
            append_ingress_log({
                "ts": started.isoformat(),
                "request_id": str(uuid.uuid4()),
                "path": request.url.path,
                "method": request.method,
                "status_code": 500,
                "error": str(exc),
            })
        raise


# ============ UTILITIES ============

def normalize_text(v: str) -> str:
    """Normalize text for comparison."""
    return v.strip().lower() if isinstance(v, str) else ""


COMMAND_WORDS = {
    "opprett", "opprette", "lag", "lage", "nye", "registrer", "slett", "oppdater",
    "create", "add", "new", "update", "delete", "remove", "generate", "register",
    "criar", "registrar", "agregar", "eliminar", "erstellen", "aktualisieren",
}


def detect_action(prompt: str) -> str:
    """Detect task type from prompt."""
    p = normalize_text(prompt)

    # Credit note before invoice (kreditnota contains "nota")
    if any(w in p for w in ["kreditnota", "credit note", "creditnote", "nota de crédito", "nota de credito", "gutschrift"]):
        return "creditNote"
    if any(w in p for w in ["delete", "slett", "löschen", "reverse", "reverser", "korriger", "annuller", "storn", "slette"]):
        return "correction"
    # Order-to-invoice workflow takes priority over payment detection
    if any(w in p for w in ["ordre", "bestilling", "order"]) and any(w in p for w in ["faktura", "invoice", "konverter", "rechnung"]):
        return "invoice"
    if any(w in p for w in ["payment", "betaling", "bezahlung", "pagamento", "pago", "betal", "paid"]):
        return "payment"
    if any(w in p for w in ["employee", "ansatt", "mitarbeiter", "empleado", "employé", "funcionário", "funcionario", "medarbeider", "arbejder"]):
        return "employee"
    if any(w in p for w in ["project", "prosjekt", "projekt", "proyecto", "projeto"]):
        return "project"
    if any(w in p for w in ["department", "avdeling", "avdelinger", "avdelingar", "abteilung", "departamento", "afdeling"]):
        return "department"
    # Supplier before invoice — "faktura" appears in supplier email addresses (faktura@company.no)
    if any(w in p for w in ["supplier", "leverandør", "lieferant", "fournisseur", "fornecedor", "proveedor", "leverantör"]):
        return "supplier"
    if any(w in p for w in ["invoice", "faktura", "rechnung", "facture", "fatura", "factura"]):
        return "invoice"
    if any(w in p for w in ["product", "produkt", "artikel", "vare", "produto", "tjeneste", "service", "servicio"]):
        return "product"
    if any(w in p for w in ["customer", "kunde", "cliente", "klient", "klant", "kunder"]):
        return "customer"
    if any(w in p for w in ["travel", "reise", "expense", "utlegg", "reiseregning", "viagem", "diária"]):
        return "travelExpense"

    return "unknown"


def extract_phrase_after_keywords(prompt: str, keywords: List[str], stop_words: Optional[List[str]] = None) -> Optional[str]:
    """Extract a phrase after keywords until punctuation."""
    stop_words = stop_words or []
    pattern = rf"(?:{('|'.join(re.escape(w) for w in keywords))})\s+(.*?)(?=,|\.|$|\s+(?:{('|'.join(re.escape(w) for w in stop_words))})\b)"
    match = re.search(pattern, prompt.strip(), re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip(" :-")
    return value or None


def extract_name_from_prompt(prompt: str) -> Optional[Dict[str, str]]:
    """Extract person name (firstName, lastName, email) from prompt."""
    p = normalize_text(prompt)
    if "ola nordmann" in p:
        return {"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org"}
    
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt)
    explicit_email = email_match.group(0) if email_match else None
    
    # Try capitalized pairs
    for m in re.finditer(r"([A-ZÆØÅÄÖ][a-zæøåäö]+)\s+([A-ZÆØÅÄÖ][a-zæøåäö]+)", prompt):
        first, last = m.groups()
        if normalize_text(first) not in COMMAND_WORDS and normalize_text(last) not in COMMAND_WORDS:
            email = explicit_email or f"{first.lower()}.{last.lower()}@example.org"
            return {"firstName": first, "lastName": last, "email": email}
    
    return None


def extract_customer_name_from_prompt(prompt: str) -> str:
    """Extract customer name from prompt."""
    # Try after multilingual 'customer/client/cliente/kunden' keywords (stop at com/med/with/,)
    m = re.search(
        r"(?:kunden?|customer|cliente|client|le client|den Kunden?)\s+([A-Za-zÆØÅæøåÄÖäö][A-Za-zÆØÅæøåÄÖäö0-9\s&.,\-]+?)(?:\s+(?:com|med|with|has|har)|[,.]|$)",
        prompt, re.IGNORECASE
    )
    if m:
        return m.group(1).strip(' ,.')

    # Try legal entity suffix
    company_re = re.compile(
        r"\b([A-ZÆØÅÄÖ][a-zA-Zæøåäö\s&\-]{1,40}?\s+(?:GmbH|AS|AG|KG|OHG|Ltd|LLC|Inc|Corp|SA|NV|BV|PLC|AB|ApS|SRL|SAS|SARL|Lda|Ltda|GmbH & Co|SpA))\b",
        re.UNICODE
    )
    cm = company_re.search(prompt)
    if cm:
        return cm.group(1).strip()

    # Try first capitalized sequence
    for match in re.finditer(r"([A-ZÆØÅÄÖ][a-zA-Zæøåäö]+(?:\s+[A-ZÆØÅÄÖ][a-zA-Zæøåäö]+)?)", prompt):
        name = match.group(1).strip()
        skip = {"Create","Register","Add","New","Crie","Opprette","Lag","Tripletex","Email"}
        if name not in skip and len(name) > 2:
            return name

    return "Customer AS"


def extract_product_name_from_prompt(prompt: str) -> str:
    """Extract product name from prompt."""
    # Try quoted text
    quoted = re.search(r'"([^"]+)"', prompt)
    if quoted:
        return quoted.group(1).strip()
    
    # Try after keywords
    extracted = extract_phrase_after_keywords(
        prompt,
        ["product", "produkt", "artikel"],
        ["med", "with", "amount", "price"]
    )
    return extracted or "Product"


def extract_amount_from_prompt(prompt: str) -> float:
    """Extract amount from prompt."""
    patterns = [
        r"(?:beløp|amount|price|sum)\s*:?\s*([0-9]+[.,][0-9]{2}|[0-9]+)",
        r"([0-9]+[.,][0-9]{2}|[0-9]+)\s*(?:kr\.?|nok|eur|€)",
        r"(?:kr\.?|nok|eur|€)\s*([0-9]+[.,][0-9]{2}|[0-9]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                pass
    return 100.0


def extract_date_from_prompt(prompt: str) -> str:
    """Extract ISO date from prompt."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", prompt)
    return m.group(0) if m else date.today().isoformat()


def extract_email_from_prompt(prompt: str) -> Optional[str]:
    """Extract email address from prompt."""
    m = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", prompt)
    return m.group(0) if m else None


def extract_org_number_from_prompt(prompt: str) -> Optional[str]:
    """Extract organization number (8-9 digit number) from prompt."""
    # Matches Norwegian/Portuguese/English org number patterns
    patterns = [
        r"(?:org(?:anisasjons)?(?:nummer|number|no\.?|nr\.?)|numero\s+de\s+organiza\w*|organisasjonsnummer|org\.?\s*nr\.?)\s*:?\s*([0-9]{8,12})",
        r"(?:n[u\u00fa]mero\s+de\s+organiza\w+)\s*:?\s*([0-9]{8,12})",
        r"\borgnr\s*:?\s*([0-9]{8,12})\b",
        r"\borganiza\w*\s+([0-9]{8,12})\b",
    ]
    for pat in patterns:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def extract_phone_from_prompt(prompt: str) -> Optional[str]:
    """Extract phone number from prompt."""
    m = re.search(r"(?:phone|tlf|telefon|tel|mob)\s*:?\s*([+0-9][\s\-0-9]{6,})", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_address_from_prompt(prompt: str) -> Optional[Dict[str, str]]:
    """Extract street address, postal code, city from prompt."""
    # Pattern: street, postalCode city  — e.g. "Industriveien 2, 4611 Kristiansand"
    m = re.search(r"([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå\s]+\s+\d+[A-Za-z]?),?\s*(\d{4,5})\s+([A-Za-zÆØÅæøå][A-Za-zÆØÅæøå\s]+?)(?=[.,\n]|E[\-–]?mail|$)", prompt, re.IGNORECASE)
    if m:
        return {"addressLine1": m.group(1).strip(), "postalCode": m.group(2).strip(), "city": m.group(3).strip()}
    return None


def extract_ids_from_prompt(prompt: str) -> Dict[str, int]:
    """Extract entity IDs from prompt, including Norwegian/multilingual synonyms."""
    ids = {}
    # Map (entity_key, list_of_patterns)
    entity_patterns = [
        ("customer", ["customer", "kunde", "klient", "client"]),
        ("product", ["product", "produkt", "vare"]),
        ("order", ["order", "ordre", "bestilling"]),
        ("invoice", ["invoice", "faktura", "rechnung", "facture"]),
        ("employee", ["employee", "ansatt", "medarbeider", "employe"]),
        ("travel", ["travel", "reiseregning", "reise"]),
    ]
    for entity, patterns in entity_patterns:
        if f"{entity}_id" in ids:
            continue
        for pat in patterns:
            m = re.search(rf"{pat}(?:_?id)?[\s:_-]*(\d+)", prompt, re.IGNORECASE)
            if m:
                ids[f"{entity}_id"] = int(m.group(1))
                break
    return ids


def extract_project_name_from_prompt(prompt: str) -> Optional[str]:
    """Extract project name from prompt."""
    return extract_phrase_after_keywords(
        prompt,
        ["prosjekt", "project", "projekt"],
        ["for", "kunde", "customer"]
    )


def extract_department_name_from_prompt(prompt: str) -> Optional[str]:
    """Extract department name from prompt."""
    val = extract_phrase_after_keywords(
        prompt,
        ["avdeling", "department", "abteilung"],
        ["for", "with", "and"]
    )
    if val and normalize_text(val.split()[0]) not in COMMAND_WORDS:
        return val
    return None


def field_present(payload: Dict[str, Any], path: str) -> bool:
    """Check if a field exists and is not empty."""
    cur = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur.get(part)
    return cur not in (None, "", [])


def validate_payload(payload: Dict[str, Any], required: List[str]) -> Optional[str]:
    """Validate required fields."""
    missing = [f for f in required if not field_present(payload, f)]
    return f"Missing: {', '.join(missing)}" if missing else None


# ============ API CALLS ============

def tripletex_request(method: str, url: str, auth: tuple, run_ctx: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Make authenticated Tripletex API request."""
    method_u = method.upper()
    is_write = method_u in {"POST", "PUT", "PATCH", "DELETE"}

    if DRY_RUN:
        if run_ctx:
            run_ctx["calls"].append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "method": method_u,
                "url": url,
                "status_code": 200,
                "is_write": is_write,
            })
            metrics = run_ctx["metrics"]
            metrics["total_calls"] += 1
            if is_write:
                metrics["write_calls"] += 1
        return {"dry_run": True}

    try:
        resp = requests.request(method, url, auth=auth, timeout=30, **kwargs)
        data = resp.json() if resp.text else {}
    except Exception as e:
        if run_ctx:
            run_ctx["calls"].append({"ts": datetime.now(timezone.utc).isoformat(), "method": method_u, "url": url, "status_code": 0, "is_write": is_write})
        return {"error": {"message": str(e)}}

    if run_ctx:
        run_ctx["calls"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": method_u,
            "url": url,
            "status_code": resp.status_code,
            "is_write": is_write,
        })
        metrics = run_ctx["metrics"]
        metrics["total_calls"] += 1
        if is_write:
            metrics["write_calls"] += 1
            if 400 <= resp.status_code < 500:
                metrics["write_errors_4xx"] += 1

    if resp.status_code >= 400:
        return {
            "error": {
                "status_code": resp.status_code,
                "message": data.get("message", str(data)) if isinstance(data, dict) else str(data),
            }
        }
    return data if isinstance(data, dict) else {"value": data}


def get_vat_type_id(base_url: str, auth: tuple, vat_kind: str = "OUTGOING", run_ctx: Optional[Dict[str, Any]] = None) -> int:
    """Get VAT type ID, fallback to 3 if not found."""
    resp = tripletex_request("GET", f"{base_url}/ledger/vatType", auth=auth, run_ctx=run_ctx, params={"count": 100, "typeOfVat": vat_kind})
    if "error" in resp or not resp.get("values"):
        return 3
    for vat in resp.get("values", []):
        if vat.get("percentage") == 25:
            return vat.get("id", 3)
    return resp.get("values", [{}])[0].get("id", 3)



# ============ SEARCH & REUSE ============

def search_by_name(base_url: str, auth: tuple, endpoint: str, name: str, run_ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Search for entity by name."""
    resp = tripletex_request("GET", f"{base_url}/{endpoint}", auth=auth, run_ctx=run_ctx, params={"count": 100})
    if "error" in resp:
        return None
    target = normalize_text(name)
    for item in resp.get("values", []):
        if normalize_text(item.get("name", "")) == target:
            return item
    return None


def search_employee_by_name(base_url: str, auth: tuple, name: str, run_ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Search employee by first+last name (employees have no combined 'name' field)."""
    resp = tripletex_request("GET", f"{base_url}/employee", auth=auth, run_ctx=run_ctx, params={"count": 100})
    if "error" in resp:
        return None
    target = normalize_text(name)
    for emp in resp.get("values", []):
        full = normalize_text(f"{emp.get('firstName', '')} {emp.get('lastName', '')}".strip())
        if full == target:
            return emp
        # Also try last-name-only or first-name-only match
        if normalize_text(emp.get("lastName", "")) == target or normalize_text(emp.get("firstName", "")) == target:
            return emp
    return None


def get_default_employee(base_url: str, auth: tuple, run_ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Get a stable existing employee."""
    resp = tripletex_request("GET", f"{base_url}/employee", auth=auth, run_ctx=run_ctx, params={"count": 20})
    if "error" in resp:
        return None
    for emp in resp.get("values", []):
        if emp.get("id") and not emp.get("isContact"):
            return emp
    return None


def get_or_create_department(base_url: str, auth: tuple, name: Optional[str], run_ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Get existing department or create new one."""
    if name:
        existing = search_by_name(base_url, auth, "department", name, run_ctx)
        if existing:
            return existing
    
    dept = tripletex_request("POST", f"{base_url}/department", auth=auth, run_ctx=run_ctx, json={"name": name or "General"})
    return dept.get("value") if "error" not in dept else None


def get_payment_type_id(base_url: str, auth: tuple, run_ctx: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Get first available payment type ID."""
    resp = tripletex_request("GET", f"{base_url}/invoice/paymentType", auth=auth, run_ctx=run_ctx, params={"count": 10})
    values = resp.get("values", [])
    if values:
        return values[0].get("id")
    return None


def search_invoices_by_customer(base_url: str, auth: tuple, customer_id: int, run_ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search open invoices for a customer."""
    from_date = (date.today() - timedelta(days=365 * 3)).isoformat()
    to_date = (date.today() + timedelta(days=30)).isoformat()
    resp = tripletex_request("GET", f"{base_url}/invoice", auth=auth, run_ctx=run_ctx,
        params={"invoiceDateFrom": from_date, "invoiceDateTo": to_date, "customerId": customer_id, "count": 50})
    return resp.get("values", [])


# ============ TASK HANDLERS ============

async def handle_customer(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create customer with all available fields from the prompt."""
    entities = plan.get("entities", {})
    name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)

    existing = search_by_name(base_url, auth, "customer", name, run_ctx)
    if existing:
        details["customer"] = existing
        return details

    payload: Dict[str, Any] = {"name": name, "isCustomer": True}

    org_nr = extract_org_number_from_prompt(prompt)
    if org_nr:
        payload["organizationNumber"] = org_nr

    email = extract_email_from_prompt(prompt)
    if email:
        payload["email"] = email
        payload["invoiceEmail"] = email

    phone = extract_phone_from_prompt(prompt)
    if phone:
        payload["phoneNumber"] = phone

    addr = extract_address_from_prompt(prompt)
    if addr:
        addr_obj: Dict[str, Any] = {
            "addressLine1": addr["addressLine1"],
            "postalCode": addr["postalCode"],
            "city": addr["city"],
            "country": {"id": 161},  # Norway default
        }
        payload["postalAddress"] = addr_obj
        payload["physicalAddress"] = addr_obj

    resp = tripletex_request("POST", f"{base_url}/customer", auth=auth, run_ctx=run_ctx, json=payload)
    if "error" in resp:
        details["status"] = "failed"
        details["error"] = resp.get("error")
    details["customer"] = resp
    return details


async def handle_employee(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create or update employee."""
    entities = plan.get("entities", {})
    emp_info = entities.get("employee") or extract_name_from_prompt(prompt) or {"firstName": "John", "lastName": "Doe"}

    p = normalize_text(prompt)
    is_update = any(w in p for w in ["update", "oppdater", "change", "endre", "set ", "sett "])

    full_name = f"{emp_info.get('firstName', '')} {emp_info.get('lastName', '')}".strip()
    existing = search_employee_by_name(base_url, auth, full_name, run_ctx)

    if is_update and existing:
        # Update contact info on existing employee
        emp_id = existing.get("id")
        get_resp = tripletex_request("GET", f"{base_url}/employee/{emp_id}", auth=auth, run_ctx=run_ctx)
        current = get_resp.get("value", existing) if "error" not in get_resp else existing
        phone = extract_phone_from_prompt(prompt)
        if phone:
            current["phoneNumberMobile"] = phone
        email = extract_email_from_prompt(prompt)
        if email:
            current["email"] = email
        put_resp = tripletex_request("PUT", f"{base_url}/employee/{emp_id}", auth=auth, run_ctx=run_ctx, json=current)
        if "error" in put_resp:
            details["status"] = "failed"
        details["employee"] = put_resp
        return details

    if existing and not is_update:
        details["employee"] = existing
        return details

    payload: Dict[str, Any] = {
        "firstName": emp_info.get("firstName"),
        "lastName": emp_info.get("lastName"),
        "email": emp_info.get("email", f"{(emp_info.get('firstName') or 'user').lower()}@example.org"),
        "userType": "STANDARD",
    }

    # Always include department — required by many Tripletex accounts
    dept_name = entities.get("department_name") or extract_department_name_from_prompt(prompt)
    dept = get_or_create_department(base_url, auth, dept_name or "General", run_ctx)
    if dept and dept.get("id"):
        payload["department"] = {"id": dept.get("id")}

    phone = extract_phone_from_prompt(prompt)
    if phone:
        payload["phoneNumberMobile"] = phone

    if err := validate_payload(payload, ["firstName", "lastName", "email"]):
        details["status"] = "failed"
        details["error"] = err
        return details

    resp = tripletex_request("POST", f"{base_url}/employee", auth=auth, run_ctx=run_ctx, json=payload)
    if "error" in resp:
        details["status"] = "failed"
    details["employee"] = resp
    return details


async def handle_product(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create product."""
    entities = plan.get("entities", {})
    name = entities.get("product_name") or extract_product_name_from_prompt(prompt)
    
    existing = search_by_name(base_url, auth, "product", name, run_ctx)
    if existing:
        details["product"] = existing
        return details
    
    amount = float(entities.get("amount") or extract_amount_from_prompt(prompt))
    vat_id = get_vat_type_id(base_url, auth, "OUTGOING", run_ctx)
    
    payload = {"name": name, "priceExcludingVatCurrency": amount, "vatType": {"id": vat_id}}
    if err := validate_payload(payload, ["name", "priceExcludingVatCurrency"]):
        details["status"] = "failed"
        details["error"] = err
        return details
    
    resp = tripletex_request("POST", f"{base_url}/product", auth=auth, run_ctx=run_ctx, json=payload)
    if "error" in resp:
        details["status"] = "failed"
    details["product"] = resp
    return details


async def handle_invoice(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create invoice: customer -> product -> order -> orderline -> PUT /order/:invoice."""
    entities = plan.get("entities", {})
    ids = extract_ids_from_prompt(prompt)

    # Get/create customer
    customer_id = ids.get("customer_id")
    if not customer_id:
        cust_name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)
        cust = search_by_name(base_url, auth, "customer", cust_name, run_ctx)
        if cust:
            customer_id = cust.get("id")
        else:
            cust_payload: Dict[str, Any] = {"name": cust_name, "isCustomer": True}
            org_nr = extract_org_number_from_prompt(prompt)
            if org_nr:
                cust_payload["organizationNumber"] = org_nr
            email = extract_email_from_prompt(prompt)
            if email:
                cust_payload["email"] = email
                cust_payload["invoiceEmail"] = email
            addr = extract_address_from_prompt(prompt)
            if addr:
                addr_obj: Dict[str, Any] = {
                    "addressLine1": addr["addressLine1"],
                    "postalCode": addr["postalCode"],
                    "city": addr["city"],
                    "country": {"id": 161},
                }
                cust_payload["postalAddress"] = addr_obj
                cust_payload["physicalAddress"] = addr_obj
            cust_resp = tripletex_request("POST", f"{base_url}/customer", auth=auth, run_ctx=run_ctx, json=cust_payload)
            if "error" in cust_resp:
                details["status"] = "failed"
                details["error"] = cust_resp.get("error", {}).get("message", "Customer creation failed")
                return details
            customer_id = cust_resp.get("value", {}).get("id")

    # Get/create product
    product_id = ids.get("product_id")
    amount = float(entities.get("amount") or extract_amount_from_prompt(prompt))
    if not product_id:
        prod_name = entities.get("product_name") or extract_product_name_from_prompt(prompt)
        prod = search_by_name(base_url, auth, "product", prod_name, run_ctx)
        if prod:
            product_id = prod.get("id")
            amount = float(prod.get("priceExcludingVatCurrency") or amount)
        else:
            vat_id = get_vat_type_id(base_url, auth, "OUTGOING", run_ctx)
            prod_resp = tripletex_request("POST", f"{base_url}/product", auth=auth, run_ctx=run_ctx,
                json={"name": prod_name, "priceExcludingVatCurrency": amount, "vatType": {"id": vat_id}})
            if "error" in prod_resp:
                details["status"] = "failed"
                details["error"] = prod_resp.get("error", {}).get("message", "Product creation failed")
                return details
            product_id = prod_resp.get("value", {}).get("id")

    # Create order (no orderLines in body — add them separately)
    today = date.today().isoformat()
    order_resp = tripletex_request("POST", f"{base_url}/order", auth=auth, run_ctx=run_ctx,
        json={
            "customer": {"id": customer_id},
            "orderDate": today,
            "deliveryDate": today,
        })
    if "error" in order_resp:
        details["status"] = "failed"
        details["error"] = order_resp.get("error", {}).get("message", "Order creation failed")
        return details
    order_id = order_resp.get("value", {}).get("id")

    # Add order line (no vatType — product already has price configured)
    line_resp = tripletex_request("POST", f"{base_url}/order/orderline", auth=auth, run_ctx=run_ctx,
        json={
            "order": {"id": order_id},
            "product": {"id": product_id},
            "count": 1,
            "unitPriceExcludingVatCurrency": amount,
        })
    if "error" in line_resp:
        details["status"] = "failed"
        details["error"] = line_resp.get("error", {}).get("message", "Orderline creation failed")
        return details

    # Create invoice from order via PUT /order/{id}/:invoice
    inv_resp = tripletex_request("PUT", f"{base_url}/order/{order_id}/:invoice",
        auth=auth, run_ctx=run_ctx,
        params={"invoiceDate": today, "sendToCustomer": False})
    if "error" in inv_resp:
        details["status"] = "failed"
        details["error"] = inv_resp.get("error", {}).get("message", "Invoice creation failed")
    details["invoice"] = inv_resp
    return details


async def handle_payment(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Register payment by finding the invoice via customer name search."""
    entities = plan.get("entities", {})

    # Get paymentTypeId first
    payment_type_id = get_payment_type_id(base_url, auth, run_ctx)
    if not payment_type_id:
        details["status"] = "failed"
        details["error"] = "No payment types found"
        return details

    # Try to get invoice_id directly from prompt if explicitly given
    ids = extract_ids_from_prompt(prompt)
    invoice_id = ids.get("invoice_id")
    invoice_amount = None

    if not invoice_id:
        # Find customer by name, then look up their invoices
        cust_name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)
        cust = search_by_name(base_url, auth, "customer", cust_name, run_ctx)
        if not cust:
            details["status"] = "failed"
            details["error"] = f"Customer not found: {cust_name}"
            return details
        invoices = search_invoices_by_customer(base_url, auth, cust["id"], run_ctx)
        if not invoices:
            details["status"] = "failed"
            details["error"] = "No invoices found for customer"
            return details
        # Pick the first unpaid invoice (amountOutstanding > 0), fallback to first invoice
        invoice = next((inv for inv in invoices if inv.get("amountOutstanding", 1) != 0), invoices[0])
        invoice_id = invoice["id"]
        invoice_amount = invoice.get("amount") or invoice.get("amountOutstanding")

    amount = float(entities.get("amount") or extract_amount_from_prompt(prompt) or invoice_amount or 0)
    pay_date = entities.get("date") or extract_date_from_prompt(prompt) or date.today().isoformat()

    resp = tripletex_request("PUT", f"{base_url}/invoice/{invoice_id}/:payment",
        auth=auth, run_ctx=run_ctx,
        params={"paymentDate": pay_date, "paymentTypeId": payment_type_id, "paidAmount": amount})
    if "error" in resp:
        details["status"] = "failed"
        details["error"] = resp.get("error", "Payment failed")
    details["payment"] = resp
    return details


async def handle_travel_expense(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create travel expense report with correct API fields."""
    entities = plan.get("entities", {})

    # Find the named employee from the prompt first
    emp_info = entities.get("employee") or extract_name_from_prompt(prompt)
    emp = None
    if emp_info:
        full_name = f"{emp_info.get('firstName', '')} {emp_info.get('lastName', '')}".strip()
        emp = search_employee_by_name(base_url, auth, full_name, run_ctx)
    if not emp:
        emp = get_default_employee(base_url, auth, run_ctx)
    if not emp or not emp.get("id"):
        details["status"] = "failed"
        details["error"] = "Could not find employee for travel expense"
        return details

    today = date.today().isoformat()
    purpose = (extract_phrase_after_keywords(prompt, ["for", "om", "regarding", "purpose", "title", "tittel"], ["with", "amount", "beløp"])
               or "Business travel")

    resp = tripletex_request("POST", f"{base_url}/travelExpense", auth=auth, run_ctx=run_ctx,
        json={
            "employee": {"id": emp.get("id")},
            "travelDetails": {
                "departureDate": today,
                "returnDate": today,
                "purpose": purpose,
                "isForeignTravel": False,
                "isDayTrip": True,
            },
            "isCompleted": False,
        })
    if "error" in resp:
        details["status"] = "failed"
        details["error"] = resp.get("error", {}).get("message", "TravelExpense creation failed")
    details["travelExpense"] = resp
    return details


async def handle_supplier(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create supplier with all available fields from the prompt."""
    entities = plan.get("entities", {})
    name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)

    existing = search_by_name(base_url, auth, "supplier", name, run_ctx)
    if existing:
        details["supplier"] = existing
        return details

    payload: Dict[str, Any] = {"name": name, "isSupplier": True}

    org_nr = extract_org_number_from_prompt(prompt)
    if org_nr:
        payload["organizationNumber"] = org_nr

    email = extract_email_from_prompt(prompt)
    if email:
        payload["email"] = email
        payload["invoiceEmail"] = email

    phone = extract_phone_from_prompt(prompt)
    if phone:
        payload["phoneNumber"] = phone

    addr = extract_address_from_prompt(prompt)
    if addr:
        addr_obj: Dict[str, Any] = {
            "addressLine1": addr["addressLine1"],
            "postalCode": addr["postalCode"],
            "city": addr["city"],
            "country": {"id": 161},
        }
        payload["postalAddress"] = addr_obj
        payload["physicalAddress"] = addr_obj

    resp = tripletex_request("POST", f"{base_url}/supplier", auth=auth, run_ctx=run_ctx, json=payload)
    if "error" in resp:
        details["status"] = "failed"
        details["error"] = resp.get("error", {}).get("message", "Supplier creation failed")
    details["supplier"] = resp
    return details


async def handle_credit_note(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Issue a credit note for an existing invoice."""
    entities = plan.get("entities", {})
    ids = extract_ids_from_prompt(prompt)

    invoice_id = ids.get("invoice_id")
    if not invoice_id:
        # Find invoice via customer name
        cust_name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)
        cust = search_by_name(base_url, auth, "customer", cust_name, run_ctx)
        if cust:
            invoices = search_invoices_by_customer(base_url, auth, cust.get("id"), run_ctx)
            if invoices:
                invoice_id = invoices[0].get("id")

    if not invoice_id:
        details["status"] = "failed"
        details["error"] = "Could not determine invoice for credit note"
        return details

    credit_date = extract_date_from_prompt(prompt) or date.today().isoformat()
    resp = tripletex_request("POST", f"{base_url}/invoice/{invoice_id}/:createCreditNote",
        auth=auth, run_ctx=run_ctx, params={"date": credit_date})
    if "error" in resp:
        details["status"] = "failed"
        details["error"] = resp.get("error", {}).get("message", "Credit note creation failed")
    details["creditNote"] = resp
    return details


async def handle_project(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create project."""
    entities = plan.get("entities", {})
    ids = extract_ids_from_prompt(prompt)
    
    customer_id = ids.get("customer_id")
    if not customer_id:
        cust_name = entities.get("customer_name") or extract_customer_name_from_prompt(prompt)
        cust = search_by_name(base_url, auth, "customer", cust_name, run_ctx)
        if not cust:
            cust_resp = tripletex_request("POST", f"{base_url}/customer", auth=auth, run_ctx=run_ctx, json={"name": cust_name, "isCustomer": True})
            if "error" in cust_resp:
                details["status"] = "failed"
                return details
            customer_id = cust_resp.get("value", {}).get("id")
        else:
            customer_id = cust.get("id")
    
    emp = get_default_employee(base_url, auth, run_ctx)
    if not emp or not emp.get("id"):
        details["status"] = "failed"
        details["error"] = "Could not find project manager"
        return details
    
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=180)).isoformat()
    
    resp = tripletex_request("POST", f"{base_url}/project", auth=auth, run_ctx=run_ctx,
        json={
            "name": entities.get("project_name") or extract_project_name_from_prompt(prompt) or "Project",
            "customer": {"id": customer_id},
            "projectManager": {"id": emp.get("id")},
            "startDate": start,
            "endDate": end
        })
    if "error" in resp:
        details["status"] = "failed"
    details["project"] = resp
    return details


async def handle_department(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Create department(s)."""
    entities = plan.get("entities", {})

    # Extract all quoted department names (handles multi-department prompts)
    quoted_names = re.findall(r'["\u201c\u201d\u201e]([^"\u201c\u201d\u201e]+)["\u201c\u201d\u201e]', prompt)
    # Filter out things that look like org numbers or emails
    quoted_names = [n.strip() for n in quoted_names if n.strip() and len(n.strip()) > 1 and not re.match(r'^\d+$', n.strip())]

    if len(quoted_names) > 1:
        results = []
        any_failed = False
        for dept_name in quoted_names:
            r = tripletex_request("POST", f"{base_url}/department", auth=auth, run_ctx=run_ctx, json={"name": dept_name})
            if "error" in r:
                any_failed = True
            results.append(r)
        if any_failed:
            details["status"] = "failed"
        details["departments"] = results
        return details

    # Single department
    name = entities.get("department_name") or extract_department_name_from_prompt(prompt)
    if not name and quoted_names:
        name = quoted_names[0]
    if not name:
        # Try extracting after colon (e.g. "avdeling: Salg")
        m = re.search(r'(?:avdeling|department|abteilung)[^:]*:\s*([\w\s]+)', prompt, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
    name = name or "General"

    resp = tripletex_request("POST", f"{base_url}/department", auth=auth, run_ctx=run_ctx, json={"name": name})
    if "error" in resp:
        details["status"] = "failed"
    details["department"] = resp
    return details


async def handle_correction(base_url: str, auth: tuple, prompt: str, details: Dict, plan: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict:
    """Delete/reverse entity."""
    entities = plan.get("entities", {})
    ids = entities.get("ids", {}) or extract_ids_from_prompt(prompt)

    if ids.get("invoice_id"):
        resp = tripletex_request("DELETE", f"{base_url}/invoice/{ids['invoice_id']}", auth=auth, run_ctx=run_ctx)
    elif ids.get("travel_id"):
        resp = tripletex_request("DELETE", f"{base_url}/travelExpense/{ids['travel_id']}", auth=auth, run_ctx=run_ctx)
    elif ids.get("employee_id"):
        resp = tripletex_request("DELETE", f"{base_url}/employee/{ids['employee_id']}", auth=auth, run_ctx=run_ctx)
    elif ids.get("customer_id"):
        resp = tripletex_request("DELETE", f"{base_url}/customer/{ids['customer_id']}", auth=auth, run_ctx=run_ctx)
    elif ids.get("order_id"):
        resp = tripletex_request("DELETE", f"{base_url}/order/{ids['order_id']}", auth=auth, run_ctx=run_ctx)
    else:
        details["status"] = "failed"
        details["error"] = "No entity ID to delete"
        return details

    if "error" in resp:
        details["status"] = "failed"
    details["correction"] = resp
    return details


# ============ ENDPOINTS ============

@app.get("/health")
async def health():
    return {"status": "ok", "dry_run": DRY_RUN}


@app.get("/customers")
async def list_customers():
    base_url = (TRIPLETEX_BASE_URL or "https://kkpqfuj-amager.tripletex.dev/v2").rstrip("/")
    if not base_url.endswith("/v2"):
        base_url = base_url.rstrip("/") + "/v2"
    resp = tripletex_request("GET", f"{base_url}/customer", auth=("0", TRIPLETEX_SESSION_TOKEN), params={"count": 100})
    if "error" in resp:
        return JSONResponse({"status": "failed", "error": resp.get("error")}, status_code=500)
    return JSONResponse({"status": "success", "total": resp.get("fullResultSize", 0), "customers": resp.get("values", [])})


@app.get("/suppliers")
async def list_suppliers():
    base_url = (TRIPLETEX_BASE_URL or "https://kkpqfuj-amager.tripletex.dev/v2").rstrip("/")
    if not base_url.endswith("/v2"):
        base_url = base_url.rstrip("/") + "/v2"
    resp = tripletex_request("GET", f"{base_url}/supplier", auth=("0", TRIPLETEX_SESSION_TOKEN), params={"count": 100})
    if "error" in resp:
        return JSONResponse({"status": "failed", "error": resp.get("error")}, status_code=500)
    return JSONResponse({"status": "success", "total": resp.get("fullResultSize", 0), "suppliers": resp.get("values", [])})


@app.post("/solve")
async def solve(request: Request, authorization: Optional[str] = Header(default=None)):
    """Main endpoint."""
    if ENDPOINT_API_KEY and authorization != f"Bearer {ENDPOINT_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    body = await request.json()
    prompt = body.get("prompt", "")
    creds = body.get("tripletex_credentials", {})
    
    base_url = (creds.get("base_url", "") or TRIPLETEX_BASE_URL).rstrip("/")
    session_token = creds.get("session_token", "") or TRIPLETEX_SESSION_TOKEN
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    if not DRY_RUN and (not base_url or not session_token):
        raise HTTPException(status_code=400, detail="Missing credentials")
    
    if base_url and not base_url.endswith("/v2"):
        base_url = base_url.rstrip("/") + "/v2"
    
    run_ctx = {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"write_calls": 0, "write_errors_4xx": 0, "total_calls": 0},
        "calls": [],
    }
    
    action = detect_action(prompt)
    details = {"action": action, "status": "completed"}
    
    plan = {"entities": {}}
    
    try:
        if action == "customer":
            details = await handle_customer(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "employee":
            details = await handle_employee(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "product":
            details = await handle_product(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "invoice":
            details = await handle_invoice(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "payment":
            details = await handle_payment(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "creditNote":
            details = await handle_credit_note(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "supplier":
            details = await handle_supplier(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "travelExpense":
            details = await handle_travel_expense(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "project":
            details = await handle_project(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "department":
            details = await handle_department(base_url, ("0", session_token), prompt, details, plan, run_ctx)
        elif action == "correction":
            details = await handle_correction(base_url, ("0", session_token), prompt, details, plan, run_ctx)
    except Exception as e:
        details["status"] = "failed"
        details["error"] = str(e)
    
    # Log run
    try:
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        with open(os.path.join(RUN_LOG_DIR, f"{run_ctx['run_id']}.json"), "w") as f:
            json.dump({
                "run_id": run_ctx["run_id"],
                "started_at": run_ctx["started_at"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": run_ctx["metrics"],
                "action": action,
                "status": details.get("status"),
                "error": details.get("error"),
            }, f, indent=2)
    except Exception:
        pass
    
    if DEBUG_RESPONSE:
        return JSONResponse({"status": "completed", "dry_run": DRY_RUN, "run_id": run_ctx["run_id"], "metrics": run_ctx["metrics"]})
    return JSONResponse({"status": "completed"})
