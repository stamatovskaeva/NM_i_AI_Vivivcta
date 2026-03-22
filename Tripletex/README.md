# Tripletex AI Accounting Agent

Dette prosjektet bygger en enkel Tripletex `/solve`-endpoint for konkurransen.

## Installere

```bash
cd /Users/evastamatovska/Documents/NM/NM_i_AI_Vivivcta
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Konfigurere LLM for Entity Extraction

Agenten bruker **litellm** for å hente ut estrutererte data (kunde, produkt, beløp, osv.) fra prompts med AI. Velg en Model:

### Alternativ 1: Ollama (Lokal, ingen API-nøkler)

```bash
# Installer Ollama (macOS)
brew install ollama

# Start Ollama-serveren
ollama serve

# I en annen terminal, last ned modell (første gang)
ollama pull mistral
# eller: ollama pull neural-chat
```

Sett i `.env`:
```
LLM_ENABLED=true
LLM_MODEL=ollama/mistral
```

### Alternativ 2: Groq (Rask, gratis tier)

1. Registrer deg på https://groq.com (2 minutter, ingen kredittkort)
2. Få API-nøkkelen fra settings
3. Sett i `.env`:
```
LLM_ENABLED=true
LLM_MODEL=groq/mixtral-8x7b-32768
GROQ_API_KEY=<your_key>
```

### Alternativ 3: Deaktiverte LLM (fallback til regex)

```
LLM_ENABLED=false
```

## Kjøre lokalt

```bash
# Aktiver venv
source venv/bin/activate

# Opprett .env fra .env.example
cp .env.example .env
# ... rediger .env med dine Tripletex-credentials og LLM-valg

# Start agenten
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Eksponere via HTTPS (ngrok)

```bash
ngrok http 8000
```

Kopier URL fra ngrok (`https://xxxxxxx.ngrok.io`) og sett i konkurranse-UI som `https://xxxxxxx.ngrok.io/solve`.

## Test med curl

```bash
curl -X POST "http://localhost:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Opprett en ansatt Ola Nordmann med epost ola@example.org",
    "tripletex_credentials": {
      "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
      "session_token": "<YOUR_TOKEN>"
    }
  }'
```

## Endpoints

- **POST /solve** — Hoved-endpoint (mottaker prompt, returnerer {"status": "completed"})
- **GET /health** — Server status
- **GET /customers** — List alle kunder
- **GET /suppliers** — List alle leverandører

## Arkitektur

- **Prompt parsing**: `detect_action()` + `extract_entities_via_llm()`
- **Entity reuse**: `search_by_name()` før POST (mindre API-kall)
- **Error tracking**: Run-kontekst lagrer metrikker (write_calls, write_errors_4xx, total_calls)
- **Task handlers**: 9 handler-funksjoner for task-typene (customer, employee, product, invoice, payment, travelExpense, project, department, correction)

