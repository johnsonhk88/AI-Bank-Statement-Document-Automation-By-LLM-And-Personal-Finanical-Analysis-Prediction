#!/usr/bin/env bash
# ============================================================
# bulk-upload.sh — Upload a folder of bank statement PDFs
#                  and start async processing via llama.cpp
#
# Usage:
#   ./scripts/bulk-upload.sh /path/to/pdf/folder
#   ./scripts/bulk-upload.sh /path/to/pdf/folder "What is my monthly spending trend?"
#
# The script will:
#   1. Login to get a JWT token
#   2. Upload all PDFs and kick off processing
#   3. Poll until all items are done
# ============================================================
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@bankai.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"
LLM_PROVIDER="${LLM_PROVIDER:-llamacpp}"
LLM_MODEL="${LLM_MODEL:-openai/local-model}"

PDF_DIR="${1:-}"
QUESTION="${2:-Extract all transactions and provide a financial summary with income, expenses, and balance trends.}"

if [[ -z "$PDF_DIR" ]]; then
    echo "Usage: $0 /path/to/pdf/folder [question]"
    echo ""
    echo "Environment variables:"
    echo "  API_URL         (default: http://localhost:8000)"
    echo "  ADMIN_EMAIL     (default: admin@bankai.local)"
    echo "  ADMIN_PASSWORD  (default: changeme)"
    echo "  LLM_PROVIDER    (default: llamacpp)"
    echo "  LLM_MODEL       (default: openai/local-model)"
    exit 1
fi

if [[ ! -d "$PDF_DIR" ]]; then
    echo "ERROR: Directory not found: $PDF_DIR"
    exit 1
fi

# Count PDFs
PDF_COUNT=$(find "$PDF_DIR" -maxdepth 1 -name "*.pdf" -o -name "*.PDF" | wc -l)
if [[ "$PDF_COUNT" -eq 0 ]]; then
    echo "ERROR: No PDF files found in $PDF_DIR"
    exit 1
fi
echo "Found $PDF_COUNT PDF files in $PDF_DIR"

# --- Login ---
echo "==> Logging in as $ADMIN_EMAIL..."
LOGIN_RESP=$(curl -sf -X POST "$API_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASSWORD\"}" 2>&1) || {
    echo "ERROR: Login failed. Is the API running? Check credentials."
    echo "Response: $LOGIN_RESP"
    exit 1
}

TOKEN=$(echo "$LOGIN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null) || {
    echo "ERROR: Could not parse login response"
    echo "Response: $LOGIN_RESP"
    exit 1
}
echo "    Logged in."

# --- Build curl file args ---
FILE_ARGS=()
for pdf in "$PDF_DIR"/*.pdf "$PDF_DIR"/*.PDF; do
    [[ -f "$pdf" ]] || continue
    FILE_ARGS+=(-F "files=@$pdf")
done

# --- Bulk upload + process ---
echo "==> Uploading $PDF_COUNT PDFs and starting processing..."
echo "    Question: $QUESTION"
echo "    LLM: $LLM_PROVIDER / $LLM_MODEL"
echo ""

UPLOAD_RESP=$(curl -sf -X POST "$API_URL/api/agent-runs/bulk-process" \
    -H "Authorization: Bearer $TOKEN" \
    -F "question=$QUESTION" \
    -F "llm_provider_id=$LLM_PROVIDER" \
    -F "llm_model_id=$LLM_MODEL" \
    "${FILE_ARGS[@]}" 2>&1) || {
    echo "ERROR: Bulk upload failed"
    echo "Response: $UPLOAD_RESP"
    exit 1
}

UPLOADED=$(echo "$UPLOAD_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['uploaded_count'])" 2>/dev/null)
SKIPPED=$(echo "$UPLOAD_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['skipped_duplicates'])" 2>/dev/null)
ERRORS=$(echo "$UPLOAD_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(len(r['upload_errors']))" 2>/dev/null)
RUN_ID=$(echo "$UPLOAD_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); ar=r.get('agent_run'); print(ar['id'] if ar else 'none')" 2>/dev/null)

echo "    Uploaded: $UPLOADED new, $SKIPPED duplicates skipped, $ERRORS errors"

if [[ "$RUN_ID" == "none" ]]; then
    echo "No documents to process."
    exit 0
fi

echo "    Agent Run: $RUN_ID"
echo ""

# --- Poll for completion ---
echo "==> Processing... (this may take a while with local LLM)"
echo "    Each PDF goes through: parse → PII redact → vectorize → analyze → format"
echo ""

POLL_INTERVAL=10
TOTAL_WAIT=0
MAX_WAIT=7200  # 2 hours max

while true; do
    RUN_STATUS=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API_URL/api/agent-runs/$RUN_ID" 2>/dev/null) || {
        echo "    [${TOTAL_WAIT}s] API unreachable, retrying..."
        sleep "$POLL_INTERVAL"
        TOTAL_WAIT=$((TOTAL_WAIT + POLL_INTERVAL))
        continue
    }

    STATUS=$(echo "$RUN_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
    ITEMS_DONE=$(echo "$RUN_STATUS" | python3 -c "
import sys,json
r=json.load(sys.stdin)
done=sum(1 for i in r['items'] if i['status'] in ('succeeded','failed'))
total=len(r['items'])
print(f'{done}/{total}')
" 2>/dev/null)

    echo "    [${TOTAL_WAIT}s] Status: $STATUS  Items: $ITEMS_DONE"

    if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" || "$STATUS" == "partial" ]]; then
        break
    fi

    if [[ "$TOTAL_WAIT" -ge "$MAX_WAIT" ]]; then
        echo "WARNING: Timed out after ${MAX_WAIT}s. Run is still processing."
        echo "Check status: curl -H 'Authorization: Bearer $TOKEN' $API_URL/api/agent-runs/$RUN_ID"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
    TOTAL_WAIT=$((TOTAL_WAIT + POLL_INTERVAL))
done

# --- Print summary ---
echo ""
echo "============================================================"
echo " Processing Complete"
echo "============================================================"
echo "$RUN_STATUS" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f\"  Status : {r['status']}\")
print(f\"  Agent  : {r['agent']}\")
print(f\"  LLM    : {r['llm_provider']} / {r['llm_model']}\")
print(f\"  Items  : {len(r['items'])}\")
print()
for i, item in enumerate(r['items'], 1):
    status_icon = '  OK' if item['status'] == 'succeeded' else 'FAIL'
    tx_count = len(item.get('transactions') or [])
    print(f\"  [{status_icon}] Item {i}: {item['status']}  ({tx_count} transactions)\")
    if item.get('error'):
        print(f\"         Error: {item['error'][:200]}\")
print()
succeeded = sum(1 for i in r['items'] if i['status'] == 'succeeded')
failed = sum(1 for i in r['items'] if i['status'] == 'failed')
total_tx = sum(len(i.get('transactions') or []) for i in r['items'])
print(f\"  Total: {succeeded} succeeded, {failed} failed, {total_tx} transactions extracted\")
"
echo ""
echo "  View results: $API_URL/api/agent-runs/$RUN_ID"
echo "  API docs:     $API_URL/api/docs"
echo "============================================================"
