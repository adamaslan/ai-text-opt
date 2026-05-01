# Firestore Security Scanner

Scan Python source files in this repository for Firestore-related security vulnerabilities. Report every finding with file, line number, severity, and a recommended fix.

## Scan Targets

Search all `.py` files under the current working directory (skip `__pycache__`, `.venv`, `node_modules`, `dist`).

## Checks to Run

### 1. Unsanitised Firestore Document IDs (CRITICAL)

Look for patterns where a variable (especially user-supplied path params or body fields) is passed directly into `.document(...)` without prior regex validation.

**Detect:**
```python
# Dangerous — no sanitisation guard before document()
.document(symbol)
.document(sym)
.document(user_id)
.document(key)
```

**Safe pattern (what to look for as evidence of a fix):**
```python
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,10}$")
if not _SYMBOL_RE.match(sym):
    raise ...
col.document(sym)
```

Flag any `.document(...)` call whose argument is **not** validated by a regex or allowlist within the same function or at the call site.

---

### 2. Raw Exception Strings Returned to Callers (HIGH)

Look for patterns where internal exception messages are serialised into HTTP responses, which can leak credentials, API keys, internal paths, or stack details.

**Detect:**
```python
raise HTTPException(status_code=..., detail=str(exc))
raise HTTPException(status_code=..., detail=f"...: {exc}")
return {"error": str(exc)}
return {"error": f"failed: {e}"}
errors.append(f"{label}: {exc}")        # list later joined into response
errors.append(str(exc))
```

**Safe pattern:**
```python
logger.exception("context: %s", exc)
raise HTTPException(status_code=503, detail="Service temporarily unavailable")
```

---

### 3. API Keys or Credentials in Query Parameters (CRITICAL)

Detect any pattern where a secret (API key, token, password) is appended to a URL as a query string. Query params appear in server logs, browser history, and `Referer` headers.

**Detect:**
```python
params={"token": api_key}
params={"apikey": ...}
params={"key": ...}
f"?api_key={...}"
f"?token={...}"
requests.get(url + f"?apiKey={...}")
httpx.get(url, params={"finnhub-token": ...})
```

**Safe pattern (header-based auth):**
```python
headers={"X-Finnhub-Token": api_key}
headers={"Authorization": f"Bearer {token}"}
```

---

### 4. Missing Input Validation on Path/Query/Body Parameters (HIGH)

Flag FastAPI path parameters (`symbol: str`, `user_id: str`, etc.) that are used in Firestore calls, cache key construction, or external API calls **without** length/character validation.

**Detect:** Functions decorated with `@app.get("/{symbol}")` or `@app.post(...)` where `symbol` (or any str param) flows into:
- `get_cache(f"...:{symbol}...")`
- `set_cache(f"...:{symbol}...")`
- `.document(symbol)`
- `_fetch(symbol, ...)`
- external HTTP calls containing the symbol

without a prior call to a sanitisation function or regex match.

---

### 5. Firestore Collection Enumeration via Unauthenticated Endpoints (MEDIUM)

Flag endpoints that stream entire Firestore collections (`.stream()` or `.list_documents()`) on unauthenticated routes without pagination or size limits.

**Detect:**
```python
col.stream()          # returns all docs — no limit
col.list_documents()  # same
```

**Recommend:** Add `.limit(MAX)` or verify the endpoint requires authentication.

---

### 6. Insecure Firestore Cache Key Construction (MEDIUM)

Flag cache keys built from user-controlled strings without sanitisation. A crafted key like `analyze2:../../admin_collection:1mo` can traverse collection namespaces.

**Detect:**
```python
cache_key = f"prefix:{symbol}:{period}"   # symbol not sanitised above this line
```

**Safe pattern:** Only build cache keys from sanitised symbols (confirmed by earlier validation).

---

### 7. Broad Exception Suppression (LOW)

Detect `except Exception: pass` or `except Exception: return None` patterns that silently swallow errors, making security-relevant failures invisible.

**Detect:**
```python
except Exception:
    pass

except Exception:
    return None

except Exception:
    return {"error": "unavailable"}   # no logging
```

**Recommend:** Always `logger.warning(...)` or `logger.exception(...)` before returning a degraded result.

---

### 8. Secrets Hard-Coded or in Environment Strings (CRITICAL)

Scan for any hard-coded credential patterns:

```
AIzaSy[A-Za-z0-9_-]{35}        # GCP API key
GOCSPX-[A-Za-z0-9_-]{24}       # GCP OAuth secret
sk_live_[A-Za-z0-9]{24}        # Stripe secret key
ya29\.[A-Za-z0-9_-]{60,}       # GCP access token
-----BEGIN.*PRIVATE KEY-----   # PEM private key block
```

If any match is found in source code (not in `.env`, `*.example`, or comments), flag as CRITICAL.

---

## Output Format

For each finding produce a table row:

| Severity | File | Line | Issue | Snippet | Recommendation |
|----------|------|------|-------|---------|----------------|
| CRITICAL | `backend2/main.py` | 93 | Unsanitised symbol in cache key | `cache_key = f"analyze2:{symbol}..."` | Call `_sanitize_symbol(symbol)` before building cache key |

After the table, print a **Summary** section:

```
## Summary
- CRITICAL: N findings
- HIGH:     N findings
- MEDIUM:   N findings
- LOW:      N findings

### Quick wins (fixes < 5 lines each)
1. ...

### Larger refactors needed
1. ...
```

## How to Run This Scan

Use the Grep and Read tools to search for the patterns above. Start with:

1. `Grep` for `.document(` across all `.py` files — check each call site for upstream sanitisation.
2. `Grep` for `str(exc)` and `f".*{exc}"` patterns in `raise HTTPException` and `return {...}` statements.
3. `Grep` for `params=` and `?api` patterns in HTTP client calls.
4. `Grep` for `@app.get("/{` and `@app.post` to enumerate all endpoints, then trace each str param.
5. `Grep` for `col.stream()` and `.list_documents()` without `.limit(`.
6. `Grep` for `except Exception:` followed immediately by `pass`, `return None`, or `return {`.
7. `Grep` for hard-coded credential patterns (AIzaSy, sk_live_, GOCSPX-, BEGIN PRIVATE KEY).

Report all findings — do not fix automatically unless the user explicitly asks.
