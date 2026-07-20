import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# +1-555-0100, (555) 123-4567, 555-123-4567, +44-20-7946-0958
_PHONE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d{1,3}[-.\s]?(?:\d{1,4}[-.\s]?){1,3}\d{2,4}"
    r"|"
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r")(?!\w)"
)
_ACCOUNT_DIGITS = re.compile(r"\b\d{8,17}\b")
# 123-456-789 and similar grouped account numbers (8–17 digits total)
_ACCOUNT_GROUPED = re.compile(r"\b\d{2,4}(?:-\d{2,4}){1,4}\b")
# Label heuristic: mask the next token after Account Number / Account No / A/C
_ACCOUNT_LABEL = re.compile(
    r"(?i)(\b(?:account\s*(?:number|no\.?|#)|a\/c)\s*[:#]?\s*)(\S+)"
)


def redact_pii(text: str) -> str:
    out = _EMAIL.sub("[REDACTED_EMAIL]", text)
    out = _PHONE.sub("[REDACTED_PHONE]", out)
    out = _ACCOUNT_DIGITS.sub("[REDACTED_ACCOUNT]", out)

    def _grouped_account(match: re.Match[str]) -> str:
        digits = match.group(0).replace("-", "")
        if 8 <= len(digits) <= 17:
            return "[REDACTED_ACCOUNT]"
        return match.group(0)

    out = _ACCOUNT_GROUPED.sub(_grouped_account, out)
    out = _ACCOUNT_LABEL.sub(r"\1[REDACTED_ACCOUNT]", out)
    return out
