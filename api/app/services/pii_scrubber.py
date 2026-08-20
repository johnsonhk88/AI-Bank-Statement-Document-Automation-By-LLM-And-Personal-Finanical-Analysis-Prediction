"""PII scrubber — strips sensitive data before sending to cloud LLMs.

Uses Microsoft Presidio for NER-based detection (names, addresses, orgs)
plus custom recognizers for HK-specific patterns (HKID, HK phone format).
Falls back to regex-only mode if Presidio is not installed.

Used by the hybrid LLM router: raw PDFs go to llama.cpp (local),
but categorization of already-extracted transactions goes to OpenRouter.
This scrubber ensures no PII leaks to the cloud.
"""
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load Presidio; fall back to regex-only if unavailable
# ---------------------------------------------------------------------------
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning(
        "presidio-analyzer/anonymizer not installed — using regex-only PII scrubbing. "
        "Install with: pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg"
    )


@dataclass(frozen=True)
class ScrubResult:
    text: str
    pii_found: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HK-specific custom recognizers for Presidio
# ---------------------------------------------------------------------------

def _build_hk_recognizers() -> list:
    """Build custom PatternRecognizers for Hong Kong PII patterns."""
    recognizers = []

    # HKID: 1-2 uppercase letters + 6 digits + check digit in optional parens
    recognizers.append(PatternRecognizer(
        supported_entity="HK_ID",
        name="HongKongIDRecognizer",
        patterns=[Pattern(
            name="HKID",
            regex=r"\b[A-Z]{1,2}\d{6}\(?[0-9A]\)?\b",
            score=0.85,
        )],
        supported_language="en",
        context=["hkid", "hong kong id", "identity card", "id card", "身份證"],
    ))

    # HK phone: optional +852 prefix + 4+4 digits
    recognizers.append(PatternRecognizer(
        supported_entity="HK_PHONE",
        name="HongKongPhoneRecognizer",
        patterns=[Pattern(
            name="HK_PHONE",
            regex=r"\b(?:\+?852[-\s]?)?\d{4}[-\s]?\d{4}\b",
            score=0.6,
        )],
        supported_language="en",
        context=["phone", "tel", "mobile", "call", "whatsapp", "電話"],
    ))

    # SWIFT/BIC code
    recognizers.append(PatternRecognizer(
        supported_entity="SWIFT_CODE",
        name="SwiftCodeRecognizer",
        patterns=[Pattern(
            name="SWIFT",
            regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
            score=0.6,
        )],
        supported_language="en",
        context=["swift", "bic", "bank code", "routing"],
    ))

    # HK bank account numbers (3-3-3+ digit format)
    recognizers.append(PatternRecognizer(
        supported_entity="HK_BANK_ACCOUNT",
        name="HKBankAccountRecognizer",
        patterns=[Pattern(
            name="HK_ACCT",
            regex=r"\b\d{3}[-–]?\d{3}[-–]?\d{3,12}\b",
            score=0.5,
        )],
        supported_language="en",
        context=["account", "acct", "a/c", "bank", "帳戶"],
    ))

    return recognizers


# ---------------------------------------------------------------------------
# Presidio engine (singleton, lazy-initialized)
# ---------------------------------------------------------------------------

_analyzer: "AnalyzerEngine | None" = None
_anonymizer: "AnonymizerEngine | None" = None


def _get_engines() -> tuple:
    """Lazy-init Presidio engines with custom HK recognizers."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
        for recognizer in _build_hk_recognizers():
            _analyzer.registry.add_recognizer(recognizer)
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


# Operator config: replace each entity type with a readable placeholder
_OPERATORS: dict[str, "OperatorConfig"] = {}

def _get_operators() -> dict:
    """Build operator configs (lazy, after Presidio import)."""
    global _OPERATORS
    if not _OPERATORS:
        _OPERATORS = {
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            "ORGANIZATION": OperatorConfig("replace", {"new_value": "[ORG]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CARD]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            "HK_ID": OperatorConfig("replace", {"new_value": "[HKID]"}),
            "HK_PHONE": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "SWIFT_CODE": OperatorConfig("replace", {"new_value": "[SWIFT]"}),
            "HK_BANK_ACCOUNT": OperatorConfig("replace", {"new_value": "[ACCT]"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        }
    return _OPERATORS


# Confidence threshold — higher = fewer false positives
SCORE_THRESHOLD = 0.35


def scrub_text_presidio(text: str) -> ScrubResult:
    """Scrub PII using Presidio (NER + patterns + custom HK recognizers)."""
    analyzer, anonymizer = _get_engines()

    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=SCORE_THRESHOLD,
    )

    if not results:
        return ScrubResult(text=text)

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=_get_operators(),
    )

    pii_found = [
        f"{r.entity_type}:{r.score:.2f}"
        for r in results
    ]
    return ScrubResult(text=anonymized.text, pii_found=pii_found)


# ---------------------------------------------------------------------------
# Regex fallback (used when Presidio is not installed)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]"),
    ("account_number", re.compile(r"\b\d{3}[-–]?\d{3}[-–]?\d{3,12}\b"), "[ACCT]"),
    ("hk_id", re.compile(r"\b[A-Z]{1,2}\d{6}\(?[0-9A]\)?\b", re.IGNORECASE), "[HKID]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    ("phone_hk", re.compile(r"\b(?:\+?852[-\s]?)?\d{4}[-\s]?\d{4}\b"), "[PHONE]"),
    ("phone_intl", re.compile(r"\b\+\d{1,3}[-\s]?\d{4,14}\b"), "[PHONE]"),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"), "[IBAN]"),
    ("swift", re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"), "[SWIFT]"),
]


def scrub_text_regex(text: str) -> ScrubResult:
    """Regex-only PII scrubbing (fallback when Presidio unavailable)."""
    found: list[str] = []
    result = text
    for name, pattern, replacement in _REGEX_PATTERNS:
        matches = pattern.findall(result)
        if matches:
            found.append(f"{name}:{len(matches)}")
            result = pattern.sub(replacement, result)
    return ScrubResult(text=result, pii_found=found)


# ---------------------------------------------------------------------------
# Public API — auto-selects Presidio or regex fallback
# ---------------------------------------------------------------------------

def scrub_text(text: str) -> ScrubResult:
    """Remove all PII from free text.

    Uses Presidio (NER + patterns) if available, otherwise regex fallback.
    Presidio catches names, addresses, and organizations that regex misses.
    """
    if _PRESIDIO_AVAILABLE:
        return scrub_text_presidio(text)
    return scrub_text_regex(text)


def scrub_transaction_for_cloud(tx: dict) -> dict:
    """Prepare a transaction dict for cloud LLM categorization.

    Only sends: merchant name (scrubbed), description (scrubbed),
    amount, date, currency. Never sends account numbers, balances,
    or any raw PII.
    """
    scrubbed_desc = scrub_text(tx.get("description", "")).text
    scrubbed_merchant = scrub_text(tx.get("merchant", "")).text

    return {
        "date": tx.get("date", ""),
        "description": scrubbed_desc,
        "merchant": scrubbed_merchant,
        "amount": tx.get("amount"),
        "currency": tx.get("currency", "HKD"),
    }


def scrub_transactions_batch(transactions: list[dict]) -> list[dict]:
    """Scrub a batch of transactions for cloud categorization."""
    return [scrub_transaction_for_cloud(tx) for tx in transactions]
