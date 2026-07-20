---
name: pii-handling
description: Strict PII detection and redaction rules for financial documents.
---

# PII Handling Skill

## Mandatory Rules
- Always redact sensitive information **before** storing data in Vector DB.
- Redact: Full names, account numbers, addresses, emails, phone numbers.
- Use the `pii_redaction_tool` for redaction.
- Never store or output raw PII in final reports unless explicitly required.
- If PII is detected, mention that redaction was applied (without revealing the original data).