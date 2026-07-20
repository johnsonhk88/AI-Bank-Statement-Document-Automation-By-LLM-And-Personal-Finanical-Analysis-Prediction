---
name: bank-statement-parsing
description: Guidelines for parsing bank statements using YOLO layout detection + OCR + LLM.
compatibility: crewai>=1.0.0
---

# Bank Statement Parsing Skill

## Core Rules
- Always start by calling the `PDF Extractor` tool.
- Do **not** trust "Paid In" / "Paid Out" column headers when they are ambiguous.
- **Verify every transaction** by checking how the running Balance changes:
  - If Balance **increases** → Credit (Paid In)
  - If Balance **decreases** → Debit (Paid Out)
- Extract **Opening Balance** from "Balance Brought Forward".
- Extract **Closing Balance** from the final running balance in the transactions.
- Preserve original descriptions exactly. Do not summarize or shorten them.
- If column alignment in the extracted text is messy, rely on balance math rather than position.

## Expected Output
Return transactions in a consistent structure with: Date, Description, Debit, Credit, Balance.