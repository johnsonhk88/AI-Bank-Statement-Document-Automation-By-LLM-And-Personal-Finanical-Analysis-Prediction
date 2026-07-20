---
name: financial-analysis
description: Guidelines for income/expense categorization, trend analysis, and generating financial insights.
compatibility: crewai>=1.0.0
---

# Financial Analysis Skill

## Rules
- After parsing transactions, always calculate:
  - **Opening Balance**
  - **Closing Balance**
  - **Total Credits** (sum of all credits)
  - **Total Debits** (sum of all debits)
- Cross-verify totals using: `Opening + Total Credits - Total Debits = Closing`
- When answering "total amount" questions, clearly state whether it refers to **Closing Balance** or **Total Credits/Debits**.
- Highlight unusual or large transactions when relevant.
- Never invent or assume values not present in the extracted data.

## Output Guidelines
- Present numbers clearly with proper formatting (e.g., $1,234.56).
- Use tables when showing multiple transactions.