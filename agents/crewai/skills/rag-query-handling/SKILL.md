---
name: rag-query-handling
description: Best practices for answering financial questions using RAG over bank statement data.
compatibility: crewai>=1.0.0
---

# RAG Query Handling Skill

## Rules
- After storing data in Vector DB, **always prefer using the `rag_tool`** to answer questions instead of relying only on memory.
- Use RAG especially when the document is long or detailed.