---
name: rag-query-handling
description: Best practices for answering financial questions using RAG over bank statement data.
---

# RAG Query Handling Skill

## Rules
- After storing data in Vector DB, **always prefer using the `rag_query`** to answer questions instead of relying only on memory.
- Use RAG especially when the document is long or detailed.