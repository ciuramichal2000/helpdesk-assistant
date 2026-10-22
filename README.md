# helpdesk-assistant

An IT helpdesk assistant that answers **only from the documentation**. It is **new sample code, not the code of any client project**.

- **BM25 retrieval** over the documents the user's role may read; hidden documents leave no trace (not even in the ranking statistics), and the model is never called when nothing visible matches.
- An honest **"not found"** when the documentation does not cover the question, instead of a guess.
- An optional **language model** as a plain callable (`prompt -> text`). Its answer must cite passages as `[n]` and the citations must exist; an answer with none or with invented ones is rejected and replaced by the documentation, and a provider failure falls back the same way.
- Every question is logged with its outcome. Polish letters are matched without diacritics.

No model is bundled and none was called in the tests (a fake one is used), so the quality of real model answers is **not measured here**. `python -m pytest` runs the tests. Standard library only. Built with AI coding assistance.
