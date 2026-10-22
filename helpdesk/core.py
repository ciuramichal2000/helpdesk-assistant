"""A helpdesk assistant that answers only from the documentation: BM25 retrieval, role-based access to documents, an optional language model
whose answer must cite sources that really exist, and an honest "not found" when the documentation does not cover the question.
The model is a plain callable (prompt -> text), so any provider can be plugged in; tests use a fake one."""
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

K1, B = 1.5, 0.75
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "i", "my", "how", "do", "can", "what", "on", "for", "me", "with", "does", "you"}
NO_ANSWER = "I could not find this in the documentation. A support agent can help with it."


def fold(text: str) -> str:
    text = text.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", fold(text)) if t not in STOP]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    roles: Set[str] = field(default_factory=lambda: {"user", "agent"})       # who may read it


class Knowledge:
    def __init__(self):
        self.docs: Dict[str, Doc] = {}

    def add(self, doc: Doc) -> None:
        self.docs[doc.id] = doc

    def search(self, query: str, role: str, limit: int = 3, min_coverage: float = 0.4) -> List[Tuple[Doc, float]]:
        """BM25 over the documents this role may read (statistics computed on that subset only, so hidden documents leave no trace).
        A document must contain at least `min_coverage` of the question's distinct words: one shared word is not an answer."""
        visible = [d for d in self.docs.values() if role in d.roles]
        if not visible:
            return []
        toks = {d.id: tokens(d.title + " " + d.text) for d in visible}
        avg = sum(len(t) for t in toks.values()) / len(visible) or 1.0
        q = list(dict.fromkeys(tokens(query)))
        scored = []
        for d in visible:
            score, matched = 0.0, 0
            for term in q:
                tf = toks[d.id].count(term)
                if not tf:
                    continue
                matched += 1
                df = sum(1 for t in toks.values() if term in t)
                idf = math.log(1 + (len(visible) - df + 0.5) / (df + 0.5))
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(toks[d.id]) / avg))
            if score > 0 and q and matched / len(q) >= min_coverage:
                scored.append((d, score))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:limit]


class Assistant:
    def __init__(self, knowledge: Knowledge, llm: Optional[Callable[[str], str]] = None, min_coverage: float = 0.4):
        self.kb, self.llm, self.min_coverage = knowledge, llm, min_coverage
        self.log: List[Dict[str, object]] = []

    def ask(self, question: str, role: str = "user") -> Dict[str, object]:
        hits = self.kb.search(question, role, min_coverage=self.min_coverage)
        if not hits:
            return self._done(question, role, "no_answer", NO_ANSWER, [])
        sources = [d for d, _ in hits]
        if self.llm is None:
            return self._done(question, role, "answered", self._extract(question, sources[0]), [sources[0].id])
        prompt = ("Answer the question using ONLY the numbered passages. Cite every claim as [n]. If the passages do not answer it, say so.\n\n" +
                  "\n".join("[%d] %s: %s" % (i, d.title, d.text) for i, d in enumerate(sources, 1)) + "\n\nQuestion: " + question)
        try:
            text = self.llm(prompt).strip()
        except Exception:                                                     # noqa: BLE001 (a provider failure must not break the helpdesk)
            return self._done(question, role, "answered", self._extract(question, sources[0]), [sources[0].id], note="model_unavailable")
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
        if not cited or any(n < 1 or n > len(sources) for n in cited):
            # an answer with no citation, or one citing a passage that does not exist, is not trusted
            return self._done(question, role, "answered", self._extract(question, sources[0]), [sources[0].id], note="model_answer_rejected")
        return self._done(question, role, "answered", text, [sources[n - 1].id for n in dict.fromkeys(cited)])

    @staticmethod
    def _extract(question: str, doc: Doc) -> str:
        """Without a model: the sentence of the best document that shares the most words with the question."""
        q = set(tokens(question))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.text) if s.strip()]
        best = max(sentences, key=lambda s: (len(q & set(tokens(s))), -sentences.index(s)))
        return "%s (%s)" % (best, doc.title)

    def _done(self, question, role, status, answer, sources, note=""):
        out = {"status": status, "answer": answer, "sources": sources, "note": note}
        self.log.append({"question": question, "role": role, "status": status, "sources": sources, "note": note})
        return out
