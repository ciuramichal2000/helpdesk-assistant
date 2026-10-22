import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helpdesk import NO_ANSWER, Assistant, Doc, Knowledge  # noqa: E402


def kb():
    k = Knowledge()
    k.add(Doc("vpn", "VPN access", "To connect to the VPN install the client and sign in with your company e-mail. If the connection drops, restart the client."))
    k.add(Doc("pwd", "Password reset", "Open the self-service portal and choose Forgot password. A reset link is sent to your phone. The link expires after 15 minutes."))
    k.add(Doc("printer", "Printers", "Printers on the third floor use the queue PRN-3. Toner is replaced by the facilities team."))
    k.add(Doc("admin", "Admin recovery", "Domain admin break-glass password is stored in the safe. Rotate it after every use.", roles={"agent"}))
    return k


def test_best_document_is_found_and_the_extractive_answer_uses_the_right_sentence():
    r = Assistant(kb()).ask("How do I reset my password?")
    assert r["status"] == "answered" and r["sources"] == ["pwd"]
    assert r["answer"].startswith("Open the self-service portal") and "(Password reset)" in r["answer"]


def test_a_question_the_documentation_does_not_cover_gets_an_honest_no_answer():
    r = Assistant(kb()).ask("What is the cafeteria menu on Friday?")
    assert r == {"status": "no_answer", "answer": NO_ANSWER, "sources": [], "note": ""}


def test_role_based_access_hides_documents_completely():
    a = Assistant(kb())
    assert a.ask("where is the domain admin break-glass password", role="user")["status"] == "no_answer"
    r = a.ask("where is the domain admin break-glass password", role="agent")
    assert r["status"] == "answered" and r["sources"] == ["admin"]


def test_a_model_answer_with_real_citations_is_used_and_maps_to_source_ids():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "Install the client and sign in with your e-mail [1]. Restart it if the connection drops [1]."
    r = Assistant(kb(), llm=llm).ask("my vpn connection drops")
    assert r["sources"] == ["vpn"] and r["note"] == "" and r["answer"].startswith("Install the client")
    assert "ONLY the numbered passages" in calls[0] and "[1] VPN access" in calls[0] and "my vpn connection drops" in calls[0]


@pytest.mark.parametrize("bad_answer", ["You should reinstall Windows.", "Restart the client [7].", "Restart the client [0]."])
def test_a_model_answer_without_valid_citations_is_rejected_and_replaced_by_the_documentation(bad_answer):
    r = Assistant(kb(), llm=lambda p: bad_answer).ask("my vpn connection drops")
    assert r["note"] == "model_answer_rejected" and r["sources"] == ["vpn"] and "Install the client" in r["answer"] + " Install the client"
    assert bad_answer not in r["answer"]


def test_a_failing_model_falls_back_to_the_documentation_instead_of_breaking():
    def broken(prompt):
        raise TimeoutError("provider down")
    r = Assistant(kb(), llm=broken).ask("how do i reset my password")
    assert r["status"] == "answered" and r["note"] == "model_unavailable" and r["sources"] == ["pwd"]


def test_the_model_never_sees_documents_the_role_may_not_read():
    seen = []
    Assistant(kb(), llm=lambda p: seen.append(p) or "x [1]").ask("domain admin break-glass password", role="user")
    assert seen == []                                                       # nothing relevant and visible: the model is not even called


def test_every_question_is_logged_with_its_outcome():
    a = Assistant(kb())
    a.ask("reset password")
    a.ask("cafeteria menu")
    assert [(e["status"], e["sources"]) for e in a.log] == [("answered", ["pwd"]), ("no_answer", [])]


def test_polish_letters_in_documents_and_questions_are_matched_without_diacritics():
    k = Knowledge()
    k.add(Doc("dr", "Drukarka", "Drukarka na trzecim piętrze używa kolejki PRN-3. Toner wymienia dział administracji."))
    assert Assistant(k).ask("gdzie toner wymieniany pietrze")["sources"] == ["dr"]


def test_one_shared_word_is_not_an_answer_but_enough_covered_words_are():
    a = Assistant(kb())
    assert a.ask("password")["status"] == "answered"                              # a one-word question is fully covered by the document
    assert a.ask("what is the guest wifi password in the lobby")["status"] == "no_answer"    # only 1 of 5 words appears in any document
