from langchain.prompts import PromptTemplate
from .llm import lc_llm

_eval_prompt = PromptTemplate.from_template("""
You are an interview evaluator. Score the candidate's answer.
Return a compact JSON with fields: relevance(1-10), clarity(1-10), technical(1-10), red_flags(list of strings), note(string).

Question: {q}
Answer: {a}

Return ONLY JSON, no extra text.
""")

def evaluate_answer(question: str, answer: str) -> dict:
    llm = lc_llm()
    resp = llm.invoke(_eval_prompt.format(q=question, a=answer))
    # Best effort safe-parse (model already returns JSON)
    import json
    try:
        return json.loads(resp.content)
    except Exception:
        return {"relevance": 5, "clarity": 5, "technical": 5, "red_flags": [], "note": "fallback"}
