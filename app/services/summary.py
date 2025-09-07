from langchain.prompts import PromptTemplate
from .llm import lc_llm

_summary_prompt = PromptTemplate.from_template("""
You are a hiring assistant. Produce a concise hiring report from the transcript.

Provide a compact JSON:
{{
 "strengths": [ ... ],
 "weaknesses": [ ... ],
 "overall_score": 0-100,
 "recommendation": "Hire" | "No Hire" | "Maybe",
 "summary": "2-4 sentence overview"
}}

Transcript:
{transcript}

Return ONLY JSON.
""")

def summarize_transcript(transcript: str) -> dict:
    llm = lc_llm()
    resp = llm.invoke(_summary_prompt.format(transcript=transcript))
    import json
    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "strengths": [],
            "weaknesses": [],
            "overall_score": 60,
            "recommendation": "Maybe",
            "summary": "Fallback summary."
        }
