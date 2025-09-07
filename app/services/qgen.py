from langchain.prompts import PromptTemplate
from .llm import lc_llm

_q_prompt = PromptTemplate.from_template("""
You are an expert technical interviewer.
Given the Job Description and Candidate Resume, generate {n} targeted interview questions.
Balance: 40% core skills, 40% project experience, 20% culture/behavioral.
No numbering, one question per line, concise.

Job Description:
{jd}

Resume:
{resume}

Output only the questions, each on a new line.
""")

def generate_questions(jd: str, resume: str, n: int = 5) -> list[str]:
    llm = lc_llm()
    resp = llm.invoke(_q_prompt.format(jd=jd, resume=resume, n=n))
    lines = [ln.strip(" -•\t") for ln in resp.content.split("\n") if ln.strip()]
    return lines[:n] if len(lines) >= n else lines
