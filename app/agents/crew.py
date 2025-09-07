# app/agents/crew.py
from crewai import Agent, Task, Crew
import os

def build_agents():
    openai_key = os.getenv("OPENAI_API_KEY")

    prescreener = Agent(
        role="Prescreener",
        goal="Analyze JD and Resume to extract must-have skills and risk areas.",
        backstory="You are a senior recruiter specializing in technical roles.",
        llm="openai/gpt-4o-mini",
        verbose=False,
    )

    interviewer = Agent(
        role="Interviewer",
        goal="Ask targeted follow-up questions based on candidate's previous answers.",
        backstory="You are a pragmatic engineering manager focused on signal.",
        llm="openai/gpt-4o-mini",
        verbose=False,
    )

    summarizer = Agent(
        role="Summarizer",
        goal="Summarize the interview into strengths, weaknesses, and a decision.",
        backstory="You produce crisp reports for busy hiring committees.",
        llm="openai/gpt-4o-mini",
        verbose=False,
    )

    reporter = Agent(
        role="Reporter",
        goal="Generate a final structured report suitable for HR systems.",
        backstory="You write standardized reports following company schema.",
        llm="openai/gpt-4o-mini",
        verbose=False,
    )
    return prescreener, interviewer, summarizer, reporter


def run_prescreen(jd: str, resume: str) -> str:
    prescreener, *_ = build_agents()
    t = Task(
        description=f"""
            Extract must-have skills, nice-to-have, and potential risks from the JD and Resume.
            Output a compact bullet list grouped by category.
            JD:
            {jd}

            Resume:
            {resume}
            """,
        agent=prescreener,
        expected_output="Bullet list grouped as Must-have, Nice-to-have, Risks."
    )
    return Crew(agents=[prescreener], tasks=[t]).kickoff()


def interviewer_next_question(history_text: str) -> str:
    _, interviewer, *_ = build_agents()
    t = Task(
        description=f"""
                    Based on the conversation so far, generate ONE concise next question that probes a gap,
                    asks for specifics, or increases difficulty. Just return the question text.
                    History:
                    {history_text}
                    """,
        agent=interviewer,
        expected_output="A single question."
    )
    return Crew(agents=[interviewer], tasks=[t]).kickoff()


def run_summarizer(transcript: str) -> str:
    *_, summarizer, _ = build_agents()
    t = Task(
        description=f"""
                    Summarize the interview into: strengths, weaknesses, overall score, and recommendation.
                    Return a crisp paragraph for each section.
                    Transcript:
                    {transcript}
                    """,
        agent=summarizer,
        expected_output="Four short labeled sections."
    )
    return Crew(agents=[summarizer], tasks=[t]).kickoff()


def run_reporter(structured_json: str) -> str:
    prescreener, interviewer, summarizer, reporter = build_agents()
    t = Task(
        description=f"""
                Given this structured JSON of the interview summary, produce a final HR-friendly report.
                Keep it under 250 words with headings.
                JSON:
                {structured_json}
                """,
        agent=reporter,
        expected_output="Short report with headings."
    )
    return Crew(agents=[reporter], tasks=[t]).kickoff()
