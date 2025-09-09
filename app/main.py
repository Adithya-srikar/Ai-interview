# app/main.py
import os, uuid, json
from fastapi import FastAPI, Query,BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from mem0 import MemoryClient
from fastapi.middleware.cors import CORSMiddleware
from services.qgen import generate_questions
from services.evals import evaluate_answer
from services.summary import summarize_transcript
from agents.crew import run_prescreen, interviewer_next_question, run_summarizer, run_reporter

load_dotenv() 



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

mem_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

# Lightweight runtime state (cursor & queue); long-term content in mem0
sessions: dict[str, dict] = {}

class InitRequest(BaseModel):
    jd: str
    resume: str
    linkedin: str = None
    github: str = None

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

@app.post("/api/init")
async def init_interview(req: InitRequest, background_tasks: BackgroundTasks):
    import aiohttp
    from bs4 import BeautifulSoup
    import datetime

    session_id = str(uuid.uuid4())
    mem_client.add(messages=[{"role": "user", "content": f"JD: {req.jd}"}], user_id=session_id)
    mem_client.add(messages=[{"role": "user", "content": f"Resume: {req.resume}"}], user_id=session_id)

    extra_info = ""
    # Scrape LinkedIn
    if req.linkedin:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(req.linkedin) as resp:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    headline = soup.find("h1")
                    extra_info += f"LinkedIn Headline: {headline.text if headline else 'N/A'}\n"
        except Exception:
            extra_info += "LinkedIn scrape failed.\n"
    # Scrape GitHub
    if req.github:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(req.github) as resp:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    repo_count = soup.find("span", class_="Counter")
                    extra_info += f"GitHub Repo Count: {repo_count.text if repo_count else 'N/A'}\n"
        except Exception:
            extra_info += "GitHub scrape failed.\n"
    if extra_info:
        mem_client.add(messages=[{"role": "user", "content": extra_info}], user_id=session_id)

    try:
        prescreen_notes = run_prescreen(req.jd, req.resume)
        mem_client.add(messages=[{"role": "assistant", "content": f"Prescreen Notes: {prescreen_notes}"}], user_id=session_id)
    except Exception as e:
        prescreen_notes = "Prescreen skipped."

    try:
        questions = generate_questions(req.jd, req.resume, n=5)
    except Exception:
        questions = [
            "Tell me about your most relevant project for this role.",
            "Which part of the JD best matches your skillset, and why?"
        ]
    mem_client.add(messages=[{"role": "assistant", "content": f"Initial Questions: {questions}"}], user_id=session_id)
    sessions[session_id] = {
        "current_q": 0,
        "questions": questions,
        "start_time": datetime.datetime.utcnow().timestamp(),
        "duration": 600,  # 10 min default
    }
    return {"session_id": session_id, "prescreen": prescreen_notes}

@app.get("/api/interview/next")
async def get_next_question(session_id: str = Query(...)):
    session = sessions.get(session_id)
    if not session:
        return {"error": "Invalid session_id"}

    if session["current_q"] < len(session["questions"]):
        q = session["questions"][session["current_q"]]
        return {"question": q}

    mem_chunks = mem_client.get(user_id=session_id)
    history_text = "\n".join([m["content"] for m in mem_chunks])
    try:
        next_q = interviewer_next_question(history_text)
    except Exception:
        return {"error": "Failed to generate next question"}

    session["questions"].append(next_q)
    return {"question": next_q}

@app.post("/api/interview/answer")
async def submit_answer(req: AnswerRequest):
    import datetime
    session = sessions.get(req.session_id)
    if not session:
        return {"error": "Invalid session_id"}

    # Interview timing check
    now = datetime.datetime.utcnow().timestamp()
    if now - session["start_time"] > session["duration"]:
        return {"finished": True, "evaluation": None, "reason": "Interview time exceeded."}

    if session["current_q"] < len(session["questions"]):
        q = session["questions"][session["current_q"]]
    else:
        q = session["questions"][-1]

    # Save Q/A to memory
    mem_client.add(messages=[{"role": "assistant", "content": f"Q: {q}"}], user_id=req.session_id)
    mem_client.add(messages=[{"role": "user", "content": f"A: {req.answer}"}], user_id=req.session_id)

    try:
        eval_json = evaluate_answer(q, req.answer)
        mem_client.add(messages=[{"role": "assistant", "content": f"Evaluation: {json.dumps(eval_json)}"}], user_id=req.session_id)
    except Exception:
        eval_json = {"relevance": 5, "clarity": 5, "technical": 5, "red_flags": [], "note": "eval skipped"}

    if session["current_q"] < len(session["questions"]):
        session["current_q"] += 1

    finished = session["current_q"] >= len(session["questions"])
    # Also finish if time exceeded
    if now - session["start_time"] > session["duration"]:
        finished = True
    return {"finished": finished, "evaluation": eval_json}

@app.get("/api/summary")
async def get_summary(session_id: str = Query(...)):
    mem_data = mem_client.get_all(user_id=session_id)
    if not mem_data:
        return {"error": "No memory found for this session"}

    transcript_parts = []
    qa_log = []
    last_q = None

    for m in mem_data:
        for msg in m.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            transcript_parts.append(content)

            if content.startswith("Q: "):
                last_q = content[3:].strip()
            elif content.startswith("A: ") and last_q:
                qa_log.append({"question": last_q, "answer": content[3:].strip()})
                last_q = None

    transcript = "\n".join(transcript_parts)

    try:
        lc_summary = summarize_transcript(transcript) or {}
    except Exception:
        lc_summary = {}

    try:
        crew_summary = run_summarizer(transcript)
    except Exception:
        crew_summary = "CrewAI summarizer skipped."

    # CrewAI final HR report
    try:
        report_text = run_reporter(json.dumps(lc_summary))
    except Exception:
        report_text = "Report generation skipped."

    return {
        "jd": "Stored in memory",
        "resume": "Stored in memory",
        "interview_log": qa_log,
        "summary": lc_summary.get("summary") or crew_summary,
        "status": lc_summary.get("recommendation", "Maybe"),
        "strengths": lc_summary.get("strengths", []),
        "weaknesses": lc_summary.get("weaknesses", []),
        "overall_score": lc_summary.get("overall_score", 60),
        "hr_report": report_text
    }

# Speech-to-text endpoint
@app.post("/api/interview/speech")
async def speech_to_text(session_id: str = Query(...), file: UploadFile = File(...)):
    import tempfile
    import datetime
    try:
        import whisper
    except ImportError:
        return {"error": "Whisper not installed."}
    session = sessions.get(session_id)
    if not session:
        return {"error": "Invalid session_id"}
    now = datetime.datetime.utcnow().timestamp()
    if now - session["start_time"] > session["duration"]:
        return {"finished": True, "reason": "Interview time exceeded."}
    # Save audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    answer_text = result.get("text", "")
    # Use same logic as submit_answer
    if session["current_q"] < len(session["questions"]):
        q = session["questions"][session["current_q"]]
    else:
        q = session["questions"][-1]
    mem_client.add(messages=[{"role": "assistant", "content": f"Q: {q}"}], user_id=session_id)
    mem_client.add(messages=[{"role": "user", "content": f"A: {answer_text}"}], user_id=session_id)
    try:
        eval_json = evaluate_answer(q, answer_text)
        mem_client.add(messages=[{"role": "assistant", "content": f"Evaluation: {json.dumps(eval_json)}"}], user_id=session_id)
    except Exception:
        eval_json = {"relevance": 5, "clarity": 5, "technical": 5, "red_flags": [], "note": "eval skipped"}
    if session["current_q"] < len(session["questions"]):
        session["current_q"] += 1
    finished = session["current_q"] >= len(session["questions"])
    if now - session["start_time"] > session["duration"]:
        finished = True
    return {"finished": finished, "evaluation": eval_json, "answer_text": answer_text}


