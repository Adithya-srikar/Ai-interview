import os
from langchain_openai import ChatOpenAI

MODEL_QA = os.getenv("MODEL_QA", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("MODEL_TEMP", "0.3"))

def lc_llm():
    return ChatOpenAI(
        model=MODEL_QA,
        temperature=TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
