import os
from dotenv import load_dotenv

load_dotenv()


def check_env():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY missing from .env")
    print(f"  GROQ_API_KEY: found ({key[:8]}...)")


def test_groq():
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    response = llm.invoke(
        [
            HumanMessage(
                content="Say exactly: 'Research agent online.' and nothing else."
            )
        ]
    )
    print(f"  Groq says: {response.content}")


if __name__ == "__main__":
    print("Day 1 smoke test\n")
    check_env()
    test_groq()
    print("\nDay 1 complete. Ready for Day 2.")
