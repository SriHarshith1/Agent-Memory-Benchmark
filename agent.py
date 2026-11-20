"""The fixed answering agent.

Every memory system uses this same agent with the same prompt and settings.
Only the `memories` list changes, so the comparison between systems is fair.
"""
from ollama import chat

MODEL = "qwen3.5"
UNKNOWN = "UNKNOWN"

SYSTEM_PROMPT = f"""You answer questions about a user, using ONLY the memories given to you.

Rules:
- Use only the memories. Do not guess and do not use outside knowledge.
- Give only the short answer (a name, a place, an email). No full sentence, no explanation.
- If two memories conflict, the more recent one is correct for questions about the present.
- If the memories do not contain the answer, reply with exactly: {UNKNOWN}"""


def answer(question: str, memories: list[str]) -> str:
    memory_block = "\n".join(f"- {m}" for m in memories) if memories else "(no memories)"
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Memories:\n{memory_block}\n\nQuestion: {question}"},
        ],
        think=False,
        # num_ctx: Ollama's default is 4096 tokens and it silently cuts longer prompts.
        # FullHistory with many sessions is longer than that, so raise it.
        options={"temperature": 0, "num_ctx": 16384},
    )
    return response.message.content.strip()


if __name__ == "__main__":
    fake = ["User's dog is named Bruno."]
    print(answer("What's my dog's name?", fake))                # expect: Bruno
    print(answer("What's my favorite football team?", fake))    # expect: UNKNOWN
