import os
os.environ["MEM0_TELEMETRY"] = "False"  # turn off Mem0 usage tracking (must be set before the import)
from mem0 import Memory


CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {"model": "qwen3.5", "temperature": 0},
    },
    "embedder": {
        "provider": "ollama",
        "config": {"model": "nomic-embed-text"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "bench",
            "path": "./.mem0_qdrant",
            "embedding_model_dims": 768,
        },
    },
}


class Mem0Memory:
    """Mem0 (open-source, local) behind the 3-method memory interface.

    It knows how to talk to Mem0. It does NOT know the test data or the answers.
    """

    def __init__(self):
        self.m = Memory.from_config(CONFIG)
        # qwen3.5 thinks by default (~40 s per call). Mem0 has no option to turn it off,
        # so wrap its Ollama client and force think=False (~1 s per call).
        original_chat = self.m.llm.client.chat
        self.m.llm.client.chat = lambda **kwargs: original_chat(think=False, **kwargs)

    def reset(self, user_id: str) -> None:
        self.m.delete_all(user_id=user_id)

    def ingest(self, user_id: str, session: dict) -> None:
        self.m.add(session["messages"], user_id=user_id)

    def retrieve(self, user_id: str, question: str) -> list[str]:
        results = self.m.search(question, filters={"user_id": user_id})
        return [r["memory"] for r in results["results"]]


# Manual check only: ingest u1, then show what Mem0 returns for each question.
# No expected answers here. Scoring is the job of eval.py.
if __name__ == "__main__":
    import json

    with open("data/u1.json") as f:
        user = json.load(f)

    memory = Mem0Memory()
    memory.reset(user["user_id"])
    for session in user["sessions"]:
        memory.ingest(user["user_id"], session)

    for q in user["questions"]:
        print(f"\nQ: {q['question']}")
        for line in memory.retrieve(user["user_id"], q["question"]):
            print("  -", line)
