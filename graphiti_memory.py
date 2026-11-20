"""Zep's open-source engine (Graphiti), run locally, behind the 3-method memory interface.

Stack: Graphiti + Kuzu (embedded graph DB, no server) + Ollama (qwen3.5 for extraction, nomic-embed-text).
Zep Cloud runs the same engine with its own models, so local results are "Graphiti with a local model",
not "Zep Cloud". Say this in the write-up.

Note: Graphiti marks Kuzu as deprecated (upstream Kuzu is unmaintained). It still works. The
alternative is FalkorDB or Neo4j, and both need a server (Docker).
"""
import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
warnings.filterwarnings("ignore", message="The Kuzu backend is deprecated")
logging.getLogger("graphiti_core").setLevel(logging.WARNING)

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # noqa: E402
from graphiti_core.driver.driver import GraphProvider  # noqa: E402
from graphiti_core.driver.kuzu_driver import KuzuDriver  # noqa: E402
from graphiti_core.graph_queries import get_fulltext_indices  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

OLLAMA_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3.5"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
DB_PATH = "./.graphiti_kuzu"

# Sessions have no real dates, so give each one a fake date, one day apart.
# Graphiti uses these dates to decide which fact is newer (for updates and invalidation).
START_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ollama_client_without_thinking() -> AsyncOpenAI:
    """An OpenAI client for Ollama that turns off qwen3.5 thinking (about 2 s per call instead of 80 s)."""
    client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_URL)
    original_create = client.chat.completions.create

    async def create(*args, **kwargs):
        kwargs.setdefault("reasoning_effort", "none")
        return await original_create(*args, **kwargs)

    client.chat.completions.create = create
    return client


class GraphitiMemory:
    def __init__(self):
        llm_config = LLMConfig(api_key="ollama", model=LLM_MODEL, small_model=LLM_MODEL,
                               base_url=OLLAMA_URL, temperature=0)
        client = _ollama_client_without_thinking()
        driver = KuzuDriver(db=DB_PATH)
        # Bug in graphiti-core 0.31 + Kuzu: add_episode reads driver._database, but KuzuDriver never sets it.
        # Kuzu has one database, and its clone() returns the same driver, so any value works.
        driver._database = ""
        self.graphiti = Graphiti(
            graph_driver=driver,
            llm_client=OpenAIGenericClient(config=llm_config, client=client),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(api_key="ollama", base_url=OLLAMA_URL,
                                            embedding_model=EMBED_MODEL, embedding_dim=EMBED_DIM)
            ),
            # Not used by graphiti.search(), but Graphiti needs one at start-up (default = OpenAI key).
            cross_encoder=OpenAIRerankerClient(config=llm_config, client=client),
        )
        # Graphiti is async. Our interface is sync, so keep one event loop and run each call in it.
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self._create_kuzu_fulltext_indices())
        self.session_count: dict[str, int] = {}

    async def _create_kuzu_fulltext_indices(self):
        # Bug in graphiti-core 0.31 + Kuzu: search needs full-text indexes, but the Kuzu driver
        # never loads the FTS extension or creates them. Do it here.
        driver = self.graphiti.driver
        for q in ["INSTALL FTS;", "LOAD EXTENSION FTS;"]:
            await driver.execute_query(q)
        driver_log = logging.getLogger("graphiti_core.driver.kuzu_driver")
        driver_log.disabled = True  # the driver logs an error when an index already exists; that is expected
        try:
            for q in get_fulltext_indices(GraphProvider.KUZU):
                try:
                    await driver.execute_query(q)
                except RuntimeError as e:
                    if "already exists" not in str(e):
                        raise
        finally:
            driver_log.disabled = False

    def reset(self, user_id: str) -> None:
        self.loop.run_until_complete(clear_data(self.graphiti.driver, group_ids=[user_id]))
        self.session_count[user_id] = 0

    def ingest(self, user_id: str, session: dict) -> None:
        i = self.session_count.get(user_id, 0)
        self.session_count[user_id] = i + 1
        body = "\n".join(f"{m['role']}: {m['content']}" for m in session["messages"])
        self.loop.run_until_complete(self.graphiti.add_episode(
            name=f"{user_id}-{session['session_id']}",
            episode_body=body,
            source=EpisodeType.message,
            source_description="chat session",
            reference_time=START_DATE + timedelta(days=i),
            group_id=user_id,
        ))

    def retrieve(self, user_id: str, question: str) -> list[str]:
        edges = self.loop.run_until_complete(
            self.graphiti.search(question, group_ids=[user_id], num_results=10)
        )
        lines = []
        for e in edges:
            # Graphiti does not delete old facts. It marks them invalid with a date.
            # Pass that mark to the agent, the same way Zep puts validity dates in its context.
            ended = e.invalid_at or e.expired_at
            lines.append(f"{e.fact} (no longer true since {ended.date()})" if ended else e.fact)
        return lines


# Manual check: ingest u1, then show what Graphiti returns for each question.
if __name__ == "__main__":
    import json
    import time

    with open("data/u1.json") as f:
        user = json.load(f)

    memory = GraphitiMemory()
    memory.reset(user["user_id"])
    t = time.time()
    for session in user["sessions"]:
        memory.ingest(user["user_id"], session)
    print(f"ingest: {time.time() - t:.0f} s for {len(user['sessions'])} sessions")

    for q in user["questions"]:
        print(f"\nQ: {q['question']}")
        for line in memory.retrieve(user["user_id"], q["question"]):
            print("  -", line)
