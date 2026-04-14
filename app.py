import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr

# ── Setup ──────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="java_kb", metadata={"hnsw:space": "cosine"}
)

# ── Knowledge base ─────────────────────────────────────
knowledge = [
    "HashMap stores key-value pairs using hashCode() for bucket location and equals() for collision resolution. Not thread-safe. ConcurrentHashMap is the thread-safe alternative using segment locking.",
    "Apache Kafka partitions allow parallel processing. Each partition is an ordered log. Consumer groups distribute partitions across consumers — one partition per consumer at most. More partitions = more parallelism.",
    "Spring Boot @Autowired injects dependencies automatically. Use constructor injection over field injection — it makes dependencies explicit and enables easier testing with mocks.",
    "Java Stream terminal operations: collect() gathers results, count() counts elements, reduce() aggregates, forEach() iterates. Without a terminal op, no intermediate operations execute — streams are lazy.",
    "Kafka offset is the position of a message in a partition. Consumers commit offsets to track progress. Auto-commit can cause duplicate processing — manual commit gives at-least-once or exactly-once guarantees.",
    "Microservice patterns: Circuit Breaker (Resilience4j) prevents cascade failures. Saga pattern handles distributed transactions. API Gateway (Spring Cloud Gateway) routes and authenticates requests.",
    "Java virtual threads (Project Loom, Java 21) are lightweight threads managed by JVM, not OS. Millions can run concurrently. Ideal for IO-bound tasks like REST calls. Use Thread.ofVirtual().start().",
    "Docker multi-stage build: first stage compiles Java (uses JDK), second stage runs it (uses smaller JRE). Reduces final image size from ~600MB to ~100MB. FROM eclipse-temurin:17-jre in final stage.",
    "Spring @Transactional ensures database operations either all succeed or all rollback. Default propagation is REQUIRED — joins existing transaction or creates new one. Use readOnly=true for SELECT queries.",
    "Kubernetes Pod is the smallest deployable unit — one or more containers. Deployment manages replica count and rolling updates. Service exposes Pods via stable IP. Ingress handles external HTTP routing.",
]

# Embed and store all knowledge
vectors = embedder.encode(knowledge).tolist()
collection.add(
    documents=knowledge,
    embeddings=vectors,
    ids=[f"k{i}" for i in range(len(knowledge))]
)
print(f"Knowledge base ready: {collection.count()} entries")

# ── Core functions ─────────────────────────────────────
def retrieve(query: str, n: int = 3) -> list:
    vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=vec, n_results=n)
    return results["documents"][0]

def rag_chat(question: str, history: list) -> str:
    chunks = retrieve(question)
    context = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(chunks)])

    system = """You are JavaMentor, a Java interview coach.
You have access to a knowledge base (provided as CONTEXT in each message).
Rules:
- Answer ONLY from the context. If not in context, say so honestly.
- After answering, ask a follow-up interview question on the same topic.
- Be encouraging but precise. Point out gaps in the user's answers."""

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content

# ── Gradio UI ──────────────────────────────────────────
def chat(message, history):
    clean_history = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            clean_history.append({
                "role": item["role"],
                "content": item["content"]
            })
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # older Gradio format: [user_msg, bot_msg]
            if item[0]: clean_history.append({"role": "user", "content": item[0]})
            if item[1]: clean_history.append({"role": "assistant", "content": item[1]})

    return rag_chat(message, clean_history)

demo = gr.ChatInterface(
    fn=chat,
    title="JavaMentor AI",
    description="RAG-powered Java interview coach. Ask anything about Java, Spring Boot, Kafka, or Microservices.",
    examples=[
        "How does HashMap work internally?",
        "What is a Kafka consumer group?",
        "Explain Spring @Transactional",
        "What are Java virtual threads?",
    ]
)

demo.launch()
