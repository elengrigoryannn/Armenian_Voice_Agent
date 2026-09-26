import glob
import json
import os
import sys
import time

import chromadb
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

RAW_DATA_DIR = "raw_data"
CONFIG_PATH = "institutions.yaml"
DB_DIR = "chroma_db"
COLLECTION_NAME = "bank_docs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 5
# Gemini's embedding model returns cosine distance in [0, 2]; lower =
# more similar. Tune this after testing on your own data — if you see
# confident wrong answers, lower it; if too many false refusals, raise
# it slightly. (If you change EMBEDDING_MODEL, re-check this threshold —
# different embedding models have different distance distributions.)
MAX_RELEVANT_DISTANCE = 0.9

GEMINI_MODEL = "gemini-3.8-flash"
# Multilingual embedding model (100+ languages, Armenian included, GA as
# of writing) — this is what makes retrieval actually work for Armenian
# questions. Chroma's old default embedding function is English-tuned
# and would silently give poor results for Armenian text.
EMBEDDING_MODEL = "gemini-embedding-001"
ALLOWED_TOPICS = ["loans", "deposits", "branches"]

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2  # doubles each retry (2s, 4s, 8s...)


def _is_transient(error: Exception) -> bool:
    """503 (overloaded) and 429 (rate limited) are worth retrying;
    other errors (bad request, auth failure, etc.) are not — retrying
    those would just waste time before failing anyway."""
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    return status in (503, 429) or "UNAVAILABLE" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def generate_with_retry(prompt: str):
    """Wraps gemini_client.models.generate_content with a few retries on
    transient errors (Gemini overloaded / rate limited), since these are
    usually short-lived spikes per Google's own error message. Raises on
    the final attempt so callers' existing try/except still catches a
    genuine, persistent failure."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS and _is_transient(e):
                time.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            raise last_error


def embed_texts(texts: list, task_type: str) -> list:
    """Embed a batch of strings with Gemini's multilingual embedding
    model. task_type must be "RETRIEVAL_DOCUMENT" when embedding chunks
    to store, or "RETRIEVAL_QUERY" when embedding a user's question —
    using the matching type on each side measurably improves retrieval
    quality (this is what the model was trained for)."""
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


# ---------- Config ----------

def load_institution_names():
    if not os.path.exists(CONFIG_PATH):
        return []
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [i["name"] for i in cfg.get("institutions", [])]


# ---------- Chunking ----------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-overlap:]
            current, current_len = overlap_words, len(overlap_words)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------- Ingestion ----------

def get_collection():
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    # No embedding_function here: we embed manually with embed_texts()
    # so we can use the right task_type (document vs. query) on each side.
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def safe_id_part(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:80]


def ingest():
    collection = get_collection()
    files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.json"), recursive=True)

    if not files:
        print(f"No scraped data found in ./{RAW_DATA_DIR}/. Run scrape.py first.")
        return

    ids, documents, metadatas = [], [], []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)

        institution = record["institution"]
        url = record["url"]
        topic = record["topic"]
        title = record.get("title", "")
        collected_at = record.get("collected_at", "")
        text = record.get("text", "")

        if topic not in ALLOWED_TOPICS:
            print(f"  SKIP {filepath}: topic '{topic}' not in {ALLOWED_TOPICS}")
            continue

        chunks = chunk_text(text)
        page_id = safe_id_part(url)

        for i, chunk in enumerate(chunks):
            # Deterministic ID -> re-ingesting the same page overwrites its
            # old chunks (upsert) instead of duplicating or mixing with
            # other institutions/pages.
            chunk_id = f"{safe_id_part(institution)}__{page_id}__chunk{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "institution": institution,
                "url": url,
                "title": title,
                "topic": topic,
                "collected_at": collected_at,
                "chunk_index": i,
            })

        print(f"  {institution} [{topic}] {url}: {len(chunks)} chunks")

    if documents:
        # Embed in batches: Gemini's embedding endpoint caps how much it
        # accepts per call, and batching also makes re-ingesting large
        # sites much faster than one call per chunk.
        embeddings = []
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings.extend(embed_texts(batch, task_type="RETRIEVAL_DOCUMENT"))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas,
                           embeddings=embeddings)
        print(f"\nIngested {len(documents)} chunks from {len(files)} page(s).")
    else:
        print("Nothing valid to ingest.")


# ---------- Language detection ----------

def detect_language(text: str) -> str:
    """Lightweight script-based language guess: 'hy' (Armenian), 'ru'
    (Russian/Cyrillic), or 'en' (default — Latin script or anything
    else, including transliterated text). Good enough to pick which
    language to answer/refuse in without an extra API call; the actual
    answer generation prompt also gets told explicitly which language
    to use, so a borderline guess here doesn't need to be perfect."""
    for ch in text:
        code = ord(ch)
        if 0x0530 <= code <= 0x058F:  # Armenian Unicode block
            return "hy"
    for ch in text:
        code = ord(ch)
        if 0x0400 <= code <= 0x04FF:  # Cyrillic Unicode block
            return "ru"
    return "en"


LANGUAGE_NAMES = {"hy": "Armenian (հայերեն)", "ru": "Russian (русский)", "en": "English"}

REFUSAL_OUT_OF_SCOPE = {
    "hy": ("Ես կարող եմ պատասխանել միայն վարկերի, ավանդների և ֆինանսական "
           "հաստատությունների մասնաճյուղերի հետ կապված հարցերին՝ իմ "
           "տվյալների բազայում առկա տեղեկության շրջանակում։ Այս հարցը "
           "դուրս է իմ օգնության շրջանակից։"),
    "ru": ("Я могу отвечать только на вопросы о кредитах, вкладах и "
           "отделениях финансовых организаций, представленных в моих "
           "данных. Этот вопрос выходит за рамки того, чем я могу помочь."),
    "en": ("I can only answer questions about loans/credit products, "
           "deposits, and branch locations of the financial institutions "
           "in my data. That question is outside what I'm set up to help with."),
}

REFUSAL_NO_EVIDENCE = {
    "hy": ("Ես չկարողացա գտնել բավարար չափով համապատասխան տեղեկություն "
           "առկա պաշտոնական տվյալների մեջ այս հարցին պատասխանելու համար։ "
           "Սա կարող է նշանակել, որ այս հաստատության կամ թեմայի "
           "վերաբերյալ տվյալներ դեռ հավաքված չեն, կամ էջը դա չի ընդգրկում։"),
    "ru": ("Я не смог найти достаточно релевантную информацию в "
           "доступных официальных данных для ответа на этот вопрос. "
           "Возможно, данные по этой организации или теме еще не "
           "собраны, либо страница этого не содержит."),
    "en": ("I couldn't find sufficiently relevant information in the "
           "available official data to answer that. This may mean the "
           "data hasn't been collected for this institution/topic yet, "
           "or the page didn't cover it."),
}

SERVICE_UNAVAILABLE = {
    "hy": ("Ես ընթացիկ պահին խնդիրներ ունեմ պատասխանող ծառայության հետ "
           "կապվելու հարցում․ սա ժամանակավոր խնդիր է և կապված չէ ձեր "
           "հարցի հետ։ Խնդրում ենք փորձել կրկին մի փոքր ժամանակ անց։"),
    "ru": ("У меня сейчас проблемы с подключением к сервису ответов — "
           "это временная проблема и не связана с вашим вопросом. "
           "Пожалуйста, попробуйте еще раз через некоторое время."),
    "en": ("I'm having trouble reaching the answering service right now "
           "— this is a temporary issue, not something wrong with your "
           "question. Please try again in a moment."),
}


# ---------- Scope guard ----------

def classify_scope(question: str) -> str:
    """Ask the model to classify the question into one of the allowed
    topics, or 'out_of_scope'. This runs BEFORE any retrieval, so
    off-topic questions never reach the document store or a full answer.
    The question may be in Armenian, English, Russian, or mixed —
    classification works regardless of language; only the category
    label returned needs to be one of the fixed English words below."""
    prompt = f"""Classify the following user question into exactly one category.
The question may be written in Armenian, English, Russian, or a mix.
Respond with only one word, nothing else: loans, deposits, branches, or out_of_scope.

- "loans": credit products, interest rates on loans, repayment periods, loan
  eligibility, required documents for loans, loan fees/conditions.
- "deposits": deposit products, deposit interest rates, currencies, minimum
  amounts, deposit periods, replenishment/withdrawal conditions.
- "branches": branch addresses, cities/regions, working hours, contact info.
- "out_of_scope": anything else, including politics, entertainment, medical
  advice, programming, general knowledge, personal financial advice,
  account-specific/confidential requests, or products not covered above.

Question: {question}

Category:"""

    response = generate_with_retry(prompt)
    label = response.text.strip().lower()
    # Robust match: catches the model answering with a singular form
    # ("loan" instead of "loans"), a plural of something singular, or
    # minor wording drift in either direction — plain substring checks
    # like `topic in label` only catch one direction and silently
    # misclassify anything not phrased in the exact expected form.
    label_stem = label.rstrip("s.,!? \n")
    for topic in ALLOWED_TOPICS:
        topic_stem = topic.rstrip("s")
        if topic in label or label in topic or topic_stem in label_stem or label_stem in topic_stem:
            return topic
    return "out_of_scope"


def detect_institution(question: str, known_names):
    q_lower = question.lower()
    for name in known_names:
        if name.lower() in q_lower:
            return name
    return None


# ---------- Retrieval ----------

def retrieve(question, topic=None, institution=None, k=TOP_K):
    collection = get_collection()

    where = {}
    # if topic:
    #     where["topic"] = topic
    if institution:
        where["institution"] = institution

    query_embedding = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where if where else None,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    print(list(zip(docs, metas, distances)))
    return list(zip(docs, metas, distances))


# ---------- Generation ----------

def build_prompt(question, retrieved_chunks, answer_language: str):
    context_blocks = []
    for doc, meta, _ in retrieved_chunks:
        context_blocks.append(
            f"[Institution: {meta['institution']} | Topic: {meta['topic']} | "
            f"Source: {meta['url']} | Collected: {meta['collected_at']}]\n{doc}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are an assistant that answers ONLY questions about credit/loan
products, deposit products, and branch locations of Armenian financial
institutions, using ONLY the context below (sourced from official
institution websites).

Rules:
- Write your answer in {answer_language}, in natural, fluent, formal
  language — regardless of whether the source context below is in
  Armenian, English, or Russian. Translate any facts you use faithfully;
  never alter a number, date, rate, or address while translating.
- Answer strictly from the context. Never invent rates, addresses,
  requirements, dates, fees, or conditions not present in the context.
- If the context does not contain enough information to answer, say
  plainly, in {answer_language}, that the requested information could
  not be found in the available official data. Do not guess or fill
  gaps with general knowledge.
- Do not mix information between different institutions.
- Always state which institution and source URL your answer is based on
  (institution names and URLs may stay in their original form).
- Do not give personal financial advice or recommendations — state facts
  from the source data only.

Context:
{context}

Question: {question}

Answer (in {answer_language}):"""


def get_answer(question: str) -> dict:
    """Core answer logic, usable by both the CLI and the voice agent.
    Returns a dict: {"status": "ok"|"out_of_scope"|"no_evidence"|"unavailable",
                      "answer": str, "sources": [(institution, url, collected_at), ...]}
    "answer" is always populated (with a refusal/error message when
    status != "ok") so callers (like the voice agent) can always just
    speak it, and a transient Gemini outage never crashes the caller.
    The answer language always matches the question's language (Armenian,
    Russian, or English — detected from the question's script)."""
    lang = detect_language(question)
    # try:
    #     scope = classify_scope(question)
    # except Exception:
    #     return {"status": "unavailable", "answer": SERVICE_UNAVAILABLE[lang], "sources": []}

    # if scope == "out_of_scope":
    #     return {"status": "out_of_scope", "answer": REFUSAL_OUT_OF_SCOPE[lang], "sources": []}

    known_institutions = load_institution_names()
    institution = detect_institution(question, known_institutions)

    try:
        # retrieved = retrieve(question, topic=scope, institution=institution)
        retrieved = retrieve(question, topic=None, institution=institution)
    except Exception:
        return {"status": "unavailable", "answer": SERVICE_UNAVAILABLE[lang], "sources": []}

    if not retrieved:
        return {"status": "no_evidence", "answer": REFUSAL_NO_EVIDENCE[lang], "sources": []}

    best_distance = min(d for _, _, d in retrieved)
    if best_distance > MAX_RELEVANT_DISTANCE:
        return {"status": "no_evidence", "answer": REFUSAL_NO_EVIDENCE[lang], "sources": []}

    prompt = build_prompt(question, retrieved, answer_language=LANGUAGE_NAMES[lang])
    try:
        response = generate_with_retry(prompt)
    except Exception:
        return {"status": "unavailable", "answer": SERVICE_UNAVAILABLE[lang], "sources": []}

    answer = response.text

    seen = set()
    sources = []
    for _, meta, _ in retrieved:
        key = (meta["institution"], meta["url"])
        if key not in seen:
            seen.add(key)
            sources.append((meta["institution"], meta["url"], meta["collected_at"]))

    return {"status": "ok", "answer": answer, "sources": sources}


def ask(question):
    """CLI entry point: prints the answer and sources to the console."""
    result = get_answer(question)

    if result["status"] != "ok":
        print(result["answer"])
        return

    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Sources ---")
    for institution, url, collected_at in result["sources"]:
        print(f"  - {institution}: {url} (collected {collected_at})")


# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python rag.py ingest\n  python rag.py ask \"your question\"")
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        ingest()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Please provide a question: python rag.py ask \"your question\"")
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
