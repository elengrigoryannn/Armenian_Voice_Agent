# Armenian Bank Loan/Deposit/Branch Q&A Agent

A scoped RAG agent that answers questions only about:
- **loans/credit products** — rates, terms, eligibility, required documents, fees
- **deposits** — products, rates, currencies, minimums, periods, conditions
- **branches** — addresses, cities, hours, contact info

It refuses anything else (politics, general knowledge, personal financial
advice, account-specific data, etc.), and refuses to answer if it can't
find good evidence in the scraped official data — it never invents rates,
addresses, or conditions.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # https://aistudio.google.com/apikey
```

## 1. Configure institutions

Edit `institutions.yaml`. Three example Armenian banks are
pre-filled with confirmed official domains (Ameriabank, Ardshinbank,
Inecobank) — **but you need to verify the exact loan/deposit/branch page
URLs yourself**, since I generated placeholders and site structures
change. Open each bank's site and update the `url` fields to the real
current pages.

To add a 4th+ institution, copy one of the existing blocks and fill in
its name, base_url, and per-topic page URLs. No code changes needed.

## 2. add the content 
```bash
python add_manual_data.py --institution "Ameriabank" --topic loans \
    --url "https://ameriabank.am/hy/loans" --file loan_info.txt
```

or paste text directly without a file:
```bash
python add_manual_data.py --institution "Ameriabank" --topic loans \
    --url "https://ameriabank.am/hy/loans"
```
(paste the text, then Ctrl+D — or Ctrl+Z then Enter on Windows — to finish)

## 3. Ingest

```bash
python rag.py ingest
```

Chunks every JSON file in `raw_data/`, embeds the chunks (using Gemini's
multilingual embedding model, `gemini-embedding-001` — this is what
makes Armenian retrieval work well), and stores them in a local Chroma
DB (`chroma_db/`) with full metadata (institution, url, title, topic,
collected_at). Chunk IDs are deterministic
(`institution__url-hash__chunk-index`), so re-ingesting after a re-scrape
**updates** that page's chunks instead of duplicating them, and chunks
from different institutions/pages never collide or mix.

If you're updating from an older copy of this project that used
Chroma's default embedding function, delete `chroma_db/` once and
re-run `ingest` — old vectors aren't compatible with the new embedding
model.

## Armenian language support

- **Questions:** ask in Armenian, English, or Russian — the scope
  classifier and retrieval both work across languages.
- **Answers:** always generated in Armenian, regardless of what
  language the source page or the question was in — the model is
  instructed to translate facts faithfully rather than invent anything.
- **Retrieval quality:** relies on `gemini-embedding-001`, a multilingual
  embedding model (100+ languages, Armenian included). For the best
  results, point `institutions.yaml` at each bank's
  Armenian-language pages rather than English ones — placeholders are
  already set to `/hy/` paths, but verify these match each site's real
  Armenian locale path.
- **Voice:** the Gemini Live API auto-detects spoken language and
  responds in kind; `voice_agent.py`'s instructions additionally tell it
  to default to Armenian and to speak the RAG tool's (Armenian) answer
  back rather than re-translating it.

## 4. Ask

```bash
python rag.py ask "What is the minimum deposit amount at Idbank?"
python rag.py ask "What's the weather in Yerevan?"     # refused — out of scope
python rag.py ask "What's Ardshinbank's mortgage rate?" # refused if not yet scraped
```

Each question goes through:
1. **Scope classification** — is this about loans, deposits, or branches?
   If not, it's refused immediately, before any retrieval happens.
2. **Retrieval** — pulls the most relevant chunks, filtered by topic and,
   if the question names a known institution, by that institution too
   (so a question about one bank won't pull in another's rates).
3. **Confidence check** — if the best match is too dissimilar
   (`MAX_RELEVANT_DISTANCE` in `rag.py`), it says the information
   couldn't be found rather than answering from a weak match.
4. **Answer** — generated strictly from the retrieved context, with a
   rule against inventing missing details, plus source citations.



## Adding institution-specific scraping logic

## Tuning

- `MAX_RELEVANT_DISTANCE` in `rag.py` controls how strict the "couldn't
  find it" refusal is. Lower = stricter (more refusals, fewer wrong
  answers); higher = more lenient. Test with real questions against your
  scraped data and adjust.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` control chunking granularity — smaller
  chunks are more precise for numeric facts (rates, minimums), larger
  chunks preserve more surrounding context.

## Voice interface (self-hosted LiveKit, no Docker, not LiveKit Cloud)

Talk to the same scoped RAG pipeline out loud: your speech → text/answer
(Gemini Live + your RAG pipeline) → spoken answer. The LiveKit server
runs as a plain local binary — no Docker, no Redis, not LiveKit Cloud.

### 1. Install the LiveKit server binary

**macOS:**
```bash
brew update && brew install livekit
```
**Linux:**
```bash
curl -sSL https://get.livekit.io | bash
```
**Windows:** download the latest release from
https://github.com/livekit/livekit/releases/latest

### 2. Install Python dependencies and configure keys

```bash
pip install -r requirements.txt
cp .env.example .env
```
Fill in `GEMINI_API_KEY` in `.env`. Leave `LIVEKIT_API_KEY=devkey` and
`LIVEKIT_API_SECRET=secret` as-is — those are the fixed pair
`livekit-server --dev` always uses locally, nothing to generate.

### 3. Start the LiveKit server

```bash
livekit-server --dev
```
Leave this running in its own terminal. It binds to `127.0.0.1:7880` by
default (add `--bind 0.0.0.0` if you need to reach it from another
device on your network).

### 4a. Quickest test — talk to it in your terminal

```bash
python voice_agent.py console
```
This runs the agent locally against your mic/speakers without needing
the web client or a browser.

### 4b. Full test — real LiveKit room + browser client

```bash
python voice_agent.py dev       # terminal: the agent worker
python token_server.py          # another terminal: serves web/ + mints tokens
```
Then open **http://localhost:8000** — it fetches a fresh access token
automatically (via `GET /token`) and shows a single connect button. No
copy-pasting a token by hand: `token_server.py` mints one on page load
using `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`, and a new one each time
you disconnect and reconnect. The "Connection details" panel still
shows the URL/token it's using, read-only, for debugging.

### How the voice pipeline stays scoped

`voice_agent.py` uses the **Gemini Live API** (speech in, speech out, one
model, authenticated with just your Gemini API key) instead of separate
STT/TTS plugins — Gemini has no standalone STT reachable with only an
API key, so this sidesteps needing a Google Cloud service account.

Instead of letting the realtime model answer from its own knowledge, it's
instructed to call a `lookup_bank_info` tool for every question. That
tool runs `rag.get_answer()` — the exact same scoped RAG pipeline as the
text CLI — and the model speaks that result back. So topic refusal,
low-confidence refusal, and grounding in your scraped data all still
apply on voice.

**Trade-off to know about:** a speech-to-speech model is more likely to
lightly rephrase the tool's answer in its own words than a strict
STT → text-answer → TTS pipeline would. The system instructions push it
to stay close to the tool's wording, but this is looser than the exact
grounding you get from `rag.py ask`. If you need stricter word-for-word
fidelity, the alternative is discrete Google Cloud STT/TTS (Chirp) with
a GCP service account — happy to add that version if you want it.

### Notes

- The dev key pair (`devkey`/`secret`) only works with `livekit-server
  --dev` and is meant for local use. For anything beyond your own
  machine, follow LiveKit's production deployment guide and generate
  real keys.
- `livekit-agents` is a fast-moving framework. If `voice_agent.py`'s
  imports break after an upgrade, check
  https://docs.livekit.io/agents/ for the current equivalent pattern —
  `rag.py` itself has no dependency on any of this and keeps working
  standalone either way.
- Spoken answers say institution names instead of reading full URLs
  aloud — the text CLI (`rag.py ask`) still prints full source URLs.

## Known limitations to be aware of

- The scope classifier and confidence threshold are heuristics, not
  guarantees — test them against adversarial questions (see the earlier
  discussion on testing RAG pipelines) before treating this as
  production-ready.
- This scrapes public pages only; it has no access to account-specific
  or confidential data, by design.
