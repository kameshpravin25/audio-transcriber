"""
FastAPI server that:
  1. Accepts raw audio from ESP32 over WebSocket (same port 8000)
  2. Streams audio to Deepgram via raw WebSocket (no SDK)
  3. Pushes transcripts to a browser via WebSocket
  4. When user presses Q/Stop, sends full transcript to Gemini via LangChain
     and streams the conversational response + follow-up question back to browser
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from langchain_google_genai import ChatGoogleGenerativeAI

# ─── Configuration ───────────────────────────────────────────────────────────

DEEPGRAM_API_KEY = "7d326f4220990e0a7dfff768d613c826f539726b"
GOOGLE_API_KEY      = "AIzaSyAovA40kOlPkQNOW5MEK2Zw9xRghfluuZQ"
GOOGLE_API_KEY_CHAT = "AIzaSyB8bb-S565l7kaeG7VU3NyfIAk58WRD5Q0"

SAMPLE_RATE  = 16000
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = int(os.environ.get("PORT", 8000))

DEEPGRAM_URL = (
    f"wss://api.deepgram.com/v1/listen"
    f"?model=nova-3&language=en&encoding=linear16"
    f"&sample_rate={SAMPLE_RATE}&channels=1"
    f"&interim_results=true&punctuate=true"
    f"&smart_format=true&endpointing=500"
    f"&utterance_end_ms=1500"
)

# ─── Shared state ────────────────────────────────────────────────────────────

browser_clients: list[WebSocket] = []

# Accumulates FINAL transcript lines during an active ESP32 session
transcript_buffer: list[str] = []

# Conversation history for multi-turn LLM summarization
conversation_history: list[tuple] = []

# Separate chat history for user ↔ Sync AI conversations
chat_history: list[tuple] = []

# Stores ALL generated summaries so chat can reference them
summary_history: list[str] = []

# ─── LLM setup ───────────────────────────────────────────────────────────────

# LLM for transcript summarization (Sync AI button)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Separate LLM for chat conversations (summary-aware Q&A)
chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY_CHAT,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

LLM_SYSTEM_PROMPT = (
    "You are Sync AI, a smart and concise assistant. "
    "When the user provides a transcript or message, summarise the key points "
    "from the conversation so far in 2-3 sentences. "
    "Then ask exactly 2 relevant follow-up questions to the user "
    "to help deepen the discussion or clarify important details."
)

CHAT_SYSTEM_PROMPT = (
    "You are Sync AI, a knowledgeable and friendly conversational assistant. "
    "You have access to ALL the AI-generated summaries from the user's audio "
    "transcription session. Use these summaries as your primary knowledge base "
    "to answer the user's questions accurately. "
    "If the user asks about something covered in the summaries, reference the "
    "relevant summary details in your answer. "
    "Keep answers clear and concise (2-4 sentences) unless the user asks for detail. "
    "Be natural and conversational."
)


async def run_llm_on_transcript(full_transcript: str):
    """
    Send the buffered transcript to Gemini via LangChain.
    Uses tuple-based messages and run_in_executor (same as new_with_llmv1.py).
    Streams the response back to browser clients.
    Each call is independent — previous summaries are NOT carried forward.
    """
    if not full_transcript.strip():
        await broadcast("__LLM_ERROR__:No transcript to process.")
        return

    await broadcast("__LLM_START__")
    print(f"[LLM] Processing transcript ({len(full_transcript)} chars)...")

    # Clear conversation history so each sync is independent (no appending)
    conversation_history.clear()

    # Build message list using tuples: (role, content)
    messages = [("system", LLM_SYSTEM_PROMPT)]

    # Add current transcript as the only user turn (no previous turns)
    messages.append(("human", full_transcript))

    try:
        # Call Gemini in a thread (same pattern as new_with_llmv1.py)
        loop = asyncio.get_event_loop()
        ai_msg = await loop.run_in_executor(None, llm.invoke, messages)
        full_response = ai_msg.content

        # Send the full response to browser
        await broadcast(f"__LLM_TOKEN__:{full_response}")

        # Save this summary so chat can reference it later
        summary_history.append(full_response)

        # Clear the transcript buffer so next sync only gets NEW lines
        transcript_buffer.clear()

        await broadcast("__LLM_DONE__")
        print(f"[LLM] Response: {full_response[:100]}...")

    except Exception as e:
        err = str(e)
        print(f"[LLM] Error: {err}")
        await broadcast(f"__LLM_ERROR__:{err}")


async def run_chat_response(user_message: str):
    """
    Handle a chat message from the user.
    Uses chat_llm (separate Gemini instance) with ALL generated summaries
    as context so the user can ask questions about any summary.
    Runs in parallel with STT — does not block audio processing.
    """
    if not user_message.strip():
        await broadcast("__CHAT_ERROR__:Empty message.")
        return

    await broadcast("__CHAT_START__")
    print(f"[Chat] User: {user_message[:80]}...")

    # Build message list with chat system prompt
    messages = [("system", CHAT_SYSTEM_PROMPT)]

    # Inject ALL generated summaries as context
    if summary_history:
        summaries_context = "\n\n".join(
            f"--- Summary {i+1} ---\n{s}" for i, s in enumerate(summary_history)
        )
        messages.append((
            "system",
            f"Here are all the AI-generated summaries from this session "
            f"({len(summary_history)} total). Use these to answer the user's "
            f"questions:\n\n{summaries_context}"
        ))

    # Include current (unsummarized) transcript if available
    if transcript_buffer:
        transcript_context = " ".join(transcript_buffer)
        messages.append((
            "system",
            f"Current live transcript (not yet summarized): {transcript_context}"
        ))

    # Replay previous chat turns
    for role, content in chat_history:
        messages.append((role, content))

    # Add current user message
    messages.append(("human", user_message))
    chat_history.append(("human", user_message))

    try:
        loop = asyncio.get_event_loop()
        ai_msg = await loop.run_in_executor(None, chat_llm.invoke, messages)

        full_response = ai_msg.content

        await broadcast(f"__CHAT_TOKEN__:{full_response}")
        chat_history.append(("ai", full_response))
        await broadcast("__CHAT_DONE__")
        print(f"[Chat] Sync AI: {full_response[:100]}...")

    except Exception as e:
        err = str(e)
        print(f"[Chat] Error: {err}")
        await broadcast(f"__CHAT_ERROR__:{err}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def broadcast(message: str):
    """Send a message to every connected browser client."""
    disconnected = []
    for ws in browser_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        browser_clients.remove(ws)


# ─── FastAPI lifecycle ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Server ready")
    yield


app = FastAPI(lifespan=lifespan)


# ─── WebSocket: browser clients ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_browser(ws: WebSocket):
    await ws.accept()
    browser_clients.append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            # Browser sends "PROCESS" when user clicks Stop/Q
            if msg == "PROCESS":
                full_text = " ".join(transcript_buffer)
                asyncio.create_task(run_llm_on_transcript(full_text))
            # Browser sends chat message as "CHAT:message"
            elif msg.startswith("CHAT:"):
                user_msg = msg[5:]
                asyncio.create_task(run_chat_response(user_msg))
            # Browser sends "CLEAR_HISTORY" to reset conversation
            elif msg == "CLEAR_HISTORY":
                conversation_history.clear()
                chat_history.clear()
                transcript_buffer.clear()
                summary_history.clear()
                await broadcast("__STATUS__:history_cleared")
    except WebSocketDisconnect:
        if ws in browser_clients:
            browser_clients.remove(ws)


# ─── WebSocket: ESP32 audio → Deepgram (fully async, no threads) ────────────

@app.websocket("/ws/audio")
async def websocket_audio(esp_ws: WebSocket):
    await esp_ws.accept()
    await broadcast("__STATUS__:esp32_connected")
    print("[ESP32] Connected via WebSocket")

    # Clear buffer for new session
    transcript_buffer.clear()

    dg_ws = None
    recv_task = None
    chunks = 0
    try:
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        print(f"[Deepgram] Connecting to: {DEEPGRAM_URL[:80]}...")
        print(f"[Deepgram] API key: {DEEPGRAM_API_KEY[:8]}...{DEEPGRAM_API_KEY[-4:]}")
        dg_ws = await websockets.connect(
            DEEPGRAM_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        )
        await broadcast("__STATUS__:deepgram_connected")
        print(f"[Deepgram] Connected (state: open={dg_ws.protocol.state if hasattr(dg_ws, 'protocol') else 'n/a'})")

        async def receive_transcripts():
            """Receive and process Deepgram transcript messages."""
            nonlocal dg_ws
            print("[Deepgram] Receive task started — waiting for messages...")
            msg_count = 0
            try:
                while True:
                    try:
                        msg = await dg_ws.recv()
                    except websockets.ConnectionClosed as e:
                        print(f"[Deepgram] Connection closed during recv: code={e.code} reason={e.reason}")
                        await broadcast(f"__DEBUG__:[DG] Closed: {e.code} {e.reason}")
                        break

                    msg_count += 1
                    # Log first few raw messages for debugging
                    if msg_count <= 3:
                        raw_preview = str(msg)[:300] if isinstance(msg, str) else f"<bytes len={len(msg)}>"
                        print(f"[Deepgram] Raw msg #{msg_count}: {raw_preview}")

                    try:
                        data = json.loads(msg)
                    except (json.JSONDecodeError, TypeError) as je:
                        print(f"[Deepgram] JSON parse error: {je} — raw: {str(msg)[:200]}")
                        continue

                    msg_type = data.get("type", "")

                    if msg_type != "Results":
                        print(f"[Deepgram] Event: {msg_type} → {str(data)[:200]}")
                        await broadcast(f"__DEBUG__:[DG] {msg_type}: {str(data)[:150]}")
                        continue

                    channel = data.get("channel")
                    if not isinstance(channel, dict):
                        continue

                    alternatives = channel.get("alternatives", [])
                    if not alternatives:
                        continue

                    transcript = alternatives[0].get("transcript", "")
                    if transcript:
                        is_final = data.get("is_final", False)
                        prefix = "FINAL" if is_final else "INTERIM"
                        await broadcast(f"__{prefix}__:{transcript}")
                        print(f"[{prefix}] {transcript}")

                        # Buffer only final lines for LLM
                        if is_final:
                            transcript_buffer.append(transcript)

            except asyncio.CancelledError:
                print(f"[Deepgram] Receive task cancelled after {msg_count} messages")
            except Exception as e:
                import traceback
                print(f"[Deepgram] Receive error: {type(e).__name__}: {e}")
                traceback.print_exc()
                await broadcast(f"__DEBUG__:[DG] Recv error: {e}")

        def task_exception_callback(task: asyncio.Task):
            """Catch any unhandled exception from the receive task."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                print(f"[Deepgram] TASK EXCEPTION: {type(exc).__name__}: {exc}")

        recv_task = asyncio.create_task(receive_transcripts())
        recv_task.add_done_callback(task_exception_callback)

        while True:
            audio = await esp_ws.receive_bytes()
            try:
                await dg_ws.send(audio)
            except Exception as send_err:
                print(f"[Deepgram] Send failed: {type(send_err).__name__}: {send_err}")
                print("[Deepgram] Connection lost, reconnecting...")
                try:
                    dg_ws = await websockets.connect(
                        DEEPGRAM_URL,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=5,
                    )
                    recv_task.cancel()
                    recv_task = asyncio.create_task(receive_transcripts())
                    recv_task.add_done_callback(task_exception_callback)
                    await dg_ws.send(audio)
                    print("[Deepgram] Reconnected!")
                except Exception as re_err:
                    print(f"[Deepgram] Reconnect failed: {re_err}")
                    break

            chunks += 1
            if chunks == 1:
                print(f"[Audio] First chunk: {len(audio)} bytes")
            if chunks % 500 == 0:
                print(f"[Audio] Forwarded {chunks} chunks")
                # Check if receive task is still alive
                if recv_task.done():
                    print("[Deepgram] WARNING: Receive task died! Restarting...")
                    recv_task = asyncio.create_task(receive_transcripts())
                    recv_task.add_done_callback(task_exception_callback)

    except WebSocketDisconnect:
        print(f"[ESP32] Disconnected after {chunks} chunks")
    except Exception as e:
        import traceback
        print(f"[Error] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if recv_task and not recv_task.done():
            recv_task.cancel()
        try:
            if dg_ws:
                await dg_ws.close()
        except Exception:
            pass
        await broadcast("__STATUS__:esp32_disconnected")
        await broadcast("__STATUS__:deepgram_disconnected")
        print("[Cleanup] Session ended")


# ─── Serve the single-page UI ───────────────────────────────────────────────

PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SyncScribe</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0c0e12;
    --surface:   #14171d;
    --border:    #23272f;
    --text:      #d4d8e0;
    --muted:     #6b7280;
    --accent:    #34d399;
    --accent-dim:#1a7a52;
    --danger:    #f87171;
    --warn:      #fbbf24;
    --llm:       #818cf8;
    --llm-dim:   #312e81;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Source Serif 4', Georgia, serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Split layout wrapper ───────────────────── */
  .app-body {
    display: flex;
    flex: 1;
    overflow: hidden;
    height: calc(100vh - 80px);
  }

  /* ── Header (glassmorphism) ──────────────────── */
  header {
    padding: 28px 32px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    position: sticky;
    top: 0;
    z-index: 100;
    /* Glass background */
    background: rgba(20, 23, 29, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    /* Glass border & glow */
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: none;
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.25),
      inset 0 1px 0 rgba(255, 255, 255, 0.12),
      inset 0 -1px 0 rgba(255, 255, 255, 0.04),
      inset 0 0 12px 2px rgba(255, 255, 255, 0.03);
  }

  /* Top edge light streak */
  header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent 5%,
      rgba(255, 255, 255, 0.35),
      rgba(52, 211, 153, 0.25),
      rgba(255, 255, 255, 0.35),
      transparent 95%
    );
  }

  /* Left edge light streak */
  header::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.3),
      transparent 50%,
      rgba(255, 255, 255, 0.08)
    );
  }

  header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 18px;
    letter-spacing: -0.02em;
    color: var(--text);
  }

  .status-row {
    display: flex;
    gap: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background 0.3s;
  }
  .dot.ok    { background: var(--accent); box-shadow: 0 0 6px var(--accent-dim); }
  .dot.warn  { background: var(--warn); }
  .dot.err   { background: var(--danger); }
  .dot.llm   { background: var(--llm);  box-shadow: 0 0 6px var(--llm-dim); }

  /* ── Transcript area (left panel) ────────────── */
  main {
    flex: 1;
    padding: 24px 32px 80px;
    overflow-y: auto;
    min-width: 0;
  }

  #transcript-container {
    max-width: 720px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .line {
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 17px;
    line-height: 1.65;
    animation: fadein 0.25s ease;
  }

  .line.final {
    background: var(--surface);
    border-left: 3px solid var(--accent-dim);
    color: var(--text);
  }

  .line.interim {
    background: transparent;
    border-left: 3px solid var(--border);
    color: var(--muted);
    font-style: italic;
  }

  /* ── LLM response block ─────────────────────── */
  .llm-block {
    margin-top: 16px;
    background: #1a1b2e;
    border: 1px solid var(--llm-dim);
    border-left: 3px solid var(--llm);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 17px;
    line-height: 1.75;
    color: #c7d2fe;
    animation: fadein 0.3s ease;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .llm-block .llm-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--llm);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .llm-block .llm-label .pulse {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--llm);
    animation: blink 1s infinite;
  }
  .llm-block .llm-label .pulse.done {
    animation: none;
    background: var(--accent);
  }

  .divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 16px 0 10px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }
  .divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .line .ts {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-right: 10px;
    user-select: none;
  }

  .empty-state {
    text-align: center;
    margin-top: 20vh;
    color: var(--muted);
    font-size: 15px;
    line-height: 2;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── Footer bar (inside left panel) ──────────── */
  footer {
    position: sticky;
    bottom: 0;
    padding: 12px 24px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    gap: 12px;
    flex-wrap: wrap;
    z-index: 10;
  }

  .footer-btns { display: flex; gap: 8px; }

  footer button {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: inherit;
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  footer button:hover {
    border-color: var(--accent-dim);
    color: var(--text);
  }

  footer button#btn-stop {
    border-color: var(--llm-dim);
    color: var(--llm);
  }
  footer button#btn-stop:hover {
    border-color: var(--llm);
    background: #1e1b4b44;
  }
  footer button#btn-stop:disabled {
    opacity: 0.35;
    cursor: default;
    pointer-events: none;
  }

  footer button#btn-clear-history {
    border-color: var(--border);
    color: var(--muted);
  }

  @keyframes fadein {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
  }

  /* ── Right chat panel ────────────────────────── */
  .chat-panel {
    width: 380px;
    min-width: 320px;
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    background: var(--surface);
  }
  /* On desktop, .hidden class is ignored — both panels always show */
  @media (min-width: 769px) {
    .chat-panel.hidden { display: flex !important; }
    .left-panel.hidden { display: flex !important; }
  }

  .chat-panel-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--llm);
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(20, 23, 29, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
  }
  .chat-panel-header .panel-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--llm);
    box-shadow: 0 0 6px var(--llm-dim);
    animation: blink 2s infinite;
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .chat-empty {
    text-align: center;
    margin-top: 30%;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
    font-family: 'IBM Plex Mono', monospace;
  }

  .chat-input-bar {
    padding: 12px 14px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
  }

  .chat-input-bar input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 14px;
    padding: 10px 12px;
    border-radius: 8px;
    outline: none;
    transition: border-color 0.2s;
  }
  .chat-input-bar input:focus {
    border-color: var(--llm);
  }
  .chat-input-bar input::placeholder {
    color: var(--muted);
  }

  .chat-input-bar button {
    background: var(--llm-dim);
    border: 1px solid var(--llm);
    color: var(--llm);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .chat-input-bar button:hover {
    background: #312e81;
    color: #c7d2fe;
  }
  .chat-input-bar button:disabled {
    opacity: 0.35;
    cursor: default;
    pointer-events: none;
  }

  /* ── Chat bubbles ───────────────────────────── */
  .chat-user {
    align-self: flex-end;
    background: #1a3a2a;
    border: 1px solid var(--accent-dim);
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.6;
    color: var(--accent);
    max-width: 90%;
    animation: fadein 0.2s ease;
  }
  .chat-user .chat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--accent-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }

  .chat-ai {
    align-self: flex-start;
    background: #1a1b2e;
    border: 1px solid var(--llm-dim);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.6;
    color: #c7d2fe;
    max-width: 90%;
    animation: fadein 0.25s ease;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .chat-ai .chat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--llm);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* ── Mobile tab bar ──────────────────────────── */
  .mobile-tabs {
    display: none;
  }

  /* ── Responsive: mobile layout ─────────────── */
  @media (max-width: 768px) {
    header {
      padding: 14px 16px 12px;
    }
    header h1 {
      font-size: 15px;
    }
    .status-row {
      gap: 10px;
      font-size: 10px;
    }

    .mobile-tabs {
      display: flex;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0;
      z-index: 50;
    }
    .mobile-tabs button {
      flex: 1;
      padding: 10px 0;
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--muted);
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .mobile-tabs button.active {
      color: var(--llm);
      border-bottom-color: var(--llm);
    }

    .app-body {
      flex-direction: column;
      height: calc(100vh - 100px);
    }

    .left-panel {
      display: flex !important;
      flex: 1;
      min-height: 0;
    }
    .left-panel.hidden {
      display: none !important;
    }

    .chat-panel {
      display: flex !important;
      width: 100% !important;
      min-width: 0 !important;
      flex: 1;
      border-left: none !important;
      border-top: 1px solid var(--border);
    }
    .chat-panel.hidden {
      display: none !important;
    }

    main {
      padding: 16px 16px 60px;
    }

    footer {
      padding: 8px 12px;
    }
    .footer-btns {
      gap: 4px;
    }
    footer button {
      padding: 5px 10px;
      font-size: 11px;
    }

    .chat-input-bar {
      padding: 8px 10px;
    }

    #debug-panel {
      display: none !important;
    }
  }

</style>
</head>
<body>

<header>
  <h1>SyncScribe</h1>
  <div class="status-row">
    <div class="status-item">
      <span class="dot" id="dot-ws"></span>
      <span id="lbl-ws">WebSocket</span>
    </div>
    <div class="status-item">
      <span class="dot" id="dot-dg"></span>
      <span id="lbl-dg">Deepgram</span>
    </div>
    <div class="status-item">
      <span class="dot" id="dot-esp"></span>
      <span id="lbl-esp">ESP32</span>
    </div>
    <div class="status-item">
      <span class="dot" id="dot-llm"></span>
      <span id="lbl-llm">Sync AI</span>
    </div>
  </div>
</header>

<!-- Mobile tab bar -->
<div class="mobile-tabs" id="mobile-tabs">
  <button class="active" id="tab-transcript" onclick="switchTab('transcript')">📝 Transcript</button>
  <button id="tab-chat" onclick="switchTab('chat')">💬 Chat</button>
</div>

<div class="app-body">

<!-- Left panel: transcript + summaries -->
<div class="left-panel" id="left-panel" style="flex:1;display:flex;flex-direction:column;overflow:hidden;">
<main>
  <div id="transcript-container">
    <div class="empty-state" id="empty">
      Waiting for audio stream from ESP32 ...<br>
      ESP32 connects via WebSocket on /ws/audio
    </div>
  </div>
</main>

<footer>
  <span id="line-count">0 lines</span>
  <div class="footer-btns">
    <button id="btn-stop" onclick="triggerLLM()" disabled
            title="Process transcript with Sync AI (or press Q)">
      ⬡ Ask Sync AI
    </button>
    <button onclick="clearTranscript()">Clear</button>
    <button id="btn-clear-history" onclick="clearHistory()">Reset Chat</button>
  </div>
</footer>
</div>

<!-- Right panel: chat with Sync AI -->
<aside class="chat-panel hidden" id="chat-panel">
  <div class="chat-panel-header">
    <span class="panel-dot"></span>
    Chat with Sync AI
  </div>
  <div class="chat-messages" id="chat-messages">
    <div class="chat-empty" id="chat-empty">
      Ask questions about your<br>AI-generated summaries...
    </div>
  </div>
  <div class="chat-input-bar">
    <input type="text" id="chat-input" placeholder="Ask about summaries..." autocomplete="off" />
    <button id="btn-chat-send" onclick="sendChat()">Send</button>
  </div>
</aside>

</div>

<div id="debug-panel" style="position:fixed;bottom:50px;left:0;width:380px;max-height:30vh;overflow-y:auto;background:#0d0f14;border:1px solid #23272f;border-radius:0 8px 0 0;padding:10px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7280;z-index:999;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="color:#818cf8;">⬡ Debug Log</span>
    <button onclick="document.getElementById('debug-panel').style.display='none'" style="background:none;border:none;color:#6b7280;cursor:pointer;font-size:11px;">✕</button>
  </div>
  <div id="debug-log"></div>
</div>

<script>
  const container   = document.getElementById('transcript-container');
  const emptyEl     = document.getElementById('empty');
  const countEl     = document.getElementById('line-count');
  const dotWs       = document.getElementById('dot-ws');
  const dotDg       = document.getElementById('dot-dg');
  const dotEsp      = document.getElementById('dot-esp');
  const dotLlm      = document.getElementById('dot-llm');
  const btnStop     = document.getElementById('btn-stop');
  const chatInput   = document.getElementById('chat-input');
  const btnChatSend = document.getElementById('btn-chat-send');
  const chatContainer = document.getElementById('chat-messages');
  const chatEmptyEl   = document.getElementById('chat-empty');
  const leftPanel     = document.getElementById('left-panel');
  const chatPanel     = document.getElementById('chat-panel');
  const tabTranscript = document.getElementById('tab-transcript');
  const tabChat       = document.getElementById('tab-chat');

  // ── Mobile tab switching ──────────────────────────────────────────────────
  function switchTab(tab) {
    if (tab === 'transcript') {
      leftPanel.classList.remove('hidden');
      chatPanel.classList.add('hidden');
      tabTranscript.classList.add('active');
      tabChat.classList.remove('active');
    } else {
      leftPanel.classList.add('hidden');
      chatPanel.classList.remove('hidden');
      tabTranscript.classList.remove('active');
      tabChat.classList.add('active');
      scrollChatBottom();
    }
  }

  let interimEl    = null;
  let lineCount    = 0;
  let hasTranscript = false;   // true once at least one FINAL line exists
  let llmStreaming  = false;
  let chatStreaming = false;
  let currentLlmEl  = null;    // the active streaming .llm-block
  let currentChatAiEl = null;  // the active streaming chat-ai bubble
  let ws;

  // ── Keyboard shortcut Q (skip when typing in chat input) ──────────────────
  document.addEventListener('keydown', (e) => {
    if (document.activeElement === chatInput) return;
    if ((e.key === 'q' || e.key === 'Q') && !e.ctrlKey && !e.metaKey) {
      triggerLLM();
    }
  });

  // ── Enter key in chat input ───────────────────────────────────────────────
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  function now() {
    return new Date().toLocaleTimeString('en-GB', { hour12: false });
  }

  function scrollBottom() {
    const mainEl = document.querySelector('main');
    if (mainEl) mainEl.scrollTo({ top: mainEl.scrollHeight, behavior: 'smooth' });
  }

  function scrollChatBottom() {
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
  }

  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Transcript lines ──────────────────────────────────────────────────────
  function addFinalLine(text) {
    if (interimEl) { interimEl.remove(); interimEl = null; }
    emptyEl.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'line final';
    div.innerHTML = '<span class="ts">' + now() + '</span>' + escHtml(text);
    container.appendChild(div);
    lineCount++;
    countEl.textContent = lineCount + ' line' + (lineCount === 1 ? '' : 's');
    hasTranscript = true;
    btnStop.disabled = false;
    scrollBottom();
  }

  function showInterim(text) {
    if (!interimEl) {
      interimEl = document.createElement('div');
      interimEl.className = 'line interim';
      container.appendChild(interimEl);
    }
    interimEl.innerHTML = '<span class="ts">' + now() + '</span>' + escHtml(text);
    scrollBottom();
  }

  // ── LLM UI ────────────────────────────────────────────────────────────────
  function triggerLLM() {
    if (!hasTranscript || llmStreaming) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send('PROCESS');
  }

  function clearHistory() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send('CLEAR_HISTORY');
  }

  // ── Chat functions ────────────────────────────────────────────────────────
  function sendChat() {
    const msg = chatInput.value.trim();
    if (!msg || chatStreaming) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Hide empty state
    if (chatEmptyEl) chatEmptyEl.style.display = 'none';

    // Show user bubble in chat panel
    const userBubble = document.createElement('div');
    userBubble.className = 'chat-user';
    userBubble.innerHTML = '<div class="chat-label">You</div>' + escHtml(msg);
    chatContainer.appendChild(userBubble);
    scrollChatBottom();

    // Send to server
    ws.send('CHAT:' + msg);
    chatInput.value = '';
    chatInput.focus();
  }

  function startChatBlock() {
    chatStreaming = true;
    dotLlm.className = 'dot llm';
    btnChatSend.disabled = true;

    currentChatAiEl = document.createElement('div');
    currentChatAiEl.className = 'chat-ai';
    currentChatAiEl.innerHTML =
      '<div class="chat-label"><span class="pulse chat-pulse"></span> Sync AI</div>' +
      '<span class="chat-ai-text"></span>';
    chatContainer.appendChild(currentChatAiEl);
    scrollChatBottom();
  }

  function appendChatToken(token) {
    if (!currentChatAiEl) return;
    const textEl = currentChatAiEl.querySelector('.chat-ai-text');
    if (textEl) {
      textEl.textContent += token;
      scrollChatBottom();
    }
  }

  function finishChatBlock() {
    chatStreaming = false;
    dotLlm.className = 'dot ok';
    btnChatSend.disabled = false;

    if (currentChatAiEl) {
      const pulse = currentChatAiEl.querySelector('.chat-pulse');
      if (pulse) pulse.className = 'pulse done chat-pulse';
    }

    scrollChatBottom();
    chatInput.focus();
  }

  function showChatError(msg) {
    chatStreaming = false;
    dotLlm.className = 'dot err';
    btnChatSend.disabled = false;
    if (currentChatAiEl) {
      const textEl = currentChatAiEl.querySelector('.chat-ai-text');
      if (textEl) textEl.textContent = '⚠ ' + msg;
    }
  }



  function startLlmBlock() {
    llmStreaming = true;
    dotLlm.className = 'dot llm';
    btnStop.disabled = true;

    // Add a divider
    const div = document.createElement('div');
    div.className = 'divider';
    div.textContent = 'Sync AI';
    container.appendChild(div);

    // Create the LLM response block
    currentLlmEl = document.createElement('div');
    currentLlmEl.className = 'llm-block';
    currentLlmEl.innerHTML =
      '<div class="llm-label"><span class="pulse llm-pulse"></span> Sync AI is thinking...</div>' +
      '<span class="llm-text"></span>';
    container.appendChild(currentLlmEl);
    scrollBottom();
  }

  function appendLlmToken(token) {
    if (!currentLlmEl) return;
    const textEl = currentLlmEl.querySelector('.llm-text');
    if (textEl) {
      textEl.textContent += token;
      scrollBottom();
    }
  }

  function finishLlmBlock() {
    llmStreaming = false;
    dotLlm.className = 'dot ok';

    const pulse = currentLlmEl && currentLlmEl.querySelector('.llm-pulse');
    if (pulse) pulse.className = 'pulse done llm-pulse';

    const label = currentLlmEl && currentLlmEl.querySelector('.llm-label');
    if (label) label.innerHTML = '<span class="pulse done"></span> Sync AI';

    btnStop.disabled = !hasTranscript;
    scrollBottom();

    // Re-enable the button after a brief pause so user can ask follow-up
    setTimeout(() => {
      if (hasTranscript) btnStop.disabled = false;
    }, 500);
  }

  function showLlmError(msg) {
    llmStreaming = false;
    dotLlm.className = 'dot err';
    if (currentLlmEl) {
      const textEl = currentLlmEl.querySelector('.llm-text');
      if (textEl) textEl.textContent = '⚠ ' + msg;
      const label = currentLlmEl.querySelector('.llm-label');
      if (label) label.innerHTML = '<span class="pulse done" style="background:var(--danger)"></span> Error';
    }
    btnStop.disabled = false;
  }

  // ── Clear functions ───────────────────────────────────────────────────────
  function clearTranscript() {
    container.querySelectorAll('.line, .llm-block, .divider').forEach(el => el.remove());
    interimEl       = null;
    currentLlmEl    = null;
    lineCount       = 0;
    hasTranscript   = false;
    countEl.textContent = '0 lines';
    emptyEl.style.display = '';
    btnStop.disabled = true;
  }

  function clearChatPanel() {
    chatContainer.querySelectorAll('.chat-user, .chat-ai').forEach(el => el.remove());
    currentChatAiEl = null;
    if (chatEmptyEl) chatEmptyEl.style.display = '';
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(proto + '://' + location.host + '/ws');

    ws.onopen = () => {
      dotWs.className = 'dot ok';
    };

    ws.onclose = () => {
      dotWs.className = 'dot err';
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      dotWs.className = 'dot err';
    };

    ws.onmessage = (e) => {
      const msg = e.data;

      // ── Status updates ───────────────────────────────────────────────────
      if (msg.startsWith('__STATUS__:')) {
        const status = msg.split(':')[1];
        if (status === 'deepgram_connected')    dotDg.className  = 'dot ok';
        if (status === 'deepgram_disconnected') dotDg.className  = 'dot err';
        if (status === 'esp32_connected')       dotEsp.className = 'dot ok';
        if (status === 'esp32_disconnected')    dotEsp.className = 'dot err';
        if (status === 'history_cleared') {
          dotLlm.className = 'dot';
          clearChatPanel();
          console.log('[Chat] History cleared');
        }
        return;
      }

      // ── Transcript lines ─────────────────────────────────────────────────
      if (msg.startsWith('__FINAL__:')) {
        addFinalLine(msg.substring('__FINAL__:'.length));
        return;
      }
      if (msg.startsWith('__INTERIM__:')) {
        showInterim(msg.substring('__INTERIM__:'.length));
        return;
      }

      // ── LLM events ───────────────────────────────────────────────────────
      if (msg === '__LLM_START__') {
        startLlmBlock();
        return;
      }
      if (msg.startsWith('__LLM_TOKEN__:')) {
        appendLlmToken(msg.substring('__LLM_TOKEN__:'.length));
        return;
      }
      if (msg === '__LLM_DONE__') {
        finishLlmBlock();
        return;
      }
      if (msg.startsWith('__LLM_ERROR__:')) {
        showLlmError(msg.substring('__LLM_ERROR__:'.length));
        return;
      }

      // ── Chat events ───────────────────────────────────────────────────────
      if (msg === '__CHAT_START__') {
        startChatBlock();
        return;
      }
      if (msg.startsWith('__CHAT_TOKEN__:')) {
        appendChatToken(msg.substring('__CHAT_TOKEN__:'.length));
        return;
      }
      if (msg === '__CHAT_DONE__') {
        finishChatBlock();
        return;
      }
      if (msg.startsWith('__CHAT_ERROR__:')) {
        showChatError(msg.substring('__CHAT_ERROR__:'.length));
        return;
      }


      // ── Debug log ─────────────────────────────────────────────────────
      if (msg.startsWith('__DEBUG__:')) {
        const logEl = document.getElementById('debug-log');
        if (logEl) {
          const d = document.createElement('div');
          d.style.borderBottom = '1px solid #1a1d24';
          d.style.padding = '3px 0';
          d.style.wordBreak = 'break-all';
          d.textContent = new Date().toLocaleTimeString() + ' ' + msg.substring('__DEBUG__:'.length);
          logEl.appendChild(d);
          logEl.scrollTop = logEl.scrollHeight;
        }
        return;
      }
    };
  }

  connect();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)