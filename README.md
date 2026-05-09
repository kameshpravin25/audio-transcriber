# 🎙️ ESP32 Real-Time Audio Transcriber

A real-time speech-to-text system that captures audio from an **ESP32 + INMP441** microphone, streams it to a cloud-hosted **FastAPI** server, transcribes it via **Deepgram Nova-3**, and generates conversational AI responses using **Google Gemini 2.5 Flash** — all viewable in a sleek live web dashboard.

---

## ✨ Features

- **Real-time transcription** — Live interim + final transcripts powered by Deepgram Nova-3
- **AI conversation** — Press **Q** or click *Ask Gemini* to get a conversational response from Gemini 2.5 Flash with multi-turn context
- **WiFi captive portal** — No hardcoded credentials; configure WiFi via phone/laptop on first boot
- **One-click deploy** — Server deploys to [Railway](https://railway.app) with zero configuration
- **Live status dashboard** — Monitor WebSocket, Deepgram, ESP32, and Gemini connection states in real time
- **Debug panel** — Built-in debug log overlay for troubleshooting

---

## 🏗️ Architecture

```
┌─────────────┐        WSS (binary audio)        ┌──────────────────┐       WSS        ┌───────────┐
│   ESP32 +   │ ──────────────────────────────▶   │  FastAPI Server  │ ──────────────▶  │ Deepgram  │
│   INMP441   │                                   │  (Railway)       │ ◀──────────────  │ Nova-3    │
└─────────────┘                                   │                  │   transcripts    └───────────┘
                                                  │                  │
                                                  │                  │       REST        ┌───────────┐
┌─────────────┐        WSS (text events)          │                  │ ──────────────▶  │  Gemini   │
│   Browser   │ ◀──────────────────────────────   │                  │ ◀──────────────  │  2.5 Flash│
│   Dashboard │ ──────────────────────────────▶   └──────────────────┘   AI response    └───────────┘
└─────────────┘        "PROCESS" / "CLEAR"
```

---

## 📁 Project Structure

```
audio_transcriber/
├── server.py                        # FastAPI server (transcription + LLM + web UI)
├── esp32_firmware/
│   └── esp32_firmware.ino           # Arduino sketch for ESP32 + INMP441
├── requirements.txt                 # Python dependencies
├── Procfile                         # Railway process definition
├── runtime.txt                      # Python version for Railway
├── .gitignore                       # Ignored files (venv, cache, etc.)
└── README.md                        # This file
```

---

## 🔧 Hardware Requirements

| Component | Description |
|---|---|
| **ESP32 Dev Board** | Any ESP32-WROOM-32 based board |
| **INMP441** | I2S MEMS microphone module |
| **Jumper wires** | 5 connections (see wiring below) |
| **USB cable** | For flashing and serial monitor |

### Wiring Diagram

| ESP32 Pin | INMP441 Pin | Function |
|---|---|---|
| `GPIO 26` | `SCK` | Bit Clock |
| `GPIO 32` | `WS` | Word Select (LRCLK) |
| `GPIO 33` | `SD` | Serial Data Out |
| `3.3V` | `VDD` | Power |
| `GND` | `GND` + `L/R` | Ground (L/R → GND for left channel) |

---

## 🚀 Getting Started

### 1. Deploy the Server to Railway

1. Push this repository to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub Repo**
3. Railway auto-detects the `Procfile` and `runtime.txt` — no extra config needed
4. Note your deployed URL (e.g. `audio-transcriber.up.railway.app`)

### 2. Flash the ESP32 Firmware

1. Open `esp32_firmware/esp32_firmware.ino` in **Arduino IDE**
2. Install the required libraries via **Library Manager**:
   - `WiFiManager` by tzapu (≥ 2.0)
   - `WebSockets` by Markus Sattler (≥ 2.4)
3. Update `SERVER_HOST` in the sketch if your Railway URL differs:
   ```cpp
   const char* SERVER_HOST = "your-app-name.up.railway.app";
   ```
4. Select board: **ESP32 Dev Module**
5. Flash and open the Serial Monitor at **115200 baud**

### 3. Connect to WiFi

1. On first boot, the ESP32 creates a WiFi AP named **`Transcriber-Setup`**
2. Connect to it from your phone or laptop
3. Select your home WiFi network and enter the password
4. The ESP32 saves the credentials and auto-connects on subsequent boots

> 💡 **Reset WiFi:** Hold the **BOOT** button (GPIO 0) while powering on to clear saved credentials.

### 4. View the Dashboard

Open your Railway URL in a browser:
```
https://audio-transcriber.up.railway.app
```

You'll see live transcription appear as the ESP32 streams audio.

---

## 🧠 Using Gemini AI

Once transcription lines appear:

1. Press **Q** on your keyboard or click the **⬡ Ask Gemini** button
2. Gemini reads the buffered transcript and responds in a casual, conversational tone
3. The conversation supports **multi-turn context** — keep talking and asking
4. Click **Reset Chat** to clear the conversation history

---

## ⚙️ Configuration

### Server Environment

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP port (auto-set by Railway) |

### API Keys

API keys are currently embedded in `server.py`. For production, move them to environment variables:

```python
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GOOGLE_API_KEY   = os.environ.get("GOOGLE_API_KEY")
```

### ESP32 Firmware

| Constant | Default | Description |
|---|---|---|
| `SERVER_HOST` | `audio-transcriber.up.railway.app` | Railway domain |
| `SERVER_PORT` | `443` | HTTPS/WSS port |
| `WS_PATH` | `/ws/audio` | WebSocket endpoint |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `I2S_READ_LEN` | `256` | Samples per I2S read cycle |

---

## 🛠️ Local Development

```bash
# Clone the repo
git clone https://github.com/<your-username>/audio_transcriber.git
cd audio_transcriber

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the server locally
python server.py
# → Server starts at http://localhost:8000
```

> Update `SERVER_HOST` in the ESP32 sketch to your local IP for local testing.

---

## 📦 Dependencies

### Python (Server)

| Package | Purpose |
|---|---|
| `fastapi` | Async web framework & WebSocket handling |
| `uvicorn` | ASGI server |
| `websockets` | Async WebSocket client for Deepgram |
| `langchain-google-genai` | LangChain integration for Gemini |

### Arduino (ESP32)

| Library | Purpose |
|---|---|
| `WiFiManager` | Captive portal for WiFi provisioning |
| `WebSockets` | WebSocket client with SSL support |

---

## 📝 API Endpoints

| Endpoint | Type | Description |
|---|---|---|
| `GET /` | HTTP | Serves the live transcription dashboard |
| `WS /ws` | WebSocket | Browser ↔ Server (transcripts + LLM events) |
| `WS /ws/audio` | WebSocket | ESP32 → Server (raw 16-bit PCM audio) |

### Browser WebSocket Messages

| Direction | Message | Description |
|---|---|---|
| Browser → Server | `PROCESS` | Trigger Gemini on buffered transcript |
| Browser → Server | `CLEAR_HISTORY` | Reset conversation history |
| Server → Browser | `__FINAL__:<text>` | Final transcript line |
| Server → Browser | `__INTERIM__:<text>` | Interim (partial) transcript |
| Server → Browser | `__LLM_START__` | Gemini processing started |
| Server → Browser | `__LLM_TOKEN__:<text>` | Gemini response text |
| Server → Browser | `__LLM_DONE__` | Gemini processing complete |
| Server → Browser | `__LLM_ERROR__:<msg>` | Gemini error message |
| Server → Browser | `__STATUS__:<state>` | Connection state change |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using ESP32 · FastAPI · Deepgram · Gemini
</p>
