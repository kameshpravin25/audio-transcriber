/*
 * ESP32 Real-Time Audio Transcriber Firmware
 * 
 * Hardware: ESP32 + INMP441 MEMS Microphone
 * 
 * Captures 16-bit 16 kHz mono audio from an INMP441 microphone via I2S
 * and streams it over a secure WebSocket connection to a FastAPI server
 * hosted on Railway for real-time transcription via Deepgram.
 *
 * WiFi credentials are managed via WiFiManager (captive portal).
 * Hold GPIO 0 (BOOT) low during startup to reset saved WiFi settings.
 *
 * Wiring (ESP32 → INMP441):
 *   GPIO 26  → SCK  (Bit Clock)
 *   GPIO 32  → WS   (Word Select / LRCLK)
 *   GPIO 33  → SD   (Serial Data)
 *   3.3 V    → VDD
 *   GND      → GND  &  L/R (left channel = GND)
 *
 * Dependencies (install via Arduino Library Manager):
 *   - WiFiManager  by tzapu  (>= 2.0)
 *   - WebSockets   by Markus Sattler (>= 2.4)
 *
 * Board: ESP32 Dev Module  |  Partition: Default 4 MB
 */

#include <WiFiManager.h>
#include <WebSocketsClient.h>
#include "driver/i2s.h"

// ─── Server Configuration ───────────────────────────────────────────────────
const char* SERVER_HOST = "audio-transcriber.up.railway.app";
const uint16_t SERVER_PORT = 443;
const char* WS_PATH = "/ws/audio";

// ─── Pin Definitions ────────────────────────────────────────────────────────
#define WIFI_RESET_PIN 0     // BOOT button – hold LOW on startup to reset WiFi
#define STATUS_LED     2     // On-board LED – HIGH = WebSocket connected

// ─── Audio Configuration ────────────────────────────────────────────────────
#define SAMPLE_RATE    16000
#define I2S_READ_LEN   256   // Samples per I2S read cycle

// ─── Globals ────────────────────────────────────────────────────────────────
WebSocketsClient webSocket;
bool wsConnected = false;

// ─── I2S Configuration (INMP441 in master-receive mode) ─────────────────────
i2s_config_t i2s_config = {
  .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
  .sample_rate = SAMPLE_RATE,
  .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
  .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
  .communication_format = I2S_COMM_FORMAT_STAND_I2S,
  .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
  .dma_buf_count = 8, .dma_buf_len = 256,
  .use_apll = false, .tx_desc_auto_clear = false, .fixed_mclk = 0
};

i2s_pin_config_t pin_config = {
  .bck_io_num = 26, .ws_io_num = 32,
  .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = 33
};

// ─── WebSocket Event Handler ────────────────────────────────────────────────
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      digitalWrite(STATUS_LED, HIGH);
      Serial.println("[WS] Connected!");
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      digitalWrite(STATUS_LED, LOW);
      Serial.println("[WS] Disconnected");
      break;
    case WStype_TEXT:
      Serial.printf("[WS] Server: %s\n", payload);
      break;
    default:
      break;
  }
}

// ─── Setup ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(WIFI_RESET_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED, OUTPUT);

  Serial.println("\n=== ESP32 Transcriber v2.0 ===");

  // ── WiFi (captive portal) ──
  WiFiManager wm;
  if (digitalRead(WIFI_RESET_PIN) == LOW) {
    wm.resetSettings();
    Serial.println("[WiFi] Reset!");
  }
  wm.setConfigPortalTimeout(180);
  if (!wm.autoConnect("Transcriber-Setup")) { ESP.restart(); }

  IPAddress dns(8, 8, 8, 8);
  WiFi.config(WiFi.localIP(), WiFi.gatewayIP(), WiFi.subnetMask(), dns);
  Serial.println("[WiFi] Connected: " + WiFi.localIP().toString());

  // ── I2S (INMP441 microphone) ──
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_zero_dma_buffer(I2S_NUM_0);

  // Flush initial garbage samples
  int32_t dummy[64];
  size_t br;
  for (int i = 0; i < 5; i++) {
    i2s_read(I2S_NUM_0, dummy, sizeof(dummy), &br, pdMS_TO_TICKS(10));
  }
  Serial.println("[Mic] INMP441 ready");

  // ── WebSocket (SSL) ──
  webSocket.beginSSL(SERVER_HOST, SERVER_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  webSocket.enableHeartbeat(15000, 10000, 2);
  Serial.println("[WS] Connecting...");
}

// ─── Main Loop ──────────────────────────────────────────────────────────────
void loop() {
  webSocket.loop();
  if (!wsConnected) return;

  static int32_t raw32[I2S_READ_LEN];
  static int16_t buffer16[I2S_READ_LEN];
  size_t bytesRead;

  i2s_read(I2S_NUM_0, raw32, sizeof(raw32), &bytesRead, pdMS_TO_TICKS(100));

  if (bytesRead > 0) {
    int samples = bytesRead / 4;
    for (int i = 0; i < samples; i++) {
      buffer16[i] = (int16_t)(raw32[i] >> 14);
    }
    webSocket.sendBIN((uint8_t*)buffer16, samples * 2);
  }
}
