'use strict';

/**
 * Simple Twitch Chat Bot backend upgraded for GPT-5 with robust error handling.
 *
 * What's new vs the previous version:
 *  - Retries on transient errors (429/500/502/503/504, connect resets, timeouts).
 *  - Per-request timeout so your proxy doesn't 502 while waiting forever.
 *  - Automatic fallback model if primary model is briefly unavailable.
 *  - Clear diagnostics so you can tell if the 502 is from your host or OpenAI.
 *  - /healthz endpoint and stricter Express timeouts.
 *
 * Env vars you can set:
 *   OPENAI_API_KEY      = sk-... (required)
 *   OPENAI_MODEL        = gpt-5            (default)
 *   FALLBACK_MODEL      = gpt-5-mini       (default)
 *   GPT_MODE            = CHAT             (default; anything else => PROMPT)
 *   HISTORY_LENGTH      = 6                (CHAT mode memory pairs)
 *   OPENAI_RETRIES      = 3                (# of retry attempts on retriable errors)
 *   OPENAI_TIMEOUT_MS   = 20000            (timeout for the OpenAI call)
 *   SERVER_TIMEOUT_MS   = 25000            (timeout for Express response)
 *   PORT                = 3000
 */

const express = require('express');
const fs = require('fs');
const { promisify } = require('util');

const readFile = promisify(fs.readFile);
const app = express();

// ---- Environment & config ---------------------------------------------------

const GPT_MODE = process.env.GPT_MODE || 'CHAT';               // 'CHAT' or anything else => PROMPT
const HISTORY_LENGTH = Number(process.env.HISTORY_LENGTH || 6);
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5';
const FALLBACK_MODEL = process.env.FALLBACK_MODEL || 'gpt-5-mini';
const OPENAI_RETRIES = Number(process.env.OPENAI_RETRIES || 3);
const OPENAI_TIMEOUT_MS = Number(process.env.OPENAI_TIMEOUT_MS || 20000);
const SERVER_TIMEOUT_MS = Number(process.env.SERVER_TIMEOUT_MS || 25000);

app.set('trust proxy', true); // helps when behind proxies
app.use(express.json({ extended: true, limit: '1mb' }));

// Express-level response timeout so the proxy doesn’t 502 on long waits
app.use((req, res, next) => {
  res.setTimeout(SERVER_TIMEOUT_MS, () => {
    console.error(`[Timeout] ${req.method} ${req.originalUrl} exceeded ${SERVER_TIMEOUT_MS}ms`);
    if (!res.headersSent) res.status(504).send('Gateway Timeout: server took too long to respond.');
  });
  next();
});

// Basic diagnostics without leaking your API key
const keyTail = (process.env.OPENAI_API_KEY || '').slice(-6);
console.log('[Boot] GPT_MODE:', GPT_MODE);
console.log('[Boot] History length:', HISTORY_LENGTH);
console.log('[Boot] Model:', OPENAI_MODEL, 'Fallback:', FALLBACK_MODEL);
console.log('[Boot] Retries:', OPENAI_RETRIES, 'OpenAI timeout (ms):', OPENAI_TIMEOUT_MS, 'Server timeout (ms):', SERVER_TIMEOUT_MS);
console.log('[Boot] OpenAI API Key present:', Boolean(process.env.OPENAI_API_KEY), keyTail ? `(ending with ...${keyTail})` : '');

// ---- OpenAI client (modern SDK) --------------------------------------------

const OpenAI = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---- Helpers: retry/backoff + timeout wrappers -----------------------------

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function withRetry(fn, { retries = 3, baseDelay = 400, factor = 2, onRetry = () => {} } = {}) {
  let attempt = 0;
  // We allow <= retries attempts after the first (i.e., total attempts = retries + 1)
  for (;;) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      const status = err?.status || err?.response?.status;
      const code = err?.code;
      const retriableStatuses = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
      const retriableCodes = new Set([
        'ECONNRESET', 'EPIPE', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT',
        'UND_ERR_BODY_TIMEOUT', 'ETIMEDOUT', 'ESOCKETTIMEDOUT'
      ]);
      const retriable = retriableStatuses.has(status) || retriableCodes.has(code);

      if (!retriable || attempt > retries) {
        throw err;
      }

      const delay = Math.round(baseDelay * Math.pow(factor, attempt - 1) + Math.random() * 200);
      onRetry({ attempt, status, code, delay, message: err?.message });
      await sleep(delay);
    }
  }
}

async function withTimeout(promise, ms) {
  let timer;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Client timeout after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timer);
  }
}

// ---- State & context --------------------------------------------------------

let file_context = 'You are a helpful Twitch Chatbot.';
const messages = [{ role: 'system', content: 'You are a helpful Twitch Chatbot.' }];

async function loadFileContextForChatMode() {
  try {
    const data = await readFile('./file_context.txt', 'utf8');
    console.log('[Context] Loaded system context for CHAT mode.');
    messages[0].content = data;
  } catch (err) {
    console.error('[Context] Error reading file_context.txt for CHAT mode:', err);
  }
}

async function loadFileContextForPromptMode() {
  try {
    const data = await readFile('./file_context.txt', 'utf8');
    console.log('[Context] Loaded prefix context for PROMPT mode:\n', data);
    file_context = data;
  } catch (err) {
    console.error('[Context] Error reading file_context.txt for PROMPT mode:', err);
  }
}

if (GPT_MODE === 'CHAT') {
  loadFileContextForChatMode();
} else {
  loadFileContextForPromptMode();
}

// ---- Health & root ----------------------------------------------------------

app.all('/', (_req, res) => res.send('Yo!'));
app.get('/healthz', (_req, res) => res.status(200).send('ok'));

// ---- OpenAI invocation (shared) --------------------------------------------

async function callOpenAIChat({ model, msgArray, temperature, maxTokens }) {
  return await withTimeout(
    withRetry(
      () =>
        openai.chat.completions.create({
          model,
          messages: msgArray,
          temperature,
          max_tokens: maxTokens,
          top_p: 1,
          frequency_penalty: 0,
          presence_penalty: 0
        }),
      {
        retries: OPENAI_RETRIES,
        baseDelay: 400,
        factor: 2,
        onRetry: ({ attempt, status, code, delay, message }) => {
          console.warn(
            `[Retry] attempt=${attempt} in ${delay}ms; status=${status || 'n/a'} code=${code || 'n/a'} msg=${message}`
          );
        }
      }
    ),
    OPENAI_TIMEOUT_MS
  );
}

function trimForTwitch(text) {
  if (!text) return '';
  if (text.length <= 1000) return text;
  console.log('[Trim] Cutting response to first 1000 characters for Twitch.');
  return text.substring(0, 1000);
}

// ---- Core route -------------------------------------------------------------

app.get('/gpt/:text', async (req, res) => {
  const text = req.params.text || '';
  const primaryModel = OPENAI_MODEL;
  const fallbackModel = FALLBACK_MODEL;

  const start = Date.now();
  console.log(`[Req] /gpt called with text="${text}" mode=${GPT_MODE}`);

  try {
    let agent_response = '';

    if (GPT_MODE === 'CHAT') {
      // --- CHAT MODE ---
      messages.push({ role: 'user', content: text });

      const maxLen = (HISTORY_LENGTH * 2) + 1; // system + N pairs
      const convPairs = Math.max(0, Math.floor((messages.length - 1) / 2));
      console.log(`[Chat] pairs=${convPairs}/${HISTORY_LENGTH} totalMsgs=${messages.length}`);
      if (messages.length > maxLen) {
        console.log('[Chat] History exceeded; dropping oldest user+assistant pair.');
        messages.splice(1, 2);
      }

      // Try primary model
      let response;
      try {
        response = await callOpenAIChat({
          model: primaryModel,
          msgArray: messages,
          temperature: 0.5,
          maxTokens: 592
        });
      } catch (err) {
        const status = err?.status || err?.response?.status;
        console.error(`[OpenAI][Primary:${primaryModel}] Failed with status=${status} msg=${err?.message}`);
        // Fallback to a smaller model if configured
        if (fallbackModel && fallbackModel !== primaryModel) {
          console.warn(`[OpenAI] Falling back to ${fallbackModel}`);
          response = await callOpenAIChat({
            model: fallbackModel,
            msgArray: messages,
            temperature: 0.5,
            maxTokens: 592
          });
        } else {
          throw err;
        }
      }

      agent_response = String(response?.choices?.[0]?.message?.content || '').trim();
      console.log('[OpenAI] Received response.');
      messages.push({ role: 'assistant', content: agent_response });

    } else {
      // --- PROMPT MODE ---
      const promptAsSingleMessage = `${file_context}\n\nQ: ${text}\nA:`;
      let response;
      try {
        response = await callOpenAIChat({
          model: primaryModel,
          msgArray: [
            { role: 'system', content: 'You are a helpful Twitch Chatbot.' },
            { role: 'user', content: promptAsSingleMessage }
          ],
          temperature: 0.5,
          maxTokens: 256
        });
      } catch (err) {
        const status = err?.status || err?.response?.status;
        console.error(`[OpenAI][Primary:${primaryModel}] Failed with status=${status} msg=${err?.message}`);
        if (fallbackModel && fallbackModel !== primaryModel) {
          console.warn(`[OpenAI] Falling back to ${fallbackModel}`);
          response = await callOpenAIChat({
            model: fallbackModel,
            msgArray: [
              { role: 'system', content: 'You are a helpful Twitch Chatbot.' },
              { role: 'user', content: promptAsSingleMessage }
            ],
            temperature: 0.5,
            maxTokens: 256
          });
        } else {
          throw err;
        }
      }

      agent_response = String(response?.choices?.[0]?.message?.content || '').trim();
      console.log('[OpenAI] Received response.');
    }

    const out = trimForTwitch(agent_response);
    const ms = Date.now() - start;
    console.log(`[OK] Responding (${ms}ms). Size=${out.length}`);
    res.set('Cache-Control', 'no-store');
    res.status(200).send(out);

  } catch (err) {
    const status = err?.status || err?.response?.status;
    const detail =
      (err?.response?.data && JSON.stringify(err.response.data)) ||
      err?.message ||
      'Unknown error';

    console.error('[Fail] OpenAI call failed:', { status, detail });

    // Use 503 on upstream failures so proxies don’t mislabel as 502
    const httpStatus = [429, 500, 502, 503, 504].includes(status) ? 503 : 500;
    if (!res.headersSent) {
      res
        .status(httpStatus)
        .send(
          `AI backend temporarily unavailable (status=${status || httpStatus}). Details: ${detail}`
        );
    }
  }
});

// ---- Start server -----------------------------------------------------------

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  console.log(`[Boot] Bot server listening on port ${PORT}`);
});

// Make Node’s own server timeouts explicit (helps some hosts)
server.headersTimeout = Math.max(SERVER_TIMEOUT_MS + 5000, 30000); // time to receive all headers
server.requestTimeout = Math.max(SERVER_TIMEOUT_MS + 5000, 30000); // total request lifetime
