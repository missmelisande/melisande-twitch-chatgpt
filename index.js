'use strict';

/**
 * Simple Twitch Chat Bot backend upgraded for GPT-5.
 * - Keeps the same GET /gpt/:text contract (expects "Username:Message")
 * - Supports two modes:
 *    - CHAT: multi-turn using messages[]
 *    - PROMPT: single-shot with file_context prefixed (emulates your old Q/A style)
 * - Trims replies to <= 1000 chars for Twitch
 * - Uses modern OpenAI Node SDK client with Chat Completions (minimal code change)
 * - Model can be overridden via OPENAI_MODEL env (defaults to 'gpt-5')
 */

const express = require('express');
const fs = require('fs');
const { promisify } = require('util');

// ---- Environment & config ---------------------------------------------------

const app = express();
const readFile = promisify(fs.readFile);

const GPT_MODE = process.env.GPT_MODE || 'CHAT';               // 'CHAT' or anything else => PROMPT
const HISTORY_LENGTH = Number(process.env.HISTORY_LENGTH || 6); // number of user/assistant exchange pairs to keep
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5';       // e.g., 'gpt-5', 'gpt-5-mini'

// Log basic diagnostics without leaking your API key
console.log('GPT_MODE:', GPT_MODE);
console.log('History length:', HISTORY_LENGTH);
const keyTail = (process.env.OPENAI_API_KEY || '').slice(-6);
console.log('OpenAI API Key present:', Boolean(process.env.OPENAI_API_KEY), keyTail ? `(ending with ...${keyTail})` : '');

// ---- OpenAI client (modern SDK) --------------------------------------------

/**
 * Using the current OpenAI Node SDK.
 * Docs (API Reference / migration toward Responses API):
 * https://platform.openai.com/docs/api-reference/introduction
 */
const OpenAI = require('openai');
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// ---- State & context --------------------------------------------------------

let file_context = 'You are a helpful Twitch Chatbot.';
const messages = [
  { role: 'system', content: 'You are a helpful Twitch Chatbot.' }
];

app.use(express.json({ extended: true, limit: '1mb' }));

// Basic health check
app.all('/', (_req, res) => {
  console.log('Just got a request!');
  res.send('Yo!');
});

// Load file_context.txt the same way you had it
if (GPT_MODE === 'CHAT') {
  fs.readFile('./file_context.txt', 'utf8', (err, data) => {
    if (err) {
      console.error('Error reading file_context.txt:', err);
      return;
    }
    console.log('Reading context file and adding it as system message for the agent.');
    messages[0].content = data;
  });
} else {
  fs.readFile('./file_context.txt', 'utf8', (err, data) => {
    if (err) {
      console.error('Error reading file_context.txt:', err);
      return;
    }
    console.log('Reading context file and adding it in front of user prompts:');
    file_context = data;
    console.log(file_context);
  });
}

// ---- Core route -------------------------------------------------------------

app.get('/gpt/:text', async (req, res) => {
  const text = req.params.text || '';

  try {
    if (GPT_MODE === 'CHAT') {
      // -------------------------- CHAT MODE ----------------------------------

      // Push the new user message
      messages.push({ role: 'user', content: text });

      // Keep a rolling window: 1 system + (HISTORY_LENGTH * 2) (user/assistant pairs)
      // Your original formula used ((HISTORY_LENGTH * 2) + 1)
      const maxLen = (HISTORY_LENGTH * 2) + 1;
      const convPairs = Math.max(0, Math.floor((messages.length - 1) / 2));
      console.log(`Conversations in History: ${convPairs}/${HISTORY_LENGTH}`);
      if (messages.length > maxLen) {
        console.log('Message amount in history exceeded. Removing oldest user and agent messages.');
        messages.splice(1, 2); // remove first user+assistant pair after system
      }

      console.log('Messages:', JSON.stringify(messages, null, 2));
      console.log('User Input:', text);

      // Minimal change: use modern client + chat.completions with GPT-5
      // (If your org lacks access, set OPENAI_MODEL=gpt-5-mini)
      const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages,
        temperature: 0.5,
        max_tokens: 592,   // keep your old ceiling
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0
      });

      let agent_response = response?.choices?.[0]?.message?.content ?? '';
      agent_response = String(agent_response).trim();

      console.log('Agent answer:', agent_response);
      messages.push({ role: 'assistant', content: agent_response });

      // Twitch hard cap (keep your 1000-char slice)
      if (agent_response.length > 1000) {
        console.log('Agent answer exceeds Twitch chat limit. Slicing to first 1000 characters.');
        agent_response = agent_response.substring(0, 1000);
        console.log('Sliced agent answer:', agent_response);
      }

      res.status(200).send(agent_response);

    } else {
      // -------------------------- PROMPT MODE --------------------------------

      // Emulate your previous “Q: … A:” style by stuffing into a single user turn.
      // (We still call chat.completions so you don’t rely on deprecated completions.)
      const promptAsSingleMessage =
        `${file_context}\n\nQ: ${text}\nA:`;

      console.log('User Input:', text);

      const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [
          { role: 'system', content: 'You are a helpful Twitch Chatbot.' },
          { role: 'user', content: promptAsSingleMessage }
        ],
        temperature: 0.5,
        max_tokens: 256,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0
      });

      let agent_response = response?.choices?.[0]?.message?.content ?? '';
      agent_response = String(agent_response).trim();

      console.log('Agent answer:', agent_response);

      if (agent_response.length > 1000) {
        console.log('Agent answer exceeds Twitch chat limit. Slicing to first 1000 characters.');
        agent_response = agent_response.substring(0, 1000);
        console.log('Sliced agent answer:', agent_response);
      }

      res.status(200).send(agent_response);
    }

  } catch (err) {
    // Helpful diagnostics if the model isn’t available or token/account issues occur
    console.error('OpenAI error:', err?.status, err?.message);
    const detail = (err?.response?.data?.error?.message) || err?.message || 'Unknown error';
    res
      .status(500)
      .send(`Something went wrong with the AI call: ${detail}`);
  }
});

// ---- Start server -----------------------------------------------------------

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Bot server listening on port ${PORT}`);
});
