// ds-mcp-server web UI client

const $ = (sel) => document.querySelector(sel);

const els = {
  messages: $('#messages'),
  input: $('#input'),
  sendBtn: $('#send-btn'),
  composer: $('#composer'),
  providerPill: $('#provider-pill'),
  modelPill: $('#model-pill'),
  statusPill: $('#status-pill'),
  toolList: $('#tool-list'),
  toolCount: $('#tool-count'),
  toolSearch: $('#tool-search'),
  resetBtn: $('#reset-btn'),
  sidebarToggle: $('#sidebar-toggle'),
  // Settings modal
  settingsBtn: $('#settings-btn'),
  settingsModal: $('#settings-modal'),
  settingsClose: $('#settings-close'),
  settingsCancel: $('#settings-cancel'),
  settingsApply: $('#settings-apply'),
  settingsNote: $('#settings-note'),
  toggleSystemTools: $('#toggle-system-tools'),
  toggleUnrestrictedExec: $('#toggle-unrestricted-exec'),
};

const state = {
  ws: null,
  ready: false,
  busy: false,
  tools: [],
  currentBotMsg: null, // {body, chipsRow, textEl}
};

// ---------- helpers ----------
function setStatus(kind, label) {
  els.statusPill.className = 'pill pill--status status-' + kind;
  els.statusPill.innerHTML = '<span class="dot"></span>' + label;
}

function autoResizeTextarea() {
  els.input.style.height = 'auto';
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + 'px';
  els.sendBtn.disabled = !els.input.value.trim() || state.busy || !state.ready;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.messages.scrollTop = els.messages.scrollHeight;
  });
}

function dismissWelcome() {
  const w = els.messages.querySelector('.welcome');
  if (w) w.remove();
}

// Minimal safe markdown-ish renderer: escape HTML, then apply code / bold /
// italic / inline-code / newlines. Deliberately tiny — no external deps.
function renderText(raw) {
  const escaped = raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  let html = escaped;
  // triple-backtick code blocks first
  html = html.replace(/```([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${code.replace(/^\n+|\n+$/g, '')}</code></pre>`);
  // inline code
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  // bold **text**
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // italic *text*
  html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  // paragraphs / line breaks
  html = html.split(/\n{2,}/).map(chunk => `<p>${chunk.replace(/\n/g, '<br/>')}</p>`).join('');
  return html;
}

function addMessage(role, opts = {}) {
  dismissWelcome();
  const meta = opts.meta || {
    user: 'you',
    bot:  'assistant',
    tool: 'tool',
    error: 'error',
  }[role];
  const avatarText = {
    user: 'YOU',
    bot: 'AI',
    tool: '⚙',
    error: '!',
  }[role];

  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;
  msg.innerHTML = `
    <div class="msg-avatar">${avatarText}</div>
    <div class="msg-body">
      <div class="msg-meta">${meta}</div>
      <div class="msg-content"></div>
    </div>
  `;
  const body = msg.querySelector('.msg-body');
  const content = msg.querySelector('.msg-content');
  if (opts.text) content.innerHTML = renderText(opts.text);
  els.messages.appendChild(msg);
  scrollToBottom();
  return { msg, body, content };
}

function ensureBotMessage() {
  if (!state.currentBotMsg) {
    const parts = addMessage('bot');
    parts.chipsRow = document.createElement('div');
    parts.chipsRow.className = 'tool-chips';
    parts.content.parentNode.insertBefore(parts.chipsRow, parts.content);
    state.currentBotMsg = parts;
  }
  return state.currentBotMsg;
}

function addToolChip(name) {
  const bot = ensureBotMessage();
  const chip = document.createElement('span');
  chip.className = 'tool-chip';
  chip.textContent = name;
  bot.chipsRow.appendChild(chip);
  scrollToBottom();
}

function addPlot(plot) {
  const bot = ensureBotMessage();
  const wrap = document.createElement('div');
  wrap.className = 'plot-frame';
  const url = `/api/plot?path=${encodeURIComponent(plot.path)}`;
  if (plot.kind === 'html') {
    wrap.innerHTML = `
      <iframe src="${url}" loading="lazy" title="plot"></iframe>
      <div class="plot-caption">
        <span>${plot.path.split(/[\\/]/).pop()}</span>
        <a href="${url}" target="_blank" rel="noopener">open ↗</a>
      </div>`;
  } else {
    wrap.innerHTML = `
      <img src="${url}" alt="plot" loading="lazy" />
      <div class="plot-caption">
        <span>${plot.path.split(/[\\/]/).pop()}</span>
        <a href="${url}" target="_blank" rel="noopener">open ↗</a>
      </div>`;
  }
  bot.body.appendChild(wrap);
  scrollToBottom();
}

function addAssistantText(text) {
  const bot = ensureBotMessage();
  bot.content.innerHTML = renderText(text);
  scrollToBottom();
}

// ---------- WebSocket ----------
function connect() {
  setStatus('busy', 'connecting');
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    switch (data.type) {
      case 'ready':
        state.ready = true;
        setStatus('ready', 'ready');
        autoResizeTextarea();
        break;
      case 'tool_call':
        addToolChip(data.name);
        break;
      case 'tool_result':
        if (data.plot) addPlot(data.plot);
        break;
      case 'text':
        addAssistantText(data.text);
        break;
      case 'done':
        state.busy = false;
        state.currentBotMsg = null;
        setStatus('ready', 'ready');
        autoResizeTextarea();
        els.input.focus();
        break;
      case 'error':
        addMessage('error', { text: data.message });
        state.busy = false;
        state.currentBotMsg = null;
        setStatus('error', 'error');
        autoResizeTextarea();
        break;
    }
  };

  ws.onclose = () => {
    state.ready = false;
    state.busy = false;
    setStatus('error', 'disconnected');
    autoResizeTextarea();
    setTimeout(connect, 2500);
  };

  ws.onerror = () => setStatus('error', 'connection error');
}

// ---------- Config / tools ----------
async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    els.providerPill.textContent = `provider · ${cfg.provider}`;
    els.modelPill.textContent = `model · ${cfg.model}`;
    state.tools = cfg.tools || [];
    renderTools(state.tools);
  } catch (e) {
    els.providerPill.textContent = 'provider · ?';
    els.modelPill.textContent = 'model · ?';
  }
}

function renderTools(tools) {
  els.toolCount.textContent = tools.length;
  els.toolList.innerHTML = '';
  tools.forEach(t => {
    const li = document.createElement('li');
    li.className = 'tool-item';
    li.textContent = t.name;
    li.title = t.description || t.name;
    els.toolList.appendChild(li);
  });
}

// ---------- Send ----------
function sendMessage() {
  const text = els.input.value.trim();
  if (!text || state.busy || !state.ready) return;

  addMessage('user', { text });
  els.input.value = '';
  autoResizeTextarea();
  state.busy = true;
  setStatus('busy', 'thinking');
  ensureBotMessage();
  state.currentBotMsg.content.innerHTML =
    '<div class="typing"><span></span><span></span><span></span></div>';

  state.ws.send(JSON.stringify({ message: text }));
}

// ---------- Events ----------
els.input.addEventListener('input', autoResizeTextarea);
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
els.composer.addEventListener('submit', (e) => { e.preventDefault(); sendMessage(); });

els.toolSearch.addEventListener('input', () => {
  const q = els.toolSearch.value.toLowerCase();
  const filtered = state.tools.filter(t => t.name.toLowerCase().includes(q));
  renderTools(filtered);
});

document.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (chip && chip.dataset.prompt) {
    els.input.value = chip.dataset.prompt;
    autoResizeTextarea();
    els.input.focus();
  }
});

els.resetBtn.addEventListener('click', () => {
  els.messages.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">
        <svg viewBox="0 0 24 24" width="42" height="42" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 3v18h18"></path>
          <polyline points="7 15 11 11 14 14 20 8"></polyline>
        </svg>
      </div>
      <h2>New conversation</h2>
      <p>Note: the server still keeps this session's conversation memory. To
      fully reset history, restart the server.</p>
    </div>`;
  state.currentBotMsg = null;
});

els.sidebarToggle.addEventListener('click', () => {
  document.body.classList.toggle('sidebar-collapsed');
});

// ---------- Settings modal ----------
const settings = { current: { system_tools: false, unrestricted_exec: false } };

function setNote(kind, text) {
  if (!text) { els.settingsNote.hidden = true; els.settingsNote.textContent = ''; return; }
  els.settingsNote.hidden = false;
  els.settingsNote.className = 'setting-note ' + (kind || '');
  els.settingsNote.textContent = text;
}

async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    const data = await r.json();
    settings.current = data.settings || settings.current;
    els.toggleSystemTools.checked = !!settings.current.system_tools;
    els.toggleUnrestrictedExec.checked = !!settings.current.unrestricted_exec;
  } catch (e) { /* ignore */ }
}

function openSettings() {
  loadSettings();
  setNote('', '');
  els.settingsModal.hidden = false;
}
function closeSettings() { els.settingsModal.hidden = true; }

async function applySettings() {
  const proposed = {
    system_tools: els.toggleSystemTools.checked,
    unrestricted_exec: els.toggleUnrestrictedExec.checked,
  };
  const cur = settings.current;
  const changed = proposed.system_tools !== cur.system_tools ||
                  proposed.unrestricted_exec !== cur.unrestricted_exec;
  if (!changed) { closeSettings(); return; }

  els.settingsApply.disabled = true;
  els.toggleSystemTools.disabled = true;
  els.toggleUnrestrictedExec.disabled = true;
  setNote('busy', 'Restarting MCP server with new settings…');

  try {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: proposed }),
    });
    const data = await r.json();
    if (!r.ok) {
      setNote('error', data.error || 'Failed to update settings.');
      return;
    }
    settings.current = data.settings;
    if (Array.isArray(data.tools)) {
      state.tools = data.tools;
      renderTools(state.tools);
    }
    const enabled = [];
    if (data.settings.system_tools) enabled.push('system tools');
    if (data.settings.unrestricted_exec) enabled.push('unrestricted exec');
    setNote(
      'success',
      enabled.length
        ? 'Applied. Now enabled: ' + enabled.join(', ') + '.'
        : 'Applied. Sandbox is fully on.',
    );
    setTimeout(closeSettings, 900);
  } catch (e) {
    setNote('error', 'Network error: ' + e.message);
  } finally {
    els.settingsApply.disabled = false;
    els.toggleSystemTools.disabled = false;
    els.toggleUnrestrictedExec.disabled = false;
  }
}

els.settingsBtn.addEventListener('click', openSettings);
els.settingsClose.addEventListener('click', closeSettings);
els.settingsCancel.addEventListener('click', closeSettings);
els.settingsApply.addEventListener('click', applySettings);
els.settingsModal.addEventListener('click', (e) => {
  if (e.target === els.settingsModal) closeSettings();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !els.settingsModal.hidden) closeSettings();
});

// ---------- Boot ----------
loadConfig();
loadSettings();
connect();
autoResizeTextarea();
