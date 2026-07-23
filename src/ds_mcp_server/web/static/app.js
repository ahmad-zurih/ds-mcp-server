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
  // Multi-agent
  toggleMultiAgent: $('#toggle-multi-agent'),
  toggleMultiAgentModal: $('#toggle-multi-agent-modal'),
  maRow: $('#ma-row'),
  maBadge: $('#ma-badge'),
  maSub: $('#ma-sub'),
  agentFields: $('#agent-fields'),
  cfgPlannerModel: $('#cfg-planner-model'),
  cfgWorkerModel: $('#cfg-worker-model'),
  cfgMaxRounds: $('#cfg-max-rounds'),
  cfgMaxRetries: $('#cfg-max-retries'),
  cfgMaxSteps: $('#cfg-max-steps'),
};

const state = {
  ws: null,
  ready: false,
  busy: false,
  tools: [],
  currentBotMsg: null, // {body, chipsRow, agentRow, textEl}
  multiAgent: false,
  plannerModel: null,
  workerModel: null,
  agentConfig: {},
};

// ---------- helpers ----------
function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

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

// Render markdown safely: use marked.js when available, fall back to a simple
// inline formatter so the UI still works if the library fails to load.
function renderText(raw) {
  // Strip model reasoning blocks (<think>...</think>) that some models emit.
  let src = raw.replace(/<think>[\s\S]*?<\/think>/gi, '').trimStart();
  // Also drop a stray unclosed <think> ... to end-of-text.
  src = src.replace(/<think>[\s\S]*$/i, '').trimStart();

  if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
    return DOMPurify.sanitize(
      marked.parse(src, { breaks: true, gfm: true }),
      { USE_PROFILES: { html: true } }
    );
  }
  // Fallback: minimal renderer (no tables / headings, but always safe)
  const escaped = src
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  let html = escaped;
  html = html.replace(/```([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${code.replace(/^\n+|\n+$/g, '')}</code></pre>`);
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
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
    parts.agentRow = document.createElement('div');
    parts.agentRow.className = 'agent-stream';
    parts.content.parentNode.insertBefore(parts.agentRow, parts.content);
    parts.chipsRow = document.createElement('div');
    parts.chipsRow.className = 'tool-chips';
    parts.content.parentNode.insertBefore(parts.chipsRow, parts.content);
    parts.lastWorkerNode = null;
    state.currentBotMsg = parts;
  }
  return state.currentBotMsg;
}

// ---------- Multi-agent event rendering ----------
function addPlan(ev) {
  const bot = ensureBotMessage();
  const card = document.createElement('div');
  card.className = 'agent-plan';
  const tasks = (ev.tasks || [])
    .map(
      (t) =>
        `<li><span class="cat-badge">${escapeHtml(t.category || '?')}</span>` +
        `<span>${escapeHtml(t.task || '')}</span></li>`
    )
    .join('');
  const statusCls = ev.status === 'done' ? 'done' : '';
  const taskCount = (ev.tasks || []).length;
  const delegateLabel = taskCount
    ? `<div class="agent-delegate">Delegating ${taskCount} task${taskCount > 1 ? 's' : ''} to workers:</div>`
    : '';
  card.innerHTML = `
    <div class="agent-plan-head">
      <span class="agent-plan-icon">🧭</span>
      <span class="agent-plan-title">Supervisor · round ${escapeHtml(ev.round)}</span>
      <span class="agent-plan-status ${statusCls}">${escapeHtml(ev.status || '')}</span>
    </div>
    ${ev.reasoning ? `<div class="agent-reason">${escapeHtml(ev.reasoning)}</div>` : ''}
    ${delegateLabel}
    ${taskCount ? `<ul class="agent-tasks">${tasks}</ul>` : ''}`;
  bot.agentRow.appendChild(card);
  scrollToBottom();
}

function addWorkerStart(ev) {
  const bot = ensureBotMessage();
  const node = document.createElement('div');
  node.className = 'agent-worker running';
  node.innerHTML = `
    <span class="agent-worker-spin"></span>
    <span class="cat-badge">${escapeHtml(ev.category || '?')}</span>
    <span class="agent-worker-task">${escapeHtml(ev.task || '')}</span>`;
  bot.agentRow.appendChild(node);
  // Tasks are dispatched sequentially, so the most recent start pairs with
  // the next result.
  bot.lastWorkerNode = node;
  scrollToBottom();
}

function addWorkerResult(ev) {
  const bot = ensureBotMessage();
  const node = bot.lastWorkerNode;
  if (!node) return;
  const taskEl = node.querySelector('.agent-worker-task');
  const taskText = taskEl ? taskEl.textContent : '';
  const icon = ev.success ? '✓' : '✗';
  const bits = [];
  if ((ev.tool_calls || []).length) bits.push(ev.tool_calls.join(', '));
  if (ev.attempts && ev.attempts > 1) bits.push(`${ev.attempts} attempts`);
  node.className = 'agent-worker ' + (ev.success ? 'ok' : 'failed');
  node.innerHTML = `
    <span class="agent-worker-icon">${icon}</span>
    <span class="cat-badge">${escapeHtml(ev.category || '?')}</span>
    <span class="agent-worker-task">${escapeHtml(taskText)}</span>
    ${bits.length ? `<span class="agent-worker-meta">${escapeHtml(bits.join(' · '))}</span>` : ''}`;
  if (!ev.success && ev.error) {
    const err = document.createElement('div');
    err.className = 'agent-worker-err';
    err.textContent = ev.error;
    node.appendChild(err);
  }
  bot.lastWorkerNode = null;
  scrollToBottom();
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
      case 'plan':
        addPlan(data);
        break;
      case 'worker_start':
        addWorkerStart(data);
        break;
      case 'worker_result':
        addWorkerResult(data);
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
    state.plannerModel = cfg.planner_model || null;
    state.workerModel = cfg.worker_model || null;
    if (cfg.agent_config) state.agentConfig = cfg.agent_config;
    updateMultiAgentUI(!!cfg.multi_agent);
  } catch (e) {
    els.providerPill.textContent = 'provider · ?';
    els.modelPill.textContent = 'model · ?';
  }
}

// ---------- Multi-agent toggle ----------
function updateMultiAgentUI(on) {
  state.multiAgent = on;
  els.toggleMultiAgent.checked = on;
  if (els.toggleMultiAgentModal) els.toggleMultiAgentModal.checked = on;
  document.body.classList.toggle('multi-agent-on', on);

  els.maBadge.textContent = on ? 'on' : 'off';
  els.maBadge.className = 'ma-badge ' + (on ? 'ma-badge--on' : 'ma-badge--off');

  const planner = state.agentConfig.planner_model || state.plannerModel || '?';
  const worker = state.agentConfig.worker_model || state.workerModel || '?';
  els.maSub.textContent = on
    ? `supervisor: ${planner} · workers: ${worker}`
    : 'single model handles everything';
  if (els.agentFields) els.agentFields.classList.toggle('is-disabled', !on);
}

async function setMultiAgent(on) {
  els.toggleMultiAgent.disabled = true;
  try {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: { multi_agent: on } }),
    });
    const data = await r.json();
    if (data && data.settings) settings.current = data.settings;
    if (data && data.agent_config) state.agentConfig = data.agent_config;
    updateMultiAgentUI(on);
  } catch (e) {
    // Revert the checkbox if the request failed.
    updateMultiAgentUI(state.multiAgent);
  } finally {
    els.toggleMultiAgent.disabled = false;
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

els.toggleMultiAgent.addEventListener('change', () => {
  setMultiAgent(els.toggleMultiAgent.checked);
});

// ---------- Settings modal ----------
const settings = { current: { system_tools: false, unrestricted_exec: false, multi_agent: false } };

function setNote(kind, text) {
  if (!text) { els.settingsNote.hidden = true; els.settingsNote.textContent = ''; return; }
  els.settingsNote.hidden = false;
  els.settingsNote.className = 'setting-note ' + (kind || '');
  els.settingsNote.textContent = text;
}

function fillAgentFields(cfg) {
  if (!cfg) return;
  els.cfgPlannerModel.value = cfg.planner_model || '';
  els.cfgWorkerModel.value = cfg.worker_model || '';
  els.cfgMaxRounds.value = cfg.max_rounds != null ? cfg.max_rounds : '';
  els.cfgMaxRetries.value = cfg.max_worker_retries != null ? cfg.max_worker_retries : '';
  els.cfgMaxSteps.value = cfg.max_worker_steps != null ? cfg.max_worker_steps : '';
}

async function loadSettings() {
  try {
    const r = await fetch('/api/settings');
    const data = await r.json();
    settings.current = data.settings || settings.current;
    els.toggleSystemTools.checked = !!settings.current.system_tools;
    els.toggleUnrestrictedExec.checked = !!settings.current.unrestricted_exec;
    if (els.toggleMultiAgentModal) {
      els.toggleMultiAgentModal.checked = !!settings.current.multi_agent;
    }
    if (data.agent_config) {
      state.agentConfig = data.agent_config;
      fillAgentFields(data.agent_config);
    }
    if (els.agentFields) {
      els.agentFields.classList.toggle('is-disabled', !settings.current.multi_agent);
    }
  } catch (e) { /* ignore */ }
}

function openSettings() {
  loadSettings();
  setNote('', '');
  els.settingsModal.hidden = false;
}
function closeSettings() { els.settingsModal.hidden = true; }

async function applySettings() {
  const proposedSettings = {
    system_tools: els.toggleSystemTools.checked,
    unrestricted_exec: els.toggleUnrestrictedExec.checked,
    multi_agent: els.toggleMultiAgentModal ? els.toggleMultiAgentModal.checked : state.multiAgent,
  };
  const proposedAgent = {
    planner_model: els.cfgPlannerModel.value.trim(),
    worker_model: els.cfgWorkerModel.value.trim(),
    max_rounds: parseInt(els.cfgMaxRounds.value, 10),
    max_worker_retries: parseInt(els.cfgMaxRetries.value, 10),
    max_worker_steps: parseInt(els.cfgMaxSteps.value, 10),
  };
  // Drop empty/NaN fields so we don't overwrite with junk.
  Object.keys(proposedAgent).forEach((k) => {
    const v = proposedAgent[k];
    if (v === '' || v == null || (typeof v === 'number' && Number.isNaN(v))) {
      delete proposedAgent[k];
    }
  });

  const cur = settings.current;
  const dangerChanged =
    proposedSettings.system_tools !== cur.system_tools ||
    proposedSettings.unrestricted_exec !== cur.unrestricted_exec;

  const controls = [
    els.settingsApply, els.toggleSystemTools, els.toggleUnrestrictedExec,
    els.toggleMultiAgentModal, els.cfgPlannerModel, els.cfgWorkerModel,
    els.cfgMaxRounds, els.cfgMaxRetries, els.cfgMaxSteps,
  ];
  controls.forEach((c) => { if (c) c.disabled = true; });
  setNote('busy', dangerChanged
    ? 'Restarting MCP server with new settings…'
    : 'Saving settings…');

  try {
    const r = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: proposedSettings, agent_config: proposedAgent }),
    });
    const data = await r.json();
    if (!r.ok) {
      setNote('error', data.error || 'Failed to update settings.');
      return;
    }
    settings.current = data.settings;
    if (data.agent_config) {
      state.agentConfig = data.agent_config;
      fillAgentFields(data.agent_config);
    }
    if (Array.isArray(data.tools)) {
      state.tools = data.tools;
      renderTools(state.tools);
    }
    // Sync the sidebar multi-agent indicator with the saved state.
    updateMultiAgentUI(!!data.settings.multi_agent);

    const enabled = [];
    if (data.settings.system_tools) enabled.push('system tools');
    if (data.settings.unrestricted_exec) enabled.push('unrestricted exec');
    let msg = 'Settings saved.';
    if (data.settings.multi_agent) msg += ' Multi-agent is ON.';
    if (enabled.length) msg += ' Enabled: ' + enabled.join(', ') + '.';
    setNote('success', msg);
    setTimeout(closeSettings, 900);
  } catch (e) {
    setNote('error', 'Network error: ' + e.message);
  } finally {
    controls.forEach((c) => { if (c) c.disabled = false; });
  }
}

// Keep the modal's agent-config fields enabled/disabled with its toggle.
if (els.toggleMultiAgentModal) {
  els.toggleMultiAgentModal.addEventListener('change', () => {
    if (els.agentFields) {
      els.agentFields.classList.toggle('is-disabled', !els.toggleMultiAgentModal.checked);
    }
  });
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
