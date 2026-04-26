from openenv.core.env_server import create_fastapi_app
from .fleet_environment import FleetResumeEnvironment
from models import FleetAction, FleetObservation

# Create the FastAPI app for the multi-agent fleet environment
app = create_fastapi_app(
    FleetResumeEnvironment,
    action_cls=FleetAction,
    observation_cls=FleetObservation,
)

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
import logging

logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


INTERACTIVE_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hiring Fleet — Interactive Demo</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg: #0b0f1a;
    --surface: #111827;
    --surface2: #1a2332;
    --border: #1e2d45;
    --accent: #3b82f6;
    --accent2: #7c3aed;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Syne', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
  }

  header {
    padding: 16px 32px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 2px 8px;
    border-radius: 4px;
  }
  header h2 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    flex: 1;
  }
  header p {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: var(--mono);
  }

  .demo-section {
    padding: 24px 32px 8px;
  }
  .demo-section h1 {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 4px;
  }
  .demo-section .sub {
    font-size: 0.875rem;
    color: var(--muted);
  }

  .layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 16px;
    padding: 16px 32px 32px;
    min-height: calc(100vh - 160px);
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .panel-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }

  label {
    font-size: 0.75rem;
    color: var(--muted);
    display: block;
    margin-bottom: 4px;
    font-family: var(--mono);
  }

  select, input[type="number"] {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.82rem;
    padding: 8px 10px;
    outline: none;
    transition: border-color 0.15s;
  }
  select:focus, input[type="number"]:focus {
    border-color: var(--accent);
  }

  .btn-reset {
    width: 100%;
    padding: 11px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 7px;
    font-family: var(--sans);
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    transition: background 0.15s, transform 0.1s;
  }
  .btn-reset:hover { background: #2563eb; }
  .btn-reset:active { transform: scale(0.98); }
  .btn-reset:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }

  .actions-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2px;
  }

  .actions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }

  .btn-action {
    padding: 7px 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.7rem;
    cursor: pointer;
    text-align: center;
    transition: border-color 0.15s, background 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .btn-action:hover:not(:disabled) {
    border-color: var(--accent);
    background: rgba(59,130,246,0.08);
  }
  .btn-action:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  /* Observation panel */
  .obs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .obs-stats {
    display: flex;
    gap: 16px;
    align-items: center;
    font-family: var(--mono);
    font-size: 0.72rem;
  }
  .stat-item { display: flex; align-items: center; gap: 5px; }
  .stat-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot-blue { background: var(--accent); }
  .dot-green { background: var(--green); }
  .dot-red { background: var(--red); }

  .ctrl-row { display: flex; gap: 6px; }
  .ctrl-btn {
    width: 28px; height: 28px;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    transition: opacity 0.15s;
  }
  .ctrl-btn:hover { opacity: 0.8; }

  .obs-output {
    flex: 1;
    background: #070b12;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 16px;
    font-family: var(--mono);
    font-size: 0.78rem;
    line-height: 1.6;
    overflow-y: auto;
    min-height: 400px;
    max-height: 60vh;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .obs-output .prompt-hint {
    color: var(--muted);
    font-style: italic;
  }
  .obs-output .key { color: var(--accent); }
  .obs-output .val-str { color: #a5f3fc; }
  .obs-output .val-num { color: var(--yellow); }
  .obs-output .val-bool-t { color: var(--green); }
  .obs-output .val-bool-f { color: var(--red); }
  .obs-output .section-hdr { color: var(--accent2); font-weight: 600; }
  .obs-output .reward-pos { color: var(--green); font-weight: 600; }
  .obs-output .reward-neg { color: var(--red); font-weight: 600; }
  .obs-output .feedback { color: var(--yellow); }
  .obs-output .done-msg { color: var(--green); font-weight: 600; }

  .loading-line {
    display: inline-block;
    animation: blink 0.9s step-end infinite;
    color: var(--accent);
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  .phase-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: var(--mono);
    margin-bottom: 6px;
  }
  .phase-fraud { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
  .phase-skills { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
  .phase-timeline { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
  .phase-overseer { background: rgba(124,58,237,0.15); color: #c4b5fd; border: 1px solid rgba(124,58,237,0.3); }
  .phase-complete { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

  .error-msg {
    color: var(--red);
    font-family: var(--mono);
    font-size: 0.78rem;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 6px;
    padding: 10px 12px;
  }
</style>
</head>
<body>

<header>
  <div class="badge">Interactive Demo</div>
  <h2>Try the Environment</h2>
  <p>Run a live episode against the deployed environment. Start with Reset, then step through each agent phase.</p>
</header>

<div class="layout">
  <!-- LEFT: Controls -->
  <div class="panel">
    <div class="panel-title">Controls</div>

    <div>
      <label>Difficulty</label>
      <select id="difficulty">
        <option value="easy">Easy — Clear fraud (8 steps)</option>
        <option value="medium" selected>Medium — Subtle fraud (11 steps)</option>
        <option value="hard">Hard — Sophisticated fraud (15 steps)</option>
      </select>
    </div>

    <div>
      <label>Seed</label>
      <input type="number" id="seed" value="42" min="0" max="9999">
    </div>

    <button class="btn-reset" id="btnReset" onclick="resetEpisode()">
      ↺ Reset Episode
    </button>

    <div>
      <div class="actions-title">Actions</div>
      <div class="actions-grid" id="actionsGrid">
        <!-- populated dynamically -->
        <button class="btn-action" disabled>verify_credential</button>
        <button class="btn-action" disabled>check_reference ref2</button>
        <button class="btn-action" disabled>check_reference ref1</button>
        <button class="btn-action" disabled>view experience</button>
        <button class="btn-action" disabled>view education</button>
        <button class="btn-action" disabled>view skills</button>
        <button class="btn-action" disabled>view header</button>
        <button class="btn-action" disabled>view references</button>
        <button class="btn-action" disabled>ask_clarification</button>
        <button class="btn-action" disabled>submit_specialist_report</button>
        <button class="btn-action" disabled>read fraud report</button>
        <button class="btn-action" disabled>read skills report</button>
        <button class="btn-action" disabled>read timeline report</button>
        <button class="btn-action" disabled>submit reject</button>
        <button class="btn-action" disabled>submit accept</button>
      </div>
    </div>
  </div>

  <!-- RIGHT: Observation -->
  <div class="panel">
    <div class="obs-header">
      <div class="panel-title">Observation</div>
      <div class="obs-stats">
        <div class="stat-item">
          <span class="stat-dot dot-blue"></span>
          Phase steps: <span id="statPhase">—</span>
        </div>
        <div class="stat-item">
          <span class="stat-dot dot-green"></span>
          Total left: <span id="statTotal">—</span>
        </div>
        <div class="stat-item">
          <span class="stat-dot dot-red"></span>
          Violations: <span id="statViolations">—</span>
        </div>
        <div class="ctrl-row">
          <button class="ctrl-btn" style="background:#3b4252;" onclick="clearOutput()" title="Clear">✕</button>
          <button class="ctrl-btn" style="background:#2d7a3a;" onclick="toggleWrap()" title="Toggle wrap">≡</button>
        </div>
      </div>
    </div>
    <div class="obs-output" id="obsOutput">
      <span class="prompt-hint">Click "↺ Reset Episode" to start a new episode.</span>
    </div>
  </div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
let episodeId = null;
let currentObs = null;
let busy = false;
let wrapEnabled = true;

// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(yes) {
  busy = yes;
  document.getElementById('btnReset').disabled = yes;
  document.querySelectorAll('.btn-action').forEach(b => {
    if (yes) b.disabled = true;
  });
}

function clearOutput() {
  document.getElementById('obsOutput').innerHTML =
    '<span class="prompt-hint">Output cleared.</span>';
}

function toggleWrap() {
  wrapEnabled = !wrapEnabled;
  const el = document.getElementById('obsOutput');
  el.style.whiteSpace = wrapEnabled ? 'pre-wrap' : 'pre';
}

function phaseClass(phase) {
  if (!phase) return '';
  if (phase.includes('fraud')) return 'phase-fraud';
  if (phase.includes('skills')) return 'phase-skills';
  if (phase.includes('timeline')) return 'phase-timeline';
  if (phase.includes('overseer')) return 'phase-overseer';
  if (phase.includes('complete')) return 'phase-complete';
  return '';
}

function renderObs(obs, reward, action_taken) {
  if (!obs) return;

  const out = document.getElementById('obsOutput');

  // Stats bar
  document.getElementById('statPhase').textContent =
    obs.steps_remaining !== undefined ? obs.steps_remaining : '—';
  document.getElementById('statTotal').textContent =
    obs.total_steps_remaining !== undefined ? obs.total_steps_remaining : '—';
  document.getElementById('statViolations').textContent =
    obs.violations_count !== undefined ? obs.violations_count : '—';

  let html = '';

  // Phase badge
  if (obs.current_phase) {
    html += `<span class="phase-badge ${phaseClass(obs.current_phase)}">${obs.current_phase.replace('_', ' ')}</span>\\n`;
  }

  // Reward line
  if (reward !== null && reward !== undefined) {
    const cls = reward >= 0 ? 'reward-pos' : 'reward-neg';
    html += `<span class="${cls}">reward: ${reward >= 0 ? '+' : ''}${Number(reward).toFixed(4)}</span>\\n`;
  }

  // Done
  if (obs.done) {
    html += `<span class="done-msg">✓ Episode complete</span>\\n`;
  }

  html += '\\n';

  // Feedback / violations
  if (obs.feedback) {
    html += `<span class="feedback">⚠ ${escHtml(obs.feedback)}</span>\\n\\n`;
  }

  // Role instructions (truncated)
  if (obs.role_instructions) {
    const instr = obs.role_instructions.length > 300
      ? obs.role_instructions.slice(0, 300) + '…'
      : obs.role_instructions;
    html += `<span class="section-hdr">── role_instructions ──</span>\\n${escHtml(instr)}\\n\\n`;
  }

  // Job description (short)
  if (obs.job_description) {
    const jd = obs.job_description.length > 200
      ? obs.job_description.slice(0, 200) + '…'
      : obs.job_description;
    html += `<span class="section-hdr">── job_description ──</span>\\n${escHtml(jd)}\\n\\n`;
  }

  // Visible sections
  if (obs.visible_sections && Object.keys(obs.visible_sections).length > 0) {
    html += `<span class="section-hdr">── visible_sections ──</span>\\n`;
    for (const [k, v] of Object.entries(obs.visible_sections)) {
      html += `<span class="key">${escHtml(k)}</span>: ${escHtml(String(v).slice(0, 200))}${v && v.length > 200 ? '…' : ''}\\n`;
    }
    html += '\\n';
  }

  // Specialist reports
  if (obs.specialist_reports && obs.specialist_reports.length > 0) {
    html += `<span class="section-hdr">── specialist_reports ──</span>\\n`;
    obs.specialist_reports.forEach(r => {
      html += `<span class="key">${escHtml(r.role || '?')}</span>: has_issues=<span class="${r.has_issues ? 'val-bool-t' : 'val-bool-f'}">${r.has_issues}</span>, confidence=${r.confidence || r.specialist_confidence || '?'}\\n`;
      if (r.findings) html += `  ${escHtml(String(r.findings).slice(0, 150))}\\n`;
    });
    html += '\\n';
  }

  // Reference / verification results
  if (obs.reference_response) {
    html += `<span class="section-hdr">── reference_response ──</span>\\n${escHtml(obs.reference_response)}\\n\\n`;
  }
  if (obs.verification_result) {
    html += `<span class="section-hdr">── verification_result ──</span>\\n${escHtml(obs.verification_result)}\\n\\n`;
  }
  if (obs.clarification_response) {
    html += `<span class="section-hdr">── clarification_response ──</span>\\n${escHtml(obs.clarification_response)}\\n\\n`;
  }

  // Read report details
  if (obs.read_report_details && Object.keys(obs.read_report_details).length > 0) {
    html += `<span class="section-hdr">── read_report_details ──</span>\\n`;
    for (const [k, v] of Object.entries(obs.read_report_details)) {
      html += `<span class="key">${escHtml(k)}</span>: ${escHtml(String(v).slice(0, 250))}…\\n`;
    }
    html += '\\n';
  }

  // Available actions
  if (obs.available_actions && obs.available_actions.length > 0) {
    html += `<span class="section-hdr">── available_actions ──</span>\\n`;
    html += obs.available_actions.map(a => `<span class="val-str">${escHtml(a)}</span>`).join('  ');
    html += '\\n';
  }

  // Steps
  html += `\\n<span class="key">steps_remaining</span>: <span class="val-num">${obs.steps_remaining ?? '?'}</span>   `;
  html += `<span class="key">total_steps_remaining</span>: <span class="val-num">${obs.total_steps_remaining ?? '?'}</span>   `;
  html += `<span class="key">violations_count</span>: <span class="${(obs.violations_count || 0) > 0 ? 'val-bool-f' : 'val-bool-t'}">${obs.violations_count ?? 0}</span>`;

  out.innerHTML = html;
  out.scrollTop = out.scrollHeight;

  // Rebuild action buttons
  rebuildActions(obs);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function rebuildActions(obs) {
  const grid = document.getElementById('actionsGrid');
  grid.innerHTML = '';

  const available = (obs && obs.available_actions) ? obs.available_actions : [];
  const phase = obs ? obs.current_phase : null;
  const done = obs ? obs.done : false;

  // Build buttons based on available actions from the server
  const actionDefs = buildActionDefs(available, phase);

  if (actionDefs.length === 0) {
    const msg = document.createElement('div');
    msg.style.cssText = 'color:var(--muted);font-family:var(--mono);font-size:0.75rem;grid-column:1/-1;padding:4px 0;';
    msg.textContent = done ? '✓ Episode ended — click Reset to start again.' : 'No actions available. Reset the episode first.';
    grid.appendChild(msg);
    return;
  }

  actionDefs.forEach(({ label, action, disabled }) => {
    const btn = document.createElement('button');
    btn.className = 'btn-action';
    btn.textContent = label;
    btn.disabled = disabled || busy || done;
    btn.onclick = () => takeAction(action);
    grid.appendChild(btn);
  });
}

function buildActionDefs(available, phase) {
  if (!available || available.length === 0) return [];

  const defs = [];
  const avSet = new Set(available.map(a => a.toLowerCase()));

  // Helper: check if action type is available
  const has = (t) => avSet.has(t.toLowerCase()) || available.some(a => a.toLowerCase().includes(t.toLowerCase()));

  if (has('verify_credential')) {
    defs.push({ label: 'verify_credential', action: { action_type: 'verify_credential' } });
  }
  if (has('check_reference')) {
    defs.push({ label: 'check_reference ref1', action: { action_type: 'check_reference', reference_id: 'ref1' } });
    defs.push({ label: 'check_reference ref2', action: { action_type: 'check_reference', reference_id: 'ref2' } });
  }
  if (has('view_section')) {
    const sections = ['header', 'summary', 'experience', 'education', 'skills', 'projects', 'references'];
    sections.forEach(s => {
      defs.push({ label: `view ${s}`, action: { action_type: 'view_section', section: s } });
    });
  }
  if (has('ask_clarification')) {
    defs.push({ label: 'ask_clarification', action: { action_type: 'ask_clarification', question: 'Can you clarify your most recent role and responsibilities?' } });
  }
  if (has('submit_specialist_report')) {
    defs.push({ label: 'submit report (issues)', action: { action_type: 'submit_specialist_report', findings: 'Suspicious indicators found.', has_issues: true, specialist_confidence: 0.7 } });
    defs.push({ label: 'submit report (clean)', action: { action_type: 'submit_specialist_report', findings: 'No issues detected.', has_issues: false, specialist_confidence: 0.7 } });
  }
  if (has('read_reports')) {
    defs.push({ label: 'read fraud report', action: { action_type: 'read_reports', report_target: 'fraud_specialist' } });
    defs.push({ label: 'read skills report', action: { action_type: 'read_reports', report_target: 'skills_specialist' } });
    defs.push({ label: 'read timeline report', action: { action_type: 'read_reports', report_target: 'timeline_specialist' } });
  }
  if (has('request_reinvestigation')) {
    defs.push({ label: 'reinvestigate fraud', action: { action_type: 'request_reinvestigation', reinvestigation_target: 'fraud_specialist', reinvestigation_reason: 'Need deeper credential check.' } });
  }
  if (has('submit_final_decision')) {
    defs.push({ label: 'submit accept', action: { action_type: 'submit_final_decision', decision: 'accept', fraud_flag: false, confidence: 0.8, fraud_reasoning: 'No fraud detected.' } });
    defs.push({ label: 'submit reject', action: { action_type: 'submit_final_decision', decision: 'reject', fraud_flag: true, confidence: 0.8, fraud_reasoning: 'Fraudulent indicators detected.' } });
  }

  return defs;
}

// ── API calls ────────────────────────────────────────────────────────────────
async function resetEpisode() {
  if (busy) return;
  setLoading(true);

  const difficulty = document.getElementById('difficulty').value;
  const seed = parseInt(document.getElementById('seed').value) || 42;

  const out = document.getElementById('obsOutput');
  out.innerHTML = '<span class="loading-line">▌</span> Resetting episode…';

  // Reset stats
  document.getElementById('statPhase').textContent = '—';
  document.getElementById('statTotal').textContent = '—';
  document.getElementById('statViolations').textContent = '—';

  try {
    const body = { task_type: difficulty, seed };
    const res = await fetch('/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status}: ${err}`);
    }

    const data = await res.json();
    // Normalize: the response may wrap obs in an "observation" key
    let obs = data;
    if (data.observation && typeof data.observation === 'object') {
      obs = { ...data, ...data.observation };
    }

    episodeId = obs.episode_id || obs.id || null;
    currentObs = obs;

    renderObs(obs, null, null);
  } catch (e) {
    out.innerHTML = `<div class="error-msg">Reset failed: ${escHtml(e.message)}\\n\\nMake sure the server is running and accessible.</div>`;
    rebuildActions(null);
  } finally {
    setLoading(false);
  }
}

async function takeAction(action) {
  if (busy || !currentObs) return;
  setLoading(true);

  const out = document.getElementById('obsOutput');
  const prevContent = out.innerHTML;
  out.innerHTML = prevContent + '\\n<span class="loading-line">▌</span>';

  try {
    // Attach episode_id if we have one
    const payload = { ...action };
    if (episodeId) payload.episode_id = episodeId;

    const res = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status}: ${err}`);
    }

    const data = await res.json();
    let obs = data.observation || data;
    if (data.observation && typeof data.observation === 'object') {
      obs = { ...data, ...data.observation };
    }
    const reward = data.reward !== undefined ? data.reward : obs.reward;

    currentObs = obs;
    renderObs(obs, reward, action);
  } catch (e) {
    out.innerHTML = prevContent;
    const errDiv = document.createElement('div');
    errDiv.className = 'error-msg';
    errDiv.textContent = 'Step failed: ' + e.message;
    out.appendChild(errDiv);
  } finally {
    setLoading(false);
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def home():
    """Interactive demo home page for the Hugging Face Space."""
    return INTERACTIVE_DEMO_HTML


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Entry point for the fleet environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
