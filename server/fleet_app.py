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
<title>Hiring Fleet — AI Oversight System</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700;800&display=swap');

  :root {
    --bg: #0b0f1a;
    --surface: #111827;
    --surface2: #1a2332;
    --border: #1e2d45;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.5);
    --purple: #7c3aed;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Outfit', sans-serif;
    --grad: linear-gradient(135deg, #3b82f6, #7c3aed);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    line-height: 1.6;
    overflow-x: hidden;
  }

  /* ── Background Glow ── */
  .glow {
    position: fixed; top: -10%; left: -10%; width: 40%; height: 40%;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    filter: blur(80px); z-index: -1; opacity: 0.3; pointer-events: none;
  }

  /* ── Typography ── */
  h1, h2, h3, h4 { font-weight: 700; letter-spacing: -0.02em; }
  .gradient-text {
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  /* ── Navigation ── */
  nav {
    position: sticky; top: 0; z-index: 1000;
    background: rgba(11, 15, 26, 0.8);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    height: 70px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.2rem; }
  .nav-links { display: flex; gap: 24px; }
  .nav-links a { 
    color: var(--muted); text-decoration: none; font-size: 0.9rem; font-weight: 600; 
    transition: color 0.2s; 
  }
  .nav-links a:hover { color: var(--text); }
  .nav-cta {
    background: var(--accent); color: white; padding: 8px 16px; border-radius: 8px;
    font-size: 0.85rem; font-weight: 700; text-decoration: none;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    transition: transform 0.2s, background 0.2s;
  }
  .nav-cta:hover { transform: translateY(-1px); background: #2563eb; }

  /* ── Hero Section ── */
  .hero {
    padding: 100px 20px 60px;
    text-align: center;
    max-width: 900px;
    margin: 0 auto;
  }
  .hero-tag {
    display: inline-block; padding: 4px 12px; background: rgba(59, 130, 246, 0.1);
    border: 1px solid var(--accent); color: var(--accent); border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 24px;
  }
  .hero h1 { font-size: 3.5rem; line-height: 1.1; margin-bottom: 24px; }
  .hero p { color: var(--muted); font-size: 1.2rem; margin-bottom: 40px; max-width: 700px; margin-left: auto; margin-right: auto; }

  /* ── Interactive Demo ── */
  .demo-container {
    max-width: 1200px; margin: 0 auto 100px;
    padding: 0 40px;
  }
  .demo-grid {
    display: grid; grid-template-columns: 380px 1fr; gap: 24px;
  }
  .glass-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  }
  
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px;
  }
  .panel-title { font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: var(--muted); letter-spacing: 0.1em; }

  .control-group { margin-bottom: 20px; }
  label { font-size: 0.8rem; font-weight: 600; color: var(--muted); display: block; margin-bottom: 8px; }
  
  select, input {
    width: 100%; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); padding: 12px; font-family: var(--mono);
    font-size: 0.85rem; outline: none; transition: border-color 0.2s;
  }
  select:focus, input:focus { border-color: var(--accent); }

  .btn-reset {
    width: 100%; background: var(--grad); color: white; border: none;
    padding: 14px; border-radius: 10px; font-weight: 800; font-family: var(--sans);
    cursor: pointer; transition: transform 0.2s, opacity 0.2s; margin-bottom: 24px;
  }
  .btn-reset:hover { opacity: 0.9; transform: scale(1.01); }
  .btn-reset:active { transform: scale(0.99); }

  .action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .btn-action {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); padding: 10px; font-family: var(--mono);
    font-size: 0.7rem; cursor: pointer; transition: all 0.2s;
    text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .btn-action:hover:not(:disabled) { border-color: var(--accent); background: rgba(59, 130, 246, 0.1); }
  .btn-action:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Output Display ── */
  .output-view { display: flex; flex-direction: column; height: 100%; min-height: 600px; }
  .obs-meta {
    display: flex; align-items: center; gap: 16px; margin-bottom: 16px;
    background: var(--surface2); padding: 12px 20px; border-radius: 12px; border: 1px solid var(--border);
  }
  .meta-item { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-family: var(--mono); }
  .meta-val { font-weight: 700; color: var(--accent); }
  .badge-phase { 
    padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.7rem; 
    text-transform: uppercase; background: var(--purple); color: white;
  }
  .reward-val { font-size: 1.2rem; font-weight: 800; color: var(--green); }

  .code-container { flex: 1; position: relative; }
  pre {
    background: #05070a; border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; font-family: var(--mono); font-size: 0.85rem; color: #a1a1aa;
    height: 100%; overflow-y: auto; line-height: 1.6;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  pre::-webkit-scrollbar { width: 6px; }
  pre::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

  .feedback-banner {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(59, 130, 246, 0.1); border-top: 1px solid var(--accent);
    padding: 12px 20px; font-size: 0.85rem; color: var(--accent); font-weight: 600;
    border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;
    backdrop-filter: blur(4px);
  }

  /* ── Sections ── */
  .section { padding: 80px 0; border-top: 1px solid var(--border); }
  .container { max-width: 1100px; margin: 0 auto; padding: 0 40px; }
  .section-header { margin-bottom: 48px; }
  .section-label { color: var(--accent); font-weight: 800; text-transform: uppercase; font-size: 0.75rem; margin-bottom: 12px; display: block; }
  .section h2 { font-size: 2.2rem; margin-bottom: 16px; }

  /* ── Agent Pipeline ── */
  .agent-pipeline {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  }
  .agent-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 24px; position: relative; transition: border-color 0.3s;
  }
  .agent-card:hover { border-color: var(--accent); }
  .agent-idx { font-weight: 900; font-size: 3rem; color: rgba(255, 255, 255, 0.05); position: absolute; top: 10px; right: 20px; }
  .agent-icon { font-size: 2rem; margin-bottom: 16px; }
  .agent-name { font-size: 1.1rem; font-weight: 800; margin-bottom: 8px; }
  .agent-desc { color: var(--muted); font-size: 0.85rem; }

  footer {
    text-align: center; padding: 60px 0; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border);
  }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .demo-grid { grid-template-columns: 1fr; }
    .agent-pipeline { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 600px) {
    .agent-pipeline { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.5rem; }
  }
</style>
</head>
<body>
<div class="glow"></div>

<nav>
  <div class="brand">🛡️ HIRING FLEET</div>
  <div class="nav-links">
    <a href="#pipeline">Pipeline</a>
    <a href="#demo">Live Demo</a>
    <a href="/docs">API Docs</a>
  </div>
  <a href="https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment" class="nav-cta">View Source</a>
</nav>

<section class="hero">
  <span class="hero-tag">Meta OpenEnv Hackathon — Day 3</span>
  <h1>Scalable Oversight for <span class="gradient-text">Adversarial Recruiting</span></h1>
  <p>A multi-agent hiring pipeline where specialists investigate isolated resume segments to prevent fraud. The Overseer synthesizes truth from reasoning, not raw data.</p>
</section>

<div class="demo-container" id="demo">
  <div class="demo-grid">
    <div class="glass-panel">
      <div class="panel-header"><span class="panel-title">Controls</span></div>
      
      <div class="control-group">
        <label>Tier Difficulty</label>
        <select id="taskType">
          <option value="easy">Easy — Basic Discrepancies</option>
          <option value="medium" selected>Medium — Subtle Fabrications</option>
          <option value="hard">Hard — Sophisticated Adversaries</option>
        </select>
      </div>
      
      <div class="control-group">
        <label>Seed (Randomness Control)</label>
        <input type="number" id="seed" value="42" min="0">
      </div>

      <button class="btn-reset" onclick="resetEpisode()">↺ RESET EPISODE</button>

      <div class="panel-header"><span class="panel-title">Available Actions</span></div>
      <div class="action-grid" id="action-grid">
        <button class="btn-action" onclick="step({action_type:'verify_credential'})" id="btn-verify">verify_credential</button>
        <button class="btn-action" onclick="step({action_type:'check_reference',reference_id:'ref2'})" id="btn-ref2">check_ref ref2</button>
        <button class="btn-action" onclick="step({action_type:'view_section',section:'experience'})" id="btn-exp">view experience</button>
        <button class="btn-action" onclick="step({action_type:'view_section',section:'education'})" id="btn-edu">view education</button>
        <button class="btn-action" onclick="step({action_type:'view_section',section:'references'})" id="btn-refs">view references</button>
        <button class="btn-action" onclick="submitSpecialistReport()" id="btn-submit-spec">submit_report</button>
        <button class="btn-action" onclick="step({action_type:'read_reports',report_target:'fraud_specialist'})" id="btn-read-fraud">read fraud report</button>
        <button class="btn-action" onclick="step({action_type:'read_reports',report_target:'skills_specialist'})" id="btn-read-skills">read skills report</button>
        <button class="btn-action" onclick="submitFinalDecision('reject')" id="btn-reject">REJECT</button>
        <button class="btn-action" onclick="submitFinalDecision('accept')" id="btn-accept">ACCEPT</button>
      </div>
    </div>

    <div class="output-view">
      <div class="obs-meta">
        <span id="phase-badge" class="badge-phase">Idle</span>
        <div class="meta-item">Steps Left: <span id="steps-left" class="meta-val">—</span></div>
        <div class="meta-item">Total: <span id="total-left" class="meta-val">—</span></div>
        <div class="meta-item">Violations: <span id="violations" class="meta-val" style="color:var(--red)">0</span></div>
        <div style="flex:1"></div>
        <div class="reward-val" id="reward-display">+0.0000</div>
      </div>

      <div class="code-container">
        <pre id="obs-output">Click "RESET EPISODE" to start a new investigation.
The environment will populate with the first agent (Fraud Specialist).</pre>
        <div id="feedback-banner" class="feedback-banner" style="display:none"></div>
      </div>
    </div>
  </div>
</div>

<section class="section" id="pipeline">
  <div class="container">
    <div class="section-header">
      <span class="section-label">System Architecture</span>
      <h2>The Hiring Fleet Protocol</h2>
    </div>
    
    <div class="agent-pipeline">
      <div class="agent-card">
        <div class="agent-idx">01</div>
        <div class="agent-icon">🔍</div>
        <div class="agent-name">Fraud Specialist</div>
        <div class="agent-desc">Validates credentials and references. Prevents degree-milling and fabrication.</div>
      </div>
      <div class="agent-card">
        <div class="agent-idx">02</div>
        <div class="agent-icon">🛠️</div>
        <div class="agent-name">Skills Specialist</div>
        <div class="agent-desc">Maps experience and projects to job requirements. Detects skill exaggeration.</div>
      </div>
      <div class="agent-card">
        <div class="agent-idx">03</div>
        <div class="agent-icon">📅</div>
        <div class="agent-name">Timeline Specialist</div>
        <div class="agent-desc">Identifies chronological overlaps and employment gaps. Verifies history.</div>
      </div>
      <div class="agent-card">
        <div class="agent-idx">04</div>
        <div class="agent-icon">⚖️</div>
        <div class="agent-name">Fleet Overseer</div>
        <div class="agent-desc">Synthesizes reports into a final verdict. No direct data access — pure oversight.</div>
      </div>
    </div>
  </div>
</section>

<footer>
  Hiring Fleet v3.0.0 &nbsp;•&nbsp; Built for Meta OpenEnv Hackathon &nbsp;•&nbsp; IshikaMahadar
</footer>

<script>
  // Use absolute paths for API calls to avoid 404s in HF Space iframes
  const API_PREFIX = window.location.origin;

  async function resetEpisode() {
    const taskType = document.getElementById('taskType').value;
    const seed = parseInt(document.getElementById('seed').value) || 42;
    
    setLoading(true);
    const url = `${API_PREFIX}/reset`;
    console.log(`Fetching: ${url}`);
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_type: taskType, seed })
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      updateUI(data.observation || data, data.reward || 0);
    } catch (err) {
      document.getElementById('obs-output').textContent = `Error: ${err.message}\nURL: ${url}\nCheck the browser console for details.`;
    } finally {
      setLoading(false);
    }
  }

  async function step(action) {
    setLoading(true);
    const url = `${API_PREFIX}/step`;
    console.log(`Fetching: ${url}`);
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action)
      });
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      updateUI(data.observation || data, data.reward || 0);
    } catch (err) {
      document.getElementById('obs-output').textContent = `Error: ${err.message}\nURL: ${url}`;
    } finally {
      setLoading(false);
    }
  }

  function submitSpecialistReport() {
    step({
      action_type: 'submit_specialist_report',
      findings: 'Investigation complete. Signal collected via role tools.',
      has_issues: false,
      specialist_confidence: 0.8
    });
  }

  function submitFinalDecision(decision) {
    const isFraud = decision === 'reject';
    step({
      action_type: 'submit_final_decision',
      decision: decision,
      fraud_flag: isFraud,
      confidence: 0.9,
      fraud_reasoning: isFraud ? 'Discrepancies found in specialist reports.' : 'All checks passed.'
    });
  }

  function updateUI(obs, reward) {
    // Meta update
    const phase = obs.current_phase || 'Complete';
    document.getElementById('phase-badge').textContent = phase.replace(/_/g, ' ');
    document.getElementById('steps-left').textContent = obs.steps_remaining ?? '0';
    document.getElementById('total-left').textContent = obs.total_steps_remaining ?? '0';
    document.getElementById('violations').textContent = obs.violations_count ?? 0;
    
    const rDisplay = document.getElementById('reward-display');
    const rVal = typeof reward === 'number' ? reward : (obs.reward ?? 0);
    rDisplay.textContent = `${rVal >= 0 ? '+' : ''}${rVal.toFixed(4)}`;
    rDisplay.style.color = rVal > 0 ? 'var(--green)' : rVal < 0 ? 'var(--red)' : 'var(--muted)';

    // Feedback banner
    const banner = document.getElementById('feedback-banner');
    if (obs.feedback) {
      banner.textContent = obs.feedback;
      banner.style.display = 'block';
    } else {
      banner.style.display = 'none';
    }

    // Clean observation for display
    const clean = { ...obs };
    delete clean.role_instructions; // Hide long instructions from JSON view
    if (clean.visible_sections && Object.keys(clean.visible_sections).length > 0) {
        // Truncate sections for readability
        Object.keys(clean.visible_sections).forEach(k => {
            if (typeof clean.visible_sections[k] === 'string')
                clean.visible_sections[k] = clean.visible_sections[k].substring(0, 80) + '...';
        });
    }
    
    document.getElementById('obs-output').textContent = JSON.stringify(clean, null, 2);
    
    // Auto-scroll output
    const pre = document.getElementById('obs-output');
    pre.scrollTop = 0;
  }

  function setLoading(loading) {
    const btn = document.querySelector('.btn-reset');
    if (loading) {
      btn.style.opacity = '0.7';
      btn.textContent = 'WORKING...';
    } else {
      btn.style.opacity = '1';
      btn.textContent = '↺ RESET EPISODE';
    }
  }
</script>
</body>
</html>"""


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
