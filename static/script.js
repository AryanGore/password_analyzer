/* ─────────────────────────────────────────────────────
   PassShield — Frontend Logic
   • Real-time password analysis (debounced 500ms)
   • Animated radial gauge (Canvas API)
   • Animated model probability bars
   • Crack-time table renderer
   • LLM suggestions with copy-to-clipboard
   ───────────────────────────────────────────────────── */

'use strict';

// ── DOM Refs ─────────────────────────────────────────
const pwInput       = document.getElementById('passwordInput');
const eyeBtn        = document.getElementById('eyeBtn');
const strengthPill  = document.getElementById('strengthPill');
const strengthLabel = document.getElementById('strengthLabel');
const metaLength    = document.getElementById('metaLength');
const metaEntropy   = document.getElementById('metaEntropy');
const metaScore     = document.getElementById('metaScore');
const gaugeCanvas   = document.getElementById('gaugeCanvas');
const gaugeValue    = document.getElementById('gaugeValue');
const strengthBig   = document.getElementById('strengthBig');
const improveBtn    = document.getElementById('improveBtn');
const improveBtnTxt = document.getElementById('improveBtnText');
const improveSpinner= document.getElementById('improveSpinner');
const suggestGrid   = document.getElementById('suggestionsGrid');
const entropyDisp   = document.getElementById('entropyDisplay');

// Gauge canvas context
const ctx = gaugeCanvas ? gaugeCanvas.getContext('2d') : null;

// Current animated gauge value
let currentGaugeVal = 0;
let targetGaugeVal  = 0;
let animFrameId     = null;

// ── Eye Toggle ───────────────────────────────────────
eyeBtn.addEventListener('click', () => {
  const isPassword = pwInput.type === 'password';
  pwInput.type = isPassword ? 'text' : 'password';
  eyeBtn.textContent = isPassword ? '🙈' : '👁';
});

// ── Debounced Real-time Analysis ─────────────────────
let debounceTimer = null;

pwInput.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  const val = pwInput.value;

  // Instant lightweight UI update
  updateMetaRow(val);

  if (!val) {
    resetUI();
    return;
  }

  debounceTimer = setTimeout(() => fetchAnalysis(val), 500);
});

// ── Reset UI to empty state ───────────────────────────
function resetUI() {
  setPill('—', '');
  metaLength.textContent   = '0 chars';
  metaEntropy.textContent  = '0 bits entropy';
  metaScore.textContent    = 'Score: —';
  setGaugeTarget(0);
  strengthBig.textContent  = '—';
  strengthBig.className    = 'strength-label';
  entropyDisp.textContent  = '0';

  // Reset bars
  ['Svm','Rf','Xgb','Lstm'].forEach(m => {
    document.getElementById('bar' + m).style.width = '0%';
    document.getElementById('pct' + m).textContent = '0%';
  });

  // Reset bonuses
  ['Base','Pat','Ent','Spec'].forEach(b => {
    document.getElementById('bon' + b).textContent = '—';
  });

  // Reset crack times
  ['Online','Cpu','Gpu','Cluster'].forEach(c => {
    document.getElementById('crack' + c).textContent = '—';
  });

  // Reset checklist
  ['Length','Upper','Lower','Digit','Special'].forEach(f => {
    const el = document.getElementById('chk' + f);
    el.className = 'feat-item feat-pending';
    el.querySelector('.feat-icon').textContent = '○';
  });
}

// ── Instant lightweight update (no API) ─────────────
function updateMetaRow(pw) {
  metaLength.textContent = `${pw.length} chars`;

  // Quick client-side entropy hint
  let charset = 0;
  if (/[a-z]/.test(pw)) charset += 26;
  if (/[A-Z]/.test(pw)) charset += 26;
  if (/[0-9]/.test(pw)) charset += 10;
  if (/[^a-zA-Z0-9]/.test(pw)) charset += 32;
  const entropy = charset > 0 ? (pw.length * Math.log2(charset)).toFixed(1) : 0;
  metaEntropy.textContent = `${entropy} bits entropy`;
}

// ── Fetch Full Analysis from Backend ─────────────────
async function fetchAnalysis(pw) {
  try {
    const res = await fetch('/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ password: pw }),
    });
    if (!res.ok) return;
    const data = await res.json();
    renderAnalysis(data);
  } catch (err) {
    console.warn('Analysis request failed:', err);
  }
}

// ── Render Full Analysis ─────────────────────────────
function renderAnalysis(data) {
  const score    = data.final_score ?? 0;
  const strength = data.strength ?? '—';
  const entropy  = data.entropy ?? 0;
  const pct      = Math.round(score * 100);

  // Pill + big label
  const cls = strength === 'STRONG' ? 'strong' : (strength === 'MEDIUM' ? 'medium' : 'weak');
  setPill(strength, cls);
  strengthBig.textContent = strength;
  strengthBig.className   = `strength-label ${cls}`;

  // Meta row
  metaScore.textContent  = `Score: ${(score * 100).toFixed(1)}%`;
  entropyDisp.textContent = entropy;

  // Gauge
  setGaugeTarget(pct, cls);

  // Model bars (LSTM may be null if TF unavailable)
  const models = data.models ?? {};
  setBar('Svm',  models.svm  ?? 0, false);
  setBar('Rf',   models.rf   ?? 0, false);
  setBar('Xgb',  models.xgb  ?? 0, false);
  setBar('Lstm', models.lstm ?? null, models.lstm === null || models.lstm === undefined);

  // Bonuses
  const bonuses = data.bonuses ?? {};
  document.getElementById('bonBase').textContent = fmtBonus(data.base_score);
  document.getElementById('bonPat').textContent  = fmtBonus(bonuses.pattern_adjustment);
  document.getElementById('bonEnt').textContent  = fmtBonus(bonuses.entropy_bonus);
  document.getElementById('bonSpec').textContent = fmtBonus(bonuses.special_bonus);

  // Crack times
  const ttc = data.time_to_crack ?? {};
  const vals = Object.values(ttc);
  document.getElementById('crackOnline').textContent  = vals[0] ?? '—';
  document.getElementById('crackCpu').textContent     = vals[1] ?? '—';
  document.getElementById('crackGpu').textContent     = vals[2] ?? '—';
  document.getElementById('crackCluster').textContent = vals[3] ?? '—';

  // Checklist
  const missing = data.missing_features ?? [];
  const pw = pwInput.value;
  updateChecklist(pw, missing);

  // Breach Check Status
  const bc = data.breach_check ?? {};
  const breachBanner = document.getElementById('breachBanner');
  const breachIcon   = document.getElementById('breachIcon');
  const breachTitle  = document.getElementById('breachTitle');
  const breachSub    = document.getElementById('breachSub');

  if (bc.breached) {
    breachBanner.className = 'breach-banner breached';
    breachIcon.textContent = '🚨';
    breachTitle.textContent = `CRITICAL BREACH WARNING: Found in ${bc.count.toLocaleString()} data leaks!`;
    breachSub.textContent = 'This password is compromised in public database dumps (Checked via HIBP k-Anonymity API).';
  } else {
    breachBanner.className = 'breach-banner safe';
    breachIcon.textContent = '🛡️';
    breachTitle.textContent = 'Breach Database Status: Clean (0 Breaches Found)';
    breachSub.textContent = 'Verified using HaveIBeenPwned Zero-Knowledge k-Anonymity protocol.';
  }
}

// ── Helpers ──────────────────────────────────────────
function setPill(text, cls) {
  strengthLabel.textContent = text;
  strengthPill.className    = `strength-pill ${cls}`;
}

function setBar(model, prob, unavailable = false) {
  const bar  = document.getElementById('bar' + model);
  const lbl  = document.getElementById('pct' + model);
  if (unavailable) {
    bar.style.width = '0%';
    bar.style.background = 'rgba(255,255,255,0.06)';
    lbl.textContent = 'N/A';
    lbl.style.color = 'var(--text-muted)';
    return;
  }
  const pct = Math.round((prob ?? 0) * 100);
  bar.style.background = '';  // reset to CSS gradient
  bar.style.width = pct + '%';
  lbl.textContent = pct + '%';
  lbl.style.color = '';
}

function fmtBonus(val) {
  if (val === undefined || val === null) return '—';
  const n = parseFloat(val);
  if (isNaN(n)) return '—';
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(3)}`;
}

function updateChecklist(pw, missing) {
  const checks = {
    Length:  { pass: pw.length >= 8, label: 'Min 8 characters' },
    Upper:   { pass: /[A-Z]/.test(pw), label: 'Uppercase letter' },
    Lower:   { pass: /[a-z]/.test(pw), label: 'Lowercase letter' },
    Digit:   { pass: /[0-9]/.test(pw), label: 'Number' },
    Special: { pass: /[^a-zA-Z0-9]/.test(pw), label: 'Special character' },
  };
  for (const [key, chk] of Object.entries(checks)) {
    const el   = document.getElementById('chk' + key);
    const icon = el.querySelector('.feat-icon');
    if (chk.pass) {
      el.className   = 'feat-item feat-pass';
      icon.textContent = '✅';
    } else {
      el.className   = 'feat-item feat-fail';
      icon.textContent = '❌';
    }
  }
}

// ── Radial Gauge (Canvas) ─────────────────────────────
const GAUGE_COLORS = {
  weak:   '#ff4d6d',
  medium: '#ffb347',
  strong: '#3dfaaf',
  empty:  'rgba(255,255,255,0.06)',
};

function drawGauge(value, cls) {
  if (!ctx) return;
  const W  = gaugeCanvas.width;
  const H  = gaugeCanvas.height;
  const cx = W / 2;
  const cy = H - 20;
  const r  = Math.min(W, H * 2) / 2 - 16;

  ctx.clearRect(0, 0, W, H);

  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;

  // Track (background arc)
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.lineWidth   = 14;
  ctx.strokeStyle = GAUGE_COLORS.empty;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Filled arc
  const fillColor = GAUGE_COLORS[cls] || GAUGE_COLORS.weak;
  const fillEnd   = startAngle + (Math.PI * value / 100);
  if (value > 0) {
    // Glow effect
    ctx.shadowBlur  = 16;
    ctx.shadowColor = fillColor;

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillEnd);
    ctx.strokeStyle = fillColor;
    ctx.lineWidth   = 14;
    ctx.lineCap     = 'round';
    ctx.stroke();

    ctx.shadowBlur = 0;
  }

  // Tick marks
  for (let i = 0; i <= 100; i += 25) {
    const angle = startAngle + (Math.PI * i / 100);
    const x1 = cx + (r - 20) * Math.cos(angle);
    const y1 = cy + (r - 20) * Math.sin(angle);
    const x2 = cx + (r - 26) * Math.cos(angle);
    const y2 = cy + (r - 26) * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function setGaugeTarget(val, cls = 'weak') {
  targetGaugeVal = Math.max(0, Math.min(100, val));
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animateGauge(cls);
}

function animateGauge(cls) {
  const diff = targetGaugeVal - currentGaugeVal;
  if (Math.abs(diff) < 0.5) {
    currentGaugeVal = targetGaugeVal;
    gaugeValue.textContent = Math.round(currentGaugeVal);
    drawGauge(currentGaugeVal, cls);
    return;
  }
  currentGaugeVal += diff * 0.12;
  gaugeValue.textContent = Math.round(currentGaugeVal);
  drawGauge(currentGaugeVal, cls);
  animFrameId = requestAnimationFrame(() => animateGauge(cls));
}

// Draw empty gauge on load
drawGauge(0, 'empty');

// ── LLM Suggestions ──────────────────────────────────
improveBtn.addEventListener('click', async () => {
  const pw = pwInput.value.trim();
  if (!pw) {
    flashInput();
    return;
  }

  setImproveLoading(true);
  suggestGrid.innerHTML = '';

  try {
    const res = await fetch('/improve', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ password: pw }),
    });
    const data = await res.json();

    if (data.error) {
      showSuggestError(data.error);
      return;
    }

    if (!data.suggestions || !data.suggestions.length) {
      showSuggestError('No suggestions returned.');
      return;
    }

    data.suggestions.forEach((s, i) => renderSuggestionCard(s, i));
  } catch (err) {
    showSuggestError('Network error — please try again.');
  } finally {
    setImproveLoading(false);
  }
});

function renderSuggestionCard(s, idx) {
  const strengthCls = s.strength === 'STRONG' ? 'strong' : (s.strength === 'MEDIUM' ? 'medium' : 'weak');
  const tagCls = `tag-${strengthCls}`;
  const strategyTitle = s.strategy || `Strategy ${idx + 1}`;

  const card = document.createElement('div');
  card.className = 'suggest-card glass-card';
  card.style.animationDelay = `${idx * 0.08}s`;
  card.innerHTML = `
    <div class="strategy-badge">⚡ ${escHtml(strategyTitle)}</div>
    <div class="suggest-pw" id="suggestPw${idx}">${escHtml(s.password)}</div>
    <div class="suggest-meta">
      <span class="suggest-tag ${tagCls}">${s.strength}</span>
      <span class="suggest-tag suggest-entropy">Score: ${(s.score * 100).toFixed(1)}%</span>
    </div>
    <div class="suggest-entropy" style="margin-bottom:12px">Entropy: ${s.entropy} bits</div>
    <button class="copy-btn" id="copyBtn${idx}" onclick="copyPassword(${idx})">📋 Copy Password</button>
  `;
  suggestGrid.appendChild(card);
}

window.copyPassword = function(idx) {
  const el   = document.getElementById(`suggestPw${idx}`);
  const btn  = document.getElementById(`copyBtn${idx}`);
  const text = el.textContent;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✅ Copied!';
    btn.classList.add('copied');
    showToast('Copied password to clipboard!');
    setTimeout(() => {
      btn.textContent = '📋 Copy Password';
      btn.classList.remove('copied');
    }, 2000);
  });
};

// ── Random Generator Tool Logic ───────────────────────
const genLenSlider   = document.getElementById('genLenSlider');
const genLenVal      = document.getElementById('genLenVal');
const quickGenBtn    = document.getElementById('quickGenBtn');
const genResultInput = document.getElementById('genResultInput');
const genCopyBtn     = document.getElementById('genCopyBtn');

if (genLenSlider) {
  genLenSlider.addEventListener('input', () => {
    genLenVal.textContent = genLenSlider.value;
  });
}

function generateRandomPassword() {
  const len     = parseInt(genLenSlider.value, 10);
  const useUpper= document.getElementById('chkGenUpper').checked;
  const useLower= document.getElementById('chkGenLower').checked;
  const useDigits=document.getElementById('chkGenDigits').checked;
  const useSyms = document.getElementById('chkGenSymbols').checked;

  let chars = '';
  if (useUpper)  chars += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  if (useLower)  chars += 'abcdefghijklmnopqrstuvwxyz';
  if (useDigits) chars += '0123456789';
  if (useSyms)   chars += '!@#$%^&*()_+-=[]{}|;:,.<>?';

  if (!chars) chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$';

  let result = '';
  const array  = new Uint32Array(len);
  window.crypto.getRandomValues(array);
  for (let i = 0; i < len; i++) {
    result += chars[array[i] % chars.length];
  }
  return result;
}

if (quickGenBtn) {
  quickGenBtn.addEventListener('click', () => {
    const pw = generateRandomPassword();
    genResultInput.value = pw;
  });
}

if (genCopyBtn) {
  genCopyBtn.addEventListener('click', () => {
    const pw = genResultInput.value;
    if (!pw) {
      const newPw = generateRandomPassword();
      genResultInput.value = newPw;
      pwInput.value = newPw;
      fetchAnalysis(newPw);
      return;
    }
    navigator.clipboard.writeText(pw).then(() => {
      // Load into analyzer
      pwInput.value = pw;
      fetchAnalysis(pw);
      showToast('Copied & loaded into analyzer!');
    });
  });
}

// ── Export Security Audit Report ──────────────────────
const exportReportBtn = document.getElementById('exportReportBtn');
if (exportReportBtn) {
  exportReportBtn.addEventListener('click', async () => {
    const pw = pwInput.value.trim();
    if (!pw) {
      flashInput();
      showToast('Please enter a password first!');
      return;
    }

    exportReportBtn.textContent = '⏳ Exporting…';
    try {
      const res = await fetch('/export_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: pw }),
      });
      const data = await res.json();
      if (data.report) {
        const blob = new Blob([data.report], { type: 'text/markdown' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = data.filename || 'PassShield_Audit_Report.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Audit Report downloaded!');
      } else {
        showToast('Failed to export report.');
      }
    } catch (err) {
      showToast('Export failed.');
    } finally {
      exportReportBtn.textContent = '📄 Export Report';
    }
  });
}

function showSuggestError(msg) {
  suggestGrid.innerHTML = `
    <div class="suggest-placeholder glass-card">
      <span>⚠️</span>
      <p>${escHtml(msg)}</p>
    </div>
  `;
}

function setImproveLoading(loading) {
  improveBtn.disabled     = loading;
  improveBtnTxt.textContent = loading ? 'Generating…' : 'Generate Suggestions';
  improveSpinner.classList.toggle('hidden', !loading);
}

function flashInput() {
  pwInput.style.transition = 'border-color 0.1s';
  const wrapper = pwInput.closest('.input-wrapper');
  wrapper.style.boxShadow = '0 0 0 3px rgba(255, 77, 109, 0.35)';
  wrapper.style.borderColor = 'rgba(255, 77, 109, 0.7)';
  setTimeout(() => {
    wrapper.style.boxShadow = '';
    wrapper.style.borderColor = '';
  }, 800);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
