/**
 * main.js
 * -------
 * Entry point: wires the router, the analysis console, and the model-status
 * poller.
 *
 * The console keeps its own small state object rather than re-reading the DOM,
 * because the same values feed three places (the drop zones, the submit
 * button's enabled state, and the request body) and keeping one source of
 * truth is cheaper than keeping three in sync.
 */

import { animate, stagger } from './anime.esm.min.js';
import { Router, playOrbitalScan } from './router.js';

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  modes: [],
  mode: null,
  files: [null, null],
  bandPresets: [],
  running: false,
};

const $ = (id) => document.getElementById(id);

/* ====================================================== model status ==== */

const TELEMETRY_STATES = {
  ready:    { cls: 'state-ready',   text: 'READY' },
  loading:  { cls: 'state-loading', text: 'LOADING…' },
  idle:     { cls: 'state-loading', text: 'NOT LOADED' },
  error:    { cls: 'state-error',   text: 'UNAVAILABLE' },
  disabled: { cls: 'state-error',   text: 'DISABLED' },
};

let pollTimer = null;

async function pollModel() {
  try {
    const response = await fetch('/api/model');
    const status = await response.json();
    renderModel(status);
    // Poll fast while weights are loading, then back off to a slow heartbeat.
    const next = status.state === 'loading' ? 2000 : 15000;
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollModel, next);
  } catch (error) {
    renderModel({ state: 'error', message: 'Backend unreachable.', progress: [], warnings: [] });
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollModel, 8000);
  }
}

function renderModel(status) {
  const preset = TELEMETRY_STATES[status.state] || TELEMETRY_STATES.error;
  const strip = $('model-state');
  strip.className = preset.cls;
  strip.innerHTML = `<i class="pip"></i>MODEL <b>${preset.text}</b>`;

  const config = status.config || {};
  $('model-repo').textContent = shorten(status.model_id || config.model_id || '—', 34);
  $('model-device').textContent = (status.device || config.device || '—').toUpperCase();

  // Model page detail
  if ($('m-state')) {
    $('m-state').textContent = preset.text;
    $('m-repo').textContent = status.model_id || '—';
    $('m-base').textContent = status.base_model_id || '—';
    $('m-device').textContent = status.device || '—';
    $('m-dtype').textContent = config.dtype
      ? `${config.dtype}${config.load_in_4bit ? ' · 4-bit quantised' : ''}`
      : '—';
    $('m-token').textContent = config.token === 'set'
      ? 'set (never displayed)'
      : 'not set';
    $('m-load').textContent = status.load_seconds
      ? `${status.load_seconds.toFixed(0)}s`
      : '—';
    $('m-message').textContent = status.message || '—';

    const lines = [...(status.progress || [])];
    (status.warnings || []).forEach((warning) => lines.push(`! ${warning}`));
    $('m-progress').textContent = lines.length ? lines.join('\n') : 'No activity yet.';
  }
}

function shorten(text, max) {
  const value = String(text || '');
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

/* ========================================================== console ===== */

async function loadOptions() {
  const response = await fetch('/api/options');
  const data = await response.json();
  state.modes = data.modes;
  state.bandPresets = data.band_presets;

  const select = $('band-preset');
  select.innerHTML = state.bandPresets
    .map((name, index) => `<option value="${index === 0 ? '' : escapeAttr(name)}">${escapeHtml(name)}</option>`)
    .join('');

  $('modes').innerHTML = state.modes
    .map((mode) => `
      <button type="button" class="mode" data-mode="${escapeAttr(mode.id)}"
              aria-pressed="false">
        ${escapeHtml(mode.name)}
        <span>${mode.images} image${mode.images > 1 ? 's' : ''}</span>
      </button>`)
    .join('');

  $('modes').addEventListener('click', (event) => {
    const button = event.target.closest('.mode');
    if (button) selectMode(button.dataset.mode);
  });

  selectMode(state.modes[0].id);
}

function selectMode(id) {
  const mode = state.modes.find((entry) => entry.id === id);
  if (!mode) return;
  state.mode = mode;
  state.files = [null, null];

  document.querySelectorAll('.mode').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.mode === id));
  });

  $('mode-blurb').textContent = mode.blurb;

  $('drops').innerHTML = mode.labels
    .map((label, index) => `
      <div class="drop is-empty" data-slot="${index}" data-required="true">
        <input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" aria-label="${escapeAttr(label)}">
        <div class="drop-body">
          <div>${escapeHtml(label)}</div>
          <div class="drop-name">drop a file or click to browse</div>
        </div>
      </div>`)
    .join('');

  $('drops').querySelectorAll('.drop').forEach(wireDropZone);

  // Chips show a trimmed label so several fit on a line, but insert the full
  // question. The complete text stays available as the tooltip.
  $('examples').innerHTML = mode.examples
    .map((text) => `<button type="button" class="chip" title="${escapeAttr(text)}"
            data-query="${escapeAttr(text)}">${escapeHtml(shorten(text, 30))}</button>`)
    .join('');
  $('examples').querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $('query').value = chip.dataset.query;
      $('query').focus();
    });
  });

  if (!REDUCED) {
    animate($('drops').querySelectorAll('.drop'), {
      opacity: [0, 1],
      translateY: [8, 0],
      duration: 280,
      delay: stagger(60),
      ease: 'out(3)',
    });
  }
}

function wireDropZone(zone) {
  const slot = Number(zone.dataset.slot);
  const input = zone.querySelector('input');

  const accept = (file) => {
    if (!file) return;
    state.files[slot] = file;
    zone.classList.remove('is-empty');

    zone.querySelector('.drop-name').textContent = shorten(file.name, 28);
    // Preview only what the browser can decode; a GeoTIFF will not render, and
    // failing silently is better than showing a broken-image icon.
    const existing = zone.querySelector('img');
    if (existing) existing.remove();
    if (/\.(png|jpe?g)$/i.test(file.name)) {
      const preview = document.createElement('img');
      preview.alt = '';
      preview.src = URL.createObjectURL(file);
      preview.addEventListener('load', () => URL.revokeObjectURL(preview.src), { once: true });
      zone.prepend(preview);
    }
  };

  input.addEventListener('change', () => accept(input.files[0]));

  ['dragenter', 'dragover'].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add('is-over');
    }));

  ['dragleave', 'drop'].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove('is-over');
    }));

  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      input.files = event.dataTransfer.files;
      accept(file);
    }
  });
}

/* ========================================================== analysis ==== */

async function runAnalysis(event) {
  event.preventDefault();
  if (state.running) return;

  const needed = state.mode.images;
  const missing = state.files.slice(0, needed).some((file) => !file);
  if (missing) {
    showResult({
      success: false,
      answer: needed > 1
        ? 'This analysis requires two images. Add both before running.'
        : 'Add a satellite image before running the analysis.',
      measurements: [], trace: [], evidence: null,
    });
    return;
  }

  const body = new FormData();
  body.append('mode', state.mode.id);
  body.append('query', $('query').value.trim());
  body.append('band_preset', $('band-preset').value);
  body.append('image_a', state.files[0]);
  if (needed > 1) body.append('image_b', state.files[1]);

  setRunning(true);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body });
    showResult(await response.json());
  } catch (error) {
    showResult({
      success: false,
      answer: `Could not reach the backend: ${error.message}`,
      measurements: [], trace: [], evidence: null,
    });
  } finally {
    setRunning(false);
  }
}

function setRunning(running) {
  state.running = running;
  $('run').disabled = running;
  $('run-label').textContent = running ? 'Analysing…' : 'Analyze';
  $('results-panel').classList.toggle('scanning', running);
  if (running) $('results-sub').textContent = 'Running the pipeline…';
}

function showResult(result) {
  $('results-empty').hidden = true;
  $('results-body').hidden = false;

  const confidenceLabel = result.confidence == null
    ? 'not reported'
    : `${Number(result.confidence).toFixed(2)}${
        result.confidence_kind === 'generation' ? ' (generation)' :
        result.confidence_kind === 'heuristic' ? ' (heuristic)' : ''}`;

  $('verdict').innerHTML = result.success
    ? [
        ['task', result.task],
        ['workflow', result.scenario],
        ['confidence', confidenceLabel],
        ['answered by', result.model_source || '—'],
        ['elapsed', `${result.elapsed_seconds ?? '—'}s`],
      ].map(([key, value]) =>
        `<span><i>${key}</i> <b>${escapeHtml(String(value))}</b></span>`).join('')
    : `<span><i>status</i> <b style="color:var(--bad)">rejected</b></span>`;

  // The API answer is self-contained so `curl` users get everything in one
  // string. In the browser the measured values have their own panel, so the
  // duplicate block is stripped here rather than shown twice.
  $('answer').textContent = stripMeasuredBlock(result.answer || '');
  $('results-sub').textContent = result.success
    ? 'Model output and measured values, with sources labelled.'
    : 'The request was not run.';

  $('labels').innerHTML = (result.model_labels || [])
    .map((label) => `<span class="label-pill">${escapeHtml(label)}</span>`)
    .join('');

  const evidencePanel = $('evidence-panel');
  if (result.evidence) {
    $('evidence').src = result.evidence;
    $('evidence-sub').textContent = evidenceCaption(result.scenario);
    evidencePanel.hidden = false;
  } else {
    evidencePanel.hidden = true;
  }

  const detailPanel = $('detail-panel');
  const measurements = result.measurements || [];
  if (measurements.length || (result.trace || []).length) {
    $('measurements').textContent = measurements.length
      ? measurements.join('\n')
      : 'No measurements were produced for this request.';
    $('trace').innerHTML = (result.trace || [])
      .map((step) => {
        const symbol = { ok: '✓', warn: '!', fail: '✗', info: '·' }[step.status] || '·';
        const timing = step.elapsed_ms >= 1 ? ` (${Math.round(step.elapsed_ms)} ms)` : '';
        const detail = step.detail
          ? `<span class="detail">${escapeHtml(step.detail)}</span>` : '';
        return `<li class="${step.status}"><span class="sym">${symbol}</span>
                  <span>${escapeHtml(step.label)}${timing}${detail}</span></li>`;
      })
      .join('');
    detailPanel.hidden = false;
  } else {
    detailPanel.hidden = true;
  }

  if (!REDUCED) {
    const panels = [$('results-body'), evidencePanel, detailPanel]
      .filter((node) => node && !node.hidden);
    animate(panels, {
      opacity: [0, 1],
      translateY: [12, 0],
      duration: 380,
      delay: stagger(70),
      ease: 'out(3)',
    });
  }
}

/**
 * Remove the "MEASURED FROM THE IMAGE" section from an answer string.
 * Sections are separated by a blank line, so splitting on that and dropping
 * the one block is safer than a regex over the whole body.
 */
function stripMeasuredBlock(answer) {
  return answer
    .split('\n\n')
    .filter((block) => !block.startsWith('MEASURED FROM THE IMAGE'))
    .join('\n\n')
    .trim();
}

function evidenceCaption(scenario) {
  return {
    single: 'Original scene beside its measured land-cover classification.',
    bi_temporal: 'T1, T2, the difference magnitude, and the change mask on T2.',
    cross_modal: 'Optical, SAR, radar structure, and the fused detection.',
  }[scenario] || '';
}

/* ============================================================ utils ===== */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/`/g, '&#96;');
}

/* ============================================================= boot ===== */

document.addEventListener('DOMContentLoaded', async () => {
  const router = new Router({
    pageSelector: '.page',
    linkSelector: '[data-link]',
    onEnter: (path) => {
      if (path === '/model') pollModel();
    },
  });
  router.start();

  playOrbitalScan();
  pollModel();

  try {
    await loadOptions();
  } catch (error) {
    $('mode-blurb').textContent =
      'Could not load workflow options from the backend. Is the server running?';
  }

  $('analysis-form').addEventListener('submit', runAnalysis);
  $('reload-model')?.addEventListener('click', async () => {
    await fetch('/api/model/load', { method: 'POST' });
    pollModel();
  });
});
