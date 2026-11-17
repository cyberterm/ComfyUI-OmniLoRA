/**
 * ComfyUI-OmniLoRA: TensorBoard-Grade Real-Time Loss & Telemetry Dashboard (`omni_lora_tracker.js`).
 *
 * Key Capabilities & Anti-Bug Architecture:
 * - TensorBoard Debiased EMA Smoothing (`S_t / (1 - w^(t+1))`) + optional +-1 sigma rolling volatility band.
 * - TensorBoard Outlier-Resistant Auto-Scaling ("Ignore Outliers" [P05-P95 + EMA framing] by default)
 *    plus 3 Y-Axis Fit Modes (`Auto Clip`, `EMA Focus`, `Full Raw`) and span-proportional headroom
 *   so diffusion timestep spikes or narrow loss bands NEVER flatten the curve.
 * - 3 X-Axis Modes: `Step (Auto-Fit)` (never squishes early steps to the left when max_steps is large),
 *   `Step (0..Max Target)`, and `Relative Time (s/m)`.
 * - Interactive Box-Drag Zoom, Wheel Zoom, Shift-Drag Pan, and 1-Click Reset Zoom.
 * - Transform-aware pointer coordinate math that stays 100% aligned at any ComfyUI canvas zoom (`ds.scale`)
 *   and dynamically adapts DOMWidget height to the LiteGraph node size so the plot never collapses.
 * - Cloudflare Tunnel (`*.trycloudflare.com`) & Reverse-Proxy Resilient Sync:
 *   multi-endpoint fallback fetch, cache-busting query/headers (`cache: "no-store"`), unconditional
 *   periodic delta polling (recovers even if initial WebSocket events were dropped by Cloudflare),
 *   upstream graph link traversal across Reroute nodes, and ComfyUI `onExecuted` UI payload ingestion.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getComfyApp() {
  return window.comfyAPI?.app?.app || app;
}

function getComfyApi() {
  return window.comfyAPI?.api?.api || api;
}

function getGraphZoomScale() {
  const cApp = getComfyApp();
  const s = Number(cApp?.canvas?.ds?.scale);
  return Number.isFinite(s) && s > 0.05 ? s : 1.0;
}

// TensorBoard-inspired high-contrast run palette
const PALETTE = [
  "#00e5ff", // 0: Cyan (Primary Active)
  "#ff9100", // 1: TensorBoard Orange
  "#00e676", // 2: Emerald Green
  "#f50057", // 3: Crimson Rose
  "#b388ff", // 4: Violet Lavender
  "#ffd600", // 5: Amber Gold
  "#40c4ff", // 6: Sky Blue
  "#69f0ae", // 7: Mint Teal
];

function ensureTrackerStyles() {
  const STYLE_ID = "omni-lora-tracker-styles-v2";
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .omni-tracker-root {
      display: flex;
      flex-direction: column;
      width: 100%;
      height: 100%;
      min-height: 240px;
      box-sizing: border-box;
      background: #0d1017;
      border: 1px solid #252a3a;
      border-radius: 6px;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      color: #e2e8f0;
      user-select: none;
    }

    .omni-tracker-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 5px 9px;
      background: #141824;
      border-bottom: 1px solid #222738;
      font-size: 11px;
      flex-shrink: 0;
    }

    .omni-tracker-status {
      display: flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      min-width: 0;
      flex: 1;
    }

    .omni-tracker-title-text {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: #f1f5f9;
    }

    .omni-tracker-status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #64748b;
      flex-shrink: 0;
      transition: background 0.25s, box-shadow 0.25s;
    }
    .omni-tracker-status-dot.active {
      background: #00e5ff;
      box-shadow: 0 0 8px rgba(0, 229, 255, 0.75);
    }
    .omni-tracker-status-dot.completed {
      background: #00e676;
      box-shadow: 0 0 6px rgba(0, 230, 118, 0.45);
    }
    .omni-tracker-status-dot.interrupted {
      background: #ff5252;
    }

    .omni-tracker-header-actions {
      display: flex;
      align-items: center;
      gap: 4px;
      flex-shrink: 0;
    }

    .omni-tracker-btn {
      background: #1e2436;
      border: 1px solid #323b54;
      color: #cbd5e1;
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 10px;
      font-weight: 500;
      cursor: pointer;
      line-height: 1.3;
      white-space: nowrap;
      transition: background 0.12s, color 0.12s, border-color 0.12s;
    }
    .omni-tracker-btn:hover {
      background: #2a324b;
      color: #ffffff;
      border-color: #4b587c;
    }
    .omni-tracker-btn.active {
      background: #112a3d;
      color: #38bdf8;
      border-color: #0284c7;
    }
    .omni-tracker-btn.warn {
      background: #2b1d12;
      color: #fbbf24;
      border-color: #d97706;
    }

    .omni-tracker-kpi-strip {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 4px;
      padding: 4px 9px;
      background: #111520;
      border-bottom: 1px solid #1e2333;
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      flex-shrink: 0;
    }

    .omni-tracker-kpi-cell {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .omni-tracker-kpi-label {
      font-size: 9px;
      color: #64748b;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .omni-tracker-kpi-val {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 10.5px;
      font-weight: 600;
      color: #e2e8f0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .omni-tracker-canvas-wrap {
      position: relative;
      width: 100%;
      flex: 1 1 170px;
      min-height: 145px;
      background: #090b10;
      cursor: crosshair;
      overflow: hidden;
      touch-action: none;
    }

    .omni-tracker-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: block;
    }

    .omni-tracker-tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      background: rgba(13, 16, 24, 0.95);
      border: 1px solid #334155;
      border-radius: 5px;
      padding: 6px 8px;
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      line-height: 1.4;
      color: #f1f5f9;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.7);
      z-index: 20;
      white-space: nowrap;
      max-width: 280px;
    }

    .omni-tracker-controls {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 5px 9px;
      background: #121622;
      border-top: 1px solid #202536;
      font-size: 10px;
      flex-shrink: 0;
    }

    .omni-tracker-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      flex-wrap: wrap;
    }

    .omni-tracker-slider-wrap {
      display: flex;
      align-items: center;
      gap: 5px;
      flex: 1 1 110px;
      min-width: 105px;
    }
    .omni-tracker-slider {
      flex: 1;
      height: 4px;
      accent-color: #00e5ff;
      cursor: pointer;
      min-width: 45px;
    }

    .omni-tracker-select {
      background: #1e2436;
      border: 1px solid #323b54;
      color: #cbd5e1;
      border-radius: 4px;
      padding: 1.5px 5px;
      font-size: 10px;
      cursor: pointer;
      max-width: 115px;
    }

    .omni-tracker-history-drawer {
      display: none;
      flex-direction: column;
      gap: 4px;
      padding: 6px 9px;
      background: #0c0e16;
      border-top: 1px solid #222738;
      font-size: 10px;
      max-height: 125px;
      overflow-y: auto;
      flex-shrink: 0;
    }
    .omni-tracker-history-drawer.open {
      display: flex;
    }

    .omni-tracker-run-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 2px 4px;
      border-radius: 3px;
      font-variant-numeric: tabular-nums;
    }
    .omni-tracker-run-item:hover {
      background: #161b29;
    }

    .omni-tracker-run-label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      min-width: 0;
      flex: 1;
    }

    .omni-tracker-modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(3, 5, 10, 0.82);
      backdrop-filter: blur(4px);
      z-index: 10000;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .omni-tracker-modal-box {
      width: 1040px;
      max-width: 96vw;
      height: 680px;
      max-height: 92vh;
      background: #0d1017;
      border: 1px solid #334155;
      border-radius: 8px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #e2e8f0;
    }
  `;
  document.head.appendChild(style);
}

/**
 * Determine how many initial warmup points to skip when computing Min-EMA markers
 * and EMA-focused Y-axis bounds so step-1 single-timestep draws never skew long runs.
 */
function getEmaWarmupSkip(pointCount) {
  if (pointCount > 40) return Math.min(Math.floor(pointCount * 0.04), 20);
  if (pointCount > 12) return 3;
  if (pointCount > 5) return 1;
  return 0;
}

/**
 * TensorBoard Debiased Exponential Moving Average (EMA) with initial window prior stabilization.
 * Prevents single-step extreme timestep draws at step 1-2 from creating artificial vertical
 * spikes/dips when high smoothing (e.g. 0.95 - 0.995) is used on stochastic diffusion losses.
 */
function computeEma(values, smoothing) {
  if (!Array.isArray(values) || values.length === 0) return [];
  const s = Math.max(0.0, Math.min(0.995, Number(smoothing) || 0));
  const ema = new Array(values.length);
  if (s <= 0.001) {
    let lastValid = 0;
    for (let i = 0; i < values.length; i++) {
      const v = Number(values[i]);
      if (Number.isFinite(v)) lastValid = v;
      ema[i] = lastValid;
    }
    return ema;
  }

  // Compute robust trimmed mean of the initial window to stabilize step 1-5 prior
  const initWinLen = Math.min(values.length, Math.max(4, Math.min(16, Math.round(0.6 / (1.0 - s)))));
  const initValid = [];
  for (let i = 0; i < initWinLen; i++) {
    const v = Number(values[i]);
    if (Number.isFinite(v)) initValid.push(v);
  }

  let priorMean = initValid.length > 0 ? initValid[0] : 0;
  if (initValid.length >= 4) {
    const sortedInit = initValid.slice().sort((a, b) => a - b);
    const trim = sortedInit.length >= 6 ? 1 : 0;
    let sum = 0;
    let cnt = 0;
    for (let i = trim; i < sortedInit.length - trim; i++) {
      sum += sortedInit[i];
      cnt++;
    }
    if (cnt > 0) priorMean = sum / cnt;
  } else if (initValid.length > 1) {
    priorMean = initValid.reduce((a, b) => a + b, 0) / initValid.length;
  }

  const priorSteps =
    initValid.length >= 3 && s >= 0.6
      ? Math.min(initValid.length - 1, Math.max(2, Math.min(8, Math.round(0.35 / (1.0 - s)))))
      : 0;
  let weightSum = priorSteps * (1.0 - s);
  let accum = weightSum * priorMean;
  let lastValid = Number.isFinite(Number(values[0])) ? Number(values[0]) : priorMean;

  for (let i = 0; i < values.length; i++) {
    const raw = Number(values[i]);
    const cur = Number.isFinite(raw) ? raw : lastValid;
    lastValid = cur;
    accum = s * accum + (1.0 - s) * cur;
    weightSum = s * weightSum + (1.0 - s);
    ema[i] = weightSum > 1e-12 ? accum / weightSum : cur;
  }
  return ema;
}

/**
 * Compute rolling local standard deviation around the EMA curve for TensorBoard +-1 sigma shading.
 */
function computeRollingStd(rawValues, emaValues, smoothing) {
  if (!Array.isArray(rawValues) || rawValues.length === 0) return [];
  const s = Math.max(0.5, Math.min(0.98, Number(smoothing) || 0.85));
  const stdOut = new Array(rawValues.length);
  let varAccum = 0;
  let weightPow = 1.0;
  for (let i = 0; i < rawValues.length; i++) {
    const r = Number(rawValues[i]);
    const m = Number(emaValues[i]);
    const diff = Number.isFinite(r) && Number.isFinite(m) ? r - m : 0;
    varAccum = s * varAccum + (1.0 - s) * (diff * diff);
    weightPow *= s;
    const debias = 1.0 - weightPow;
    const variance = debias > 1e-12 ? varAccum / debias : 0;
    stdOut[i] = Math.sqrt(Math.max(0, variance));
  }
  return stdOut;
}

/**
 * Compute quantile of a numeric array (p in [0, 1]).
 */
function computeQuantile(sortedNums, p) {
  if (!sortedNums || sortedNums.length === 0) return 0;
  if (sortedNums.length === 1) return sortedNums[0];
  const idx = Math.max(0, Math.min(sortedNums.length - 1, p * (sortedNums.length - 1)));
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sortedNums[lo] * (1 - frac) + sortedNums[hi] * frac;
}

/**
 * Format axis numbers with dynamic precision based on axis span so tick labels
 * are never identical (e.g. `0.125, 0.125, 0.125`) and never `0.000` for small losses.
 */
function formatAxisValue(val, span) {
  const num = Number(val);
  if (!Number.isFinite(num)) return "--";
  if (num === 0) return "0";
  const absVal = Math.abs(num);
  if (absVal < 1e-4 || absVal >= 1e4) {
    return num.toExponential(2);
  }
  const cleanSpan = Number.isFinite(span) && span > 0 ? span : absVal * 0.1;
  const stepApprox = cleanSpan / 4;
  let decimals = 3;
  if (stepApprox > 0) {
    decimals = Math.max(2, Math.min(6, Math.ceil(-Math.log10(stepApprox)) + 1));
  }
  return num.toFixed(decimals);
}

class OmniLossTrackerWidget {
  constructor(node, options = {}) {
    this.node = node;
    this.options = options;

    this.runs = [];
    this.activeRunId = null;

    // TensorBoard View & Scaling Controls
    this.smoothing = 0.85;
    this.noiseBin = "all"; // 'all', 'high', 'mid', 'low'
    this.selectedBucket = "all";
    this.scaleMode = "linear"; // 'linear' | 'log'
    this.yFitMode = "auto"; // 'auto' (TensorBoard Outlier-Free) | 'ema' (EMA Focus) | 'full' (Raw Min-Max)
    this.ignoreOutliers = true;
    this.xAxisMode = "step"; // 'step' (Auto-fit observed steps) | 'target' (0..max_steps) | 'relative' (seconds)
    this.showRaw = true;
    this.showEma = true;
    this.showBand = true; // TensorBoard +-1 sigma volatility envelope
    this.showLr = true;

    // Interactive Zoom / Pan viewport override (null = auto-fit)
    this.zoomDomain = null; // { xMin, xMax, yMin, yMax }
    this._dragState = null; // { mode: 'zoom' | 'pan', startX, startY, curX, curY, initialDomain }

    // Available buckets & dismissed runs
    this.knownBuckets = new Set();
    this._dismissedRunIds = new Set();

    // Idle / sleep / Cloudflare / WebSocket reconnect catch-up state
    this._lastWsMessageAt = 0;
    this._lastSyncAt = 0;
    this._syncPromise = null;
    this._pendingForceSync = false;
    this._gapSyncTimer = null;
    this._workflowActive = false;
    this._modalCanvas = null;
    this._modalTooltip = null;
    this._hoverXVal = null;

    // Root DOM Element
    this.element = document.createElement("div");
    this.element.className = "omni-tracker-root";

    ensureTrackerStyles();
    this.buildUI();
    this.loadState();

    this.bindServerEvents();

    // Immediately reconcile with server buffer (handles Cloudflare page refresh or mid-run open)
    this.syncFromServer({ force: true });

    // Prevent LiteGraph canvas drag / zoom when interacting with widget controls
    const stopProp = (e) => e.stopPropagation();
    this.element.addEventListener("pointerdown", stopProp);
    this.element.addEventListener("mousedown", stopProp);
    this.element.addEventListener("wheel", stopProp);
  }

  buildUI() {
    this.element.innerHTML = `
      <div class="omni-tracker-header">
        <div class="omni-tracker-status">
          <span class="omni-tracker-status-dot"></span>
          <span class="omni-tracker-title-text">OmniLoRA TensorBoard Tracker</span>
        </div>
        <div class="omni-tracker-header-actions">
          <button class="omni-tracker-btn omni-tracker-btn-reset-zoom" style="display:none;" title="Reset Zoom / Auto-Fit Axes (Double-Click Chart)">Reset Zoom</button>
          <button class="omni-tracker-btn omni-tracker-btn-ckpt" title="Request Immediate Checkpoint Save at Current Step (Saves to models/loras/)">Save Ckpt</button>
          <button class="omni-tracker-btn omni-tracker-btn-sync" title="Force Sync Telemetry from Server (Cloudflare / Proxy Recovery)">↻ Sync</button>
          <button class="omni-tracker-btn omni-tracker-btn-history" title="Toggle Multi-Run Comparison & Export">Runs (0)</button>
          <button class="omni-tracker-btn omni-tracker-btn-clear-top" title="Delete All Training Runs & Reset Tracker" style="color:#f87171;">Clear</button>
          <button class="omni-tracker-btn omni-tracker-btn-popout" title="Open Enlarged TensorBoard Inspection Modal">⛶</button>
        </div>
      </div>

      <div class="omni-tracker-kpi-strip">
        <div class="omni-tracker-kpi-cell">
          <span class="omni-tracker-kpi-label">Progress</span>
          <span class="omni-tracker-kpi-val omni-tracker-stat-step">0 / 0</span>
        </div>
        <div class="omni-tracker-kpi-cell">
          <span class="omni-tracker-kpi-label">EMA Loss (Raw)</span>
          <span class="omni-tracker-kpi-val omni-tracker-stat-loss">--</span>
        </div>
        <div class="omni-tracker-kpi-cell">
          <span class="omni-tracker-kpi-label">Min EMA (Slope)</span>
          <span class="omni-tracker-kpi-val omni-tracker-stat-min">--</span>
        </div>
        <div class="omni-tracker-kpi-cell">
          <span class="omni-tracker-kpi-label">Learning Rate</span>
          <span class="omni-tracker-kpi-val omni-tracker-stat-lr">--</span>
        </div>
        <div class="omni-tracker-kpi-cell">
          <span class="omni-tracker-kpi-label">Throughput / ETA</span>
          <span class="omni-tracker-kpi-val omni-tracker-stat-speed">--</span>
        </div>
      </div>

      <div class="omni-tracker-canvas-wrap">
        <canvas class="omni-tracker-canvas"></canvas>
        <div class="omni-tracker-tooltip"></div>
      </div>

      <div class="omni-tracker-controls">
        <div class="omni-tracker-row">
          <div class="omni-tracker-slider-wrap" title="TensorBoard Debiased EMA Smoothing Weight (0.00 to 0.99)">
            <span style="color:#94a3b8;">Smooth:</span>
            <input type="range" class="omni-tracker-slider" min="0" max="0.99" step="0.01" value="0.85">
            <span class="omni-tracker-smooth-val" style="font-family:monospace; min-width:26px;">0.85</span>
          </div>
          <div style="display:flex; gap:3px; align-items:center;">
            <button class="omni-tracker-btn omni-tracker-btn-yfit active" title="Y-Axis Scaling Mode: Auto (Ignore Outliers) -> EMA Focus -> Full Raw">Fit: Clip Outliers</button>
            <button class="omni-tracker-btn omni-tracker-btn-scale" title="Toggle Linear vs Logarithmic Y-Axis">Linear</button>
            <button class="omni-tracker-btn omni-tracker-btn-band active" title="Toggle +-1 Sigma Rolling Volatility Band">±1σ</button>
            <button class="omni-tracker-btn omni-tracker-btn-raw active" title="Toggle Raw Unsmoothed Step Scatter">Raw</button>
            <button class="omni-tracker-btn omni-tracker-btn-lr active" title="Toggle Learning Rate Schedule Overlay">LR</button>
          </div>
        </div>

        <div class="omni-tracker-row">
          <div style="display:flex; gap:3px; align-items:center;">
            <button class="omni-tracker-btn omni-tracker-bin-btn active" data-bin="all" title="All Timesteps [0.0 - 1.0]">All t</button>
            <button class="omni-tracker-btn omni-tracker-bin-btn" data-bin="high" title="High Noise Timesteps [0.67 - 1.0]">High t</button>
            <button class="omni-tracker-btn omni-tracker-bin-btn" data-bin="mid" title="Mid Noise Timesteps [0.33 - 0.67]">Mid t</button>
            <button class="omni-tracker-btn omni-tracker-bin-btn" data-bin="low" title="Low Noise Timesteps [0.0 - 0.33]">Low t</button>
          </div>
          <div style="display:flex; gap:4px; align-items:center;">
            <button class="omni-tracker-btn omni-tracker-btn-xaxis" title="Switch Horizontal Axis: Auto-Fit Steps -> Target Max Steps -> Relative Time">X: Fit Steps</button>
            <select class="omni-tracker-select omni-tracker-bucket-select" title="Filter by Aspect Ratio Resolution Bucket">
              <option value="all">All Buckets</option>
            </select>
          </div>
        </div>
      </div>

      <div class="omni-tracker-history-drawer">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
          <span style="font-weight:600; color:#94a3b8;">Training Runs (Overlay Comparison)</span>
          <div style="display:flex; gap:4px;">
            <button class="omni-tracker-btn omni-tracker-btn-export-csv" title="Export Active Run Telemetry as CSV">CSV</button>
            <button class="omni-tracker-btn omni-tracker-btn-export-png" title="Export High-Res Chart Snapshot as PNG">PNG</button>
            <button class="omni-tracker-btn omni-tracker-btn-clear" title="Clear Inactive Run History" style="color:#f87171;">Clear</button>
          </div>
        </div>
        <div class="omni-tracker-run-list"></div>
      </div>
    `;

    this.canvasWrap = this.element.querySelector(".omni-tracker-canvas-wrap");
    this.canvas = this.element.querySelector(".omni-tracker-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.tooltip = this.element.querySelector(".omni-tracker-tooltip");

    this.statusDot = this.element.querySelector(".omni-tracker-status-dot");
    this.titleText = this.element.querySelector(".omni-tracker-title-text");
    this.statStep = this.element.querySelector(".omni-tracker-stat-step");
    this.statLoss = this.element.querySelector(".omni-tracker-stat-loss");
    this.statMin = this.element.querySelector(".omni-tracker-stat-min");
    this.statLr = this.element.querySelector(".omni-tracker-stat-lr");
    this.statSpeed = this.element.querySelector(".omni-tracker-stat-speed");

    this.slider = this.element.querySelector(".omni-tracker-slider");
    this.smoothVal = this.element.querySelector(".omni-tracker-smooth-val");
    this.yFitBtn = this.element.querySelector(".omni-tracker-btn-yfit");
    this.scaleBtn = this.element.querySelector(".omni-tracker-btn-scale");
    this.bandBtn = this.element.querySelector(".omni-tracker-btn-band");
    this.rawBtn = this.element.querySelector(".omni-tracker-btn-raw");
    this.lrBtn = this.element.querySelector(".omni-tracker-btn-lr");
    this.xAxisBtn = this.element.querySelector(".omni-tracker-btn-xaxis");
    this.resetZoomBtn = this.element.querySelector(".omni-tracker-btn-reset-zoom");
    this.ckptBtn = this.element.querySelector(".omni-tracker-btn-ckpt");
    this.syncBtn = this.element.querySelector(".omni-tracker-btn-sync");
    this.bucketSelect = this.element.querySelector(".omni-tracker-bucket-select");
    this.historyBtn = this.element.querySelector(".omni-tracker-btn-history");
    this.clearTopBtn = this.element.querySelector(".omni-tracker-btn-clear-top");
    this.historyDrawer = this.element.querySelector(".omni-tracker-history-drawer");
    this.runList = this.element.querySelector(".omni-tracker-run-list");

    this.bindUIEvents();
    this.resizeCanvas();
  }

  syncControlButtonsUI() {
    if (this.yFitBtn) {
      const labels = {
        auto: "Fit: Clip Outliers",
        ema: "Fit: EMA Focus",
        full: "Fit: Full Raw",
      };
      this.yFitBtn.textContent = labels[this.yFitMode] || "Fit: Clip Outliers";
      this.yFitBtn.classList.toggle("active", this.yFitMode !== "full");
    }
    if (this.scaleBtn) {
      this.scaleBtn.textContent = this.scaleMode === "log" ? "Log10" : "Linear";
      this.scaleBtn.classList.toggle("active", this.scaleMode === "log");
    }
    if (this.bandBtn) {
      this.bandBtn.classList.toggle("active", Boolean(this.showBand));
    }
    if (this.rawBtn) {
      this.rawBtn.classList.toggle("active", Boolean(this.showRaw));
    }
    if (this.lrBtn) {
      this.lrBtn.classList.toggle("active", Boolean(this.showLr));
    }
    if (this.xAxisBtn) {
      const xLabels = {
        step: "X: Fit Steps",
        target: "X: 0..Max",
        relative: "X: Wall Time",
      };
      this.xAxisBtn.textContent = xLabels[this.xAxisMode] || "X: Fit Steps";
      this.xAxisBtn.classList.toggle("active", this.xAxisMode !== "target");
    }
    if (this.resetZoomBtn) {
      this.resetZoomBtn.style.display = this.zoomDomain ? "inline-block" : "none";
      this.resetZoomBtn.classList.toggle("warn", Boolean(this.zoomDomain));
    }
  }

  bindUIEvents() {
    // Smoothing slider
    this.slider.addEventListener("input", (e) => {
      this.smoothing = parseFloat(e.target.value);
      this.smoothVal.textContent = this.smoothing.toFixed(2);
      const activeRun = this.getActiveRun();
      if (activeRun) this.updateStatsFromRun(activeRun);
      this.render();
    });

    // Y-Axis Fit Mode cycle: Auto (Clip Outliers) -> EMA Focus -> Full Raw
    this.yFitBtn.addEventListener("click", () => {
      const order = ["auto", "ema", "full"];
      const nextIdx = (order.indexOf(this.yFitMode) + 1) % order.length;
      this.yFitMode = order[nextIdx];
      this.ignoreOutliers = this.yFitMode !== "full";
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // Scale mode toggle (Linear vs Log10)
    this.scaleBtn.addEventListener("click", () => {
      this.scaleMode = this.scaleMode === "linear" ? "log" : "linear";
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // +-1 sigma volatility band toggle
    this.bandBtn.addEventListener("click", () => {
      this.showBand = !this.showBand;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // Raw scatter toggle
    this.rawBtn.addEventListener("click", () => {
      this.showRaw = !this.showRaw;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // LR overlay toggle
    this.lrBtn.addEventListener("click", () => {
      this.showLr = !this.showLr;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // X-Axis mode toggle: step (auto-fit) -> target (0..max_steps) -> relative (time)
    this.xAxisBtn.addEventListener("click", () => {
      const order = ["step", "target", "relative"];
      const nextIdx = (order.indexOf(this.xAxisMode) + 1) % order.length;
      this.xAxisMode = order[nextIdx];
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.saveStateThrottled();
      this.render();
    });

    // Reset Zoom button
    this.resetZoomBtn.addEventListener("click", () => {
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.render();
    });

    // Save Checkpoint Now button
    if (this.ckptBtn) {
      this.ckptBtn.addEventListener("click", async () => {
        const origText = this.ckptBtn.textContent;
        this.ckptBtn.textContent = "Saving...";
        this.ckptBtn.classList.add("active");
        try {
          const fetchFn =
            api && typeof api.fetchApi === "function"
              ? (url, options) => api.fetchApi(url, options)
              : fetch;
          await fetchFn(`/omni_lora/metrics?save_checkpoint=1&_ts=${Date.now()}`, { cache: "no-store" });
          this.ckptBtn.textContent = "Queued ✓";
        } catch (_err) {
          this.ckptBtn.textContent = "Queued";
        }
        setTimeout(() => {
          if (this.ckptBtn) {
            this.ckptBtn.textContent = origText;
            this.ckptBtn.classList.remove("active");
          }
        }, 2000);
      });
    }

    // Force Sync button (Cloudflare / proxy manual catch-up)
    this.syncBtn.addEventListener("click", () => {
      this.syncFromServer({ force: true });
      this.resizeCanvas();
      this.render();
    });

    // Noise bin tabs
    const binButtons = this.element.querySelectorAll(".omni-tracker-bin-btn");
    binButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        binButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.noiseBin = btn.dataset.bin;
        this.zoomDomain = null;
        this.syncControlButtonsUI();
        this.render();
      });
    });

    // Bucket dropdown
    this.bucketSelect.addEventListener("change", (e) => {
      this.selectedBucket = e.target.value;
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.render();
    });

    // Run history drawer toggle
    this.historyBtn.addEventListener("click", () => {
      this.historyDrawer.classList.toggle("open");
      this.historyBtn.classList.toggle("active", this.historyDrawer.classList.contains("open"));
      setTimeout(() => {
        this.resizeCanvas();
        this.render();
      }, 20);
    });

    // Export CSV
    this.element.querySelector(".omni-tracker-btn-export-csv").addEventListener("click", () => {
      this.exportCsv();
    });

    // Export PNG
    this.element.querySelector(".omni-tracker-btn-export-png").addEventListener("click", () => {
      this.exportPng();
    });

    // Clear all runs (both drawer Clear button and top header Clear button)
    this.element.querySelector(".omni-tracker-btn-clear").addEventListener("click", () => {
      this.clearAllRuns();
    });
    if (this.clearTopBtn) {
      this.clearTopBtn.addEventListener("click", () => {
        this.clearAllRuns();
      });
    }

    // Popout modal button
    this.element.querySelector(".omni-tracker-btn-popout").addEventListener("click", () => {
      this.openModal();
    });

    // Interactive Canvas Pointer / Zoom / Crosshair Events
    this.bindCanvasInteractions(this.canvasWrap, this.canvas, this.tooltip);

    // ResizeObserver + IntersectionObserver so canvas is always sharp even after tab/Cloudflare layout shifts
    if (window.ResizeObserver) {
      this._resizeObserver = new ResizeObserver(() => {
        this.resizeCanvas();
        this.render();
      });
      this._resizeObserver.observe(this.canvasWrap);
      this._resizeObserver.observe(this.element);
    }
    if (window.IntersectionObserver) {
      this._intersectionObserver = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          this.resizeCanvas();
          this.render();
        }
      });
      this._intersectionObserver.observe(this.element);
    }
  }

  /**
   * Convert screen pointer coordinates (`e.clientX`, `e.clientY`) into untransformed
   * canvas logical CSS pixels (`[0..clientW, 0..clientH]`), immune to LiteGraph `transform: scale(...)`.
   */
  getPointerCanvasCoords(e, wrapEl, targetCanvas) {
    const rect = wrapEl.getBoundingClientRect();
    const clientW = targetCanvas._cssW || wrapEl.clientWidth || 560;
    const clientH = targetCanvas._cssH || wrapEl.clientHeight || 220;
    const scaleX = rect.width > 1 ? clientW / rect.width : 1.0;
    const scaleY = rect.height > 1 ? clientH / rect.height : 1.0;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
      clientW,
      clientH,
      rect,
    };
  }

  bindCanvasInteractions(wrapEl, targetCanvas, tooltipEl) {
    wrapEl.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 && e.button !== 1) return;
      const coords = this.getPointerCanvasCoords(e, wrapEl, targetCanvas);
      const layout = targetCanvas._lastLayout;
      if (!layout) return;

      // Only initiate drag inside plot area
      if (
        coords.x < layout.pad.left ||
        coords.x > layout.pad.left + layout.plotW ||
        coords.y < layout.pad.top ||
        coords.y > layout.pad.top + layout.plotH
      ) {
        return;
      }

      try {
        wrapEl.setPointerCapture(e.pointerId);
      } catch (_err) {
        // Ignore pointer capture errors
      }

      const mode = e.shiftKey || e.button === 1 ? "pan" : "zoom";
      this._dragState = {
        pointerId: e.pointerId,
        mode,
        startX: coords.x,
        startY: coords.y,
        curX: coords.x,
        curY: coords.y,
        initialDomain: {
          xMin: layout.xMin,
          xMax: layout.xMax,
          yMin: layout.yMin,
          yMax: layout.yMax,
        },
      };
    });

    wrapEl.addEventListener("pointermove", (e) => {
      const coords = this.getPointerCanvasCoords(e, wrapEl, targetCanvas);
      const layout = targetCanvas._lastLayout;

      if (this._dragState && layout) {
        this._dragState.curX = Math.max(layout.pad.left, Math.min(layout.pad.left + layout.plotW, coords.x));
        this._dragState.curY = Math.max(layout.pad.top, Math.min(layout.pad.top + layout.plotH, coords.y));

        if (this._dragState.mode === "pan") {
          const dxFrac = (this._dragState.curX - this._dragState.startX) / Math.max(1, layout.plotW);
          const dyFrac = (this._dragState.curY - this._dragState.startY) / Math.max(1, layout.plotH);
          const init = this._dragState.initialDomain;
          const xSpan = init.xMax - init.xMin;
          const ySpan = init.yMax - init.yMin;
          this.zoomDomain = {
            xMin: Math.max(0, init.xMin - dxFrac * xSpan),
            xMax: Math.max(1e-3, init.xMax - dxFrac * xSpan),
            yMin: Math.max(0, init.yMin + dyFrac * ySpan),
            yMax: Math.max(1e-6, init.yMax + dyFrac * ySpan),
          };
          this.syncControlButtonsUI();
        }
        if (tooltipEl) tooltipEl.style.display = "none";
        this.render(targetCanvas);
        return;
      }

      this.handlePointerHover(coords, targetCanvas, tooltipEl);
    });

    const finishDrag = (e) => {
      if (!this._dragState) return;
      const drag = this._dragState;
      this._dragState = null;
      try {
        wrapEl.releasePointerCapture(e.pointerId);
      } catch (_err) {
        // Ignore
      }

      const layout = targetCanvas._lastLayout;
      if (drag.mode === "zoom" && layout) {
        const dx = Math.abs(drag.curX - drag.startX);
        const dy = Math.abs(drag.curY - drag.startY);
        if (dx >= 10 && dy >= 8) {
          const leftFrac = (Math.min(drag.startX, drag.curX) - layout.pad.left) / Math.max(1, layout.plotW);
          const rightFrac = (Math.max(drag.startX, drag.curX) - layout.pad.left) / Math.max(1, layout.plotW);
          const topFrac = (Math.min(drag.startY, drag.curY) - layout.pad.top) / Math.max(1, layout.plotH);
          const botFrac = (Math.max(drag.startY, drag.curY) - layout.pad.top) / Math.max(1, layout.plotH);

          const xSpan = layout.xMax - layout.xMin;
          const newXMin = layout.xMin + leftFrac * xSpan;
          const newXMax = layout.xMin + rightFrac * xSpan;

          let newYMin;
          let newYMax;
          if (this.scaleMode === "log") {
            const logSpan = layout.logMax - layout.logMin;
            newYMax = Math.pow(10, layout.logMax - topFrac * logSpan);
            newYMin = Math.pow(10, layout.logMax - botFrac * logSpan);
          } else {
            const ySpan = layout.yMax - layout.yMin;
            newYMax = layout.yMax - topFrac * ySpan;
            newYMin = layout.yMax - botFrac * ySpan;
          }

          if (newXMax > newXMin && newYMax > newYMin) {
            this.zoomDomain = {
              xMin: Math.max(0, newXMin),
              xMax: newXMax,
              yMin: Math.max(0, newYMin),
              yMax: newYMax,
            };
            this.syncControlButtonsUI();
          }
        }
      }
      this.render(targetCanvas);
    };

    wrapEl.addEventListener("pointerup", finishDrag);
    wrapEl.addEventListener("pointercancel", finishDrag);

    wrapEl.addEventListener("pointerleave", () => {
      if (!this._dragState) {
        this._hoverXVal = null;
        if (tooltipEl) tooltipEl.style.display = "none";
        this.render(targetCanvas);
      }
    });

    // Double-click resets zoom and auto-fits
    wrapEl.addEventListener("dblclick", () => {
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.render(targetCanvas);
    });

    // Mouse wheel zoom around cursor
    wrapEl.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        const layout = targetCanvas._lastLayout;
        if (!layout) return;
        const coords = this.getPointerCanvasCoords(e, wrapEl, targetCanvas);
        const factor = e.deltaY < 0 ? 0.82 : 1.22;

        const xFrac = Math.max(0, Math.min(1, (coords.x - layout.pad.left) / Math.max(1, layout.plotW)));
        const yFrac = Math.max(0, Math.min(1, 1 - (coords.y - layout.pad.top) / Math.max(1, layout.plotH)));

        const xCenter = layout.xMin + xFrac * (layout.xMax - layout.xMin);
        const yCenter = layout.yMin + yFrac * (layout.yMax - layout.yMin);
        const newXSpan = (layout.xMax - layout.xMin) * factor;
        const newYSpan = (layout.yMax - layout.yMin) * factor;

        this.zoomDomain = {
          xMin: Math.max(0, xCenter - xFrac * newXSpan),
          xMax: Math.max(1e-3, xCenter + (1 - xFrac) * newXSpan),
          yMin: Math.max(0, yCenter - yFrac * newYSpan),
          yMax: Math.max(1e-6, yCenter + (1 - yFrac) * newYSpan),
        };
        this.syncControlButtonsUI();
        this.render(targetCanvas);
      },
      { passive: false }
    );
  }

  /**
   * Measure true untransformed CSS pixel size of canvas wrapper and set high-DPI backing store.
   * Never collapses to 60px even if mounted offscreen or inside a zoomed LiteGraph node.
   */
  resizeCanvas(targetCanvas = this.canvas) {
    if (!targetCanvas) return;
    const parent = targetCanvas.parentElement;
    if (!parent) return;

    const graphScale = getGraphZoomScale();

    // `clientWidth` and `offsetWidth` are untransformed CSS pixels (immune to LiteGraph CSS scale)
    let cssW = parent.clientWidth || parent.offsetWidth || 0;
    let cssH = parent.clientHeight || parent.offsetHeight || 0;

    if (cssW < 60 || cssH < 60) {
      const rect = parent.getBoundingClientRect();
      if (rect.width > 20 && rect.height > 20) {
        cssW = Math.round(rect.width / graphScale);
        cssH = Math.round(rect.height / graphScale);
      }
    }

    // Fallback to LiteGraph node size if DOM layout has not completed yet
    if (cssW < 100) {
      const nodeW = Array.isArray(this.node?.size) ? Number(this.node.size[0]) : 620;
      cssW = Math.max(360, (Number.isFinite(nodeW) && nodeW > 150 ? nodeW : 620) - 20);
    }
    if (cssH < 100) {
      const nodeH = Array.isArray(this.node?.size) ? Number(this.node.size[1]) : 460;
      cssH = Math.max(165, (Number.isFinite(nodeH) && nodeH > 180 ? nodeH : 460) - 165);
    }

    const screenDpr = window.devicePixelRatio || 1;
    const dpr = Math.min(4.0, Math.max(2.0, screenDpr * Math.max(1.0, graphScale)));

    const bufferW = Math.round(cssW * dpr);
    const bufferH = Math.round(cssH * dpr);

    if (targetCanvas.width !== bufferW || targetCanvas.height !== bufferH) {
      targetCanvas.width = bufferW;
      targetCanvas.height = bufferH;
    }
    targetCanvas._cssW = cssW;
    targetCanvas._cssH = cssH;
    targetCanvas._dpr = dpr;
  }

  getActiveRun() {
    if (this.activeRunId) {
      const found = this.runs.find((r) => r.id === this.activeRunId);
      if (found) return found;
    }
    if (this.runs.length > 0) {
      this.activeRunId = this.runs[this.runs.length - 1].id;
      return this.runs[this.runs.length - 1];
    }
    return null;
  }

  createOrGetRun(runId, runTitle, maxSteps, nodeId = "", setActive = true) {
    let run = this.runs.find((r) => r.id === runId);
    if (!run) {
      this._dismissedRunIds.delete(runId);
      const colorIdx = this.runs.length % PALETTE.length;
      run = {
        id: runId,
        title: runTitle || `Run ${this.runs.length + 1}`,
        node_id: nodeId || "",
        max_steps: maxSteps || 1000,
        color: PALETTE[colorIdx],
        visible: true,
        status: "active",
        final_step: 0,
        final_loss: null,
        final_ema_loss: null,
        reason: null,
        _prefixTruncated: false,
        _localCreatedAt: Date.now(),
        data: [],
      };
      this.runs.push(run);
      if (this.runs.length > 10) {
        this.runs.shift();
        this.rebuildKnownBuckets();
      }
      this.updateRunHistoryUI();
    } else {
      if (runTitle) run.title = runTitle;
      if (maxSteps) run.max_steps = maxSteps;
      if (nodeId) run.node_id = nodeId;
    }
    if (setActive) {
      this.activeRunId = runId;
    }
    return run;
  }

  /**
   * Sanitize and insert/update a step point in sorted order.
   */
  upsertStepPoint(run, rawPt) {
    if (!run || !rawPt) return false;
    const step = Math.round(Number(rawPt.step) || 0);
    if (step <= 0) return false;

    const rawLoss = Number(rawPt.loss);
    const rawEma = Number(rawPt.ema_loss);
    const safeLoss = Number.isFinite(rawLoss) ? rawLoss : Number.isFinite(rawEma) ? rawEma : 0;
    const safeEma = Number.isFinite(rawEma) ? rawEma : safeLoss;
    const safeLr = Number.isFinite(Number(rawPt.lr)) ? Number(rawPt.lr) : Number(run.lr) || 0;
    const safeT = Number.isFinite(Number(rawPt.timestep)) ? Number(rawPt.timestep) : 0.5;
    const safeStepSec = Number.isFinite(Number(rawPt.step_sec)) ? Math.max(0, Number(rawPt.step_sec)) : 0;
    const safeVram = Number.isFinite(Number(rawPt.vram_gb)) ? Math.max(0, Number(rawPt.vram_gb)) : 0;

    const data = run.data;
    const len = data.length;
    const prevRelTime = len > 0 ? Number(data[len - 1].rel_time_sec) || 0 : 0;
    const safeRelTime = Number.isFinite(Number(rawPt.rel_time_sec))
      ? Number(rawPt.rel_time_sec)
      : Number((prevRelTime + safeStepSec).toFixed(3));

    const pt = {
      ...rawPt,
      step,
      max_steps: Number(rawPt.max_steps) || run.max_steps || 1000,
      loss: safeLoss,
      ema_loss: safeEma,
      lr: safeLr,
      timestep: safeT,
      bucket: String(rawPt.bucket || "default"),
      step_sec: safeStepSec,
      rel_time_sec: safeRelTime,
      eta_sec: Number.isFinite(Number(rawPt.eta_sec)) ? Math.max(0, Number(rawPt.eta_sec)) : 0,
      vram_gb: safeVram,
    };

    if (len === 0 || step > (Number(data[len - 1].step) || 0)) {
      data.push(pt);
      return true;
    }
    if (step === (Number(data[len - 1].step) || 0)) {
      data[len - 1] = pt;
      return false;
    }

    let lo = 0;
    let hi = len - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const midStep = Number(data[mid].step) || 0;
      if (midStep === step) {
        data[mid] = pt;
        return false;
      }
      if (midStep < step) lo = mid + 1;
      else hi = mid - 1;
    }
    data.splice(lo, 0, pt);
    return true;
  }

  getSyncCursor(run) {
    if (!run || !Array.isArray(run.data) || run.data.length === 0) return 0;
    const firstStep = Number(run.data[0].step) || 0;
    if (firstStep > 1 && !run._prefixTruncated) {
      return 0;
    }
    let cursor = firstStep;
    for (let i = 1; i < run.data.length; i++) {
      const s = Number(run.data[i].step) || 0;
      if (s === cursor) continue;
      if (s === cursor + 1) {
        cursor = s;
      } else {
        break;
      }
    }
    return cursor;
  }

  updateStatsFromRun(run) {
    if (!run) return;
    const data = Array.isArray(run.data) ? run.data : [];
    const latest = data.length > 0 ? data[data.length - 1] : null;

    if (run.status === "completed") {
      this.statusDot.className = "omni-tracker-status-dot completed";
      this.titleText.textContent = `${run.title || "Training"} · Completed`;
    } else if (run.status === "interrupted") {
      this.statusDot.className = "omni-tracker-status-dot interrupted";
      this.titleText.textContent = `${run.title || "Training"} · Interrupted (Step ${run.final_step || latest?.step || "?"})`;
    } else {
      this.statusDot.className = "omni-tracker-status-dot active";
      this.titleText.textContent = `${run.title || "Training"} · Live`;
    }

    if (latest) {
      const maxSteps = Math.max(1, Number(run.max_steps || latest.max_steps) || 1000);
      const stepNum = Number(latest.step) || 0;
      const pct = Math.min(100, (stepNum / maxSteps) * 100).toFixed(1);
      this.statStep.textContent = `${stepNum}/${maxSteps} (${pct}%)`;

      const rawLosses = data.map((p) => Number(p.loss) || 0);
      const emaSeries = computeEma(rawLosses, this.smoothing);
      const curRaw = Number(run.status === "completed" && run.final_loss != null ? run.final_loss : latest.loss) || 0;
      const curEma = emaSeries.length > 0 ? emaSeries[emaSeries.length - 1] : curRaw;

      this.statLoss.textContent = `${curEma.toFixed(4)} (${curRaw.toFixed(4)})`;

      // Compute Min EMA & recent convergence slope (skipping initial warmup transient)
      let minEma = Infinity;
      let minStep = stepNum;
      const minStartIdx = getEmaWarmupSkip(emaSeries.length);
      for (let i = minStartIdx; i < emaSeries.length; i++) {
        if (emaSeries[i] < minEma) {
          minEma = emaSeries[i];
          minStep = data[i].step;
        }
      }
      if ( this.statMin ) {
        if (emaSeries.length >= 4) {
          const lookbackIdx = Math.max(0, emaSeries.length - Math.min(25, emaSeries.length));
          const deltaEma = curEma - emaSeries[lookbackIdx];
          const sign = deltaEma <= 0 ? "↓" : "↑";
          this.statMin.textContent = `${minEma.toFixed(4)} @${minStep} (${sign}${Math.abs(deltaEma).toFixed(4)})`;
          this.statMin.style.color = deltaEma <= 0 ? "#34d399" : "#fbbf24";
        } else {
          this.statMin.textContent = `${minEma.toFixed(4)} @${minStep}`;
          this.statMin.style.color = "#e2e8f0";
        }
      }

      const lrVal = Number(latest.lr);
      this.statLr.textContent = Number.isFinite(lrVal) && lrVal > 0 ? lrVal.toExponential(2) : "--";

      if (run.status === "completed") {
        const stepSec = Number(latest.step_sec);
        this.statSpeed.textContent = Number.isFinite(stepSec) && stepSec > 0 ? `${stepSec.toFixed(2)}s/st · Done` : "Done";
      } else {
        const etaSec = Math.max(0, Math.round(Number(latest.eta_sec) || 0));
        const eta =
          etaSec >= 3600
            ? `${Math.floor(etaSec / 3600)}h ${Math.floor((etaSec % 3600) / 60)}m`
            : etaSec >= 60
            ? `${Math.floor(etaSec / 60)}m ${etaSec % 60}s`
            : `${etaSec}s`;
        const stepSec = Number(latest.step_sec) || 0;
        this.statSpeed.textContent = `${stepSec.toFixed(2)}s/st · ETA ${eta}`;
      }
    } else {
      this.statStep.textContent = `0 / ${run.max_steps || 0}`;
      this.statLoss.textContent = "--";
      if (this.statMin) this.statMin.textContent = "--";
      if (run.lr != null && Number(run.lr) > 0) {
        this.statLr.textContent = Number(run.lr).toExponential(2);
      } else {
        this.statLr.textContent = "--";
      }
      this.statSpeed.textContent = "Waiting for step 1...";
    }
  }

  resetStatsUI() {
    if (this.statusDot) this.statusDot.className = "omni-tracker-status-dot";
    if (this.titleText) this.titleText.textContent = "OmniLoRA TensorBoard Tracker";
    if (this.statStep) this.statStep.textContent = "0 / 0";
    if (this.statLoss) this.statLoss.textContent = "--";
    if (this.statMin) {
      this.statMin.textContent = "--";
      this.statMin.style.color = "#e2e8f0";
    }
    if (this.statLr) this.statLr.textContent = "--";
    if (this.statSpeed) this.statSpeed.textContent = "--";
  }

  deleteRun(runId) {
    if (!runId) return;
    const targetId = String(runId);
    this._dismissedRunIds.add(targetId);
    this.runs = this.runs.filter((r) => r.id !== targetId);
    if (this.activeRunId === targetId) {
      this.activeRunId = this.runs.length > 0 ? this.runs[this.runs.length - 1].id : null;
    }
    this.rebuildKnownBuckets();
    this.updateRunHistoryUI();
    const active = this.getActiveRun();
    if (active) {
      this.updateStatsFromRun(active);
    } else {
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.resetStatsUI();
    }
    this.saveState();
    this.render();
    if (this._modalCanvas) {
      this.render(this._modalCanvas);
    }
    this.fetchMetricsSnapshot(`delete_run=${encodeURIComponent(targetId)}&_t=${Date.now()}`).catch(() => {});
  }

  clearAllRuns() {
    this.runs.forEach((r) => {
      if (r && r.id) this._dismissedRunIds.add(String(r.id));
    });
    this.runs = [];
    this.activeRunId = null;
    this.zoomDomain = null;
    this.rebuildKnownBuckets();
    this.updateRunHistoryUI();
    this.syncControlButtonsUI();
    this.resetStatsUI();
    this.saveState();
    this.render();
    if (this._modalCanvas) {
      this.render(this._modalCanvas);
    }
    this.fetchMetricsSnapshot(`clear_runs=all&_t=${Date.now()}`).catch(() => {});
  }

  updateRunHistoryUI() {
    this.historyBtn.textContent = `Runs (${this.runs.length})`;
    this.runList.innerHTML = "";

    this.runs.forEach((run) => {
      const item = document.createElement("div");
      item.className = "omni-tracker-run-item";

      const left = document.createElement("label");
      left.className = "omni-tracker-run-label";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = run.visible !== false;
      cb.addEventListener("change", (e) => {
        run.visible = e.target.checked;
        this.render();
      });

      const dot = document.createElement("span");
      dot.style.cssText = `width:8px; height:8px; border-radius:50%; background:${run.color}; display:inline-block; flex-shrink:0;`;

      const lastPt = run.data && run.data.length > 0 ? run.data[run.data.length - 1] : null;
      const lossBadge = lastPt ? ` · ${Number(lastPt.ema_loss ?? lastPt.loss).toFixed(4)}` : "";

      const title = document.createElement("span");
      title.textContent = `${run.title} (${run.data.length} st${lossBadge})`;
      title.style.cssText = `white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:${
        run.id === this.activeRunId ? "#ffffff" : "#94a3b8"
      }; font-weight:${run.id === this.activeRunId ? "600" : "400"};`;
      title.addEventListener("click", (ev) => {
        ev.preventDefault();
        this.activeRunId = run.id;
        run.visible = true;
        this.updateStatsFromRun(run);
        this.updateRunHistoryUI();
        this.render();
      });

      left.appendChild(cb);
      left.appendChild(dot);
      left.appendChild(title);

      const del = document.createElement("button");
      del.className = "omni-tracker-btn";
      del.textContent = "×";
      del.style.padding = "0 5px";
      del.title = "Delete run from tracker & server";
      del.addEventListener("click", () => {
        this.deleteRun(run.id);
      });

      item.appendChild(left);
      item.appendChild(del);
      this.runList.appendChild(item);
    });
  }

  rebuildKnownBuckets() {
    this.knownBuckets.clear();
    this.runs.forEach((r) => {
      if (!r || !Array.isArray(r.data)) return;
      r.data.forEach((pt) => {
        if (pt && pt.bucket && pt.bucket !== "default") {
          this.knownBuckets.add(pt.bucket);
        }
      });
    });
    if (this.selectedBucket !== "all" && !this.knownBuckets.has(this.selectedBucket)) {
      this.selectedBucket = "all";
    }
    this.updateBucketOptions();
  }

  updateBucketOptions() {
    const prev = this.selectedBucket || this.bucketSelect.value;
    this.bucketSelect.innerHTML = `<option value="all">All Buckets</option>`;
    Array.from(this.knownBuckets)
      .sort()
      .forEach((b) => {
        const opt = document.createElement("option");
        opt.value = b;
        opt.textContent = b;
        this.bucketSelect.appendChild(opt);
      });
    if (this.knownBuckets.has(prev)) {
      this.bucketSelect.value = prev;
    } else {
      this.selectedBucket = "all";
      this.bucketSelect.value = "all";
    }
  }

  /**
   * Safely resolve upstream connected node IDs across both Legacy LiteGraph (`graph.links[id]`)
   * and Modern ComfyUI_frontend (`graph.links.get(id)`), walking through Reroute nodes.
   */
  getConnectedOriginIds() {
    const connectedOriginIds = [];
    const graph = this.node?.graph || getComfyApp()?.graph;
    if (!this.node || !Array.isArray(this.node.inputs) || !graph || !graph.links) {
      return connectedOriginIds;
    }

    const resolveLink = (linkId) => {
      if (linkId == null) return null;
      if (typeof graph.links.get === "function") {
        return graph.links.get(linkId) || graph.links[linkId] || null;
      }
      return graph.links[linkId] || null;
    };

    const getNodeById = (id) => {
      if (id == null) return null;
      if (typeof graph.getNodeById === "function") return graph.getNodeById(id);
      return null;
    };

    for (const input of this.node.inputs) {
      let curLinkId = input?.link;
      let hops = 0;
      while (curLinkId != null && hops < 8) {
        hops++;
        const link = resolveLink(curLinkId);
        if (!link || link.origin_id == null) break;
        const originIdStr = String(link.origin_id);
        connectedOriginIds.push(originIdStr);

        const upstreamNode = getNodeById(link.origin_id);
        if (
          upstreamNode &&
          (upstreamNode.type === "Reroute" || upstreamNode.comfyClass === "Reroute") &&
          Array.isArray(upstreamNode.inputs) &&
          upstreamNode.inputs[0]?.link != null
        ) {
          curLinkId = upstreamNode.inputs[0].link;
        } else {
          break;
        }
      }
    }
    return connectedOriginIds;
  }

  /**
   * Check whether an incoming telemetry event belongs to this widget.
   * Never drops events when connected through passthrough/subgraph nodes or when no runs exist yet.
   */
  isTargetNode(detail) {
    if (!detail) return false;
    const connectedOriginIds = this.getConnectedOriginIds();
    if (connectedOriginIds.length > 0 && detail.node_id) {
      const detailId = String(detail.node_id);
      const tailId = detailId.includes(":") ? detailId.split(":").pop() : detailId;
      if (connectedOriginIds.includes(detailId) || connectedOriginIds.includes(tailId)) {
        return true;
      }
      // If this tracker currently has no active run from the connected ID, still accept the run
      // so Reroute/Subgraph/Cloudflare workflows never show an empty graph.
      const hasMatchingRun = this.runs.some((r) => connectedOriginIds.includes(String(r.node_id)));
      return !hasMatchingRun;
    }
    return true;
  }

  scheduleGapSync(runId) {
    if (this._gapSyncTimer) return;
    this._gapSyncTimer = setTimeout(() => {
      this._gapSyncTimer = null;
      this.syncFromServer({ force: true, runId });
    }, 250);
  }

  /**
   * Cloudflare Tunnel & Reverse-Proxy Resilient Metrics Fetcher.
   * Uses `cache: "no-store"` + timestamp query parameter so Cloudflare never serves stale cached JSON,
   * and tries `comfyApi.fetchApi`, `comfyApi.apiURL`, relative URL, and root path.
   */
  async fetchMetricsSnapshot(queryString) {
    const cApi = getComfyApi();
    const fetchOpts = {
      method: "GET",
      cache: "no-store",
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        "Cache-Control": "no-cache, no-store, max-age=0",
        Pragma: "no-cache",
      },
    };

    const relPath = `/omni_lora/metrics?${queryString}`;
    const candidateUrls = [];
    if (typeof cApi?.apiURL === "function") {
      try {
        candidateUrls.push(cApi.apiURL(relPath));
      } catch (_e) {
        // Ignore
      }
    }
    candidateUrls.push(`omni_lora/metrics?${queryString}`);
    candidateUrls.push(relPath);

    if (typeof cApi?.fetchApi === "function") {
      try {
        const res = await cApi.fetchApi(relPath, fetchOpts);
        if (res && res.ok) {
          return await res.json();
        }
      } catch (_e) {
        // Fallback to direct fetch candidates below
      }
    }

    for (const url of candidateUrls) {
      try {
        const res = await fetch(url, fetchOpts);
        if (res && res.ok) {
          return await res.json();
        }
      } catch (_e) {
        // Try next candidate URL
      }
    }
    return null;
  }

  async syncFromServer({ force = false, runId = null } = {}) {
    if (this._destroyed) return;
    const now = Date.now();
    if (!force && now - this._lastSyncAt < 1200) {
      return;
    }
    if (this._syncPromise) {
      if (force) this._pendingForceSync = true;
      return this._syncPromise;
    }

    this._lastSyncAt = now;
    const params = new URLSearchParams();

    if (runId) {
      const targetRun = this.runs.find((r) => r.id === runId);
      params.set("run_id", runId);
      params.set("after_step", String(this.getSyncCursor(targetRun)));
    } else {
      const cursorPairs = this.runs
        .filter((r) => r && r.id)
        .map((r) => `${r.id}:${this.getSyncCursor(r)}`);
      if (cursorPairs.length > 0) {
        params.set("cursors", cursorPairs.join(","));
      }
    }
    // Cache-busting timestamp for Cloudflare tunnels and reverse proxies
    params.set("_t", String(now));

    this._syncPromise = (async () => {
      try {
        const payload = await this.fetchMetricsSnapshot(params.toString());
        if (payload && Array.isArray(payload.runs)) {
          this.mergeServerRuns(payload.runs, !runId);
        }
      } catch (_err) {
        // Server unreachable or outside ComfyUI runtime; ignore silently
      } finally {
        this._syncPromise = null;
        if (this._pendingForceSync && !this._destroyed) {
          this._pendingForceSync = false;
          this.syncFromServer({ force: true });
        }
      }
    })();

    return this._syncPromise;
  }

  mergeServerRuns(serverRuns, isFullSync = true) {
    if (!Array.isArray(serverRuns) || this._destroyed) return;

    const connectedOriginIds = this.getConnectedOriginIds();
    const activeOnServer = new Set();
    let changed = false;
    let statusChanged = false;
    let bucketsAdded = false;

    serverRuns.forEach((sr) => {
      if (!sr || !sr.run_id) return;
      if (sr.status === "active") {
        activeOnServer.add(sr.run_id);
      }

      const existing = this.runs.find((r) => r.id === sr.run_id);
      if (this._dismissedRunIds.has(sr.run_id)) {
        return;
      }

      const isPreferredOrigin =
        connectedOriginIds.length === 0 ||
        (sr.node_id && connectedOriginIds.includes(String(sr.node_id)));

      const currentActive = this.getActiveRun();
      const shouldSelect =
        !this.activeRunId ||
        (sr.status === "active" && isPreferredOrigin) ||
        (!existing && (!currentActive || currentActive.status !== "active") && isPreferredOrigin);

      const run = this.createOrGetRun(
        sr.run_id,
        sr.run_title,
        sr.max_steps,
        sr.node_id || "",
        shouldSelect
      );
      if (!existing) changed = true;

      if (sr.lr != null) run.lr = sr.lr;
      if (sr.status && run.status !== sr.status) {
        run.status = sr.status;
        statusChanged = true;
      }
      if (sr.final_step != null) run.final_step = sr.final_step;
      if (sr.final_loss != null) run.final_loss = sr.final_loss;
      if (sr.final_ema_loss != null) run.final_ema_loss = sr.final_ema_loss;
      if (sr.reason != null) run.reason = sr.reason;

      if (Array.isArray(sr.steps) && sr.steps.length > 0) {
        sr.steps.forEach((pt) => {
          this.upsertStepPoint(run, pt);
          changed = true;
          if (pt.bucket && pt.bucket !== "default" && !this.knownBuckets.has(pt.bucket)) {
            this.knownBuckets.add(pt.bucket);
            bucketsAdded = true;
          }
        });
      }
    });

    if (isFullSync) {
      const now = Date.now();
      const serverRunIds = new Set(serverRuns.map((sr) => sr && sr.run_id).filter(Boolean));
      const beforeCount = this.runs.length;
      this.runs = this.runs.filter(
        (r) => serverRunIds.has(r.id) || (r._localCreatedAt && now - r._localCreatedAt < 3000)
      );
      if (this.runs.length !== beforeCount) {
        changed = true;
        if (!this.runs.some((r) => r.id === this.activeRunId)) {
          this.activeRunId = this.runs.length > 0 ? this.runs[this.runs.length - 1].id : null;
        }
        this.rebuildKnownBuckets();
      }

      this.runs.forEach((r) => {
        if (r.status === "active" && !activeOnServer.has(r.id)) {
          const matched = serverRuns.find((sr) => sr.run_id === r.id);
          if (matched && matched.status) {
            r.status = matched.status;
            statusChanged = true;
          } else if (!r._localCreatedAt || now - r._localCreatedAt > 15000) {
            r.status = r.data.length >= (r.max_steps || Infinity) ? "completed" : "interrupted";
            statusChanged = true;
          }
        }
      });
    }

    if (bucketsAdded) {
      this.updateBucketOptions();
    }

    if (changed || statusChanged) {
      const activeRun = this.getActiveRun();
      if (activeRun) {
        this.updateStatsFromRun(activeRun);
      } else {
        this.resetStatsUI();
      }
      this.updateRunHistoryUI();
      if (statusChanged) {
        this.saveState();
      } else {
        this.saveStateThrottled();
      }
      this.resizeCanvas();
      this.render();
    }
  }

  bindServerEvents() {
    const cApi = getComfyApi();
    const unwrapDetail = (e) => {
      if (!e) return null;
      if (e.detail && typeof e.detail === "object") {
        return e.detail.data && typeof e.detail.data === "object" && !e.detail.run_id
          ? e.detail.data
          : e.detail;
      }
      return null;
    };

    this._onStart = (e) => {
      this._lastWsMessageAt = Date.now();
      const d = unwrapDetail(e);
      if (!this.isTargetNode(d)) return;
      if (d.run_id) {
        this._dismissedRunIds.delete(String(d.run_id));
      }
      const run = this.createOrGetRun(d.run_id, d.run_title, d.max_steps, d.node_id || "", true);
      run.data = [];
      run.max_steps = d.max_steps;
      run.lr = d.lr;
      run.status = "active";
      run.final_step = 0;
      run.final_loss = null;
      run.final_ema_loss = null;
      run.reason = null;
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.updateStatsFromRun(run);
      this.updateRunHistoryUI();
      this.saveState();
      this.resizeCanvas();
      this.render();
    };

    this._onStep = (e) => {
      this._lastWsMessageAt = Date.now();
      const d = unwrapDetail(e);
      if (!this.isTargetNode(d)) return;
      if (d.run_id && this._dismissedRunIds.has(String(d.run_id))) return;
      const run = this.createOrGetRun(d.run_id, d.run_title, d.max_steps, d.node_id || "", true);
      const prevCursor = this.getSyncCursor(run);
      const stepNum = Number(d.step) || 0;

      run.status = "active";
      this.upsertStepPoint(run, d);

      // Detect any steps missed during Cloudflare WS hiccup / PC idle / tab sleep
      if (stepNum > 1 && prevCursor < stepNum - 1) {
        this.scheduleGapSync(d.run_id);
      }

      if (d.bucket && d.bucket !== "default" && !this.knownBuckets.has(d.bucket)) {
        this.knownBuckets.add(d.bucket);
        this.updateBucketOptions();
      }

      this.updateStatsFromRun(run);
      if (run.data.length === 1 || stepNum % 10 === 0) {
        this.updateRunHistoryUI();
      }
      this.saveStateThrottled();
      this.render();
    };

    this._onComplete = (e) => {
      this._lastWsMessageAt = Date.now();
      const d = unwrapDetail(e);
      if (!this.isTargetNode(d)) return;
      if (d.run_id && this._dismissedRunIds.has(String(d.run_id))) return;
      const run = this.createOrGetRun(d.run_id, d.run_title, d.final_step, d.node_id || "", true);
      const prevCursor = this.getSyncCursor(run);
      run.status = "completed";
      run.final_step = d.final_step;
      run.final_loss = d.final_loss;
      run.final_ema_loss = d.final_ema_loss;

      this.updateStatsFromRun(run);
      this.updateRunHistoryUI();
      this.saveState();
      this.render();

      if ((Number(d.final_step) || 0) > prevCursor) {
        this.syncFromServer({ force: true, runId: d.run_id });
      }
    };

    this._onInterrupted = (e) => {
      this._lastWsMessageAt = Date.now();
      const d = unwrapDetail(e);
      if (!this.isTargetNode(d)) return;
      if (d.run_id && this._dismissedRunIds.has(String(d.run_id))) return;
      const run = d.run_id
        ? this.createOrGetRun(d.run_id, d.run_title, null, d.node_id || "", true)
        : this.getActiveRun();
      if (run) {
        run.status = "interrupted";
        run.final_step = d.step || run.final_step;
        run.reason = d.reason || null;
        this.updateStatsFromRun(run);
      } else {
        this.statusDot.className = "omni-tracker-status-dot interrupted";
        this.titleText.textContent = `Interrupted (Step ${d?.step || "?"})`;
      }
      this.updateRunHistoryUI();
      this.saveState();
      this.render();
      if (d.run_id) {
        this.syncFromServer({ force: true, runId: d.run_id });
      }
    };

    this._onApiReconnected = () => {
      this.resizeCanvas();
      this.syncFromServer({ force: true });
    };

    this._onExecutionStart = () => {
      this._workflowActive = true;
      this.syncFromServer({ force: true });
    };

    this._onExecuting = (e) => {
      if (e?.detail != null) {
        this._workflowActive = true;
      } else {
        this._workflowActive = false;
        this.syncFromServer({ force: true });
      }
    };

    this._onExecuted = (e) => {
      const outputRuns = e?.detail?.output?.runs;
      if (Array.isArray(outputRuns) && outputRuns.length > 0) {
        this.mergeServerRuns(outputRuns, false);
      }
      this.syncFromServer({ force: true });
    };

    this._onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        this.resizeCanvas();
        this.render();
        this.syncFromServer({ force: true });
      }
    };

    this._onWindowWake = () => {
      this.resizeCanvas();
      this.render();
      this.syncFromServer({ force: true });
    };

    if (cApi && typeof cApi.addEventListener === "function") {
      cApi.addEventListener("omni_lora_training_start", this._onStart);
      cApi.addEventListener("omni_lora_step_metric", this._onStep);
      cApi.addEventListener("omni_lora_training_complete", this._onComplete);
      cApi.addEventListener("omni_lora_training_interrupted", this._onInterrupted);
      cApi.addEventListener("reconnected", this._onApiReconnected);
      cApi.addEventListener("execution_start", this._onExecutionStart);
      cApi.addEventListener("executing", this._onExecuting);
      cApi.addEventListener("executed", this._onExecuted);
    }

    document.addEventListener("visibilitychange", this._onVisibilityChange);
    window.addEventListener("focus", this._onWindowWake);
    window.addEventListener("online", this._onWindowWake);
    window.addEventListener("pageshow", this._onWindowWake);
    window.addEventListener("resize", this._onWindowWake);

    // Unconditional Watchdog Timer:
    // Over Cloudflare tunnels (`*.trycloudflare.com`), WebSocket frames can be dropped or buffered,
    // or a user may open/refresh the browser while training is already in progress (`hasActiveRun === false`).
    // Polling delta cursors every 2.5s when visible ensures the chart ALWAYS receives live telemetry.
    let lastTickAt = Date.now();
    this._reconcileInterval = setInterval(() => {
      const now = Date.now();
      const elapsed = now - lastTickAt;
      lastTickAt = now;

      if (elapsed > 6500) {
        this.resizeCanvas();
        this.syncFromServer({ force: true });
        return;
      }

      if (document.visibilityState !== "hidden") {
        const activeRun = this.getActiveRun();
        const hasActiveRun = Boolean(activeRun && activeRun.status === "active");
        const wsQuietMs = now - this._lastWsMessageAt;
        if ((hasActiveRun || this._workflowActive) && wsQuietMs > 1800) {
          this.syncFromServer();
        } else if (wsQuietMs > 3000) {
          this.syncFromServer();
        }
      }
    }, 2200);
  }

  destroy() {
    this._destroyed = true;
    if (this._gapSyncTimer) {
      clearTimeout(this._gapSyncTimer);
      this._gapSyncTimer = null;
    }
    if (this._saveTimeout) {
      clearTimeout(this._saveTimeout);
      this._saveTimeout = null;
    }
    if (this._reconcileInterval) {
      clearInterval(this._reconcileInterval);
      this._reconcileInterval = null;
    }
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    if (this._intersectionObserver) {
      this._intersectionObserver.disconnect();
      this._intersectionObserver = null;
    }
    const cApi = getComfyApi();
    if (cApi && typeof cApi.removeEventListener === "function") {
      if (this._onStart) cApi.removeEventListener("omni_lora_training_start", this._onStart);
      if (this._onStep) cApi.removeEventListener("omni_lora_step_metric", this._onStep);
      if (this._onComplete) cApi.removeEventListener("omni_lora_training_complete", this._onComplete);
      if (this._onInterrupted) cApi.removeEventListener("omni_lora_training_interrupted", this._onInterrupted);
      if (this._onApiReconnected) cApi.removeEventListener("reconnected", this._onApiReconnected);
      if (this._onExecutionStart) cApi.removeEventListener("execution_start", this._onExecutionStart);
      if (this._onExecuting) cApi.removeEventListener("executing", this._onExecuting);
      if (this._onExecuted) cApi.removeEventListener("executed", this._onExecuted);
    }
    if (this._onVisibilityChange) document.removeEventListener("visibilitychange", this._onVisibilityChange);
    if (this._onWindowWake) {
      window.removeEventListener("focus", this._onWindowWake);
      window.removeEventListener("online", this._onWindowWake);
      window.removeEventListener("pageshow", this._onWindowWake);
      window.removeEventListener("resize", this._onWindowWake);
    }
  }

  filterData(points) {
    if (!Array.isArray(points) || points.length === 0) return [];
    return points.filter((p) => {
      if (!p || !Number.isFinite(Number(p.loss))) return false;
      const t = Number(p.timestep);
      if (this.noiseBin === "high" && t < 0.67) return false;
      if (this.noiseBin === "mid" && (t < 0.33 || t >= 0.67)) return false;
      if (this.noiseBin === "low" && t >= 0.33) return false;
      if (this.selectedBucket !== "all" && p.bucket && p.bucket !== this.selectedBucket) return false;
      return true;
    });
  }

  getPointX(pt, idx = 0) {
    if (this.xAxisMode === "relative") {
      const rel = Number(pt.rel_time_sec);
      if (Number.isFinite(rel) && rel >= 0) return rel;
      return (Number(pt.step) || idx + 1) * (Number(pt.step_sec) || 1.0);
    }
    return Number(pt.step) || idx + 1;
  }

  /**
   * TensorBoard-Grade Axis Domain Computation.
   * Prevents single high-timestep diffusion spikes or narrow loss ranges from flattening the graph.
   */
  computeAxisDomains(visibleSeries) {
    let xMin = Infinity;
    let xMax = -Infinity;
    let maxTargetStep = 10;
    let maxLr = 0;

    const allRawLosses = [];
    const allEmaLosses = [];

    visibleSeries.forEach((s) => {
      if (s.run.max_steps && s.run.max_steps > maxTargetStep) {
        maxTargetStep = Number(s.run.max_steps);
      }
      // Skip initial warmup points for EMA domain framing when enough points exist
      const emaStartIdx = getEmaWarmupSkip(s.points.length);
      s.points.forEach((p, idx) => {
        const xVal = s.xValues[idx];
        if (Number.isFinite(xVal)) {
          if (xVal < xMin) xMin = xVal;
          if (xVal > xMax) xMax = xVal;
        }
        const rawL = Number(p.loss);
        if (Number.isFinite(rawL)) {
          if (this.scaleMode !== "log" || rawL > 0) {
            allRawLosses.push(rawL);
          }
        }
        const emaL = Number(s.emaLosses[idx]);
        if (idx >= emaStartIdx && Number.isFinite(emaL)) {
          if (this.scaleMode !== "log" || emaL > 0) {
            allEmaLosses.push(emaL);
          }
        }
        const lr = Number(p.lr);
        if (Number.isFinite(lr) && lr > maxLr) {
          maxLr = lr;
        }
      });
    });

    if (!Number.isFinite(xMin)) xMin = 0;
    if (!Number.isFinite(xMax) || xMax <= xMin) xMax = xMin + 10;

    if (this.xAxisMode === "target") {
      xMin = 0;
      xMax = Math.max(xMax, maxTargetStep);
    } else if (this.xAxisMode === "step") {
      // TensorBoard Auto-Fit Steps: frame actual observed steps so early steps are never squished
      if (xMin <= 2) xMin = Math.min(1, xMin);
      const xSpan = Math.max(1, xMax - xMin);
      xMax = xMax + xSpan * 0.02;
    } else {
      xMin = Math.max(0, xMin);
      const xSpan = Math.max(0.5, xMax - xMin);
      xMax = xMax + xSpan * 0.02;
    }

    if (maxLr <= 0) maxLr = 1e-4;

    // Compute Y-axis domain according to yFitMode ('auto' | 'ema' | 'full')
    let yMin = 0.0;
    let yMax = 0.25;

    const sortedRaw = allRawLosses.slice().sort((a, b) => a - b);
    const sortedEma = allEmaLosses.slice().sort((a, b) => a - b);

    if (sortedRaw.length > 0 || sortedEma.length > 0) {
      if (this.yFitMode === "ema" && sortedEma.length > 0) {
        yMin = sortedEma[0];
        yMax = sortedEma[sortedEma.length - 1];
      } else if (this.yFitMode === "auto") {
        // TensorBoard "Ignore Outliers in Chart Scaling":
        // Frame the full EMA convergence range + [P08, P92] quantile of raw points
        const rawLo = sortedRaw.length >= 6 ? computeQuantile(sortedRaw, 0.08) : sortedRaw[0];
        const rawHi = sortedRaw.length >= 6 ? computeQuantile(sortedRaw, 0.92) : sortedRaw[sortedRaw.length - 1];
        if (sortedEma.length > 0 && this.showEma) {
          const emaLo = sortedEma[0];
          const emaHi = sortedEma[sortedEma.length - 1];
          const emaSpan = Math.max(1e-6, emaHi - emaLo);
          // Cap how far raw scatter can stretch the EMA viewport (max 2.2x EMA span)
          yMin = Math.max(rawLo, emaLo - emaSpan * 0.65);
          yMax = Math.min(rawHi, emaHi + emaSpan * 0.85);
          if (yMin > emaLo) yMin = emaLo;
          if (yMax < emaHi) yMax = emaHi;
        } else {
          yMin = rawLo;
          yMax = rawHi;
        }
      } else {
        // 'full': unclipped raw min/max
        yMin = sortedRaw.length > 0 ? sortedRaw[0] : sortedEma[0];
        yMax = sortedRaw.length > 0 ? sortedRaw[sortedRaw.length - 1] : sortedEma[sortedEma.length - 1];
      }
    }

    // Apply user interactive zoom override if active
    if (this.zoomDomain) {
      xMin = this.zoomDomain.xMin;
      xMax = this.zoomDomain.xMax;
      yMin = this.zoomDomain.yMin;
      yMax = this.zoomDomain.yMax;
    } else {
      // Span-proportional vertical headroom (NEVER multiply by 0.95/1.05 which flattens small-span curves!)
      let span = yMax - yMin;
      if (!Number.isFinite(span) || span <= 0) {
        span = Math.abs(yMax) > 0 ? Math.abs(yMax) * 0.16 : 0.05;
        yMin = Math.max(0, yMin - span * 0.5);
        yMax = yMax + span * 0.5;
      } else {
        const minAllowedSpan = Math.max(1e-6, Math.abs(yMax) * 1e-4);
        if (span < minAllowedSpan) {
          const mid = 0.5 * (yMin + yMax);
          yMin = Math.max(0, mid - minAllowedSpan * 0.5);
          yMax = mid + minAllowedSpan * 0.5;
          span = yMax - yMin;
        }
        const pad = span * 0.12;
        yMin = yMin >= 0 ? Math.max(0, yMin - pad) : yMin - pad;
        yMax = yMax + pad;
      }
    }

    const isLog = this.scaleMode === "log";
    const safeLogMinVal = Math.max(1e-9, yMin > 0 ? yMin : (sortedRaw.find((v) => v > 0) || 1e-4) * 0.8);
    const safeLogMaxVal = Math.max(safeLogMinVal * 1.15, yMax);
    const logMin = isLog ? Math.log10(safeLogMinVal) : 0;
    const logMax = isLog ? Math.log10(safeLogMaxVal) : 1;

    return {
      xMin,
      xMax,
      yMin: isLog ? safeLogMinVal : yMin,
      yMax: isLog ? safeLogMaxVal : yMax,
      logMin,
      logMax,
      maxLr,
    };
  }

  /**
   * Core TensorBoard Canvas Rendering Engine.
   */
  render(targetCanvas = this.canvas, hoverXVal = this._hoverXVal, options = {}) {
    if (!targetCanvas) return;
    if (!targetCanvas._cssW || !targetCanvas._cssH) {
      this.resizeCanvas(targetCanvas);
    }

    const ctx = targetCanvas.getContext("2d");
    const parent = targetCanvas.parentElement;
    const clientW = targetCanvas._cssW || (parent ? parent.clientWidth : 580) || 580;
    const clientH = targetCanvas._cssH || (parent ? parent.clientHeight : 220) || 220;
    const dpr = targetCanvas._dpr || targetCanvas.width / Math.max(1, clientW);

    ctx.save();
    ctx.scale(dpr, dpr);
    if (!options.skipClear) {
      ctx.clearRect(0, 0, clientW, clientH);
      ctx.fillStyle = "#090b10";
      ctx.fillRect(0, 0, clientW, clientH);
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const topOffset = Number(options.topOffset) || 0;
    const compactH = clientH < 175;
    const pad = {
      top: (compactH ? 18 : 22) + topOffset,
      right: this.showLr ? 54 : 16,
      bottom: compactH ? 22 : 26,
      left: 56,
    };
    const plotW = Math.max(40, clientW - pad.left - pad.right);
    const plotH = Math.max(40, clientH - pad.top - pad.bottom);

    // Prepare filtered + smoothed series for each visible run
    const visibleRuns = this.runs.filter((r) => r && r.visible !== false && Array.isArray(r.data) && r.data.length > 0);
    const visibleSeries = visibleRuns
      .map((run) => {
        const points = this.filterData(run.data);
        const rawLosses = points.map((p) => Number(p.loss) || 0);
        const emaLosses = computeEma(rawLosses, this.smoothing);
        const stdLosses = this.showBand ? computeRollingStd(rawLosses, emaLosses, this.smoothing) : [];
        const xValues = points.map((p, idx) => this.getPointX(p, idx));
        return { run, points, rawLosses, emaLosses, stdLosses, xValues };
      })
      .filter((s) => s.points.length > 0);

    const domains = this.computeAxisDomains(visibleSeries);
    const { xMin, xMax, yMin, yMax, logMin, logMax, maxLr } = domains;
    const isLog = this.scaleMode === "log";
    const ySpan = Math.max(1e-9, yMax - yMin);
    const xSpan = Math.max(1e-9, xMax - xMin);

    const toX = (xVal) => pad.left + ((xVal - xMin) / xSpan) * plotW;
    const toYLoss = (loss) => {
      if (isLog) {
        const val = Math.log10(Math.max(1e-9, Number(loss) || 1e-9));
        const norm = (val - logMin) / Math.max(1e-6, logMax - logMin);
        return pad.top + plotH - norm * plotH;
      }
      return pad.top + plotH - ((Number(loss) - yMin) / ySpan) * plotH;
    };
    const toYLr = (lr) => pad.top + plotH - (Math.max(0, Number(lr) || 0) / Math.max(1e-9, maxLr * 1.05)) * plotH;

    // Save layout metadata on canvas for accurate pointer zoom/hover lookup
    targetCanvas._lastLayout = {
      pad,
      plotW,
      plotH,
      xMin,
      xMax,
      yMin,
      yMax,
      logMin,
      logMax,
      maxLr,
    };

    const FONT_MONO = "600 10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    const FONT_HEADER = "600 10.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

    // Plot background & subtle border
    ctx.fillStyle = "#0c0f17";
    ctx.fillRect(pad.left, pad.top, plotW, plotH);

    // Axis Headers
    ctx.font = FONT_HEADER;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = "#38bdf8";
    const fitBadge = this.yFitMode === "ema" ? " (EMA)" : this.yFitMode === "auto" ? " (P08–P92)" : "";
    ctx.fillText(`Loss${isLog ? " [Log]" : ""}${fitBadge}`, pad.left, pad.top - 5);

    if (this.showLr) {
      ctx.textAlign = "right";
      ctx.fillStyle = "#fbbf24";
      ctx.fillText("LR", pad.left + plotW + pad.right - 6, pad.top - 5);
    }

    // Horizontal Grid Lines & Y-Axis Tick Labels
    const yTicks = compactH ? 3 : 4;
    ctx.font = FONT_MONO;
    ctx.lineWidth = 1;

    for (let i = 0; i <= yTicks; i++) {
      const frac = i / yTicks;
      const y = Math.round(pad.top + plotH - frac * plotH) + 0.5;
      ctx.strokeStyle = i === 0 || i === yTicks ? "rgba(148, 163, 184, 0.22)" : "rgba(148, 163, 184, 0.10)";
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      const val = isLog ? Math.pow(10, logMin + frac * (logMax - logMin)) : yMin + frac * ySpan;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(formatAxisValue(val, ySpan), pad.left - 6, y);

      if (this.showLr && (i === 0 || i === Math.floor(yTicks / 2) || i === yTicks)) {
        const lrTick = frac * maxLr * 1.05;
        ctx.textAlign = "left";
        ctx.fillStyle = "#fcd34d";
        ctx.fillText(lrTick === 0 ? "0" : lrTick.toExponential(1), pad.left + plotW + 6, y);
      }
    }

    // Vertical Grid Lines & X-Axis Tick Labels
    const xTicks = clientW < 420 ? 4 : 5;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= xTicks; i++) {
      const frac = i / xTicks;
      const xVal = xMin + frac * xSpan;
      const x = Math.round(pad.left + frac * plotW) + 0.5;
      ctx.strokeStyle = i === 0 || i === xTicks ? "rgba(148, 163, 184, 0.22)" : "rgba(148, 163, 184, 0.10)";
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      ctx.fillStyle = "#94a3b8";
      ctx.font = FONT_MONO;
      const xLabel =
        this.xAxisMode === "relative"
          ? xVal >= 60
            ? `${(xVal / 60).toFixed(1)}m`
            : `${xVal.toFixed(0)}s`
          : String(Math.round(xVal));
      ctx.fillText(xLabel, x, pad.top + plotH + 6);
    }

    // Empty state guidance when no runs have been recorded yet
    if (visibleSeries.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "500 11.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        "Waiting for OmniLoRA training telemetry...",
        pad.left + plotW / 2,
        pad.top + plotH / 2
      );
      ctx.restore();
      return;
    }

    // Clip all plotted curves, bands, and scatter points strictly inside the plot box
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    visibleSeries.forEach((series) => {
      const { run, points, rawLosses, emaLosses, stdLosses, xValues } = series;
      const isCurrent = run.id === this.activeRunId;

      // 1. TensorBoard +-1 Sigma Rolling Volatility Band (behind curves)
      if (this.showBand && stdLosses.length > 1) {
        ctx.fillStyle = run.color;
        ctx.globalAlpha = isCurrent ? 0.13 : 0.06;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const px = toX(xValues[i]);
          const pyUpper = toYLoss(emaLosses[i] + stdLosses[i]);
          if (i === 0) ctx.moveTo(px, pyUpper);
          else ctx.lineTo(px, pyUpper);
        }
        for (let i = points.length - 1; i >= 0; i--) {
          const px = toX(xValues[i]);
          const lowerVal = isLog ? Math.max(1e-9, emaLosses[i] - stdLosses[i]) : Math.max(0, emaLosses[i] - stdLosses[i]);
          const pyLower = toYLoss(lowerVal);
          ctx.lineTo(px, pyLower);
        }
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }

      // 2. TensorBoard Ghost Raw Trace (density-adaptive faint polyline + subtle scatter dots)
      if (this.showRaw && points.length > 0) {
        const density = points.length / Math.max(120, plotW);
        const densityScale = density > 0.5 ? 1.0 / Math.pow(density * 2.0, 0.55) : 1.0;
        const baseAlpha = isCurrent ? 0.22 : 0.12;
        ctx.strokeStyle = run.color;
        ctx.globalAlpha = Math.max(0.045, Math.min(baseAlpha, baseAlpha * densityScale));
        ctx.lineWidth = density > 1.2 ? 0.75 : 1.0;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const px = toX(xValues[i]);
          const py = toYLoss(rawLosses[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();

        if (points.length <= 400) {
          ctx.fillStyle = run.color;
          ctx.globalAlpha = isCurrent ? 0.35 : 0.18;
          for (let i = 0; i < points.length; i++) {
            const px = toX(xValues[i]);
            const py = toYLoss(rawLosses[i]);
            ctx.beginPath();
            ctx.arc(px, py, 1.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.globalAlpha = 1.0;
      }

      // 3. Primary Debiased EMA Convergence Curve
      if (this.showEma && emaLosses.length > 0) {
        ctx.strokeStyle = run.color;
        ctx.lineWidth = isCurrent ? 2.25 : 1.5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";

        ctx.beginPath();
        let minEmaVal = Infinity;
        let minEmaIdx = 0;
        const minSearchStart = getEmaWarmupSkip(points.length);
        for (let i = 0; i < points.length; i++) {
          const px = toX(xValues[i]);
          const py = toYLoss(emaLosses[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
          if (i >= minSearchStart && emaLosses[i] < minEmaVal) {
            minEmaVal = emaLosses[i];
            minEmaIdx = i;
          }
        }
        ctx.stroke();

        // Highlight minimum EMA marker on active run
        if (isCurrent && points.length >= 3) {
          const minPx = toX(xValues[minEmaIdx]);
          const minPy = toYLoss(minEmaVal);
          ctx.fillStyle = "#10b981";
          ctx.beginPath();
          ctx.arc(minPx, minPy, 3.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#090b10";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }

      // 4. Learning Rate Schedule Overlay (Active run)
      if (this.showLr && isCurrent && points.length > 1) {
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = "#fbbf24";
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const px = toX(xValues[i]);
          const py = toYLr(points[i].lr);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.restore();
      }
    });

    // 5. Interactive Box-Zoom Selection Overlay
    if (this._dragState && this._dragState.mode === "zoom") {
      const rx = Math.min(this._dragState.startX, this._dragState.curX);
      const ry = Math.min(this._dragState.startY, this._dragState.curY);
      const rw = Math.abs(this._dragState.curX - this._dragState.startX);
      const rh = Math.abs(this._dragState.curY - this._dragState.startY);
      ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(rx + 0.5, ry + 0.5, rw, rh);
    }

    // 6. Interactive Hover Crosshair & Multi-Run Comparison Markers
    if (hoverXVal != null && hoverXVal >= xMin && hoverXVal <= xMax && !this._dragState) {
      const hx = toX(hoverXVal);
      ctx.strokeStyle = "rgba(226, 232, 240, 0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      visibleSeries.forEach((series) => {
        const idx = this.findClosestXIndex(series.xValues, hoverXVal);
        if (idx >= 0) {
          const px = toX(series.xValues[idx]);
          const yVal = this.showEma ? series.emaLosses[idx] : series.rawLosses[idx];
          const py = toYLoss(yVal);
          ctx.fillStyle = series.run.color;
          ctx.beginPath();
          ctx.arc(px, py, 3.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }
      });
    }

    ctx.restore(); // End plot clip
    ctx.restore(); // End DPR scale

    // Keep pop-out modal canvas live-synced when open
    if (targetCanvas === this.canvas && this._modalCanvas && document.body.contains(this._modalCanvas)) {
      this.render(this._modalCanvas, hoverXVal, options);
    }
  }

  findClosestXIndex(xValues, targetX) {
    if (!Array.isArray(xValues) || xValues.length === 0) return -1;
    let bestIdx = 0;
    let minDiff = Math.abs(xValues[0] - targetX);
    for (let i = 1; i < xValues.length; i++) {
      const diff = Math.abs(xValues[i] - targetX);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  findClosestPoint(points, targetStep) {
    if (!points || points.length === 0) return null;
    let closest = points[0];
    let minDiff = Math.abs(closest.step - targetStep);
    for (let i = 1; i < points.length; i++) {
      const diff = Math.abs(points[i].step - targetStep);
      if (diff < minDiff) {
        minDiff = diff;
        closest = points[i];
      }
    }
    return closest;
  }

  handlePointerHover(coords, targetCanvas, tooltipEl) {
    const layout = targetCanvas._lastLayout;
    if (!layout) return;

    const { pad, plotW, xMin, xMax } = layout;
    if (coords.x < pad.left || coords.x > pad.left + plotW) {
      this._hoverXVal = null;
      if (tooltipEl) tooltipEl.style.display = "none";
      this.render(targetCanvas);
      return;
    }

    const frac = Math.max(0, Math.min(1, (coords.x - pad.left) / Math.max(1, plotW)));
    const xVal = xMin + frac * (xMax - xMin);
    this._hoverXVal = xVal;

    const visibleRuns = this.runs.filter((r) => r && r.visible !== false && Array.isArray(r.data) && r.data.length > 0);
    if (visibleRuns.length === 0 || !tooltipEl) {
      if (tooltipEl) tooltipEl.style.display = "none";
      this.render(targetCanvas);
      return;
    }

    const activeRun = this.getActiveRun() || visibleRuns[0];
    const activeFiltered = this.filterData(activeRun.data);
    if (activeFiltered.length === 0) {
      tooltipEl.style.display = "none";
      this.render(targetCanvas);
      return;
    }

    const activeXValues = activeFiltered.map((p, idx) => this.getPointX(p, idx));
    const bestIdx = this.findClosestXIndex(activeXValues, xVal);
    const p = activeFiltered[bestIdx];
    if (!p) {
      tooltipEl.style.display = "none";
      this.render(targetCanvas);
      return;
    }

    const activeEma = computeEma(
      activeFiltered.map((pt) => Number(pt.loss) || 0),
      this.smoothing
    );
    const emaVal = activeEma[bestIdx] ?? p.ema_loss ?? p.loss;
    const noiseCat = p.timestep >= 0.67 ? "High t" : p.timestep >= 0.33 ? "Mid t" : "Low t";
    const bucketStr = p.bucket || "default";

    let comparisonRowsHtml = "";
    if (visibleRuns.length > 1) {
      comparisonRowsHtml = `<div style="margin-top:4px; padding-top:4px; border-top:1px solid #1e293b;">`;
      visibleRuns.slice(0, 4).forEach((r) => {
        const fPts = this.filterData(r.data);
        if (fPts.length === 0) return;
        const rX = fPts.map((pt, i) => this.getPointX(pt, i));
        const rIdx = this.findClosestXIndex(rX, xVal);
        const rEma = computeEma(
          fPts.map((pt) => Number(pt.loss) || 0),
          this.smoothing
        );
        const rPt = fPts[rIdx];
        comparisonRowsHtml += `
          <div style="display:flex; justify-content:space-between; gap:8px; align-items:center;">
            <span style="display:flex; align-items:center; gap:4px;">
              <span style="width:6px; height:6px; border-radius:50%; background:${r.color}; display:inline-block;"></span>
              <span>${r.title.slice(0, 16)}</span>
            </span>
            <b>${Number(rEma[rIdx]).toFixed(4)} <span style="color:#64748b; font-weight:400;">(@${rPt.step})</span></b>
          </div>
        `;
      });
      comparisonRowsHtml += `</div>`;
    }

    tooltipEl.innerHTML = `
      <div style="font-weight:600; color:${activeRun.color}; margin-bottom:2px;">
        Step ${p.step} / ${activeRun.max_steps || p.max_steps}
        <span style="color:#64748b; font-weight:400;">(${Number(p.rel_time_sec || 0).toFixed(1)}s)</span>
      </div>
      <div>EMA Loss: <b style="color:#38bdf8;">${Number(emaVal).toFixed(5)}</b> <span style="color:#94a3b8;">(raw ${Number(p.loss).toFixed(5)})</span></div>
      <div>LR: <b>${Number(p.lr).toExponential(2)}</b> · t: <b>${Number(p.timestep).toFixed(3)}</b> <span style="color:#94a3b8;">(${noiseCat})</span></div>
      <div style="color:#94a3b8;">Bucket: <b style="color:#e2e8f0;">${bucketStr}</b> · ${Number(p.step_sec).toFixed(2)}s/st · ${Number(p.vram_gb).toFixed(2)}GB</div>
      ${comparisonRowsHtml}
    `;

    // Position tooltip in unscaled wrapper coordinates
    let tipX = coords.x + 14;
    let tipY = coords.y - 16;
    if (tipX + 235 > coords.clientW) tipX = Math.max(6, coords.x - 240);
    if (tipY + 95 > coords.clientH) tipY = Math.max(6, coords.clientH - 100);
    if (tipY < 6) tipY = 6;

    tooltipEl.style.left = `${Math.round(tipX)}px`;
    tooltipEl.style.top = `${Math.round(tipY)}px`;
    tooltipEl.style.display = "block";

    this.render(targetCanvas);
  }

  exportCsv() {
    const run = this.getActiveRun();
    if (!run || !Array.isArray(run.data) || run.data.length === 0) {
      if (this.statLoss) {
        this.statLoss.textContent = "No data to export";
      }
      return;
    }

    const emaLosses = computeEma(
      run.data.map((d) => Number(d.loss) || 0),
      this.smoothing
    );
    let csv = "step,loss,debiased_ema_loss,lr,timestep,noise_bin,bucket,step_sec,rel_time_sec,vram_gb\n";

    run.data.forEach((p, idx) => {
      const noiseCat = p.timestep >= 0.67 ? "high" : p.timestep >= 0.33 ? "mid" : "low";
      csv += `${p.step},${p.loss},${emaLosses[idx].toFixed(6)},${p.lr},${p.timestep},${noiseCat},"${
        p.bucket || "default"
      }",${p.step_sec},${p.rel_time_sec || 0},${p.vram_gb}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${run.id}_tensorboard_metrics.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  exportPng() {
    const offscreen = document.createElement("canvas");
    offscreen.width = 1600;
    offscreen.height = 900;
    offscreen._cssW = 1600;
    offscreen._cssH = 900;
    offscreen._dpr = 1;
    const ctx = offscreen.getContext("2d");

    ctx.fillStyle = "#090b10";
    ctx.fillRect(0, 0, 1600, 900);

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 24px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText("OmniLoRA TensorBoard Loss & Telemetry Report", 56, 44);

    const activeRun = this.getActiveRun();
    if (activeRun) {
      ctx.fillStyle = activeRun.color;
      ctx.font = "600 16px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      ctx.fillText(
        `${activeRun.title} · Smooth: ${this.smoothing.toFixed(2)} · Fit: ${this.yFitMode.toUpperCase()} · Scale: ${this.scaleMode.toUpperCase()}`,
        56,
        72
      );
    }

    this.render(offscreen, null, { skipClear: true, topOffset: 76 });

    offscreen.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${activeRun?.id || "omni_lora"}_tensorboard_plot.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  openModal() {
    const overlay = document.createElement("div");
    overlay.className = "omni-tracker-modal-overlay";

    const modalBox = document.createElement("div");
    modalBox.className = "omni-tracker-modal-box";

    modalBox.innerHTML = `
      <div class="omni-tracker-header" style="padding:10px 16px; font-size:13px;">
        <span style="font-weight:600; color:#38bdf8;">OmniLoRA TensorBoard Scalar Inspector (Drag to Box-Zoom · Scroll to Zoom · Shift+Drag to Pan · Double-Click to Reset)</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <button class="omni-tracker-btn omni-tracker-modal-reset" style="font-size:11px; padding:3px 8px;">Reset Zoom</button>
          <button class="omni-tracker-btn omni-tracker-modal-sync" style="font-size:11px; padding:3px 8px;">↻ Sync Server</button>
          <button class="omni-tracker-btn omni-tracker-modal-clear" style="font-size:11px; padding:3px 8px; color:#f87171;">Clear Runs</button>
          <button class="omni-tracker-btn omni-tracker-modal-close" style="font-size:13px; padding:2px 9px;">✕</button>
        </div>
      </div>
      <div class="omni-tracker-modal-canvas-wrap" style="flex:1; position:relative; background:#090b10; cursor:crosshair; overflow:hidden;">
        <canvas class="omni-tracker-modal-canvas" style="width:100%; height:100%; display:block;"></canvas>
        <div class="omni-tracker-tooltip omni-tracker-modal-tooltip"></div>
      </div>
      <div class="omni-tracker-header" style="padding:8px 16px; font-size:11px; border-top:1px solid #222738; border-bottom:none;">
        <span style="color:#94a3b8;">Tip: Click 'Fit: Clip Outliers' or 'Fit: EMA Focus' in the node toolbar to inspect subtle convergence slopes.</span>
        <div style="display:flex; gap:6px;">
          <button class="omni-tracker-btn omni-tracker-modal-csv">Export CSV</button>
          <button class="omni-tracker-btn omni-tracker-modal-png">Export PNG</button>
        </div>
      </div>
    `;

    overlay.appendChild(modalBox);
    document.body.appendChild(overlay);

    const modalWrap = modalBox.querySelector(".omni-tracker-modal-canvas-wrap");
    const modalCanvas = modalBox.querySelector(".omni-tracker-modal-canvas");
    const modalTooltip = modalBox.querySelector(".omni-tracker-modal-tooltip");
    this._modalCanvas = modalCanvas;
    this._modalTooltip = modalTooltip;

    this.bindCanvasInteractions(modalWrap, modalCanvas, modalTooltip);

    const close = () => {
      this._modalCanvas = null;
      this._modalTooltip = null;
      document.removeEventListener("keydown", onKey);
      overlay.remove();
    };

    const onKey = (e) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);

    modalBox.querySelector(".omni-tracker-modal-close").addEventListener("click", close);
    modalBox.querySelector(".omni-tracker-modal-reset").addEventListener("click", () => {
      this.zoomDomain = null;
      this.syncControlButtonsUI();
      this.render(modalCanvas);
    });
    modalBox.querySelector(".omni-tracker-modal-sync").addEventListener("click", () => {
      this.syncFromServer({ force: true });
    });
    modalBox.querySelector(".omni-tracker-modal-clear").addEventListener("click", () => {
      this.clearAllRuns();
    });
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });

    modalBox.querySelector(".omni-tracker-modal-csv").addEventListener("click", () => this.exportCsv());
    modalBox.querySelector(".omni-tracker-modal-png").addEventListener("click", () => this.exportPng());

    requestAnimationFrame(() => {
      this.resizeCanvas(modalCanvas);
      this.render(modalCanvas);
    });
  }

  saveState() {
    if (!this.node || this.node.id == null) return;
    try {
      // Only persist lightweight view preferences and dismissed run tombstones.
      // Never cache run step arrays in localStorage so stale runs from old sessions
      // or different workflows sharing the same node ID never stick to the tracker.
      const state = {
        smoothing: this.smoothing,
        noiseBin: this.noiseBin,
        scaleMode: this.scaleMode,
        yFitMode: this.yFitMode,
        ignoreOutliers: this.ignoreOutliers,
        xAxisMode: this.xAxisMode,
        showRaw: this.showRaw,
        showBand: this.showBand,
        showLr: this.showLr,
        dismissedRunIds: Array.from(this._dismissedRunIds).slice(-64),
      };
      localStorage.setItem(`omni_lora_tracker_${this.node.id}`, JSON.stringify(state));
    } catch (_err) {
      // Storage quota or sandboxed iframe
    }
  }

  saveStateThrottled() {
    if (this._saveTimeout) return;
    this._saveTimeout = setTimeout(() => {
      this._saveTimeout = null;
      this.saveState();
    }, 2500);
  }

  loadState() {
    if (!this.node || this.node.id == null) return;
    try {
      const raw = localStorage.getItem(`omni_lora_tracker_${this.node.id}`);
      if (!raw) return;
      const state = JSON.parse(raw);
      if (state.smoothing != null && Number.isFinite(Number(state.smoothing))) {
        this.smoothing = Number(state.smoothing);
        this.slider.value = String(this.smoothing);
        this.smoothVal.textContent = this.smoothing.toFixed(2);
      }
      if (state.scaleMode === "log" || state.scaleMode === "linear") {
        this.scaleMode = state.scaleMode;
      }
      if (state.yFitMode && ["auto", "ema", "full"].includes(state.yFitMode)) {
        this.yFitMode = state.yFitMode;
        this.ignoreOutliers = this.yFitMode !== "full";
      }
      if (state.xAxisMode && ["step", "target", "relative"].includes(state.xAxisMode)) {
        this.xAxisMode = state.xAxisMode;
      }
      if (state.showRaw != null) this.showRaw = Boolean(state.showRaw);
      if (state.showBand != null) this.showBand = Boolean(state.showBand);
      if (state.showLr != null) this.showLr = Boolean(state.showLr);
      if (state.noiseBin) {
        this.noiseBin = state.noiseBin;
        const binButtons = this.element.querySelectorAll(".omni-tracker-bin-btn");
        binButtons.forEach((b) => b.classList.toggle("active", b.dataset.bin === this.noiseBin));
      }
      if (Array.isArray(state.dismissedRunIds)) {
        state.dismissedRunIds.forEach((id) => {
          if (id) this._dismissedRunIds.add(String(id));
        });
      }
      // If a legacy localStorage entry contained cached `runs`, scrub it immediately
      if (Array.isArray(state.runs)) {
        this.saveState();
      }
      this.syncControlButtonsUI();
      this.render();
    } catch (_err) {
      // Ignore corrupted local cache
    }
  }
}

// Register ComfyUI Web Extension
getComfyApp().registerExtension({
  name: "OmniLoRA.LossTracker",

  commands: [
    {
      id: "OmniLoRA.openLossTracker",
      label: "Open OmniLoRA TensorBoard Loss Tracker",
      function: () => {
        const dummyNode = { id: "global_tracker", size: [760, 520] };
        const tracker = new OmniLossTrackerWidget(dummyNode);
        tracker.openModal();
      },
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "OmniLoRALossTracker") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

        // Generous default canvas node dimensions so TensorBoard plot has ample vertical resolution
        this.setSize([640, 460]);
        this.min_width = 420;
        this.min_height = 280;

        this.computeSize = function () {
          return [420, 280];
        };

        const tracker = new OmniLossTrackerWidget(this);

        // Dynamic height getter ensures ComfyUI DOMWidget container always stretches with the node
        // instead of clamping to a fixed 180px height and flattening the canvas!
        const getDynamicWidgetHeight = () => {
          const nodeH = Array.isArray(this.size) ? Number(this.size[1]) : Number(this.size?.[1]) || 460;
          return Math.max(240, (Number.isFinite(nodeH) && nodeH > 150 ? nodeH : 460) - 42);
        };

        this.omniLossTracker = this.addDOMWidget(
          "omni_loss_tracker",
          "omniLossTrackerWidget",
          tracker.element,
          {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => 240,
            getMaxHeight: () => 4000,
            getHeight: getDynamicWidgetHeight,
          }
        );
        this.omniLossTracker.computeSize = (width) => [
          Math.max(380, (Number(width) || 640) - 20),
          getDynamicWidgetHeight(),
        ];

        // Post-mount layout pass so initial canvas buffer matches actual DOM client dimensions
        requestAnimationFrame(() => {
          tracker.resizeCanvas();
          tracker.render();
        });

        const onResize = this.onResize;
        this.onResize = function () {
          const r = onResize ? onResize.apply(this, arguments) : undefined;
          tracker.resizeCanvas();
          tracker.render();
          return r;
        };

        // Ensure saved workflows with legacy small node sizes automatically expand to TensorBoard aspect ratio
        const onConfigure = this.onConfigure;
        this.onConfigure = function (info) {
          const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
          const curW = Array.isArray(this.size) ? Number(this.size[0]) : Number(this.size?.[0]) || 0;
          const curH = Array.isArray(this.size) ? Number(this.size[1]) : Number(this.size?.[1]) || 0;
          if (curW < 540 || curH < 360) {
            this.setSize([Math.max(640, curW), Math.max(440, curH)]);
          }
          requestAnimationFrame(() => {
            tracker.resizeCanvas();
            tracker.render();
            tracker.syncFromServer({ force: true });
          });
          return r;
        };

        // Ingest server runs delivered via ComfyUI's native execution pipeline (Cloudflare fallback)
        const onExecuted = this.onExecuted;
        this.onExecuted = function (message) {
          const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
          if (message && Array.isArray(message.runs) && message.runs.length > 0) {
            tracker.mergeServerRuns(message.runs, false);
          }
          tracker.syncFromServer({ force: true });
          return r;
        };

        const onRemoved = this.onRemoved;
        this.onRemoved = function () {
          tracker.destroy();
          return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };

        return ret;
      };
    }
  },
});
