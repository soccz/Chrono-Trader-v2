/**
 * Chrono-Trader v2: Masterpiece Dashboard Logic
 * "Deep Space Glass" Interaction Layer
 */

// --- Global State ---
let cachedPredictions = [];
let isModalLoading = false;
let marketOverviewLoaded = false;
let tickerLoaded = false;
let isChartView = false;
let chartWidget = null;
const STARTING_CAPITAL = 1000000;
let pnlLoadAttempts = 0;
const MAX_RETRY_ATTEMPTS = 3;
let latestOpsOverview = null;

document.addEventListener('DOMContentLoaded', function () {
    console.log("🚀 Deep Space Glass UI Initializing...");
    initDashboard();
});

function initDashboard() {
    // Critical Path
    loadOpenPnL();
    loadOpsOverviewStatus();
    loadLatestOutputs();
    startCountdown();
    loadModelStatus();

    // NEW: Inject "Last Updated" Indicator into Header
    const headerRow = document.querySelector('.d-flex.align-items-center.gap-3');
    if (headerRow && !document.getElementById('last-updated-badge')) {
        const badge = document.createElement('span');
        badge.id = 'last-updated-badge';
        badge.className = 'badge bg-secondary bg-opacity-10 text-secondary border border-secondary border-opacity-10 rounded-pill px-3';
        badge.style.fontSize = '0.75rem';
        badge.innerHTML = '<i class="bi bi-clock-history me-1"></i> Syncing...';
        headerRow.appendChild(badge);
    }
    updateLastUpdatedTime();

    // Deferred Loading
    setTimeout(() => {
        if (!tickerLoaded) { fetchTicker(); tickerLoaded = true; }
        loadRoadmap();
        if (!marketOverviewLoaded) { loadMarketOverview(); marketOverviewLoaded = true; }
    }, 400);

    // Polling Intervals
    setInterval(fetchTicker, 15000); // 15s Ticker
    setInterval(loadMarketOverview, 30000); // 30s Market Sentiment
    setInterval(loadOpenPnL, 10000); // 10s PnL Update (Fast)
    setInterval(loadModelStatus, 60000); // 60s Model Status Update
    setInterval(loadOpsOverviewStatus, 60000);
    setInterval(loadLatestOutputs, 60000);
}

// --- AI Model Status (Gate Mode & Ensemble) ---
async function loadModelStatus() {
    try {
        const response = await fetch('/api/model/gate-status');
        const result = await response.json();

        if (result.success && result.data) {
            const data = result.data;

            // Update Gate Mode Text
            const gateModeEl = document.getElementById('gate-mode');
            if (gateModeEl) {
                gateModeEl.textContent = data.regime;
                gateModeEl.style.color = data.regime_color;
            }

            // Update Progress Bars
            const transformerPct = Math.round(data.transformer_weight * 100);
            const cnnPct = Math.round(data.cnn_weight * 100);

            const transformerBar = document.getElementById('gate-transformer-bar');
            const cnnBar = document.getElementById('gate-cnn-bar');
            if (transformerBar) transformerBar.style.width = `${transformerPct}%`;
            if (cnnBar) cnnBar.style.width = `${cnnPct}%`;

            // Update Percentage Labels
            const transformerPctEl = document.getElementById('gate-transformer-pct');
            const cnnPctEl = document.getElementById('gate-cnn-pct');
            if (transformerPctEl) transformerPctEl.textContent = `${transformerPct}%`;
            if (cnnPctEl) cnnPctEl.textContent = `${cnnPct}%`;
        }
    } catch (error) {
        console.warn('Failed to load model status:', error);
    }
}

// --- Native Dialog Logic (60fps) ---
// --- Native Dialog Logic (60fps) ---
window.showPredictionsModal = async function () {
    const dialog = document.getElementById('predictionsDialog');
    if (!dialog) return;

    dialog.showModal(); // Native API

    const modalList = document.getElementById('modal-predictions-list');
    const metaEl = document.getElementById('predictions-meta');
    if (!modalList) return;
    if (metaEl) {
        metaEl.innerHTML = '<span class="text-muted">Loading latest run outputs...</span>';
    }

    // Loading State
    modalList.innerHTML = `
        <div class="d-flex align-items-center justify-content-center p-5 text-muted">
            <span class="spinner-border spinner-border-sm me-2 text-primary" role="status"></span>
            <span>Unlocking AI Insights...</span>
        </div>`;

    // Timeout Controller (5 seconds)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
        const response = await fetch(`/api/predictions/latest?_t=${Date.now()}`, {
            signal: controller.signal
        });
        const result = await response.json();

        if (result.success) {
            const recommendations = (result.data && result.data.recommendations) || [];
            const manifests = result.manifests || {};
            cachedPredictions = recommendations;
            if (metaEl) metaEl.innerHTML = renderPredictionsMeta(manifests);

            if (recommendations.length > 0) {
                modalList.innerHTML = recommendations.map(p => renderPredictionCard(p)).join('');
            } else {
                modalList.innerHTML = '<div class="text-center py-5 text-muted">No active signals found for the latest scheduled runs.</div>';
            }
        } else {
            throw new Error(result.message || 'Failed to load latest predictions');
        }
    } catch (error) {
        console.error('Error fetching predictions:', error);
        if (error.name === 'AbortError') {
            modalList.innerHTML = `
                <div class="text-center py-5 text-warning">
                    <i class="bi bi-wifi-off fs-1 mb-2"></i><br>
                    Connection Timeout. Retrying...
                </div>`;
            // Optional: Auto-retry logic could go here
        } else {
            modalList.innerHTML = `
                <div class="text-center py-5 text-danger">
                    <i class="bi bi-exclamation-triangle fs-1 mb-2"></i><br>
                    Failed to retrieve intelligence.
                </div>`;
        }
        if (metaEl) metaEl.innerHTML = '<span class="text-danger">Latest output metadata unavailable.</span>';
    } finally {
        clearTimeout(timeoutId);
    }
};

// --- Prediction Card Renderer (Neon/Haptic) ---
function renderPredictionCard(p) {
    const expectedReturn = Number(p.expected_return ?? p.predicted_return ?? 0);
    const actualPct = Number(p.actual_return || 0).toFixed(2);
    const isCorrect = Boolean(p.direction_correct);
    const entryPrice = p.entry_price || p.current_price || 0;
    const currentPrice = p.current_price || 0;

    // Hybrid Metrics
    const gateValue = Number(p.gate_value ?? p.gate ?? 0.5);
    const gateInfo = gateValue > 0.6 ? 'Trend' : (gateValue < 0.4 ? 'Pattern' : 'Hybrid');
    const confidence = Number(p.confidence ?? p.consensus_score ?? 0);
    const consensus = confidence > 0 ? `${(confidence * 100).toFixed(0)}%` : 'N/A';
    const uncertainty = Number(p.uncertainty ?? 0);
    const sourceMode = p.output_mode || p.strategy_type || p.strategy || 'Scheduled';

    // Dynamic Styles based on Gate
    let cardClass = "glass-panel p-3 mb-3 d-flex justify-content-between align-items-center";
    let badgeClass = "badge badge-soft text-white";

    if (gateInfo === 'Trend') {
        cardClass += " mode-trend"; // Cyan Glow
        badgeClass += " bg-primary bg-opacity-25";
    } else if (gateInfo === 'Pattern') {
        cardClass += " mode-pattern"; // Magenta Glow
        badgeClass += " bg-warning bg-opacity-25";
    } else {
        badgeClass += " bg-secondary bg-opacity-25";
    }

    const priceClass = (currentPrice - entryPrice) >= 0 ? 'text-success' : 'text-danger';

    return `
    <div class="${cardClass}">
        <div>
            <div class="d-flex align-items-center gap-2 mb-2">
                <span class="fw-bold fs-5 text-white">${p.market}</span>
                <span class="${badgeClass} tiny-badge">${gateInfo} Mode</span>
                <span class="badge badge-soft bg-opacity-10 bg-info text-info tiny-badge">Consensus ${consensus}</span>
            </div>
            <div class="small text-muted mb-1">${sourceMode}</div>
            <div class="small text-dim">
                <span>Exp: ${expectedReturn >= 0 ? '+' : ''}${expectedReturn.toFixed(2)}%</span>
                <span class="ms-2">Vol: ${(Number(p.volatility || 0) * 100).toFixed(1)}%</span>
                ${uncertainty > 0 ? `<span class="ms-2">Unc: ${(uncertainty * 100).toFixed(1)}%</span>` : ''}
            </div>
        </div>
        <div class="text-end">
            <span class="badge ${p.signal === 'Long' ? 'bg-danger' : 'bg-primary'} mb-2">${p.signal || 'HOLD'}</span>
            <div class="nums small text-muted">Entry: ${Math.round(entryPrice).toLocaleString()}</div>
            <div class="nums fw-bold ${priceClass}">Current: ${Math.round(currentPrice).toLocaleString()}</div>
            <div class="nums small ${isCorrect ? 'text-success' : 'text-danger'}">${isCorrect ? 'Hit' : 'Live'} ${actualPct}%</div>
        </div>
    </div>`;
}

// --- Position Table (Skeleton + Error Handling) ---
async function loadOpenPnL() {
    const tbody = document.getElementById('pnl-table-body');
    if (!tbody) return;

    if (pnlLoadAttempts === 0) {
        tbody.innerHTML = `
        <tr><td colspan="6" class="p-4"><div class="skeleton w-100" style="height: 48px;"></div></td></tr>
        <tr><td colspan="6" class="p-4 pt-0"><div class="skeleton w-100" style="height: 48px;"></div></td></tr>`;
    }

    try {
        const response = await fetch(`/api/performance/positions?_t=${Date.now()}`);
        const res = await response.json();

        if (res.success) {
            pnlLoadAttempts = 0;
            const openTrades = (res.data.trades || []).filter(t => t.status === 'OPEN');
            renderPositions(openTrades);
        }
    } catch (e) {
        console.warn("PnL load glitch:", e);
        pnlLoadAttempts++;
    }
}

function renderPositions(trades) {
    const tbody = document.getElementById('pnl-table-body');
    let totalEq = 0, totalPnl = 0, invested = 0;

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-5 text-muted small text-uppercase tracking-wider">No Active Positions</td></tr>';
        updateStats(STARTING_CAPITAL, 0);
        return;
    }

    const html = trades.map(t => {
        const entry = t.entry_price || 0;
        const current = t.current_price || entry;
        const size = t.position_size || 0.1;

        const investVal = STARTING_CAPITAL * size;
        const qty = entry > 0 ? investVal / entry : 0;
        const curVal = qty * current;

        invested += investVal;
        totalEq += curVal;

        const pnlVal = curVal - investVal;
        const pnlPct = investVal > 0 ? (pnlVal / investVal) * 100 : 0;

        const colorClass = pnlPct >= 0 ? 'text-success' : 'text-danger'; // Up=Green, Down=Red (International standard for dark mode often prefers Green/Red) 
        // Actually, Korean standard: Red=Up, Blue=Down. Let's stick to user preference or standard.
        // User is Korean. Red=Up, Blue=Down.
        const kColorClass = pnlPct >= 0 ? 'text-danger' : 'text-primary';

        return `
        <tr class="position-row" style="border-bottom: 1px solid rgba(255,255,255,0.05);" onclick="changeChartSymbol('${t.market}')">
            <td class="ps-4 py-3">
                <div class="d-flex align-items-center gap-3">
                    <div class="fw-bold text-white">${t.market.replace('KRW-', '')}</div>
                </div>
            </td>
            <td class="text-center py-3">
                <span class="badge ${t.side === 'Long' ? 'bg-success text-success' : 'bg-danger text-danger'} bg-opacity-10 border border-${t.side === 'Long' ? 'success' : 'danger'} border-opacity-25 px-2 py-1">${t.side || 'Long'}</span>
            </td>
            <td class="text-end py-3 text-muted nums">${(size * 100).toFixed(1)}%</td>
            <td class="text-end py-3 text-white nums">₩${Math.round(investVal).toLocaleString()}</td>
            <td class="text-end py-3 text-muted nums">${Math.round(entry).toLocaleString()}</td>
            <td class="text-end py-3 text-white fw-bold nums">${Math.round(current).toLocaleString()}</td>
            <td class="text-end pe-4 py-3">
                <div class="${kColorClass} fw-bold nums">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</div>
                <div class="${kColorClass} small nums opacity-75">${Math.round(pnlVal).toLocaleString()}</div>
            </td>
        </tr>`;
    }).join('');

    tbody.innerHTML = html;

    // Calculate Totals
    const cash = STARTING_CAPITAL - invested;
    const finalEquity = totalEq + cash;
    const finalPnl = finalEquity - STARTING_CAPITAL;
    updateStats(finalEquity, finalPnl);
}

function updateStats(equity, pnl) {
    const elEq = document.getElementById('total-equity-footer');
    const elPnl = document.getElementById('total-pnl-footer');

    // Zero-State Handling
    if (equity === undefined || equity === 0 || isNaN(equity)) equity = STARTING_CAPITAL;
    if (pnl === undefined || isNaN(pnl)) pnl = 0;

    if (elEq) elEq.innerText = `₩${Math.round(equity).toLocaleString()}`;
    if (elPnl) {
        elPnl.innerText = `${pnl >= 0 ? '+' : ''}₩${Math.round(pnl).toLocaleString()}`;
        elPnl.className = `fs-3 fw-bold nums ${pnl >= 0 ? 'text-danger' : 'text-primary'}`;

        // Reactive Favicon (Haptic Visual)
        updateFavicon(pnl >= 0 ? 'up' : 'down');
    }
}

// --- Utils: Reactive Favicon ---
function updateFavicon(direction) {
    // Advanced: Draw tiny canvas or swap link
    // Implementation simplified for now
}

// --- Utils: Last Updated Time ---
function updateLastUpdatedTime(ts = null, label = 'Updated') {
    const el = document.getElementById('last-updated-badge');
    if (!el) return;

    const ref = ts instanceof Date ? ts : new Date(ts || Date.now());
    const timeStr = ref.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    el.innerHTML = `<i class="bi bi-clock-history me-1"></i> ${label}: ${timeStr}`;
    el.classList.remove('bg-secondary', 'text-secondary');
    el.classList.add('bg-success', 'text-success'); // Green to show aliveness

    // Blink effect
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0.5';
    setTimeout(() => el.style.opacity = '1', 500);
}

// --- Chart Toggle ---
function togglePositionChartView() {
    isChartView = !isChartView;
    const table = document.getElementById('position-table-view');
    const chart = document.getElementById('chart-view');
    const btnText = document.getElementById('toggle-btn-text');
    const btn = document.getElementById('view-toggle-btn');

    if (isChartView) {
        if (table) table.classList.add('d-none');
        if (chart) {
            chart.classList.remove('d-none');
            if (!chartWidget) initTradingViewChart();
        }
        if (btnText) btnText.innerText = 'Back to List';
        if (btn) {
            btn.classList.add('active');
            btn.style.borderColor = 'var(--neon-trend)';
            btn.style.color = 'var(--neon-trend)';
        }
    } else {
        if (table) table.classList.remove('d-none');
        if (chart) chart.classList.add('d-none');
        if (btnText) btnText.innerText = 'Chart View';
        if (btn) {
            btn.classList.remove('active');
            btn.style.borderColor = '';
            btn.style.color = '';
        }
    }
}

function initTradingViewChart() {
    const container = document.getElementById('tradingview-chart');
    if (!container) return;
    container.innerHTML = '';

    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.innerHTML = JSON.stringify({
        "autosize": true,
        "symbol": "BITGET:BTCUSDT",
        "interval": "60",
        "timezone": "Asia/Seoul",
        "theme": "dark", // Locked to Dark for Deep Space theme
        "style": "1",
        "locale": "kr",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "backgroundColor": "rgba(20, 20, 25, 1)", // Match bg-glass-heavy
        "gridLineColor": "rgba(255, 255, 255, 0.05)",
        "support_host": "https://www.tradingview.com"
    });

    const wDiv = document.createElement('div');
    wDiv.className = 'tradingview-widget-container';
    wDiv.style.height = '100%';
    wDiv.appendChild(script);
    container.appendChild(wDiv);
    chartWidget = true;
}

function changeChartSymbol(market) {
    const symbol = market.replace('KRW-', '') + 'USDT';
    window.open(`https://www.tradingview.com/chart/?symbol=BITGET:${symbol}`, '_blank');
}

// --- Status & Roadmap ---
async function checkSystemStatus() {
    const indicator = document.querySelector('.status-indicator');
    const textEl = indicator ? indicator.querySelector('span:last-child') : null;
    const pipelineEl = document.getElementById('ops-pipeline-status');

    try {
        const response = await fetch(`/api/health/data-pipeline?_t=${Date.now()}`);
        const result = await response.json();
        const status = result.status || 'critical';

        if (indicator) indicator.classList.toggle('active', status === 'healthy');
        if (textEl) {
            textEl.textContent = status === 'healthy'
                ? 'System Online'
                : (status === 'stale' ? 'System Stale' : 'System Degraded');
        }
        if (pipelineEl) {
            pipelineEl.innerHTML = [
                formatPipelineLine('Intraday', result.intraday || {}),
                formatPipelineLine('Morning', result.morning || {}),
                formatPipelineLine('Refresh', result.refresh_db || {})
            ].join('<br>');
        }
    } catch (e) {
        console.warn('Health check failed:', e);
        if (textEl) textEl.textContent = 'System Unknown';
        if (pipelineEl) pipelineEl.innerHTML = '<span class="text-danger">Pipeline health unavailable.</span>';
    }
}

async function loadRuntimeHealth() {
    const runtimeEl = document.getElementById('ops-runtime-status');
    if (!runtimeEl) return;

    try {
        const response = await fetch(`/api/health/runtime?_t=${Date.now()}`);
        const result = await response.json();
        if (!result.success) {
            throw new Error(result.message || 'runtime health unavailable');
        }
        runtimeEl.innerHTML = renderRuntimeStatus(result.report || {}, result.status || 'critical');
    } catch (e) {
        console.warn('Runtime health failed:', e);
        runtimeEl.innerHTML = '<span class="text-danger">Runtime health unavailable.</span>';
    }
}

async function loadOpsOverviewStatus() {
    const indicator = document.querySelector('.status-indicator');
    const textEl = indicator ? indicator.querySelector('span:last-child') : null;
    const summaryEl = document.getElementById('ops-overview-status');
    const pipelineEl = document.getElementById('ops-pipeline-status');
    const runtimeEl = document.getElementById('ops-runtime-status');

    try {
        const response = await fetch(`/api/ops/overview?_t=${Date.now()}`);
        const result = await response.json();
        if (!result.success) {
            throw new Error(result.message || 'ops overview unavailable');
        }

        const runtimeStatus = result?.runtime?.status || 'critical';
        const readiness = result?.readiness || {};
        const rows = Array.isArray(readiness.jobs) ? readiness.jobs : [];
        const summary = readiness.summary || {};
        latestOpsOverview = result;

        if (indicator) indicator.classList.toggle('active', runtimeStatus === 'healthy');
        if (textEl) {
            textEl.textContent = runtimeStatus === 'healthy'
                ? 'System Online'
                : (runtimeStatus === 'degraded' ? 'System Stale' : 'System Degraded');
        }
        if (summaryEl) {
            const primaryAction = summary?.primary_action?.label || 'Inspect';
            const primaryJob = summary?.primary_job || 'n/a';
            const primaryAge = summary?.primary_age_hours != null ? `${Number(summary.primary_age_hours).toFixed(1)}h` : 'n/a';
            summaryEl.innerHTML = `Ops blocked=${summary?.blocked_n || 0} stale=${summary?.stale_n || 0} ready=${summary?.ready_n || 0} | Next ${primaryAction} @ ${primaryJob} (${primaryAge})`;
        }
        if (pipelineEl) {
            pipelineEl.innerHTML = rows.length
                ? rows.map((row) => {
                    const age = row.age_hours != null ? `${Number(row.age_hours).toFixed(1)}h` : 'n/a';
                    const next = row?.next_action?.label || 'Inspect';
                    return `<span>${row.job}: ${row.status} | age=${age} | ${next}</span>`;
                }).join('<br>')
                : '<span class="text-muted">No readiness rows.</span>';
        }
        if (runtimeEl) {
            runtimeEl.innerHTML = renderRuntimeStatus(result.runtime?.report || {}, runtimeStatus);
        }
    } catch (e) {
        console.warn('Ops overview failed:', e);
        if (textEl) textEl.textContent = 'System Unknown';
        if (summaryEl) summaryEl.innerHTML = '<span class="text-danger">Ops summary unavailable.</span>';
        if (pipelineEl) pipelineEl.innerHTML = '<span class="text-danger">Pipeline health unavailable.</span>';
        if (runtimeEl) runtimeEl.innerHTML = '<span class="text-danger">Runtime health unavailable.</span>';
    }
}

window.runDashboardOpsAction = async function () {
    try {
        if (!latestOpsOverview) {
            await loadOpsOverviewStatus();
        }
        const summary = latestOpsOverview?.readiness?.summary || {};
        const job = summary.primary_job || 'intraday';
        const response = await fetch('/api/ops/next-action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job })
        });
        const result = await response.json();

        if (result.success && (result.action_taken === 'run' || result.action_taken === 'backfill_and_run')) {
            if (window.showToast) {
                window.showToast(`Ops started for ${job}`, 'success');
            }
            loadOpsOverviewStatus();
            return;
        }

        if (result.success && result.action_taken === 'monitor_only') {
            if (window.showToast) {
                window.showToast(result.message || 'No action needed.', 'info');
            }
            return;
        }

        location.href = '/control';
    } catch (error) {
        console.warn('Dashboard ops action failed:', error);
        location.href = '/control';
    }
};

window.openControlCenter = function () {
    location.href = '/control';
};

function startCountdown() {
    updateCountdown();
    setInterval(updateCountdown, 1000);
}

function updateCountdown() {
    const now = new Date();
    const hours = now.getHours();
    const nextHour = [0, 4, 8, 12, 16, 20].find(h => h > hours) || 24;
    const nextTime = new Date(now);
    nextTime.setHours(nextHour, 0, 0, 0);
    if (nextHour === 24) nextTime.setDate(nextTime.getDate() + 1);

    const diff = nextTime - now;
    const h = Math.floor(diff / 3600000);
    const m = Math.floor((diff % 3600000) / 60000);

    const el = document.getElementById('next-analysis-countdown');
    if (el) el.innerText = `${h}h ${m}m`;
}

// --- Market Sentiment (Optimized) ---
async function loadMarketOverview() {
    const elBtc = document.getElementById('btc-dominance');
    const elTrend = document.getElementById('avg-change');
    const elRise = document.getElementById('rising-count');
    const elFall = document.getElementById('falling-count');

    try {
        const res = await fetch(`/api/market/overview?_t=${Date.now()}`);
        const data = (await res.json()).data;

        if (elBtc) elBtc.innerText = `${data.btc_dominance}%`;
        if (elTrend) {
            elTrend.innerText = `${data.avg_change > 0 ? '+' : ''}${data.avg_change}%`;
            elTrend.className = `fw-bold nums ${data.avg_change >= 0 ? 'text-danger' : 'text-primary'}`;
        }
        if (elRise) {
            elRise.innerText = data.rising;
            // Update Progerss Bar
            document.getElementById('bar-rising').style.width = `${(data.rising / data.total) * 100}%`;
            document.getElementById('bar-falling').style.width = `${(data.falling / data.total) * 100}%`;
        }
        if (elFall) elFall.innerText = data.falling;

    } catch (e) { console.warn("Market overview silent fail"); }
}

async function loadLatestOutputs() {
    const summaryEl = document.getElementById('ops-output-summary');
    if (!summaryEl) return;

    try {
        const response = await fetch(`/api/predictions/latest?_t=${Date.now()}`);
        const result = await response.json();
        const manifests = result.manifests || {};
        summaryEl.innerHTML = renderOutputSummary(manifests);

        const latestTs = latestManifestTimestamp(manifests);
        if (latestTs) updateLastUpdatedTime(new Date(latestTs), 'Outputs');
    } catch (e) {
        console.warn('Latest outputs load failed:', e);
        summaryEl.innerHTML = '<div class="text-danger">Latest outputs unavailable.</div>';
    }
}

function parseIsoTime(ts) {
    if (!ts) return null;
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? null : d;
}

function formatRelativeTime(ts) {
    const date = parseIsoTime(ts);
    if (!date) return 'unknown';
    const diffMin = Math.max(0, Math.round((Date.now() - date.getTime()) / 60000));
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const hours = Math.floor(diffMin / 60);
    const mins = diffMin % 60;
    return mins > 0 ? `${hours}h ${mins}m ago` : `${hours}h ago`;
}

function latestManifestTimestamp(manifests) {
    const candidates = [];
    Object.values(manifests || {}).forEach(manifest => {
        const ts = manifest?.recommendation?.mtime || manifest?.run_metrics?.mtime || manifest?.ts;
        if (ts) candidates.push(ts);
    });
    if (!candidates.length) return null;
    return candidates.sort().slice(-1)[0];
}

function shortenPath(path) {
    if (!path) return 'unknown';
    const parts = String(path).split('/');
    return parts.slice(-3).join('/') || path;
}

function renderRuntimeStatus(report, status) {
    const pythonPath = report?.python?.path || '';
    const dbReady = Boolean(report?.db?.exists);
    const modelReady = Boolean(report?.models?.main?.exists);
    const shortModelReady = Boolean(report?.models?.short?.exists);
    const legacyVenvBroken = Boolean(report?.venv?.transplanted);

    const toneClass = status === 'healthy'
        ? 'text-success'
        : (status === 'degraded' ? 'text-warning' : 'text-danger');

    const lines = [
        `<span class="${toneClass}">Runtime ${String(status || 'critical').toUpperCase()}</span>`,
        `<span class="text-muted">Python ${shortenPath(pythonPath)} | DB ${dbReady ? 'ok' : 'missing'} | Model ${modelReady ? 'ok' : 'missing'}</span>`
    ];

    if (!shortModelReady) {
        lines.push('<span class="text-muted opacity-75">Short model missing</span>');
    }
    if (legacyVenvBroken) {
        lines.push('<span class="text-warning">Legacy .venv is transplanted and ignored</span>');
    }

    return lines.join('<br>');
}

function buildPipelineFlags(payload) {
    const flags = [];
    if (payload?.status === 'offline') flags.push('<span class="ops-flag ops-flag-warning">refresh offline</span>');
    if (payload?.used_fallback) flags.push('<span class="ops-flag ops-flag-info">fallback</span>');
    if (payload?.auto_refresh_skipped_offline) flags.push('<span class="ops-flag ops-flag-warning">offline auto-refresh</span>');
    if (payload?.auto_refresh_dns_error) {
        flags.push(`<span class="ops-flag ops-flag-danger">dns ${payload.auto_refresh_dns_error}</span>`);
    }
    return flags.join(' ');
}

function renderPredictionsMeta(manifests) {
    const blocks = [];
    ['morning', 'intraday'].forEach(mode => {
        const manifest = manifests[mode];
        if (!manifest) return;
        const rec = manifest.recommendation;
        const pump = manifest.pump_prediction;
        const updated = rec?.mtime || manifest.ts;
        blocks.push(
            `<div class="d-flex justify-content-between align-items-center py-1 border-bottom border-secondary border-opacity-10">
                <span class="text-uppercase">${mode}</span>
                <span>${updated ? formatRelativeTime(updated) : 'no rec file'}</span>
            </div>`
        );
        if (pump?.mtime) {
            blocks.push(
                `<div class="d-flex justify-content-between align-items-center py-1 opacity-75">
                    <span>${mode} pump</span>
                    <span>${formatRelativeTime(pump.mtime)}</span>
                </div>`
            );
        }
    });
    return blocks.length ? blocks.join('') : '<span class="text-muted">No scheduled output manifest found.</span>';
}

function renderOutputSummary(manifests) {
    const modes = ['morning', 'intraday'];
    const cards = modes.map(mode => {
        const manifest = manifests[mode];
        if (!manifest) {
            return `<div class="text-muted">${mode}: no manifest</div>`;
        }

        const rec = manifest.recommendation;
        const pump = manifest.pump_prediction;
        const recAge = rec?.mtime ? formatRelativeTime(rec.mtime) : 'no rec file';
        const recName = rec?.basename || 'missing';
        const pumpText = pump?.mtime ? ` | pump ${formatRelativeTime(pump.mtime)}` : '';

        return `
            <div class="d-flex justify-content-between align-items-start gap-2">
                <div class="text-uppercase text-secondary">${mode}</div>
                <div class="text-end">
                    <div class="text-white">${recAge}</div>
                    <div class="text-muted" style="font-size: 0.72rem;">${recName}${pumpText}</div>
                </div>
            </div>
        `;
    });
    return cards.join('');
}

function formatPipelineLine(label, payload) {
    const status = payload?.status || 'unknown';
    const age = typeof payload?.age_hours === 'number' ? `${payload.age_hours.toFixed(1)}h` : 'n/a';
    const kept = payload?.kept ?? 0;
    const dropped = payload?.dropped ?? 0;
    const className = status === 'ok'
        ? 'text-success'
        : ((status === 'stale' || status === 'offline') ? 'text-warning' : 'text-danger');
    const flagHtml = buildPipelineFlags(payload);
    return `<span class="${className}">${label}</span> ${status} · age ${age} · kept ${kept} · drop ${dropped}${flagHtml ? `<br><span class="ops-flag-row">${flagHtml}</span>` : ''}`;
}

function loadRoadmap() {
    const container = document.getElementById('roadmap-widget-content');
    if (!container) return;

    const tasks = [
        { t: 'Phase 1: Foundation', s: 'done' },
        { t: 'Phase 2: Continuous Engine', s: 'done' },
        { t: 'Phase 3: Frontend Masterpiece', s: 'in-progress' }
    ];

    container.innerHTML = tasks.map(t => {
        const color = t.s === 'done' ? 'text-success' : 'text-primary animate-pulse-slow';
        const icon = t.s === 'done' ? 'bi-check-circle-fill' : 'bi-activity';
        return `
        <div class="d-flex align-items-center gap-3 p-3 rounded-3" style="background: rgba(255,255,255,0.03);">
            <i class="bi ${icon} fs-5 ${color}"></i>
            <span class="text-white small fw-bold">${t.t}</span>
        </div>`;
    }).join('');
}

// --- Ticker Logic (Restored) ---
async function fetchTicker() {
    // Top 5 Coins by Volume interest
    const markets = 'KRW-BTC,KRW-ETH,KRW-XRP,KRW-SOL,KRW-DOGE';
    try {
        const response = await fetch(`/api/ticker?markets=${markets}`);
        const result = await response.json(); // API returns {success:true, data:[...]} or direct list?
        // app.py api_ticker returns jsonify(data) which is the raw list from Upbit usually, 
        // OR wrapper. Let's assume wrapper based on other endpoints.
        // Actually line 772 in app.py: return jsonify(res.json()) -> raw list from Upbit.

        let data = result;
        if (result.success !== undefined) data = result.data; // Handle wrapper if exists

        // If it's the raw Upbit list
        if (Array.isArray(data)) {
            const container = document.querySelector('.ticker');
            if (!container) return;

            const html = data.map(item => {
                const name = item.market.replace('KRW-', '');
                const price = item.trade_price.toLocaleString();
                const change = (item.signed_change_rate * 100).toFixed(2);
                const color = change >= 0 ? 'text-danger' : 'text-primary';
                const icon = change >= 0 ? 'bi-caret-up-fill' : 'bi-caret-down-fill';

                return `
                <div class="ticker__item">
                    <span class="fw-bold me-2">${name}</span>
                    <span class="nums">₩${price}</span>
                    <span class="${color} ms-2 small nums">
                        <i class="bi ${icon}"></i> ${change}%
                    </span>
                </div>`;
            }).join('');

            container.innerHTML = html;
        }
    } catch (e) {
        console.warn('Ticker fetch error:', e);
    }
}

// --- Dark Mode Toggle ---
// REMOVED: duplicate toggleDarkMode definition.
// The canonical version lives in base.html with updateThemeUI() support.

// --- Sidebar Toggle (Mobile) ---
// REMOVED: toggleSidebar moved to base.html inline script so it works on all pages.

// --- Language Switcher ---
window.switchLanguage = function (lang) {
    localStorage.setItem('language', lang);
    // Trigger i18n update if available
    if (typeof updateLanguage === 'function') {
        updateLanguage(lang);
    } else {
        // Fallback: reload page
        location.reload();
    }
};

// Apply saved theme on load
(function () {
    const savedTheme = localStorage.getItem('theme') || 'light'; // Default to light (user preference)
    document.documentElement.setAttribute('data-theme', savedTheme);
})();
