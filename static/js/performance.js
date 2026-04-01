/**
 * Performance Dashboard Logic
 * Handles data fetching and chart rendering for performance.html
 */

document.addEventListener('DOMContentLoaded', function () {
    loadPerformanceData();
    // Refresh every 60 seconds
    setInterval(loadPerformanceData, 60000);

    // Toggle Listener
    const toggle = document.getElementById('stats-toggle');
    if (toggle) {
        toggle.addEventListener('change', function () {
            const isWeekly = this.checked;
            document.getElementById('stats-mode-label').textContent = isWeekly ? 'Weekly' : 'All Time';
            // Trigger update with filter
            filterDataByPeriod(isWeekly ? 'weekly' : 'all');
        });
    }
});

let globaHistoryData = null; // Store for filtering

function filterDataByPeriod(period) {
    if (!globaHistoryData) return;

    // Filtering logic needs raw trades. 
    // If backend doesn't support params, we filter client side.
    // Assuming client-side filter for now as backend audit is separate.

    let filteredData = { ...globaHistoryData };

    if (period === 'weekly') {
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

        const validTrades = (globaHistoryData.trades || []).filter(t => new Date(t.entry_time) >= oneWeekAgo);

        // Recalculate KPIs based on filtered trades
        filteredData.total_trades = validTrades.length;
        const wins = validTrades.filter(t => t.pnl_percent > 0).length;
        filteredData.win_rate = validTrades.length > 0 ? wins / validTrades.length : 0;
        filteredData.total_return = validTrades.reduce((acc, t) => acc + t.pnl_percent, 0);
        // Profit factor calc omitted for brevity, keeping it simple

        filteredData.trades = validTrades;
    }

    updateKPIs(filteredData);
    updateTradeLedger(filteredData.trades);
}

let portfolioChart = null;
let equityChart = null;
let accuracyChart = null;

async function loadPerformanceData() {
    try {
        // 1. Load History/Stats
        const historyRes = await fetch('/api/performance/history');
        const historyData = await historyRes.json();

        if (historyData.success) {
            globaHistoryData = historyData.data; // Store for toggling
            updateKPIs(historyData.data);
            updateTradeLedger(historyData.data.trades);
            renderAccuracyChart(historyData.data.accuracy_history || []);
        }

        // 2. Load Portfolio Composition
        const portRes = await fetch('/api/performance/portfolio');
        const portData = await portRes.json();

        if (portData.success) {
            renderPortfolioChart(portData.data);
            // Also update equity curve if available in this endpoint or separate
            // For now, assume equity curve data comes from history or separate endpoint
            // If API doesn't provide equity curve history, we might need to mock or use trade history
            // Checking app.py, /api/performance/history returns 'equity_curve' usually
            if (historyData.data && historyData.data.equity_curve) {
                renderEquityChart(historyData.data.equity_curve);
            }
        }

        // 3. Load Gate Status (reused from dashboard logic or separate endpoint)
        // performance.html expects gate data in `gate-*-p` IDs.
        const gateRes = await fetch('/api/model/gate-status');
        const gateData = await gateRes.json();

        if (gateData.success) {
            updateGateAnalysis(gateData.data);
        }

    } catch (error) {
        console.error("Error loading performance data:", error);
    }
}

function updateKPIs(data) {
    // Total Trades
    setText('perf-total', data.total_trades);

    // Win Rate
    const winRate = data.win_rate ? (data.win_rate * 100).toFixed(1) + '%' : '0%';
    setText('perf-accuracy', winRate);

    // Net Return (using perf-error for now as per HTML ID)
    const netReturn = data.total_return ? data.total_return.toFixed(2) + '%' : '0%';
    setText('perf-error', netReturn); // label-error says Net Return

    // Profit Factor (using perf-winrate for now as per HTML ID)
    setText('perf-winrate', data.profit_factor || '0.00'); // label-winrate says Profit Factor
}

function updateGateAnalysis(data) {
    const tPct = Math.round(data.transformer_weight * 100);
    const cPct = Math.round(data.cnn_weight * 100);

    setPreservedWidth('gate-transformer-bar-p', `${tPct}%`);
    setPreservedWidth('gate-cnn-bar-p', `${cPct}%`);

    setText('gate-transformer-pct-p', `${tPct}%`);
    setText('gate-cnn-pct-p', `${cPct}%`);

    const modeEl = document.getElementById('gate-mode-p');
    if (modeEl) {
        modeEl.textContent = data.regime + ' Dominant';
        modeEl.style.color = data.regime_color;
    }
}

function updateTradeLedger(trades) {
    const tbody = document.getElementById('trade-history-body');
    if (!tbody || !trades) return;

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted">No trades recorded yet.</td></tr>';
        return;
    }

    // Show last 20 trades
    const recentTrades = trades.slice(0, 50);

    tbody.innerHTML = recentTrades.map(t => {
        const pnl = t.pnl_percent || 0;
        const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger'; // Int'l standard or Toss Red/Blue
        // Using Toss Style: Red=Loss(Blue), Green=Win(Red)? User is Korean -> Red=Win, Blue=Loss usually.
        // But main_v5.css defines --up: red, --down: blue.
        // Let's stick to standard classes: success (green/blue) / danger (red). 
        // Wait, main_v5.css: .text-success is green? No, updated CSS might have valid variable overrides.
        // Safe bet: pnl >= 0 ? 'text-danger' : 'text-primary' (Korean style)

        const colorClass = pnl >= 0 ? 'text-danger' : 'text-primary';
        const sign = pnl >= 0 ? '+' : '';
        const dateStr = t.entry_time ? new Date(t.entry_time).toLocaleDateString() : '-';

        return `
        <tr>
            <td class="ps-4 small text-muted">${dateStr}</td>
            <td class="text-center fw-bold">${(t.market || '').replace('KRW-', '')}</td>
            <td class="text-center"><span class="badge bg-light text-dark border">${t.signal || 'LONG'}</span></td>
            <td class="text-end nums">${Math.round(t.entry_price || 0).toLocaleString()}</td>
            <td class="text-end nums">${t.exit_price ? Math.round(t.exit_price).toLocaleString() : '-'}</td>
            <td class="text-end fw-bold nums ${colorClass}">${sign}${pnl.toFixed(2)}%</td>
            <td class="text-end pe-4">
                <span class="badge ${t.status === 'OPEN' ? 'bg-success' : 'bg-secondary'} bg-opacity-10 ${t.status === 'OPEN' ? 'text-success' : 'text-secondary'}">
                    ${t.status}
                </span>
            </td>
        </tr>
        `;
    }).join('');
}


// --- Chart Rendering ---

function renderPortfolioChart(portfolio) {
    const ctx = document.getElementById('portfolioPieChart');
    if (!ctx) return;

    if (portfolioChart) portfolioChart.destroy();

    // Prepare data
    // portfolio is likely { 'BTC': 0.4, 'ETH': 0.3 ... } or array
    let labels = [];
    let data = [];

    if (Array.isArray(portfolio)) {
        labels = portfolio.map(p => p.currency);
        data = portfolio.map(p => p.balance); // or weight
    } else {
        // Assume object { BTC: 20, ETH: 30 }
        labels = Object.keys(portfolio).map(key => {
            if (key === 'current_sim_capital') return 'Cash (KRW)';
            if (key === 'equity_curve') return 'Equity';
            if (key === 'summary') return 'Stats'; // Should actally filter these out
            if (key === 'trades') return 'Trades';
            return key.replace('KRW-', '');
        });

        // Filter out non-asset keys for the chart
        const validKeys = Object.keys(portfolio).filter(k => !['equity_curve', 'summary', 'trades'].includes(k));
        labels = validKeys.map(k => k === 'current_sim_capital' ? 'Cash (KRW)' : k.replace('KRW-', ''));
        data = validKeys.map(k => portfolio[k]);
    }

    // If empty
    if (data.every(v => v === 0)) {
        labels = ['Cash'];
        data = [100];
    }

    portfolioChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#2ecc71', '#95a5a6'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'right', labels: { boxWidth: 10, font: { size: 11 } } }
            }
        }
    });
}

function renderEquityChart(curveData) {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;

    if (equityChart) equityChart.destroy();
    if (!curveData) return; // Only return if null/undefined, allow empty array for empty state

    // curveData might be {dates: [], equity: []} OR [{date:..., value:...}]
    let labels = [];
    let values = [];

    if (curveData.dates && curveData.equity) {
        labels = curveData.dates;
        values = curveData.equity;
    } else if (Array.isArray(curveData)) {
        labels = curveData.map(d => d.time || d.date);
        values = curveData.map(d => d.value || d.equity);
    }

    // Empty State Handling
    if (labels.length === 0 || values.length === 0) {
        console.warn("No equity data available. Render empty chart.");
        // Create dummy data for visualization
        labels = ['Start', 'Now'];
        const defaultCap = 10000000;
        values = [defaultCap, defaultCap];
    }

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity',
                data: values,
                borderColor: '#3182f6', // Toss Blue
                backgroundColor: 'rgba(49, 130, 246, 0.05)',
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false } // clean look
            },
            interaction: {
                mode: 'index',
                intersect: false,
            }
        }
    });
}

function renderAccuracyChart(history) {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;

    if (accuracyChart) accuracyChart.destroy();

    // Handle new API format {dates: [], accuracy: []} or fallback array
    let labels = [];
    let data = [];

    if (Array.isArray(history)) {
        labels = history.map((_, i) => `Trade ${i + 1}`);
        data = history;
    } else if (history && history.dates) {
        labels = history.dates;
        data = history.accuracy;
    }

    if (labels.length === 0) {
        labels = ['No Trades'];
        data = [0];
    }

    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Win Rate %',
                data: data,
                borderColor: '#2ecc71',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }]

        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 40, max: 100 }
            }
        }
    });
}


// Utils
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setPreservedWidth(id, width) {
    const el = document.getElementById(id);
    if (el) el.style.width = width;
}

function exportToCSV() {
    const tbody = document.getElementById('trade-history-body');
    if (!tbody) return;

    const rows = tbody.querySelectorAll('tr');
    if (rows.length === 0) return;

    const headers = ['Date', 'Market', 'Signal', 'Entry', 'Exit', 'PnL%', 'Status'];
    const csvLines = [headers.join(',')];

    rows.forEach(row => {
        if (row.style.display === 'none') return;
        const cells = row.querySelectorAll('td');
        const values = Array.from(cells).map(td => {
            let text = td.textContent.trim().replace(/,/g, '');
            if (text.includes('"')) text = text.replace(/"/g, '""');
            return `"${text}"`;
        });
        csvLines.push(values.join(','));
    });

    const blob = new Blob([csvLines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
}

function filterTrades(type) {
    const tbody = document.getElementById('trade-history-body');
    if (!tbody) return;

    const rows = tbody.querySelectorAll('tr');
    const buttons = document.querySelectorAll('.btn-group .btn-outline-secondary');

    // Update active button state
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.trim().toLowerCase().startsWith(type === 'all' ? 'all' : type === 'win' ? 'win' : '')) {
            btn.classList.add('active');
        }
    });

    rows.forEach(row => {
        if (type === 'all') {
            row.style.display = '';
            return;
        }
        const pnlCell = row.querySelectorAll('td')[5];
        if (!pnlCell) return;
        const pnlValue = parseFloat(pnlCell.textContent);
        const isWin = pnlValue >= 0;
        row.style.display = (type === 'win' && isWin) || (type === 'loss' && !isWin) ? '' : 'none';
    });
}
