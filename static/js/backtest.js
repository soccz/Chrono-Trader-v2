/**
 * Backtest Analysis v2.0
 * Premium Dashboard Logic with Period Switching
 */

// Global State
let currentPeriod = 'week';
let allTrades = [];
let equityChart = null;
let tradeDistChart = null;
let currentPage = 1;
const tradesPerPage = 20;

document.addEventListener('DOMContentLoaded', () => {
    console.log("Backtest v2.0 Loaded");
    initBacktest();
});

async function initBacktest() {
    await loadBacktestData(currentPeriod);
}

// ========== Period Switching ==========
window.switchPeriod = function (period) {
    currentPeriod = period;

    // Update Tab UI
    document.querySelectorAll('.period-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.period === period);
    });

    // Update Period Label
    const labelMap = { week: '이번 주', month: '이번 달', all: '전체' };
    document.getElementById('bt-period-label').textContent = labelMap[period] || period;

    // Reload Data
    loadBacktestData(period);
};

// ========== Main Data Loader ==========
async function loadBacktestData(period) {
    try {
        // Fetch 30 days by default to ensure fast loading
        // 'all' can be too slow if history is large
        const targetPeriod = (period === 'week') ? '7d' : (period === 'month' ? '30d' : period);
        const res = await fetch(`/api/backtest/metrics?period=${targetPeriod}`);
        const result = await res.json();

        if (!result.success) {
            console.error("API Error:", result.message);
            return;
        }

        const data = result.data;
        allTrades = data.trades || [];

        // Filter trades by period
        const filteredTrades = filterTradesByPeriod(allTrades, period);

        // Render KPI Cards
        renderKPIs(filteredTrades, data);

        // Render Equity Chart
        renderEquityChart(data.equity_curve || {});

        // Render Weekly Summary
        renderWeeklyList(allTrades);

        // Render Trade Table
        renderTradeTable(filteredTrades);

        // Render Monthly Heatmap
        renderMonthlyHeatmap(allTrades);

        // Render Trade Distribution Chart
        renderTradeDistChart(filteredTrades);

    } catch (e) {
        console.error("Failed to load backtest data:", e);
    }
}

function filterTradesByPeriod(trades, period) {
    const now = new Date();
    let startDate;

    if (period === 'week') {
        const day = now.getDay();
        startDate = new Date(now);
        startDate.setDate(now.getDate() - day); // Start of week (Sunday)
        startDate.setHours(0, 0, 0, 0);
    } else if (period === 'month') {
        startDate = new Date(now.getFullYear(), now.getMonth(), 1);
    } else {
        return trades; // 'all'
    }

    return trades.filter(t => {
        const tradeDate = new Date(t.entry_time);
        return tradeDate >= startDate;
    });
}

// ========== KPI Rendering ==========
function renderKPIs(trades, fullData) {
    const closedTrades = trades.filter(t => t.status === 'CLOSED');

    // Total Return
    const totalReturn = closedTrades.reduce((sum, t) => sum + (parseFloat(t.pnl_percent) || 0), 0);
    const returnEl = document.getElementById('bt-total-return');
    returnEl.textContent = (totalReturn >= 0 ? '+' : '') + totalReturn.toFixed(2) + '%';
    returnEl.style.color = totalReturn >= 0 ? 'var(--up)' : 'var(--down)';

    // Win Rate
    const wins = closedTrades.filter(t => (parseFloat(t.pnl_percent) || 0) > 0).length;
    const winRate = closedTrades.length > 0 ? (wins / closedTrades.length * 100) : 0;
    document.getElementById('bt-win-rate').textContent = winRate.toFixed(1) + '%';

    // Profit Factor
    const grossProfit = closedTrades.filter(t => (parseFloat(t.pnl_percent) || 0) > 0)
        .reduce((sum, t) => sum + parseFloat(t.pnl_percent), 0);
    const grossLoss = Math.abs(closedTrades.filter(t => (parseFloat(t.pnl_percent) || 0) < 0)
        .reduce((sum, t) => sum + parseFloat(t.pnl_percent), 0));
    const profitFactor = grossLoss > 0 ? (grossProfit / grossLoss) : grossProfit > 0 ? 999 : 0;
    document.getElementById('bt-profit-factor').textContent = profitFactor.toFixed(2);

    // Max Drawdown (simplified - running max)
    let peak = 0;
    let maxDD = 0;
    let cumulative = 0;
    closedTrades.forEach(t => {
        cumulative += parseFloat(t.pnl_percent) || 0;
        if (cumulative > peak) peak = cumulative;
        const dd = peak - cumulative;
        if (dd > maxDD) maxDD = dd;
    });
    document.getElementById('bt-mdd').textContent = '-' + maxDD.toFixed(2) + '%';

    // Calculate Professional Risk Metrics
    calculateRiskMetrics(closedTrades, totalReturn, maxDD);
}

// ========== Professional Risk Metrics ==========
function calculateRiskMetrics(trades, totalReturn, maxDD) {
    if (trades.length < 2) {
        setRiskMetricsNA();
        return;
    }

    // Get daily returns (approximate by grouping trades by day)
    const dailyReturns = getDailyReturns(trades);

    // 1. Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev
    const meanReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
    const stdDev = Math.sqrt(dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length);
    const sharpeRatio = stdDev > 0 ? (meanReturn / stdDev) * Math.sqrt(252) : 0;
    document.getElementById('bt-sharpe').textContent = sharpeRatio.toFixed(2);
    colorizeMetric('bt-sharpe', sharpeRatio, 0.5, 1.0);

    // 2. Sortino Ratio = Mean Return / Downside Std Dev
    const negativeReturns = dailyReturns.filter(r => r < 0);
    const downsideStdDev = negativeReturns.length > 0
        ? Math.sqrt(negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length)
        : 0;
    const sortinoRatio = downsideStdDev > 0 ? (meanReturn / downsideStdDev) * Math.sqrt(252) : 0;
    document.getElementById('bt-sortino').textContent = sortinoRatio.toFixed(2);
    colorizeMetric('bt-sortino', sortinoRatio, 0.5, 1.0);

    // 3. Calmar Ratio = Annual Return / Max Drawdown
    const annualizedReturn = totalReturn * (365 / Math.max(getDaySpan(trades), 30));
    const calmarRatio = maxDD > 0 ? (annualizedReturn / maxDD) : 0;
    document.getElementById('bt-calmar').textContent = calmarRatio.toFixed(2);
    colorizeMetric('bt-calmar', calmarRatio, 0.5, 1.0);

    // 4. Average Win / Average Loss Ratio
    const winTrades = trades.filter(t => (parseFloat(t.pnl_percent) || 0) > 0);
    const lossTrades = trades.filter(t => (parseFloat(t.pnl_percent) || 0) < 0);
    const avgWin = winTrades.length > 0
        ? winTrades.reduce((sum, t) => sum + parseFloat(t.pnl_percent), 0) / winTrades.length
        : 0;
    const avgLoss = lossTrades.length > 0
        ? Math.abs(lossTrades.reduce((sum, t) => sum + parseFloat(t.pnl_percent), 0) / lossTrades.length)
        : 0;
    const winLossRatio = avgLoss > 0 ? (avgWin / avgLoss) : avgWin > 0 ? 999 : 0;
    document.getElementById('bt-win-loss-ratio').textContent = winLossRatio.toFixed(2);
    colorizeMetric('bt-win-loss-ratio', winLossRatio, 1.0, 2.0);

    // 5. Max Consecutive Losses
    let maxConsecLoss = 0;
    let currentStreak = 0;
    trades.forEach(t => {
        if ((parseFloat(t.pnl_percent) || 0) < 0) {
            currentStreak++;
            if (currentStreak > maxConsecLoss) maxConsecLoss = currentStreak;
        } else {
            currentStreak = 0;
        }
    });
    document.getElementById('bt-max-consec-loss').textContent = maxConsecLoss + '회';

    // 6. Recovery Factor = Total Return / MDD
    const recoveryFactor = maxDD > 0 ? (totalReturn / maxDD) : 0;
    document.getElementById('bt-recovery-factor').textContent = recoveryFactor.toFixed(2);
    colorizeMetric('bt-recovery-factor', recoveryFactor, 1.0, 2.0);

    // 7. Alpha (simplified as total return for now)
    document.getElementById('bt-alpha').textContent = (totalReturn >= 0 ? '+' : '') + totalReturn.toFixed(1) + '%';

    // 8. Beta (requires market correlation data - not yet available from API)
    document.getElementById('bt-beta').textContent = 'N/A';

    // 9. vs BTC Buy&Hold (requires benchmark data - not yet available from API)
    document.getElementById('bt-vs-btc').textContent = 'N/A';
}

function getDailyReturns(trades) {
    const dailyMap = {};
    trades.forEach(t => {
        const date = new Date(t.entry_time).toISOString().split('T')[0];
        if (!dailyMap[date]) dailyMap[date] = 0;
        dailyMap[date] += parseFloat(t.pnl_percent) || 0;
    });
    return Object.values(dailyMap);
}

function getDaySpan(trades) {
    if (trades.length < 2) return 1;
    const dates = trades.map(t => new Date(t.entry_time).getTime());
    const minDate = Math.min(...dates);
    const maxDate = Math.max(...dates);
    return Math.max(1, (maxDate - minDate) / (1000 * 60 * 60 * 24));
}

function colorizeMetric(id, value, thresholdOk, thresholdGood) {
    const el = document.getElementById(id);
    if (!el) return;
    if (value >= thresholdGood) {
        el.style.color = 'var(--up)';
    } else if (value >= thresholdOk) {
        el.style.color = 'var(--ct-primary)';
    } else {
        el.style.color = 'var(--text-muted)';
    }
}

function setRiskMetricsNA() {
    ['bt-sharpe', 'bt-sortino', 'bt-calmar', 'bt-win-loss-ratio',
        'bt-max-consec-loss', 'bt-recovery-factor', 'bt-alpha', 'bt-beta', 'bt-vs-btc']
        .forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = 'N/A';
        });
}

// ========== Equity Chart ==========
function renderEquityChart(equityData) {
    const ctx = document.getElementById('btEquityChart');
    if (!ctx) return;

    // Destroy existing chart
    if (equityChart) {
        equityChart.destroy();
    }

    const dates = equityData.dates || [];
    const values = equityData.equity || [];

    // Get CSS variable colors
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const lineColor = isDark ? '#00f2ff' : '#3182f6';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
    const textColor = isDark ? '#9ca3af' : '#6b7280';

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: '자산',
                data: values,
                borderColor: lineColor,
                backgroundColor: isDark ? 'rgba(0,242,255,0.1)' : 'rgba(49,130,246,0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: textColor, maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: {
                        color: textColor,
                        callback: v => '₩' + (v / 10000).toFixed(0) + '만'
                    }
                }
            }
        }
    });
}

// ========== Weekly Summary List ==========
function renderWeeklyList(trades) {
    const container = document.getElementById('week-list');
    if (!container) return;

    // Group trades by week
    const weeks = {};
    trades.forEach(t => {
        const date = new Date(t.entry_time);
        const weekStart = getWeekStart(date);
        const key = weekStart.toISOString().split('T')[0];
        if (!weeks[key]) weeks[key] = [];
        weeks[key].push(t);
    });

    // Sort weeks descending
    const sortedWeeks = Object.keys(weeks).sort().reverse();

    if (sortedWeeks.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-4">데이터 없음</div>';
        return;
    }

    container.innerHTML = sortedWeeks.slice(0, 12).map(weekKey => {
        const weekTrades = weeks[weekKey];
        const closed = weekTrades.filter(t => t.status === 'CLOSED');
        const pnl = closed.reduce((sum, t) => sum + (parseFloat(t.pnl_percent) || 0), 0);
        const wins = closed.filter(t => (parseFloat(t.pnl_percent) || 0) > 0).length;
        const winRate = closed.length > 0 ? (wins / closed.length * 100).toFixed(0) : 0;

        const weekLabel = formatWeekLabel(weekKey);

        return `
            <div class="week-item" onclick="selectWeek('${weekKey}')">
                <div>
                    <div class="week-label">${weekLabel}</div>
                    <div class="text-muted small">${closed.length}건 거래</div>
                </div>
                <div class="week-stats">
                    <span class="text-muted">${winRate}%</span>
                    <span class="week-pnl ${pnl >= 0 ? 'positive' : 'negative'}">
                        ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

function getWeekStart(date) {
    const d = new Date(date);
    const day = d.getDay();
    d.setDate(d.getDate() - day);
    d.setHours(0, 0, 0, 0);
    return d;
}

function formatWeekLabel(dateStr) {
    const date = new Date(dateStr);
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const endDate = new Date(date);
    endDate.setDate(endDate.getDate() + 6);
    return `${month}/${day} ~ ${endDate.getMonth() + 1}/${endDate.getDate()}`;
}

window.selectWeek = function (weekKey) {
    document.querySelectorAll('.week-item').forEach(el => {
        el.classList.remove('active');
        if (el.onclick.toString().includes(weekKey)) {
            el.classList.add('active');
        }
    });
    // Future: Filter trades by this week
};

// ========== Trade Table ==========
function renderTradeTable(trades, filter = 'all') {
    const tbody = document.getElementById('bt-trade-body');
    if (!tbody) return;

    let filtered = trades;
    if (filter === 'win') {
        filtered = trades.filter(t => (parseFloat(t.pnl_percent) || 0) > 0);
    } else if (filter === 'loss') {
        filtered = trades.filter(t => (parseFloat(t.pnl_percent) || 0) <= 0);
    }

    // Pagination
    const totalPages = Math.ceil(filtered.length / tradesPerPage);
    const startIdx = (currentPage - 1) * tradesPerPage;
    const pageTrades = filtered.slice(startIdx, startIdx + tradesPerPage);

    if (pageTrades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted">거래 내역 없음</td></tr>';
    } else {
        tbody.innerHTML = pageTrades.map(t => {
            const pnl = typeof t.pnl_percent === 'number' ? t.pnl_percent : (parseFloat(t.pnl_percent) || 0);
            const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
            const signalBadge = t.signal === 'Long' ?
                '<span class="badge bg-success bg-opacity-10 text-success">롱</span>' :
                '<span class="badge bg-danger bg-opacity-10 text-danger">숏</span>';
            const statusBadge = t.status === 'CLOSED' ?
                '<span class="badge bg-secondary">청산</span>' :
                '<span class="badge bg-warning text-dark">보유</span>';

            const date = new Date(t.entry_time).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
            const market = (t.market || '').replace('KRW-', '');

            return `
                <tr>
                    <td class="ps-4">${date}</td>
                    <td class="text-center fw-bold">${market}</td>
                    <td class="text-center">${signalBadge}</td>
                    <td class="text-end">${formatPrice(t.entry_price)}</td>
                    <td class="text-end">${formatPrice(t.exit_price || t.current_price)}</td>
                    <td class="text-end ${pnlClass} fw-bold">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%</td>
                    <td class="text-end pe-4">${statusBadge}</td>
                </tr>
            `;
        }).join('');
    }

    // Update count
    document.getElementById('bt-trade-count').textContent = `${filtered.length}건 거래`;

    // Render pagination
    renderPagination(totalPages);
}

function formatPrice(price) {
    if (!price) return '-';
    const p = parseFloat(price);
    if (p >= 1000) return p.toLocaleString('ko-KR', { maximumFractionDigits: 0 });
    if (p >= 1) return p.toFixed(2);
    return p.toFixed(4);
}

function renderPagination(totalPages) {
    const container = document.getElementById('bt-pagination');
    if (!container) return;

    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    html += `<li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="goToPage(${currentPage - 1})">이전</a>
    </li>`;

    for (let i = 1; i <= Math.min(totalPages, 5); i++) {
        html += `<li class="page-item ${currentPage === i ? 'active' : ''}">
            <a class="page-link" href="#" onclick="goToPage(${i})">${i}</a>
        </li>`;
    }

    html += `<li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="goToPage(${currentPage + 1})">다음</a>
    </li>`;

    container.innerHTML = html;
}

window.goToPage = function (page) {
    currentPage = page;
    const filtered = filterTradesByPeriod(allTrades, currentPeriod);
    renderTradeTable(filtered);
};

window.filterBtTrades = function (filter) {
    currentPage = 1;
    const filtered = filterTradesByPeriod(allTrades, currentPeriod);
    renderTradeTable(filtered, filter);

    // Update button states
    document.querySelectorAll('.btn-group .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
};

// ========== Monthly Heatmap ==========
function renderMonthlyHeatmap(trades) {
    const container = document.getElementById('monthly-heatmap');
    if (!container) return;

    // Group by month
    const months = {};
    trades.forEach(t => {
        if (t.status !== 'CLOSED') return;
        const date = new Date(t.entry_time);
        const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        if (!months[key]) months[key] = 0;
        months[key] += parseFloat(t.pnl_percent) || 0;
    });

    const sortedMonths = Object.keys(months).sort().slice(-12);

    container.innerHTML = sortedMonths.map(m => {
        const pnl = months[m];
        let bgClass = 'bg-secondary bg-opacity-25';
        if (pnl > 5) bgClass = 'bg-success';
        else if (pnl > 0) bgClass = 'bg-success bg-opacity-50';
        else if (pnl < -5) bgClass = 'bg-danger';
        else if (pnl < 0) bgClass = 'bg-danger bg-opacity-50';

        return `
            <div class="text-center p-2 rounded ${bgClass}" style="min-width: 60px;">
                <div class="small text-muted">${m.split('-')[1]}월</div>
                <div class="fw-bold small">${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%</div>
            </div>
        `;
    }).join('');
}

// ========== Trade Distribution Chart ==========
function renderTradeDistChart(trades) {
    const ctx = document.getElementById('btTradeDistChart');
    if (!ctx) return;

    if (tradeDistChart) {
        tradeDistChart.destroy();
    }

    const closedTrades = trades.filter(t => t.status === 'CLOSED');
    const wins = closedTrades.filter(t => (parseFloat(t.pnl_percent) || 0) > 0).length;
    const losses = closedTrades.length - wins;

    tradeDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['수익', '손실'],
            datasets: [{
                data: [wins, losses],
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// ========== Export ==========
window.exportBtCSV = function () {
    const trades = filterTradesByPeriod(allTrades, currentPeriod);
    const headers = ['Date', 'Market', 'Signal', 'Entry', 'Exit', 'PnL%', 'Status'];
    const rows = trades.map(t => [
        t.entry_time,
        t.market,
        t.signal,
        t.entry_price,
        t.exit_price || t.current_price,
        t.pnl_percent,
        t.status
    ]);

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backtest_${currentPeriod}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
};
