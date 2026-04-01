/**
 * Control Panel Logic (Refactored)
 * Handles training, prediction, backtest, and system monitoring.
 */

const ControlPanel = {
    state: {
        isRunning: false,
        activeTab: 'train',
        eventSource: null,
        currentPreflight: null,
        opsJobPinned: false
    },

    init() {
        this.bindEvents();
        this.checkInitialStatus();
        this.startSystemMonitor();
        this.loadOpsOverview();
        this.loadOpsPreflight();
        console.log("Control Panel Initialized");
    },

    bindEvents() {
        // Tab Switching
        document.querySelectorAll('.nav-link-ct').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target));
        });

        // Action Buttons
        this.bindAction('btn-train', () => this.startTraining());
        this.bindAction('btn-daily', () => this.startDaily());
        this.bindAction('btn-backtest', () => this.startBacktest());
        this.bindAction('btn-quick-predict', () => this.triggerPrediction());
        this.bindAction('btn-ops-intraday', () => this.startOpsRun('intraday'));
        this.bindAction('btn-ops-morning', () => this.startOpsRun('morning-report'));
        this.bindAction('btn-ops-run', () => this.startOpsRun());
        this.bindAction('btn-ops-next-action', () => this.runRecommendedOpsAction());
        this.bindAction('btn-ops-focus-primary', () => this.focusPrimaryOpsJob());
        this.bindAction('btn-ops-preflight', () => this.loadOpsPreflight());
        this.bindAction('btn-ops-plan', () => this.previewOpsPlan());
        this.bindAction('btn-ops-backfill', () => this.backfillOpsContracts());
        this.bindAction('btn-reset-portfolio', () => this.resetPortfolio()); // NEW

        // Utils
        document.getElementById('btn-clear-logs')?.addEventListener('click', () => this.clearLogs());
        document.getElementById('ops-job')?.addEventListener('change', () => {
            this.state.opsJobPinned = true;
            this.loadOpsPreflight();
        });
    },

    bindAction(id, handler) {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', handler);
    },

    switchTab(target) {
        // Update UI Tabs
        document.querySelectorAll('.nav-link-ct').forEach(l => l.classList.remove('active'));
        target.classList.add('active');

        // Show Content
        const tabId = target.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('d-none'));
        document.getElementById(`tab-${tabId}`).classList.remove('d-none');

        this.state.activeTab = tabId;
    },

    // --- Actions ---

    async startTraining() {
        if (this.state.isRunning) return this.toast('작업이 이미 실행 중입니다', 'warning');

        const epochs = document.getElementById('train-epochs').value;
        const tune = document.getElementById('train-tune').checked;

        await this.executeTask('/api/train', { epochs: parseInt(epochs), tune }, '학습이 시작되었습니다', 'train');
    },

    async startDaily() {
        if (this.state.isRunning) return this.toast('작업이 이미 실행 중입니다', 'warning');
        const epochs = document.getElementById('daily-epochs').value;
        await this.executeTask('/api/daily', { daily_epochs: parseInt(epochs) }, '데일리 실행 시작', 'daily');
    },

    async startBacktest() {
        if (this.state.isRunning) return this.toast('작업이 이미 실행 중입니다', 'warning');
        const days = document.getElementById('backtest-days').value;
        await this.executeTask('/api/backtest', { days: parseInt(days) }, '백테스트 시작', 'backtest');
    },

    async triggerPrediction() {
        const btn = document.getElementById('btn-quick-predict');
        this.setButtonLoading(btn, true);

        try {
            await this.startOpsRun('intraday');
        } catch (e) {
            this.toast(`오류: ${e.message}`, 'danger');
            this.addLog(`❌ Ops intraday 실패: ${e.message}`, 'error');
        } finally {
            setTimeout(() => this.setButtonLoading(btn, false), 5000); // 5s cooldown
        }
    },

    getSelectedOpsJob() {
        return document.getElementById('ops-job')?.value || 'intraday';
    },

    setSelectedOpsJob(job, { pin = false } = {}) {
        const select = document.getElementById('ops-job');
        if (!select || !job) return false;
        if (![...select.options].some((option) => option.value === job)) return false;
        const changed = select.value !== job;
        select.value = job;
        this.state.opsJobPinned = !!pin;
        return changed;
    },

    async focusPrimaryOpsJob() {
        this.state.opsJobPinned = false;
        await this.loadOpsOverview();
        await this.loadOpsPreflight();
    },

    async previewOpsPlan() {
        const job = this.getSelectedOpsJob();
        const output = document.getElementById('ops-plan-output');
        if (output) output.textContent = 'Loading dry-run plan...';

        try {
            const res = await fetch('/api/ops/plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job })
            });
            const data = await res.json();

            if (!data.success) {
                throw new Error(data.message || 'dry-run failed');
            }

            if (output) output.textContent = JSON.stringify(data.plan, null, 2);
            this.addLog(`🧭 Ops dry-run ready: ${job}`, 'info');
        } catch (e) {
            if (output) output.textContent = `Dry run failed: ${e.message}`;
            this.toast(`Ops dry-run 오류: ${e.message}`, 'danger');
        }
    },

    async loadOpsPreflight() {
        const job = this.getSelectedOpsJob();
        const el = document.getElementById('ops-preflight-output');
        if (el) el.innerHTML = '<span class="text-muted">Checking preflight...</span>';

        try {
            const res = await fetch('/api/ops/preflight', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job })
            });
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.message || 'preflight unavailable');
            }
            this.state.currentPreflight = data;
            this.renderPreflightResult(data);
        } catch (e) {
            if (el) {
                el.innerHTML = `<span class="text-danger">Preflight unavailable: ${e.message}</span>`;
            }
        }
    },

    async runRecommendedOpsAction() {
        try {
            const res = await fetch('/api/ops/next-action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job: this.getSelectedOpsJob() })
            });
            const data = await res.json();

            if (data.preflight) {
                this.state.currentPreflight = data.preflight;
                this.renderPreflightResult(data.preflight);
            }

            if (data.success && (data.action_taken === 'run' || data.action_taken === 'backfill_and_run')) {
                this.toast(`운영 실행 시작: ${this.getSelectedOpsJob()}`, 'success');
                this.setRunningState(true, 'ops');
                this.connectLogStream();
                this.loadOpsReadiness();
                if (data.backfilled && Object.keys(data.backfilled).length) {
                    const output = document.getElementById('ops-plan-output');
                    if (output) output.textContent = JSON.stringify(data.backfilled, null, 2);
                }
                return;
            }

            if (data.success && data.action_taken === 'inspect_dry_run') {
                const output = document.getElementById('ops-plan-output');
                if (output) output.textContent = JSON.stringify(data.dry_run?.plan || data.dry_run || {}, null, 2);
                this.toast('dry-run을 먼저 확인합니다.', 'warning');
                return;
            }

            if (data.success && data.action_taken === 'monitor_only') {
                this.toast(data.message || '현재 상태는 정상입니다.', 'info');
                return;
            }

            this.toast(data.message || 'preflight 이슈를 먼저 확인하세요.', 'warning');
        } catch (e) {
            this.toast(`시스템 오류: ${e.message}`, 'danger');
        }
    },

    async loadOpsReadiness() {
        const el = document.getElementById('ops-readiness-output');
        if (!el) return;
        el.innerHTML = '<span class="text-muted">Syncing readiness...</span>';

        try {
            const res = await fetch('/api/ops/readiness');
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.message || 'readiness unavailable');
            }
            const rows = Array.isArray(data.jobs) ? data.jobs : [];
            if (!rows.length) {
                el.innerHTML = '<span class="text-muted">No readiness rows.</span>';
                return;
            }
            el.innerHTML = rows.map((row) => {
                const next = row.next_action || {};
                const color = row.status === 'blocked'
                    ? 'text-danger'
                    : row.status === 'ready_to_run'
                        ? 'text-success'
                        : 'text-warning';
                return `<div class="py-1 border-bottom border-secondary border-opacity-10">
                    <div class="d-flex justify-content-between gap-2">
                        <span>${row.job}</span>
                        <span class="${color}">${row.status}</span>
                    </div>
                    <div class="text-muted">${next.label || 'Inspect'} | errors=${row.blocking_errors || 0} warnings=${row.warning_count || 0}</div>
                </div>`;
            }).join('');
        } catch (e) {
            el.innerHTML = `<span class="text-danger">Readiness unavailable: ${e.message}</span>`;
        }
    },

    async loadOpsOverview() {
        try {
            const res = await fetch('/api/ops/overview?runs_limit=6&audit_limit=6');
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.message || 'overview unavailable');
            }
            this.renderOverviewRuntime(data.runtime || {});
            this.renderOverviewSummary((data.readiness || {}).summary || {});
            this.renderOverviewReadiness((data.readiness || {}).jobs || []);
            this.renderOverviewRuns(data.runs || []);
            this.renderOverviewAudit(data.audit || []);
        } catch (e) {
            console.warn('Ops overview sync failed', e);
        }
    },

    renderOverviewRuntime(runtime) {
        const el = document.getElementById('system-runtime-summary');
        if (!el) return;
        const report = runtime.report || {};
        const python = report?.python?.path ? report.python.path.split('/').slice(-3).join('/') : 'unknown';
        el.textContent = `runtime=${runtime.status || 'unknown'} | python=${python} | core=${report.core_ok ? 'ok' : 'bad'} | outputs=${report.outputs_ok ? 'ok' : 'stale'}`;
    },

    renderOverviewSummary(summary) {
        const el = document.getElementById('ops-overview-summary');
        if (!el) return;
        const primary = summary?.primary_action || {};
        const primaryJob = summary?.primary_job || '';
        const primaryAge = summary?.primary_age_hours != null ? `${Number(summary.primary_age_hours).toFixed(1)}h` : 'n/a';
        if (primaryJob && !this.state.opsJobPinned) {
            const changed = this.setSelectedOpsJob(primaryJob, { pin: false });
            if (changed) {
                this.loadOpsPreflight();
            }
        }
        el.textContent = `ops=blocked:${summary?.blocked_n || 0} stale:${summary?.stale_n || 0} empty:${summary?.empty_n || 0} ready:${summary?.ready_n || 0} | next=${primary.label || 'Inspect'}${primaryJob ? ` @ ${primaryJob}` : ''} age=${primaryAge}${this.state.opsJobPinned ? ' | pinned' : ' | auto'}`;
    },

    renderOverviewReadiness(rows) {
        const el = document.getElementById('ops-readiness-output');
        if (!el) return;
        if (!rows.length) {
            el.innerHTML = '<span class="text-muted">No readiness rows.</span>';
            return;
        }
        el.innerHTML = rows.map((row) => {
            const next = row.next_action || {};
            const color = row.status === 'blocked'
                ? 'text-danger'
                : row.status === 'ready_to_run'
                    ? 'text-success'
                    : 'text-warning';
            const age = row.age_hours != null ? `${Number(row.age_hours).toFixed(1)}h` : 'n/a';
            const flags = [];
            if (row.used_fallback) flags.push('fallback');
            if (row.target === 'refresh-db' && row.status === 'ready_but_stale') flags.push('refresh attention');
            return `<div class="py-1 border-bottom border-secondary border-opacity-10">
                <div class="d-flex justify-content-between gap-2">
                    <span>${row.job}</span>
                    <span class="${color}">${row.status}</span>
                </div>
                <div class="text-muted">${next.label || 'Inspect'} | age=${age} | errors=${row.blocking_errors || 0} warnings=${row.warning_count || 0}${flags.length ? ` | ${flags.join(' | ')}` : ''}</div>
            </div>`;
        }).join('');
    },

    renderOverviewRuns(rows) {
        const el = document.getElementById('ops-history-output');
        if (!el) return;
        if (!rows.length) {
            el.innerHTML = '<span class="text-muted">No recent ops runs.</span>';
            return;
        }
        el.innerHTML = rows.map((row) => {
            const flags = [];
            if (row.has_watch) flags.push('watch');
            if (row.has_forced) flags.push('forced');
            if (row.used_fallback) flags.push('fallback');
            if (row.auto_refresh_skipped_offline) flags.push('offline-auto-refresh');
            if (row.auto_refresh_dns_error) flags.push(`dns:${row.auto_refresh_dns_error}`);
            const elapsed = row.elapsed_sec != null ? `${Number(row.elapsed_sec).toFixed(1)}s` : 'n/a';
            return `<div class="d-flex justify-content-between gap-2 py-1 border-bottom border-secondary border-opacity-10">
                <span>${row.mode}</span>
                <span class="text-muted">recs=${row.recs_n} kept=${row.kept} drop=${row.dropped} ${flags.join('/') || 'clean'} ${elapsed}</span>
            </div>`;
        }).join('');
    },

    renderOverviewAudit(rows) {
        const el = document.getElementById('ops-audit-output');
        if (!el) return;
        if (!rows.length) {
            el.innerHTML = '<span class="text-muted">No recent control events.</span>';
            return;
        }
        el.innerHTML = rows.map((row) => {
            const payload = row.payload || {};
            const bits = [payload.job, payload.action_taken, payload.preflight_status].filter(Boolean).join(' | ');
            return `<div class="d-flex justify-content-between gap-2 py-1 border-bottom border-secondary border-opacity-10">
                <span>${row.event}</span>
                <span class="text-muted">${bits || row.ts}</span>
            </div>`;
        }).join('');
    },

    async startOpsRun(job = null) {
        if (this.state.isRunning) return this.toast('작업이 이미 실행 중입니다', 'warning');
        const selectedJob = job || this.getSelectedOpsJob();
        const firstTry = await this.requestOpsRun({ job: selectedJob });
        if (firstTry?.blocked && firstTry?.preflight) {
            const issues = (firstTry.preflight.issues || [])
                .map((issue) => `[${issue.code}] ${issue.message}`)
                .join('\n');
            const confirmed = confirm(`Preflight blocked this run.\n\n${issues || 'Unknown issue'}\n\n강제로 실행할까요?`);
            if (!confirmed) {
                this.toast('운영 실행이 취소되었습니다', 'warning');
                return;
            }
            await this.requestOpsRun({ job: selectedJob, force: true });
        }
    },

    async requestOpsRun(body) {
        try {
            const res = await fetch('/api/ops/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();

            if (data.success) {
                this.toast(`운영 실행 시작: ${body.job}`, 'success');
                this.setRunningState(true, 'ops');
                this.connectLogStream();
                this.loadOpsPreflight();
                this.loadOpsOverview();
                return data;
            }

            if (res.status === 409 && data.blocked) {
                this.toast('Preflight가 실행을 차단했습니다', 'warning');
                if (data.preflight) {
                    this.renderPreflightResult(data.preflight);
                }
                return data;
            }

            this.toast(data.message || '운영 실행 실패', 'danger');
            return data;
        } catch (e) {
            this.toast(`시스템 오류: ${e.message}`, 'danger');
            return { success: false, message: e.message };
        }
    },

    renderPreflightResult(data) {
        const el = document.getElementById('ops-preflight-output');
        const actionBtn = document.getElementById('btn-ops-next-action');
        if (!el) return;
        const issues = Array.isArray(data.issues) ? data.issues : [];
        const doctor = data.doctor || {};
        const targetHealth = doctor?.health?.[data.target] || {};
        const manifestOk = doctor?.manifests?.[data.target] ? 'ok' : 'missing';
        const python = doctor?.python?.path
            ? doctor.python.path.split('/').slice(-3).join('/')
            : 'unknown';
        const issueHtml = issues.length
            ? issues.map((issue) => `<div class="${issue.level === 'error' ? 'text-danger' : 'text-warning'}">[${issue.code}] ${issue.message}</div>`).join('')
            : '<div class="text-success">No blocking issues.</div>';
        const next = data.next_action || {};

        if (actionBtn) {
            actionBtn.innerHTML = `<i class="bi bi-magic me-2"></i>${next.label || '추천 액션 실행'}`;
            actionBtn.disabled = this.state.isRunning;
        }

        el.innerHTML = `
            <div class="d-flex justify-content-between gap-2">
                <span>status=${data.status}</span>
                <span class="text-muted">job=${data.job}</span>
            </div>
            <div class="text-muted mt-1">python=${python} | core=${doctor.core_ok ? 'ok' : 'bad'} | target=${targetHealth.status || 'unknown'} | manifest=${manifestOk}</div>
            <div class="text-muted mt-1">next=${next.label || 'Inspect'}${next.reason ? ` | ${next.reason}` : ''}</div>
            <div class="mt-2">${issueHtml}</div>
        `;
    },

    async backfillOpsContracts() {
        const output = document.getElementById('ops-plan-output');
        if (output) output.textContent = 'Backfilling output contracts...';
        try {
            const res = await fetch('/api/ops/backfill-contracts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ modes: ['intraday', 'morning', 'refresh-db'] })
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.message || 'backfill failed');
            if (output) output.textContent = JSON.stringify(data.written, null, 2);
            this.toast('Output contracts backfilled', 'success');
            this.loadOpsPreflight();
            this.loadOpsOverview();
        } catch (e) {
            if (output) output.textContent = `Backfill failed: ${e.message}`;
            this.toast(`Backfill 오류: ${e.message}`, 'danger');
        }
    },

    async loadOpsHistory() {
        const el = document.getElementById('ops-history-output');
        if (!el) return;
        try {
            const res = await fetch('/api/ops/history?limit=6');
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.message || 'history unavailable');
            }
            const rows = data.runs || [];
            if (!rows.length) {
                el.innerHTML = '<span class="text-muted">No recent ops runs.</span>';
                return;
            }
            el.innerHTML = rows.map((row) => {
                const flags = [];
                if (row.has_watch) flags.push('watch');
                if (row.has_forced) flags.push('forced');
                if (row.used_fallback) flags.push('fallback');
                const elapsed = row.elapsed_sec != null ? `${Number(row.elapsed_sec).toFixed(1)}s` : 'n/a';
                return `<div class="d-flex justify-content-between gap-2 py-1 border-bottom border-secondary border-opacity-10">
                    <span>${row.mode}</span>
                    <span class="text-muted">recs=${row.recs_n} kept=${row.kept} drop=${row.dropped} ${flags.join('/') || 'clean'} ${elapsed}</span>
                </div>`;
            }).join('');
        } catch (e) {
            el.innerHTML = '<span class="text-danger">Ops history unavailable.</span>';
        }
    },

    async loadOpsAudit() {
        const el = document.getElementById('ops-audit-output');
        if (!el) return;
        try {
            const res = await fetch('/api/ops/audit?limit=6');
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.message || 'audit unavailable');
            }
            const rows = data.events || [];
            if (!rows.length) {
                el.innerHTML = '<span class="text-muted">No recent control events.</span>';
                return;
            }
            el.innerHTML = rows.map((row) => {
                const payload = row.payload || {};
                const bits = [payload.job, payload.action_taken, payload.preflight_status].filter(Boolean).join(' | ');
                return `<div class="d-flex justify-content-between gap-2 py-1 border-bottom border-secondary border-opacity-10">
                    <span>${row.event}</span>
                    <span class="text-muted">${bits || row.ts}</span>
                </div>`;
            }).join('');
        } catch (e) {
            el.innerHTML = '<span class="text-danger">Ops audit unavailable.</span>';
        }
    },

    async resetPortfolio() {
        if (!confirm('정말로 포트폴리오를 초기화하시겠습니까? 모든 거래 기록이 삭제됩니다.')) return;

        try {
            const res = await fetch('/api/performance/history', { method: 'DELETE' });
            const data = await res.json();

            if (data.success) {
                this.toast('포트폴리오가 초기화되었습니다 (새로고침 권장)', 'success');
                this.addLog('⚠️ 포트폴리오 초기화 완료', 'warn');
                setTimeout(() => location.reload(), 2000);
            } else {
                this.toast(data.message, 'danger');
            }
        } catch (e) {
            console.error(e);
            this.toast('초기화 요청 실패', 'danger');
        }
    },

    // --- Core Task Executor ---

    async executeTask(endpoint, body, successMsg, type) {
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();

            if (data.success) {
                this.toast(successMsg, 'success');
                this.setRunningState(true, type);
                this.connectLogStream();
            } else {
                this.toast(data.message, 'danger');
            }
        } catch (e) {
            this.toast(`시스템 오류: ${e.message}`, 'danger');
        }
    },

    // --- Log Streaming (SSE) ---

    connectLogStream() {
        if (this.state.eventSource) this.state.eventSource.close();

        this.state.eventSource = new EventSource('/api/logs/stream');

        this.state.eventSource.onmessage = (e) => {
            const msg = e.data;
            if (msg === '[STREAM_END]') {
                this.finishTask();
            } else if (msg !== '[HEARTBEAT]') {
                this.addLog(msg);
            }
        };

        this.state.eventSource.onerror = (e) => {
            console.error('SSE Error', e);
            this.state.eventSource.close();
        };
    },

    finishTask() {
        if (this.state.eventSource) {
            this.state.eventSource.close();
            this.state.eventSource = null;
        }
        this.setRunningState(false);
        this.toast('작업이 완료되었습니다', 'info');
        this.addLog('=== 작업 완료 ===', 'info');
    },

    // --- UI Helpers ---

    setRunningState(isRunning, activeType = null) {
        this.state.isRunning = isRunning;

        // Disable main action buttons
        const buttons = ['btn-train', 'btn-daily', 'btn-backtest', 'btn-ops-run', 'btn-ops-intraday', 'btn-ops-morning'];
        buttons.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = isRunning;
        });

        // Update Status Indicators
        document.querySelectorAll('.status-indicator').forEach(el => el.classList.remove('active'));
        if (activeType) {
            const indicator = document.getElementById(`status-${activeType}`);
            if (indicator) indicator.classList.add('active');
        }

        const masterStatus = document.getElementById('system-status-text');
        if (masterStatus) masterStatus.textContent = isRunning ? '작업 중...' : '대기 중';
    },

    setButtonLoading(btn, isLoading) {
        if (!btn) return;
        if (isLoading) {
            btn.dataset.originalText = btn.innerHTML;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 처리중';
            btn.disabled = true;
        } else {
            btn.innerHTML = btn.dataset.originalText;
            btn.disabled = false;
        }
    },

    addLog(msg, type = 'normal') {
        const consoleEl = document.getElementById('log-console');
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;

        // formatting timestamp
        const time = new Date().toLocaleTimeString();
        entry.textContent = `[${time}] ${msg}`;

        consoleEl.appendChild(entry);
        consoleEl.scrollTop = consoleEl.scrollHeight;
    },

    toast(msg, type = 'info') {
        // Reuse global showToast if available, else console
        if (window.showToast) window.showToast(msg, type);
        else alert(msg);
    },

    clearLogs() {
        const consoleEl = document.getElementById('log-console');
        consoleEl.innerHTML = '<div class="log-entry text-muted">로그가 초기화되었습니다.</div>';
    },

    checkInitialStatus() {
        fetch('/api/status').then(r => r.json()).then(data => {
            if (data.running) {
                this.setRunningState(true, this.resolveTaskKey(data));
                const statusEl = document.getElementById('system-status-text');
                if (statusEl) {
                    statusEl.textContent = data.task_name ? `${data.task_name} 실행 중` : '작업 중...';
                }
                this.connectLogStream();
            }
        });
    },

    startSystemMonitor() {
        setInterval(() => {
            this.updateResourceMeters();
            this.syncTaskStatus();
            this.loadOpsOverview();
        }, 5000);
        setInterval(() => this.loadOpsPreflight(), 15000);
        this.updateResourceMeters();
    },

    async updateResourceMeters() {
        try {
            const res = await fetch('/api/system/resources');
            const data = await res.json();
            if (!data.success) throw new Error(data.message || 'resource unavailable');
            const metrics = data.data || {};
            this.updateMeter('cpu', Number(metrics.cpu_percent || 0));
            this.updateMeter('mem', Number(metrics.memory_percent || 0));
        } catch (e) {
            console.warn('Resource sync failed', e);
            this.updateMeter('cpu', 0);
            this.updateMeter('mem', 0);
        }
    },

    updateMeter(id, val) {
        const bar = document.getElementById(`meter-bar-${id}`);
        const text = document.getElementById(`meter-val-${id}`);
        if (bar) bar.style.width = `${val}%`;
        if (text) text.textContent = `${val}%`;
    },

    async syncRuntimeSummary() {
        const el = document.getElementById('system-runtime-summary');
        if (!el) return;
        try {
            const res = await fetch('/api/health/runtime');
            const data = await res.json();
            if (!data.success) throw new Error(data.message || 'runtime unavailable');
            const report = data.report || {};
            const python = report?.python?.path ? report.python.path.split('/').slice(-3).join('/') : 'unknown';
            const text = `runtime=${data.status} | python=${python} | core=${report.core_ok ? 'ok' : 'bad'} | outputs=${report.outputs_ok ? 'ok' : 'stale'}`;
            el.textContent = text;
        } catch (e) {
            el.textContent = 'runtime=unknown';
        }
    },

    async syncTaskStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.running) {
                this.setRunningState(true, this.resolveTaskKey(data));
                const statusEl = document.getElementById('system-status-text');
                if (statusEl) {
                    statusEl.textContent = data.task_name ? `${data.task_name} 실행 중` : '작업 중...';
                }
            } else if (data.status === 'completed' || data.status === 'failed') {
                const statusEl = document.getElementById('system-status-text');
                if (statusEl) {
                    const suffix = data.returncode != null ? ` (rc=${data.returncode})` : '';
                    statusEl.textContent = data.task_name ? `${data.task_name} ${data.status}${suffix}` : '대기 중';
                }
            }
        } catch (e) {
            console.warn('Task status sync failed', e);
        }
    }
    ,

    resolveTaskKey(data) {
        const raw = String(data?.task_key || '').trim().toLowerCase();
        if (['train', 'daily', 'backtest', 'ops'].includes(raw)) return raw;
        const taskName = String(data?.task_name || '').toLowerCase();
        if (taskName.includes('train')) return 'train';
        if (taskName.includes('daily')) return 'daily';
        if (taskName.includes('backtest')) return 'backtest';
        if (taskName.includes('ops')) return 'ops';
        return 'unknown';
    }
};

document.addEventListener('DOMContentLoaded', () => ControlPanel.init());
