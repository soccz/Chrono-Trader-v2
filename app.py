# -*- coding: utf-8 -*-
try:
    import eventlet
    eventlet.monkey_patch()
except Exception:  # pragma: no cover
    eventlet = None

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask.json.provider import DefaultJSONProvider
import json
import os
import subprocess
import sys
import math
from datetime import datetime
from functools import wraps

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_utils.task_runner import TaskRunner
from utils.portfolio_manager import portfolio_manager
from utils.research_assistant import assistant
from web_utils.data_loader import DataLoader
from web_utils.config_reader import ConfigReader
from utils.price_cache import get_prices_batch, get_cached_price

# --- Custom JSON Provider to handle NaN/Infinity (CRITICAL for frontend) ---
class NaNSafeJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that converts NaN and Infinity to null for valid JSON."""
    
    def dumps(self, obj, **kwargs):
        import json
        import numpy as np
        
        def sanitize(o):
            """Recursively sanitize NaN/Infinity values."""
            if isinstance(o, dict):
                return {k: sanitize(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [sanitize(item) for item in o]
            elif isinstance(o, float):
                if math.isnan(o) or math.isinf(o):
                    return None
                return o
            elif hasattr(np, 'floating') and isinstance(o, np.floating):
                if np.isnan(o) or np.isinf(o):
                    return None
                return float(o)
            elif hasattr(np, 'integer') and isinstance(o, np.integer):
                return int(o)
            elif hasattr(np, 'ndarray') and isinstance(o, np.ndarray):
                return sanitize(o.tolist())
            return o
        
        sanitized = sanitize(obj)
        return super().dumps(sanitized, **kwargs)

app = Flask(__name__)
app.json = NaNSafeJSONProvider(app)  # Use custom JSON provider
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'chrono-trader-secret-key-change-in-production')

WEB_HOST = os.getenv('AETHER_WEB_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('AETHER_WEB_PORT', '5001'))
LOCAL_WEB_BASE_URL = f"http://127.0.0.1:{WEB_PORT}"

# --- WebSocket Setup (Flask-SocketIO) ---
try:
    from flask_socketio import SocketIO, emit
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    
    @socketio.on('connect')
    def handle_connect():
        print("Client connected to WebSocket")
        emit('status', {'message': '연결됨', 'connected': True})
    
    @socketio.on('request_update')
    def handle_request_update():
        """Client requests data update"""
        import requests
        try:
            # Fetch market data
            resp = requests.get(f'{LOCAL_WEB_BASE_URL}/api/market/overview', timeout=5)
            if resp.ok:
                emit('market_update', resp.json()['data'])
        except Exception as e:
            emit('error', {'message': str(e)}) # Changed
            
except ImportError:
    socketio = None
    print("Warning: flask-socketio not installed. WebSocket disabled.")

# --- Background Task for Real-time Updates ---
def background_market_update():
    """Periodically broadcast market data to connected clients"""
    print("Starting background market update task...")
    import requests
    while True:
        try:
            # Sleep first to allow server startup
            socketio.sleep(10)
            
            # Fetch data locally (using localhost to trigger the API logic)
            try:
                resp = requests.get(f'{LOCAL_WEB_BASE_URL}/api/market/overview', timeout=5)
                if resp.ok:
                    data = resp.json().get('data')
                    if data:
                        # Broadcast to all connected clients
                        socketio.emit('market_update', data)
                        # print("Broadcasted market update") # Debug
            except Exception as e:
                print(f"Error fetching market data for broadcast: {e}")
                
        except Exception as e:
            print(f"Background task error: {e}")
            socketio.sleep(10) # Wait before retry

if socketio:
    socketio.start_background_task(background_market_update)

# --- Rate Limiting Setup ---
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    # limiter = Limiter(
    #     key_func=get_remote_address,
    #     app=app,
    #     default_limits=["1000 per day", "500 per hour"],
    #     storage_uri="memory://"
    # )
    limiter = None # Disabled for stability
except ImportError:
    limiter = None
    print("Warning: flask-limiter not installed. Rate limiting disabled.")

# --- Caching Setup ---
try:
    from flask_caching import Cache
    cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})
except ImportError:
    class _NullCache:
        def get(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

        def cached(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    cache = _NullCache()
    print("Warning: flask-caching not installed. Caching disabled.")

# --- API Key Authentication ---
API_KEY = os.getenv('API_KEY', 'chrono-trader-api-key-2024')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth for local requests
        if request.remote_addr in ['127.0.0.1', '::1', 'localhost']:
            return f(*args, **kwargs)
        # Check API key in header or query param
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key and key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({'error': 'Unauthorized', 'message': 'Valid API key required'}), 401
    return decorated

# Initialize utilities
task_runner = TaskRunner()
data_loader = DataLoader()
config_reader = ConfigReader()

# ============================================
# Core Page Routes
# ============================================

from flask import make_response, send_from_directory

# --- Static Route for Analysis Figures ---
@app.route('/analysis/<path:filename>')
def serve_analysis(filename):
    """Serve analysis figures (gate_distribution.png, etc.)"""
    return send_from_directory('analysis', filename)

def no_cache_response(template):
    """Helper to add no-cache headers to HTML responses"""
    response = make_response(render_template(template))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _ops_command(job: str, *, dry_run: bool = False, extra_args=None):
    command = [sys.executable or 'python3', '-u', 'scripts/run_ops_job.py', '--job', str(job)]
    for arg in (extra_args or []):
        command.append(str(arg))
    if dry_run:
        command.append('--dry_run')
    return command


def _python_command(*args):
    return [sys.executable or 'python3', '-u', *[str(arg) for arg in args]]


def _collect_ops_extra_args(job: str, data: dict | None = None):
    data = data or {}
    extra_args = []
    if bool(data.get('no_telegram', False)):
        extra_args.append('--no_telegram')
    if job == 'morning-report':
        if bool(data.get('skip_aux', False)):
            extra_args.append('--skip_aux')
        if bool(data.get('skip_pattern_followers', False)):
            extra_args.append('--skip_pattern_followers')
        if bool(data.get('skip_pump_radar', False)):
            extra_args.append('--skip_pump_radar')
    return extra_args


def _start_ops_job(job: str, data: dict | None = None, *, extra_args=None, task_name: str | None = None, success_message: str | None = None):
    data = data or {}
    job = str(job or 'intraday').strip()
    if job not in {'intraday', 'morning-report'}:
        return {'success': False, 'message': 'invalid job'}, 400

    force = bool(data.get('force', False))
    job_extra_args = list(extra_args if extra_args is not None else _collect_ops_extra_args(job, data))

    try:
        from utils.ops_preflight import build_ops_preflight
        from utils.ops_audit import append_ops_audit

        preflight = build_ops_preflight(job, extra_args=job_extra_args, repo_root=_repo_root())
    except Exception as e:
        return {'success': False, 'message': f'preflight failed: {e}'}, 500

    if not preflight.get('ready_to_run') and not force:
        append_ops_audit('ops_run_blocked', {
            'job': job,
            'forced': force,
            'preflight_status': preflight.get('status'),
            'issues': [issue.get('code') for issue in (preflight.get('issues') or [])],
        })
        return {
            'success': False,
            'blocked': True,
            'message': f'preflight blocked ops run for {job}',
            'preflight': preflight,
        }, 409

    command = _ops_command(job, extra_args=job_extra_args)
    result = task_runner.start_task(command, task_name or f"Ops {job}", cwd=_repo_root(), task_key="ops")
    result['preflight_status'] = preflight.get('status')
    result['forced'] = force
    append_ops_audit('ops_run_started', {
        'job': job,
        'forced': force,
        'preflight_status': preflight.get('status'),
        'task_name': task_name or f"Ops {job}",
    })
    if result.get('success') and success_message:
        result['message'] = success_message
        result['triggered_at'] = datetime.now().isoformat()
    return result, 200

@app.route('/')
def index():
    """Home/Dashboard page"""
    return no_cache_response('index.html')

@app.route('/control')
def control():
    """Control panel page"""
    return no_cache_response('control.html')

@app.route('/performance')
def performance():
    """Performance dashboard page"""
    return no_cache_response('performance.html')

@app.route('/model')
def model():
    """Model inspector page"""
    return no_cache_response('model.html')

@app.route('/docs')
def docs():
    """Documentation page"""
    return no_cache_response('docs.html')

@app.route('/backtest')
def backtest():
    """Backtest analysis page"""
    return no_cache_response('backtest.html')

@app.route('/tasks')
def tasks():
    """TODO/Roadmap page"""
    return no_cache_response('tasks.html')

# ============================================
# API Endpoints - Task Control
# ============================================

@app.route('/api/train', methods=['POST'])
@require_api_key
def api_train():
    """Start model training"""
    data = request.get_json(silent=True) or {}
    tune = data.get('tune', False)
    epochs = data.get('epochs', None)
    
    command = _python_command('main.py', '--mode', 'train')
    if tune:
        command.append('--tune')
    if epochs:
        command.extend(['--epochs', str(epochs)])
    
    result = task_runner.start_task(command, "Training", cwd=_repo_root(), task_key="train")
    return jsonify(result)

@app.route('/api/daily', methods=['POST'])
@require_api_key
def api_daily():
    """Start daily pipeline"""
    data = request.get_json(silent=True) or {}
    epochs = data.get('daily_epochs', 2)
    
    command = _python_command('main.py', '--mode', 'daily', '--daily_epochs', str(epochs))
    
    result = task_runner.start_task(command, "Daily Run", cwd=_repo_root(), task_key="daily")
    return jsonify(result)

@app.route('/api/continuous', methods=['POST'])
@require_api_key
def api_continuous():
    """Compatibility alias: run the current intraday ops pipeline."""
    result, status = _start_ops_job('intraday', request.get_json(silent=True) or {}, extra_args=['--no_telegram'], task_name="Ops intraday")
    return jsonify(result), status

@app.route('/api/backtest', methods=['POST'])
@require_api_key
def api_backtest():
    """Start backtest"""
    data = request.get_json(silent=True) or {}
    days = data.get('days', 30)
    
    command = _python_command('main.py', '--mode', 'backtest', '--days', str(days))
    
    result = task_runner.start_task(command, "Backtest", cwd=_repo_root(), task_key="backtest")
    return jsonify(result)


@app.route('/api/ops/run', methods=['POST'])
def api_ops_run():
    """Start scheduled ops via the unified launcher."""
    data = request.get_json(silent=True) or {}
    result, status = _start_ops_job(str(data.get('job', 'intraday') or 'intraday'), data)
    return jsonify(result), status


@app.route('/api/ops/plan', methods=['POST'])
def api_ops_plan():
    """Return the dry-run plan for a scheduled ops job."""
    data = request.get_json(silent=True) or {}
    job = str(data.get('job', 'intraday') or 'intraday').strip()
    if job not in {'intraday', 'morning-report'}:
        return jsonify({'success': False, 'message': 'invalid job'}), 400

    extra_args = _collect_ops_extra_args(job, data)
    command = _ops_command(job, dry_run=True, extra_args=extra_args)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_repo_root(),
        )
        stdout = result.stdout or ''
        start = stdout.find('{')
        payload = None
        if start >= 0:
            payload = json.loads(stdout[start:])

        return jsonify({
            'success': result.returncode == 0 and payload is not None,
            'job': job,
            'command': command,
            'plan': payload,
            'stdout': stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/ops/preflight', methods=['POST'])
def api_ops_preflight():
    """Return a one-shot readiness summary: runtime doctor + dry-run plan."""
    try:
        from utils.ops_audit import append_ops_audit
        from utils.ops_preflight import build_ops_preflight

        data = request.get_json(silent=True) or {}
        job = str(data.get('job', 'intraday') or 'intraday').strip()
        if job not in {'intraday', 'morning-report'}:
            return jsonify({'success': False, 'message': 'invalid job'}), 400

        extra_args = _collect_ops_extra_args(job, data)
        payload = build_ops_preflight(job, extra_args=extra_args, repo_root=_repo_root())
        append_ops_audit('ops_preflight', {
            'job': job,
            'status': payload.get('status'),
            'next_action': (payload.get('next_action') or {}).get('action'),
        })
        return jsonify(payload)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/ops/readiness', methods=['GET'])
def api_ops_readiness():
    """Return preflight/readiness rows for both scheduled ops jobs."""
    try:
        from utils.ops_preflight import build_ops_readiness

        payload = build_ops_readiness(repo_root=_repo_root())
        return jsonify(payload)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/ops/audit', methods=['GET'])
def api_ops_audit():
    """Expose recent ops control events such as preflight, blocked runs, and forced runs."""
    try:
        from utils.ops_audit import read_recent_ops_audit

        limit = max(1, min(20, int(request.args.get('limit', 8) or 8)))
        return jsonify({
            'success': True,
            'events': read_recent_ops_audit(limit=limit),
            'checked_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/ops/overview', methods=['GET'])
def api_ops_overview():
    """Return a compact combined payload for the ops control panel."""
    try:
        from scripts.ops_doctor import build_report
        from utils.ops_preflight import build_ops_readiness
        from utils.ops_history import read_recent_ops_runs
        from utils.ops_audit import read_recent_ops_audit

        runs_limit = max(1, min(10, int(request.args.get('runs_limit', 6) or 6)))
        audit_limit = max(1, min(20, int(request.args.get('audit_limit', 6) or 6)))

        report = build_report(max_age_intraday_h=5.0, max_age_morning_h=30.0)
        runtime_status = 'healthy' if report.get('core_ok') else 'critical'
        if report.get('core_ok') and not report.get('outputs_ok'):
            runtime_status = 'degraded'

        return jsonify({
            'success': True,
            'runtime': {
                'status': runtime_status,
                'report': report,
            },
            'readiness': build_ops_readiness(repo_root=_repo_root()),
            'runs': read_recent_ops_runs(limit_per_mode=runs_limit)[:runs_limit],
            'audit': read_recent_ops_audit(limit=audit_limit),
            'checked_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/ops/next-action', methods=['POST'])
def api_ops_next_action():
    """Execute the server-side recommended action for the selected ops job."""
    try:
        from utils.ops_audit import append_ops_audit
        from utils.ops_preflight import build_ops_preflight, run_ops_dry_plan
        from utils.output_contract import backfill_output_manifests

        data = request.get_json(silent=True) or {}
        job = str(data.get('job', 'intraday') or 'intraday').strip()
        if job not in {'intraday', 'morning-report'}:
            return jsonify({'success': False, 'message': 'invalid job'}), 400

        extra_args = _collect_ops_extra_args(job, data)
        preflight = build_ops_preflight(job, extra_args=extra_args, repo_root=_repo_root())
        action = ((preflight.get('next_action') or {}).get('action') or 'inspect_preflight').strip()
        issue_codes = {str(issue.get('code') or '') for issue in (preflight.get('issues') or [])}

        if action in {'run_now', 'run_bootstrap'}:
            written = {}
            if {'refresh_manifest_missing', 'target_manifest_missing'} & issue_codes:
                modes = ['refresh-db', 'intraday' if job == 'intraday' else 'morning']
                written = backfill_output_manifests(modes, overwrite=False)
                preflight = build_ops_preflight(job, extra_args=extra_args, repo_root=_repo_root())

            result, status = _start_ops_job(job, data, extra_args=extra_args)
            result['action_taken'] = 'backfill_and_run' if written else 'run'
            result['backfilled'] = written
            result['preflight'] = preflight
            append_ops_audit('ops_next_action', {
                'job': job,
                'action_taken': result['action_taken'],
                'preflight_status': preflight.get('status'),
            })
            return jsonify(result), status

        if action == 'inspect_dry_run':
            plan = run_ops_dry_plan(job, extra_args=extra_args, repo_root=_repo_root())
            append_ops_audit('ops_next_action', {
                'job': job,
                'action_taken': 'inspect_dry_run',
                'preflight_status': preflight.get('status'),
            })
            return jsonify({
                'success': True,
                'action_taken': 'inspect_dry_run',
                'preflight': preflight,
                'dry_run': plan,
            })

        if action == 'monitor_only':
            append_ops_audit('ops_next_action', {
                'job': job,
                'action_taken': 'monitor_only',
                'preflight_status': preflight.get('status'),
            })
            return jsonify({
                'success': True,
                'action_taken': 'monitor_only',
                'preflight': preflight,
                'message': 'No action needed. Runtime and outputs look healthy.',
            })

        append_ops_audit('ops_next_action_blocked', {
            'job': job,
            'action_taken': action or 'inspect_preflight',
            'preflight_status': preflight.get('status'),
        })
        return jsonify({
            'success': False,
            'blocked': True,
            'action_taken': action or 'inspect_preflight',
            'preflight': preflight,
            'message': 'Recommended action requires manual inspection.',
        }), 409
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get current task status"""
    return jsonify(task_runner.get_status())

@app.route('/api/logs/stream')
def api_logs_stream():
    """SSE endpoint for real-time log streaming"""
    return Response(
        stream_with_context(task_runner.stream_logs()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

# ============================================
# API Endpoints - Manual Triggers
# ============================================

@app.route('/api/trigger/prediction', methods=['POST'])
def api_trigger_prediction():
    """Compatibility alias: trigger the current intraday ops pipeline."""
    result, status = _start_ops_job(
        'intraday',
        request.get_json(silent=True) or {},
        task_name="Ops intraday",
        success_message='Intraday ops triggered in background. Results will update after refresh + inference completes.',
    )
    return jsonify(result), status

@app.route('/api/health/data-pipeline', methods=['GET'])
def api_health_data_pipeline():
    """Check health of data pipeline"""
    try:
        from utils.ops_health import mode_health

        intraday = mode_health('intraday', 5.0)
        morning = mode_health('morning', 30.0)
        refresh_db = mode_health('refresh-db', 30.0)
        overall_ok = bool(intraday.get('ok')) and bool(morning.get('ok')) and bool(refresh_db.get('ok'))
        statuses = {intraday.get('status'), morning.get('status'), refresh_db.get('status')}
        any_stale = 'stale' in statuses
        any_offline = 'offline' in statuses
        status = 'healthy' if overall_ok else ('stale' if any_stale else ('degraded' if any_offline else 'critical'))

        return jsonify({
            'success': True,
            'status': status,
            'intraday': intraday,
            'morning': morning,
            'refresh_db': refresh_db,
            'checked_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/health/runtime', methods=['GET'])
def api_health_runtime():
    """Expose runtime/core asset health for the dashboard."""
    try:
        from scripts.ops_doctor import build_report

        report = build_report(max_age_intraday_h=5.0, max_age_morning_h=30.0)
        status = 'healthy' if report.get('core_ok') else 'critical'
        if report.get('core_ok') and not report.get('outputs_ok'):
            status = 'degraded'

        return jsonify({
            'success': True,
            'status': status,
            'report': report,
            'checked_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/ops/history', methods=['GET'])
def api_ops_history():
    """Expose recent scheduled ops runs from jsonl metrics history."""
    try:
        from utils.ops_history import read_recent_ops_runs

        limit = max(1, min(10, int(request.args.get('limit', 6) or 6)))
        rows = read_recent_ops_runs(limit_per_mode=limit)
        return jsonify({
            'success': True,
            'runs': rows[:limit],
            'checked_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500


@app.route('/api/ops/backfill-contracts', methods=['POST'])
def api_ops_backfill_contracts():
    """Backfill output manifests from existing metrics/recommendation artifacts."""
    try:
        from utils.output_contract import backfill_output_manifests

        data = request.json or {}
        modes = data.get('modes') or ["intraday", "morning", "refresh-db"]
        overwrite = bool(data.get('overwrite', False))
        written = backfill_output_manifests(modes, overwrite=overwrite)
        return jsonify({
            'success': True,
            'written': written,
            'checked_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500


@app.route('/api/system/resources', methods=['GET'])
def api_system_resources():
    """Lightweight host resource snapshot for the control panel."""
    try:
        from utils.system_resources import snapshot

        return jsonify({
            'success': True,
            'data': snapshot(path=_repo_root()),
            'checked_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500

# ============================================
# API Endpoints - Model Status
# ============================================

@app.route('/api/model/ensemble-status')
def api_ensemble_status():
    """Get ensemble model weights and performance statistics."""
    try:
        from utils.model_tracker import get_tracker
        from utils.config import config as _cfg
        tracker = get_tracker(n_models=_cfg.Gan.N_ENSEMBLE_MODELS)
        weights = tracker.get_weights()
        stats = tracker.get_stats()
        try:
            metadata = tracker.get_model_metadata()
        except:
            metadata = {}
        
        return jsonify({
            'success': True,
            'data': {
                'weights': weights.tolist(),
                'stats': stats,
                'metadata': metadata,
                'n_models': 5,
                'description': 'Model weights based on rolling prediction accuracy'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'data': {
                'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
                'n_models': 5
            }
        })

@app.route('/api/model/gate-status')
def api_gate_status():
    """Get current gating mode (Trend vs Pattern) from latest predictions."""
    try:
        import glob
        import pandas as pd
        
        # Read latest gate values from analysis file
        # Use absolute path
        gate_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis', 'gate_values.csv')
        
        latest_gate = 0.5 # Default
        if os.path.exists(gate_csv):
            df = pd.read_csv(gate_csv)
            if not df.empty and 'gate_value' in df.columns:
                latest_gate = df['gate_value'].iloc[-10:].mean()  # Average of last 10
                
                # Handle NaN values (critical fix for JS rendering)
                import math
                if math.isnan(latest_gate):
                    latest_gate = 0.5  # Default to hybrid
                
                # Determine regime
                if latest_gate > 0.6:
                    regime = 'Trend'
                    regime_color = '#3498db'
                elif latest_gate < 0.4:
                    regime = 'Pattern'
                    regime_color = '#e74c3c'
                else:
                    regime = 'Hybrid'
                    regime_color = '#9b59b6'
                
                return jsonify({
                    'success': True,
                    'data': {
                        'gate_value': round(latest_gate, 4),
                        'regime': regime,
                        'regime_color': regime_color,
                        'transformer_weight': round(latest_gate, 2), # Simplified visual mapping
                        'cnn_weight': round(1.0 - latest_gate, 2)
                    }
                })
        
        # Fallback if file missing or empty
        return jsonify({
            'success': True,
            'data': {
                'gate_value': 0.5,
                'regime': 'Hybrid',
                'regime_color': '#9b59b6',
                'transformer_weight': 0.5,
                'cnn_weight': 0.5
            }
        })
    except Exception as e:
        print(f"Gate Status Error: {e}")
        return jsonify({
            'success': True,
            'data': {
                'gate_value': 0.5,
                'regime': 'Hybrid',
                'regime_color': '#9b59b6',
                'transformer_weight': 0.5,
                'cnn_weight': 0.5
            }
        })



@app.route('/research')
def research():
    """Research Lab page (Report Viewer)"""
    return no_cache_response('research.html')

# ============================================
# API Endpoints - Research Reports
# ============================================

@app.route('/api/research/reports', methods=['GET'])
def api_list_reports():
    """List available research reports"""
    try:
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        files = glob.glob(os.path.join(report_dir, "*.md"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        reports = []
        for f in files:
            stats = os.stat(f)
            reports.append({
                'filename': os.path.basename(f),
                'created': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M'),
                'size': f"{round(stats.st_size / 1024, 1)} KB"
            })
            
        return jsonify({'success': True, 'data': reports})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/research/reports/<filename>', methods=['GET'])
def api_get_report(filename):
    """Get markdown content of a report"""
    try:
        # Security check: filename must be simple
        if '..' in filename or '/' in filename:
            return jsonify({'success': False, 'message': 'Invalid filename'}), 400
            
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_reports')
        filepath = os.path.join(report_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'}), 404
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# --- Research Assistant Chat API ---
from utils.research_assistant import assistant

@app.route('/api/research/chat', methods=['POST'])
def api_research_chat():
    """Chat with the Research Assistant (RAG)"""
    data = request.json or {}
    message = data.get('message')
    if not message:
        return jsonify({'success': False, 'message': 'Message required'}), 400
    
    response_text = assistant.chat(message)
    return jsonify({'success': True, 'reply': response_text})

@app.route('/api/research/upload', methods=['POST'])
def api_research_upload():
    """Upload PDF paper to Research Assistant"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
        
    if file and file.filename.endswith('.pdf'):
        try:
            upload_dir = "data/papers"
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, file.filename)
            file.save(filepath)
            
            # Ingest
            success, msg = assistant.ingest_pdf(filepath)
            return jsonify({'success': success, 'message': msg})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    else:
        return jsonify({'success': False, 'message': 'Only PDF files allowed'}), 400


# ============================================
# API Endpoints - Data
# ============================================

@app.route('/api/performance/data', methods=['GET'])
def api_performance_data():
    """Get performance metrics (today + total stats) - Using Backtest Analyzer"""
    try:
        from utils.backtest_analyzer import backtest_analyzer
        
        # 1. All-Time Metrics (Simulation)
        all_metrics = backtest_analyzer.get_metrics_for_period('all')
        
        # 2. Recent Signals (For Home Screen Modal)
        try:
            recent_signals = backtest_analyzer.get_recent_signals(limit=100)
        except:
            recent_signals = []
            
        # Map to frontend expected format
        metrics = {
            "total": all_metrics['trade_count'],
            "accuracy": all_metrics['win_rate'], # Now this is Win Rate
            "avg_error": all_metrics['total_return'], # Using Total Return instead of Error
            "results": recent_signals,
            "max_drawdown": all_metrics['max_drawdown'],
            "profit_factor": all_metrics['profit_factor'],
            "alpha": all_metrics['alpha']
        }
        
        result = {"success": True, "data": metrics}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/performance/history', methods=['GET'])
def api_performance_history():
    """Get historical aggregated metrics for charts (from REAL Portfolio DB)"""
    try:
        from utils.portfolio_manager import portfolio_manager
        import pandas as pd
        
        limit = int(request.args.get('limit', 100))
        
        # 1. Fetch real trades from DB (Fetch all for accurate equity curve)
        trades_list = portfolio_manager.get_trade_history(limit=10000) # Fetch all history for curve
        
        # Check if empty
        if not trades_list:
            return jsonify({
                "success": True,
                "data": {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'profit_factor': 0.0,
                    'trades': [],
                    'equity_curve': [],
                    'accuracy_history': {'dates': [], 'accuracy': []}
                },
                "is_fallback": True,
                "last_updated": datetime.now().isoformat()
            })
            
        # 2. Convert to DataFrame for analysis
        df = pd.DataFrame(trades_list)
        # Columns: market, strategy, signal, entry_price, exit_price, pnl_percent, status, entry_time, exit_time
        
        # Ensure correct types
        df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce').fillna(0.0)
        
        # 3. Calculate KPI Stats
        closed_trades = df[df['status'] == 'CLOSED']
        total_trades = len(df)
        
        if len(closed_trades) > 0:
            wins = closed_trades[closed_trades['pnl_percent'] > 0]
            losses = closed_trades[closed_trades['pnl_percent'] <= 0]
            
            win_rate = len(wins) / len(closed_trades)
            total_return = closed_trades['pnl_percent'].sum()
            
            gross_profit = wins['pnl_percent'].sum()
            gross_loss = abs(losses['pnl_percent'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = 0.0
            total_return = 0.0
            profit_factor = 0.0

        # 4. Prepare Chart Data (Daily Accuracy)
        # Convert string dates to datetime - use mixed format to handle variable formats
        df['date'] = pd.to_datetime(df['entry_time'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Group by date for accuracy chart
        daily_stats = df.groupby('date').agg(
            count=('status', 'count'),
            wins=('pnl_percent', lambda x: (x > 0).sum())
        ).reset_index().sort_values('date')
        
        # Take last N days for chart
        chart_data = daily_stats.tail(30)
        
        accuracy_history = {
            'dates': chart_data['date'].tolist(),
            'accuracy': (chart_data['wins'] / chart_data['count'] * 100).round(1).tolist(),
            'counts': chart_data['count'].tolist()
        }

        # 5. Format detailed trades list for table (latest first)
        display_limit = limit
        
        # CRITICAL: Replace NaN with None to prevent invalid JSON (NaN is not valid JSON)
        df_clean = df.head(display_limit).fillna(value={
            'entry_price': 0,
            'exit_price': 0,
            'pnl_percent': 0,
            'position_value': 0
        })
        # Convert remaining NaN to None for JSON compatibility
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        display_trades = df_clean.to_dict('records')
        
        # 6. Calculate Equity Curve (Cumulative PnL for Capital Growth chart)
        df_sorted = df.sort_values('entry_time')
        starting_capital = 1000000  # 100만원 기준
        df_sorted['cumulative_pnl'] = df_sorted['pnl_percent'].cumsum()
        df_sorted['equity'] = starting_capital * (1 + df_sorted['cumulative_pnl'] / 100)
        
        equity_curve = {
            'dates': pd.to_datetime(df_sorted['entry_time'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d').tolist(),
            'equity': df_sorted['equity'].round(0).tolist()
        }
        
        response_data = {
            'total_trades': int(total_trades),
            'win_rate': float(round(win_rate, 4)) if pd.notnull(win_rate) else 0.0,
            'total_return': float(round(total_return, 2)) if pd.notnull(total_return) else 0.0,
            'profit_factor': float(round(profit_factor, 2)) if pd.notnull(profit_factor) else 0.0,
            'trades': display_trades,
            'accuracy_history': accuracy_history,
            'equity_curve': equity_curve
        }
        
        return jsonify({
            "success": True,
            "data": response_data,
            "is_fallback": False,
            "last_updated": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({ 
            "success": False, 
            "message": str(e),
            "is_fallback": True
        }), 500

@app.route('/api/performance/portfolio', methods=['GET'])
def api_performance_portfolio():
    """Get portfolio summary (Pie Chart), trade history with compounding simulation"""
    # Get period filter parameter
    period = request.args.get('period', 'all')  # 'week', 'month', 'all'
    
    # Apply caching: 30 seconds for this heavy computation (include period in cache key)
    cache_key = f'api_performance_portfolio_{period}'
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)
    
    import math
    from datetime import datetime, timedelta

    def sanitize_for_json(obj):
        """Recursively sanitize NaN/Infinity values for JSON serialization"""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        return obj
    
    try:
        summary = data_loader.get_portfolio_summary()
        # Increased limit for full history access
        trades = data_loader.get_trade_history(limit=500) 
        equity_curve = data_loader.get_equity_curve()
        
        # Apply period filter to trades
        now = datetime.now()
        if period == 'week':
            start_of_week = now - timedelta(days=now.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            trades = [t for t in trades if t.get('entry_time') and 
                     datetime.strptime(str(t['entry_time'])[:19], '%Y-%m-%d %H:%M:%S') >= start_of_week]
        elif period == 'month':
            start_of_month = now - timedelta(days=30)
            trades = [t for t in trades if t.get('entry_time') and 
                     datetime.strptime(str(t['entry_time'])[:19], '%Y-%m-%d %H:%M:%S') >= start_of_month]
        # 'all' keeps all trades
        
        # Enrich trades with current prices and position_size
        recs_df = data_loader.get_latest_recommendations()
        position_sizes = {}
        if recs_df is not None and not recs_df.empty:
            for _, row in recs_df.iterrows():
                market = row.get('market')
                if market:
                    pos_size = row.get('position_size')
                    # If position_size not in CSV, calculate using composite formula
                    if pos_size is None or (isinstance(pos_size, float) and math.isnan(pos_size)) or pos_size <= 0:
                        # Composite formula: Confidence × Volatility
                        confidence = row.get('confidence', 0.5)
                        volatility = row.get('volatility', 0.01)
                        if confidence is None or (isinstance(confidence, float) and math.isnan(confidence)):
                            confidence = 0.5
                        if volatility is None or (isinstance(volatility, float) and math.isnan(volatility)):
                            volatility = 0.01
                        
                        base_position = 0.10
                        confidence_factor = max(0.5, min(1.0, confidence))
                        volatility_factor = 1 / (1 + volatility * 5)
                        pos_size = base_position * confidence_factor * volatility_factor
                        pos_size = max(0.03, min(0.20, pos_size))
                    
                    position_sizes[market] = pos_size
        
        # Batch fetch current prices for all trades
        trade_markets = [t.get('market') for t in trades if t.get('market')]
        current_prices = get_prices_batch(trade_markets) if trade_markets else {}
        
        # Sort trades chronologically for simulation
        # entry_time might be string "YYYY-MM-DD HH:MM:SS" or datetime object
        def parse_date(date_str):
            if isinstance(date_str, str):
                try:
                    if '.' in date_str:
                        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
                    else:
                        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except: pass
                try: return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except: pass
            return date_str

        for t in trades:
            t['entry_time_dt'] = parse_date(t['entry_time'])
            
        trades.sort(key=lambda x: x['entry_time_dt'] if x.get('entry_time_dt') else datetime.min)

        # Simulation Parameters
        SIM_START_DATE = datetime(2024, 12, 22)
        current_capital = 1000000.0 # 1M KRW Start
        
        # New: Equity Curve Tracking
        sim_equity_curve = []
        sim_equity_curve.append({'time': SIM_START_DATE.strftime('%Y-%m-%d %H:%M'), 'value': current_capital})

        
        for trade in trades:
            # Basic Enrichment
            market = trade.get('market')
            if market:
                current = current_prices.get(market)
                trade['current_price'] = current if current else trade.get('entry_price', 0)
                trade['position_size'] = position_sizes.get(market, 0.095)  # Composite default

                # Recalculate PnL
                entry = trade.get('entry_price', 0) or 0
                current_p = trade.get('current_price', 0) or 0
                
                # Check if Closed
                if trade.get('status') == 'CLOSED':
                    # Use stored PnL or Calculate from exit
                    exit_p = trade.get('exit_price', current_p)
                else:
                    exit_p = current_p

                if entry > 0 and exit_p > 0:
                    signal = trade.get('signal', 'Long')
                    raw_return = (exit_p - entry) / entry
                    if signal == 'Short':
                        trade['pnl_percent'] = -raw_return
                    else:
                        trade['pnl_percent'] = raw_return
                else:
                    trade['pnl_percent'] = 0.0

            # Simulation Logic
            entry_dt = trade.get('entry_time_dt')
            
            if entry_dt and entry_dt >= SIM_START_DATE:
                pos_size = trade.get('position_size', 0.095)  # Composite default
                
                # Calculate Invested Amount based on Current Capital at that moment
                sim_invested = current_capital * pos_size
                trade['sim_invested'] = sim_invested
                
                # Calculate Resulting Value
                pnl = trade.get('pnl_percent', 0)
                profit = sim_invested * pnl
                sim_result_value = sim_invested + profit
                
                trade['sim_value'] = sim_result_value
                trade['sim_profit'] = profit
                
                if trade.get('status') == 'CLOSED':
                    current_capital += profit
                    # Record Equity Step
                    exit_t = trade.get('exit_time')
                    # Ensure string format
                    if isinstance(exit_t, datetime):
                        exit_t_str = exit_t.strftime('%Y-%m-%d %H:%M')
                    else:
                        exit_t_str = str(exit_t) if exit_t else datetime.now().strftime('%Y-%m-%d %H:%M')
                        
                    sim_equity_curve.append({'time': exit_t_str, 'value': current_capital})
                
                trade['sim_weight_at_entry'] = pos_size * 100
                trade['sim_current_capital'] = current_capital
            else:
                trade['sim_invested'] = 0
                trade['sim_value'] = 0
                trade['sim_weight_at_entry'] = 0
                trade['sim_current_capital'] = 0
        
        # Final Point: Unrealized PnL of Open Positions
        current_unrealized_pnl = 0
        for t in trades:
             # Only count OPEN trades that started after SIM_START_DATE
             entry_dt = t.get('entry_time_dt')
             if t.get('status') == 'OPEN' and entry_dt and entry_dt >= SIM_START_DATE:
                  val = t.get('sim_value', 0)
                  inv = t.get('sim_invested', 0)
                  current_unrealized_pnl += (val - inv)
        
        final_equity = current_capital + current_unrealized_pnl
        # Only add final point if it differs or strictly later? Just add it.
        sim_equity_curve.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'value': final_equity})
        
        # Sort equity curve by time and remove duplicates
        def parse_time_str(time_str):
            try:
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            except:
                try:
                    return datetime.strptime(time_str[:16], '%Y-%m-%d %H:%M')
                except:
                    return datetime.min
        
        sim_equity_curve.sort(key=lambda x: parse_time_str(x['time']))
        
        # Remove duplicate timestamps, keep last value for each time
        seen_times = {}
        for point in sim_equity_curve:
            seen_times[point['time']] = point['value']
        sim_equity_curve = [{'time': t, 'value': v} for t, v in seen_times.items()]
        sim_equity_curve.sort(key=lambda x: parse_time_str(x['time']))
        
        # Updates global var for chart
        equity_curve = sim_equity_curve

        # Reverse back to newest first for display
        trades.sort(key=lambda x: x['entry_time_dt'] if x.get('entry_time_dt') else datetime.min, reverse=True)

        # Cleanup temporary field
        for t in trades: 
            if 'entry_time_dt' in t: del t['entry_time_dt']
            
        result = {
            "success": True,
            "data": sanitize_for_json({
                "summary": summary,
                "trades": trades,
                "equity_curve": equity_curve,
                "current_sim_capital": current_capital
            })
        }
        
        # Cache the result for 30 seconds
        if cache:
            cache.set(cache_key, result, timeout=300)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({ "success": False, "message": str(e) }), 500

@app.route('/api/performance/positions', methods=['GET'])
@cache.cached(timeout=30, query_string=True)
def api_performance_positions():
    """Lightweight API for home page position table - only OPEN trades"""
    import math
    
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        return obj
    
    try:
        # Only get recent trades (limit 50) and filter to OPEN only
        trades = data_loader.get_trade_history(limit=50)
        open_trades = [t for t in trades if t.get('status') == 'OPEN']
        
        # Only fetch prices for OPEN trades (much faster)
        trade_markets = list(set([t.get('market') for t in open_trades if t.get('market')]))
        current_prices = get_prices_batch(trade_markets) if trade_markets else {}
        
        # Get position sizes from recommendations
        recs_df = data_loader.get_latest_recommendations()
        position_sizes = {}
        if recs_df is not None and not recs_df.empty:
            for _, row in recs_df.iterrows():
                market = row.get('market')
                if market:
                    pos_size = row.get('position_size')
                    if pos_size is None or (isinstance(pos_size, float) and math.isnan(pos_size)) or pos_size <= 0:
                        confidence = row.get('confidence', 0.5)
                        volatility = row.get('volatility', 0.05)
                        base_position = 0.095
                        confidence_factor = 0.5 + (confidence * 0.5)
                        volatility_factor = 1 / (1 + volatility * 5)
                        pos_size = base_position * confidence_factor * volatility_factor
                        pos_size = max(0.03, min(0.20, pos_size))
                    position_sizes[market] = pos_size
        
        # Enrich open trades
        for trade in open_trades:
            market = trade.get('market')
            if market:
                trade['current_price'] = current_prices.get(market, trade.get('entry_price', 0))
                trade['position_size'] = position_sizes.get(market, 0.095)
                
                entry = trade.get('entry_price', 0) or 0
                current_p = trade.get('current_price', 0) or 0
                if entry > 0 and current_p > 0:
                    signal = trade.get('signal', 'Long')
                    raw_return = (current_p - entry) / entry
                    trade['pnl_percent'] = -raw_return if signal == 'Short' else raw_return
                else:
                    trade['pnl_percent'] = 0.0
        
        return jsonify({
            "success": True,
            "data": {
                "trades": sanitize_for_json(open_trades),
                "current_sim_capital": 1000000.0
            }
        })
    except Exception as e:
        return jsonify({ "success": False, "message": str(e) }), 500

@app.route('/api/performance/weekly', methods=['GET'])
def api_performance_weekly():
    """Get weekly stats (Home Widget)"""
    try:
        stats = data_loader.get_weekly_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({ "success": False, "message": str(e) }), 500

@app.route('/api/market/overview', methods=['GET'])
def api_market_overview():
    """Get market overview data (CORS bypass for Upbit API) - Cached 30 seconds"""
    # Apply caching: 30 seconds for external API calls
    cache_key = 'api_market_overview'
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)
    
    import requests
    
    # Default Fallback Data (Safe Mode)
    result_data = {
        "total": 0,
        "rising": 0,
        "falling": 0,
        "avg_change": 0.0,
        "btc_dominance": 57.8
    }

    try:
        # Fetch all KRW tickers from Upbit
        res = requests.get('https://api.upbit.com/v1/ticker/all?quoteCurrencies=KRW', timeout=5)
        if res.ok:
            data = res.json()
            if data and len(data) > 0:
                rising = sum(1 for c in data if c.get('signed_change_rate', 0) > 0)
                falling = sum(1 for c in data if c.get('signed_change_rate', 0) < 0)
                total_change = sum(c.get('signed_change_rate', 0) for c in data)
                avg_change = (total_change / len(data)) * 100
                
                result_data.update({
                    "total": len(data),
                    "rising": rising,
                    "falling": falling,
                    "avg_change": round(avg_change, 2)
                })

        # Fetch BTC Dominance from CoinGecko
        try:
            cg_res = requests.get('https://api.coingecko.com/api/v3/global', timeout=3)
            if cg_res.ok:
                cg_data = cg_res.json()
                btc_dominance = round(cg_data.get('data', {}).get('market_cap_percentage', {}).get('btc', 57.8), 1)
                result_data["btc_dominance"] = btc_dominance
        except:
            pass  # Fail checking is fine, keep default

        # Cache result
        result = { "success": True, "data": result_data }
        if cache:
            cache.set(cache_key, result, timeout=30)
            
        return jsonify(result)

    except Exception as e:
        # On connection error, return fallback instead of 500
        print(f"Market Overview API Error: {e}")
        return jsonify({ "success": True, "data": result_data })



@app.route('/api/ticker', methods=['GET'])
def api_ticker():
    """Get individual coin tickers (CORS bypass for Upbit API) - Cached 10 seconds"""
    # Apply caching: 10 seconds for ticker data
    markets_param = request.args.get('markets', 'KRW-BTC,KRW-ETH,KRW-SOL,KRW-XRP,KRW-DOGE')
    cache_key = f'api_ticker_{markets_param}'
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return jsonify(cached)
    
    import requests
    try:
        markets = markets_param.split(',')
        
        # Fetch ticker data from Upbit
        url = f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        if data and len(data) > 0:
            result = {
                "success": True,
                "data": data
            }
            # Cache result for 10 seconds
            if cache:
                cache.set(cache_key, result, timeout=10)
            return jsonify(result)
        return jsonify({"success": False, "message": "No data"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- Research Assistant Routes (additional) ---

@app.route('/api/research/papers', methods=['GET'])
def list_papers():
    """Lists all uploaded PDF papers"""
    try:
        papers = []
        if os.path.exists(assistant.upload_dir):
            papers = [f for f in os.listdir(assistant.upload_dir) if f.endswith('.pdf')]
        return jsonify({'papers': papers})
    except Exception as e:
        return jsonify({'papers': [], 'error': str(e)})

@app.route('/api/research/set_key', methods=['POST'])
def set_api_key():
    data = request.json
    assistant.set_api_key(data.get('key', ''))
    return jsonify({'success': True})

@app.route('/api/research/save_note', methods=['POST'])
def save_research_note():
    data = request.json
    title = data.get('title', f"Research Note {datetime.now().strftime('%Y%m%d')}")
    content = data.get('content', '')
    
    success, msg = assistant.save_to_obsidian(title, content)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/research/summarize', methods=['POST'])
def summarize_research_chat():
    """Generates a structured summary from chat history"""
    data = request.json
    messages = data.get('messages', [])
    
    if not messages:
        return jsonify({"success": False, "message": "No messages provided"})

    if summary:
        return jsonify({"success": True, "summary": summary})
    else:
        return jsonify({"success": False, "message": "Failed to generate summary"})

@app.route('/api/research/generate', methods=['POST'])
def generate_research_report():
    """Trigger research report generation (for n8n automation)"""
    try:
        import subprocess
        import threading
        
        def run_report_async():
            try:
                result = subprocess.run(
                    _python_command('utils/research_reporter.py'),
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=_repo_root(),
                )
                print(f"Research report completed: {result.returncode}")
            except Exception as e:
                print(f"Research report failed: {e}")
        
        thread = threading.Thread(target=run_report_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Research report generation triggered in background.',
            'triggered_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/predictions/latest', methods=['GET'])
def api_predictions_latest():
    """Get latest predictions"""
    try:
        recs_df = data_loader.get_latest_recommendations()
        pump_df = data_loader.get_latest_pump_predictions()
        manifests = data_loader.get_latest_output_contracts()
        
        result = {}
        
        if recs_df is not None and not recs_df.empty:
            result['recommendations'] = recs_df.to_dict('records')
        else:
            result['recommendations'] = []
        
        if pump_df is not None and not pump_df.empty:
            result['pump_predictions'] = pump_df.to_dict('records')
        else:
            result['pump_predictions'] = []
        
        return jsonify({
            "success": True,
            "data": result,
            "manifests": manifests
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/api/model/config', methods=['GET'])
def api_model_config():
    """Get model configuration and hyperparameters"""
    try:
        model_config = config_reader.get_model_config()
        hyperparams = config_reader.get_hyperparameters()
        model_files = config_reader.get_model_files_info()
        
        return jsonify({
            "success": True,
            "data": {
                "model_config": model_config,
                "hyperparameters": hyperparams,
                "model_files": model_files
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# ============================================
# API Endpoints - Task Management
# ============================================

from utils.task_manager import task_manager

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    return jsonify(task_manager.get_tasks())

@app.route('/api/tasks', methods=['POST'])
def add_task():
    data = request.json
    title = data.get('title')
    desc = data.get('description', '')
    status = data.get('status', 'planned')
    
    if not title:
        return jsonify({'success': False, 'message': 'Title required'})
    
    tid = task_manager.add_task(title, desc, status)
    return jsonify({'success': True, 'id': tid})

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task_status(task_id):
    data = request.json
    status = data.get('status')
    if task_manager.update_status(task_id, status):
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

# ============================================
# API Endpoints - Backtest Analysis
# ============================================

@app.route('/api/backtest/metrics')
def api_backtest_metrics():
    """백테스트 성과 지표 조회"""
    from utils.backtest_analyzer import backtest_analyzer
    
    period = request.args.get('period', '30d')  # monthly, 90d, all
    month = request.args.get('month', None)  # 2026-01 형식
    
    try:
        metrics = backtest_analyzer.get_metrics_for_period(period, month)
        return jsonify({'success': True, 'data': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest/equity-curve')
def api_backtest_equity_curve():
    """누적 수익 곡선 데이터"""
    from utils.backtest_analyzer import backtest_analyzer
    from datetime import datetime, timedelta
    
    period = request.args.get('period', 'all')
    
    try:
        now = datetime.now()
        if period == '90d':
            start_date = now - timedelta(days=90)
            end_date = now
        elif period == '30d':
            start_date = now - timedelta(days=30)
            end_date = now
        elif period == '7d':
            start_date = now - timedelta(days=7)
            end_date = now
        elif period == 'monthly':
            month_str = request.args.get('month')
            if month_str:
                year, mon = map(int, month_str.split('-'))
                start_date = datetime(year, mon, 1)
                if mon == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
                else:
                    end_date = datetime(year, mon + 1, 1) - timedelta(seconds=1)
            else:
                # month 없으면 이번 달
                start_date = datetime(now.year, now.month, 1)
                end_date = now
        else:
            start_date = None
            end_date = None
        
        curve = backtest_analyzer.get_equity_curve(start_date, end_date)
        return jsonify({'success': True, 'data': curve})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest/monthly')
def api_backtest_monthly():
    """월별 수익률 데이터"""
    from utils.backtest_analyzer import backtest_analyzer
    
    try:
        monthly = backtest_analyzer.get_monthly_returns()
        return jsonify({'success': True, 'data': monthly})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# Security Headers (보안 강화 - 디자인에 영향 없음)
# ============================================

# 보안 헤더는 주석 처리 (CSP가 Bootstrap Icons 폰트 로드를 막을 수 있음)
# @app.after_request
# def set_security_headers(response):
#     """보안 헤더 추가 (모든 응답에 적용)"""
#     response.headers['X-Content-Type-Options'] = 'nosniff'
#     response.headers['X-Frame-Options'] = 'DENY'
#     response.headers['X-XSS-Protection'] = '1; mode=block'
#     response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
#     
#     # Content Security Policy (점진적 적용)
#     csp_policy = (
#         "default-src 'self'; "
#         "script-src 'self' 'unsafe-inline' "
#         "https://cdn.jsdelivr.net https://cdn.socket.io; "
#         "style-src 'self' 'unsafe-inline' "
#         "https://fonts.googleapis.com https://cdn.jsdelivr.net; "
#         "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net data:; "
#         "img-src 'self' data: https:; "
#         "connect-src 'self' "
#         "https://api.upbit.com https://api.coingecko.com wss: ws:; "
#         "frame-ancestors 'none';"
#     )
#     response.headers['Content-Security-Policy'] = csp_policy
#     return response

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# ============================================
# Main
# ============================================

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/lang', exist_ok=True)
    
    # Prefer Socket.IO when available, but keep the app bootable without it.
    if socketio is not None:
        socketio.run(app, debug=True, host=WEB_HOST, port=WEB_PORT)
    else:
        app.run(debug=True, host=WEB_HOST, port=WEB_PORT)
