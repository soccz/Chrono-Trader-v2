import requests
from datetime import datetime
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

from utils.config import config

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

# Telegram credentials from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
HISTORY_FILE = "logs/telegram_history.log"
TUNNEL_URL_FILE = "tunnel_url.txt"

def _now_report_tz() -> datetime:
    tz_name = getattr(config.General, "REPORT_TIMEZONE", "") or ""
    if ZoneInfo and tz_name:
        try:
            return datetime.now(ZoneInfo(tz_name))
        except Exception:
            return datetime.now()
    return datetime.now()

def _tz_suffix(dt: datetime) -> str:
    try:
        tzname = dt.tzname()
        return f" {tzname}" if tzname else ""
    except Exception:
        return ""

def get_dashboard_url():
    """Read current tunnel URL from file (dynamically updated by cloudflared)"""
    try:
        if os.path.exists(TUNNEL_URL_FILE):
            with open(TUNNEL_URL_FILE, 'r') as f:
                url = f.read().strip()
                if url:
                    return url
    except Exception as e:
        print(f"Failed to read tunnel URL: {e}")
    return "http://localhost:5001"  # fallback for local access

def _is_duplicate(message):
    """Checks if the message hash matches the last sent message."""
    try:
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            return False
        
        msg_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        
        with open(HISTORY_FILE, 'r') as f:
            last_hash = f.read().strip()
            
        if last_hash == msg_hash:
            return True
            
        return False
    except Exception as e:
        print(f"Dedup check failed: {e}")
        return False

def _save_message_hash(message):
    """Saves the hash of the sent message."""
    try:
        msg_hash = hashlib.md5(message.encode('utf-8')).hexdigest()
        with open(HISTORY_FILE, 'w') as f:
            f.write(msg_hash)
    except Exception as e:
        print(f"Failed to save message hash: {e}")

def send_alert(message, parse_mode='HTML', bypass_dedup=False):
    """
    텔레그램으로 메시지를 보내는 함수입니다. (기본 HTML 모드)
    중복 메시지 방지 로직이 포함되어 있습니다.
    bypass_dedup=True일 경우 중복 체크를 건너뜁니다.
    """
    if not TOKEN or not CHAT_ID:
        print("Telegram credentials missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). Skipping send.")
        return
    if not bypass_dedup and _is_duplicate(message):
        print("중복된 텔레그램 메시지 감지됨. 전송 생략.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": message, 
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    try:
        # Never let Telegram block ops. Keep it fast-fail.
        timeout_sec = float(os.getenv("AETHER_TG_TIMEOUT_SEC", "10") or 10)
        response = requests.post(url, json=payload, timeout=(3.05, timeout_sec))
        if response.status_code == 200:
            print("텔레그램 알림 전송 성공")
            _save_message_hash(message)
        else:
            print(f"텔레그램 전송 실패: {response.text}")
    except Exception as e:
        print(f"텔레그램 에러 발생: {e}")


def _format_run_summary(items, meta=None):
    meta = meta or {}
    items = items or []

    statuses = [str(it.get("status", "") or "") for it in items]
    has_watch = any(s.startswith("Watch") for s in statuses)
    has_forced = any(s.startswith("Forced") for s in statuses)

    lines = []
    if has_watch:
        lines.append("MinRec: WATCH ONLY")
    if has_forced:
        lines.append("MinRec: FORCED TRADE")

    # Market selection / freshness signals
    sel_total = meta.get("run_markets_selected")
    sel_kept = meta.get("run_markets_kept")
    dropped = meta.get("freshness_dropped")
    if sel_total is not None or sel_kept is not None or dropped is not None:
        lines.append(f"Markets: selected={sel_total} kept={sel_kept} dropped_stale={dropped}")

    rot_kept = meta.get("rotation_kept")
    rot_added = meta.get("rotation_added")
    if rot_kept is not None or rot_added is not None:
        lines.append(f"Rotation: kept={rot_kept} added={rot_added}")

    # Selection stage metrics (best-effort; only shown if present).
    try:
        core_n = meta.get("core_n")
        exploit_sel = meta.get("exploit_selected")
        exploit_target = meta.get("exploit_target")
        exploit_skip = meta.get("exploit_corr_skipped")
        explore_sel = meta.get("explore_selected")
        explore_target = meta.get("explore_target")
        explore_skip = meta.get("explore_corr_skipped")
        if any(x is not None for x in (core_n, exploit_sel, explore_sel)):
            lines.append(
                "Sel: "
                f"core={core_n} "
                f"exploit={exploit_sel}/{exploit_target} (skip={exploit_skip}) "
                f"explore={explore_sel}/{explore_target} (skip={explore_skip})"
            )
        qh, qm, ql = meta.get("bucket_quota_high"), meta.get("bucket_quota_mid"), meta.get("bucket_quota_low")
        fh, fm, fl = meta.get("bucket_filled_high"), meta.get("bucket_filled_mid"), meta.get("bucket_filled_low")
        if any(x is not None for x in (qh, qm, ql, fh, fm, fl)):
            lines.append(f"Buckets: high {fh}/{qh} mid {fm}/{qm} low {fl}/{ql}")
    except Exception:
        pass

    if meta.get("auto_refresh_skipped_offline") is True:
        lines.append("Auto-refresh: OFFLINE (skipped)")

    if meta.get("data_stale_abort") is True:
        lines.append("Data: STALE (aborted)")

    if not lines:
        return ""

    msg = "<b>🧩 RUN SUMMARY</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    for ln in lines:
        msg += f"<code>{ln}</code>\n"
    msg += "\n"
    return msg


def format_daily_report(trending_recs, pattern_recs, pump_recs, meta=None):
    """
    Generates a premium structured HTML report for Telegram.
    """
    now_dt = _now_report_tz()
    date_str = now_dt.strftime("%Y-%m-%d | %H:%M") + _tz_suffix(now_dt)
    
    msg = f"<b>✨ AETHER QUANT PREMIUM</b>\n"
    msg += f"<code>{date_str}</code>\n\n"

    # Summary block (based on any rec list)
    msg += _format_run_summary((trending_recs or []) + (pattern_recs or []), meta=meta)
    
    # 1. Trending Section
    msg += "<b>📈 TRENDING STRATEGY</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += _format_section(trending_recs)
    
    # 2. Pattern Section
    msg += "<b>🧬 PATTERN MATCHING</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += _format_section(pattern_recs)
    
    # 3. Pump Section
    if pump_recs:
        msg += "<b>🚀 PUMP RADAR (URGENT)</b>\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
        for p in pump_recs:
             market = p.get('market')
             curr = p.get('current_price', 0)
             prob = p.get('total_pump_prob', 0) * 100
             msg += f"⚠️ <b>{market}</b>\n"
             msg += f"   🔥 Prob: <b>{prob:.1f}%</b> | Price: {curr:,.0f} KRW\n\n"
    
    msg += f"\n<a href='{get_dashboard_url()}'>📊 Access Dashboard</a>"
    return msg

def _format_section(items):
    if not items:
        return "No opportunities detected.\n\n"
    
    text = ""
    for i, item in enumerate(items, 1):
        market = item.get('market')
        signal = item.get('signal', 'Neutral')
        curr = item.get('current_price', 0)
        exp_ret = item.get('expected_return', 0)
        target = curr * (1 + exp_ret)
        conf = item.get('confidence', 0) * 100
        reason = item.get('reason', 'Algorithmic Signal')
        status = str(item.get('status', '') or '')
        fallback_reason = str(item.get('fallback_reason', '') or '')
        
        # Dynamic position sizing (composite formula default)
        position_size = item.get('position_size', 0.095) * 100
        volatility = item.get('volatility', 0) * 100
        
        # Emotes based on signal
        icon = "🟢" if signal == 'Long' else ("🔴" if signal == 'Short' else "⚪️")
        
        text += f"<b>{i}. {market} ({signal} {icon})</b>\n"
        text += f"   💎 Conf: <b>{conf:.0f}%</b>\n"
        text += f"   📊 Size: <b>{position_size:.1f}%</b> (Vol: {volatility:.1f}%)\n"
        text += f"   💰 Entry: {curr:,.0f}\n"
        text += f"   🎯 Target: <b>{target:,.0f}</b> ({exp_ret*100:+.2f}%)\n"
        text += f"   🧠 <i>{reason}</i>\n"
        if status and status != "Recommended":
            text += f"   🧾 Status: <code>{status}</code>\n"
        if fallback_reason:
            # Keep Telegram readable.
            short = fallback_reason
            if len(short) > 160:
                short = short[:157] + "..."
            text += f"   🧯 Fallback: <code>{short}</code>\n"
        if (status.startswith("Watch") or position_size <= 0.01) and status:
            text += f"   ⚠️ <b>WATCH ONLY</b>\n"
        text += "\n"
    return text

def format_short_term_report(scalp_recs, pump_recs, meta=None):
    """
    Generates a premium HTML report for 4H Scalping.
    """
    now_dt = _now_report_tz()
    date_str = now_dt.strftime("%m-%d %H:%M") + _tz_suffix(now_dt)
    
    msg = f"<b>⚡ AETHER SCALP (4H)</b>\n"
    msg += f"<code>{date_str}</code>\n\n"

    msg += _format_run_summary(scalp_recs or [], meta=meta)
    
    # Scalping Section
    if scalp_recs:
        msg += "<b>⏱ SHORT-TERM SIGNALS</b>\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
        msg += _format_section(scalp_recs)
    else:
        msg += "<b>⏱ SHORT-TERM SIGNALS</b>\n"
        msg += "No actionable signals.\n\n"

    # Pump Section
    if pump_recs:
        msg += "<b>🚀 PUMP ALERT</b>\n"
        for p in pump_recs:
             market = p.get('market')
             prob = p.get('total_pump_prob', 0) * 100
             msg += f"⚠️ <b>{market}</b> (Prob: {prob:.1f}%)\n"
        msg += "\n"

    msg += f"<a href='{get_dashboard_url()}'>📊 Dashboard</a>"
    return msg

def format_trade_alert(action, market, price, reason, pnl=None):
    """
    Generates an INSTANT trade alert message with Premium Design.
    """
    now_dt = _now_report_tz()
    date_str = now_dt.strftime("%Y-%m-%d | %H:%M") + _tz_suffix(now_dt)
    
    msg = f"<b>⚡ AETHER QUANT SIGNAL</b>\n"
    msg += f"<code>{date_str}</code>\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    
    icon = "🟢" if action == "BUY" else "🔴"
    msg += f"<b>{icon} {action} EXECUTED</b>\n"
    msg += f"💎 <b>{market}</b>\n"
    msg += f"💰 Price: {price:,.0f} KRW\n"
    
    if pnl is not None:
        pnl_icon = "📈" if pnl > 0 else "📉"
        msg += f"{pnl_icon} Realized PnL: <b>{pnl:+.2f}%</b>\n"
        
    msg += f"🧠 <i>{reason}</i>\n"
    msg += f"\n<a href='{get_dashboard_url()}'>📊 Check Dashboard</a>"
    return msg

def format_status_report(active_positions, total_equity=None, profit_loss=None):
    """
    Generates a 4-hour periodic status report.
    """
    now_dt = _now_report_tz()
    date_str = now_dt.strftime("%m-%d %H:%M") + _tz_suffix(now_dt)
    msg = f"<b>🤖 SYSTEM ALIVE: {date_str}</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    
    if total_equity:
        msg += f"💰 Equity: {total_equity:,.0f} KRW\n"
    if profit_loss:
        pnl_str = f"{profit_loss:+.2f}%"
        msg += f"📈 Session PnL: {pnl_str}\n" # Placeholder for session pnl
        
    msg += f"\n<b>📦 Active Positions ({len(active_positions)})</b>\n"
    if active_positions:
        for market, data in active_positions.items():
            # data could be dict or object, handling dict for now based on typical pm structure
            # If pm.positions is simple dict: {market: {entry_price: ..., weight: ...}}
            entry = data.get('entry_price', 0)
            curr = data.get('current_price', 0)
            if curr == 0: curr = entry # Fallback
            
            pnl_pct = 0
            if entry > 0:
                pnl_pct = (curr - entry) / entry * 100
                
            pnl_icon = "🟢" if pnl_pct >= 0 else "🔴"
            msg += f"  • {market}: {pnl_icon} {pnl_pct:+.1f}%\n"
    else:
        msg += "  (No open positions)\n"
        
    msg += f"\n<a href='{get_dashboard_url()}'>📊 Dashboard</a>"
    return msg
