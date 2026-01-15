import requests
from datetime import datetime
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram credentials from environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
HISTORY_FILE = "logs/telegram_history.log"
TUNNEL_URL_FILE = "tunnel_url.txt"

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
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("텔레그램 알림 전송 성공")
            _save_message_hash(message)
        else:
            print(f"텔레그램 전송 실패: {response.text}")
    except Exception as e:
        print(f"텔레그램 에러 발생: {e}")

def format_daily_report(trending_recs, pattern_recs, pump_recs):
    """
    Generates a premium structured HTML report for Telegram.
    """
    date_str = datetime.now().strftime("%Y-%m-%d | %H:%M AM")
    
    msg = f"<b>✨ AETHER QUANT PREMIUM</b>\n"
    msg += f"<code>{date_str}</code>\n\n"
    
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
        text += f"   🧠 <i>{reason}</i>\n\n"
    return text

def format_short_term_report(scalp_recs, pump_recs):
    """
    Generates a premium HTML report for 4H Scalping.
    """
    date_str = datetime.now().strftime("%m-%d %H:%M")
    
    msg = f"<b>⚡ AETHER SCALP (4H)</b>\n"
    msg += f"<code>{date_str}</code>\n\n"
    
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
