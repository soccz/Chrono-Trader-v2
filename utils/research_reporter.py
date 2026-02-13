import os
import pandas as pd
import glob
from datetime import datetime, timedelta
import time
import logging
from utils.config import config

logger = logging.getLogger("CryptoPredictor")

class ResearchReporter:
    def __init__(self, mode='daily'):
        self.mode = mode
        self.report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'research_reports')
        self.rec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'recommendations')
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.report_dir, exist_ok=True)

    def _get_todays_files(self):
        """Finds recommendation CSVs generated today."""
        today_str = datetime.now().strftime("%Y%m%d")
        # Matches patterns like recs_daily_trending_20260120_*.csv
        pattern = os.path.join(self.rec_dir, f"recs_*{today_str}*.csv")
        files = glob.glob(pattern)
        return sorted(files)

    def _analyze_csv(self, file_path):
        """Extracts key stats from a recommendation CSV."""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            
            stats = {
                'filename': os.path.basename(file_path),
                'total_recs': len(df),
                'long_count': len(df[df['signal'] == 'Long']),
                'short_count': len(df[df['signal'] == 'Short']),
                'avg_confidence': df['confidence'].mean(),
                'max_confidence': df['confidence'].max(),
                'avg_uncertainty': df['uncertainty'].mean() if 'uncertainty' in df.columns else None,
                'strategies': df['strategy'].unique().tolist() if 'strategy' in df.columns else [],
                'top_picks': df.head(3)[['market', 'signal', 'confidence']].to_dict('records')
            }
            return stats
        except Exception as e:
            logger.error(f"Error analyzing CSV {file_path}: {e}")
            return None

    def _analyze_past_performance(self):
        """Analyzes performance of yesterday's recommendations."""
        from data.collector import get_current_price
        
        # Find yesterday's file
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        pattern = os.path.join(self.rec_dir, f"recs_*{yesterday_str}*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None
            
        results = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty: continue
                
                for _, row in df.iterrows():
                    market = row['market']
                    signal = row['signal']
                    entry_price = row['current_price']
                    current_price = get_current_price(market)
                    time.sleep(0.1) # Avoid rate limit
                    
                    if current_price:
                        # Calculate Return
                        raw_return = (current_price - entry_price) / entry_price
                        if signal == 'Short':
                            pnl = -raw_return
                        else: # Long
                            pnl = raw_return
                            
                        # Success?
                        is_correct = pnl > 0
                        
                        results.append({
                            'market': market,
                            'signal': signal,
                            'entry': entry_price,
                            'current': current_price,
                            'pnl': pnl,
                            'is_correct': is_correct,
                            'confidence': row['confidence']
                        })
            except Exception as e:
                logger.error(f"Error checking past performance for {f}: {e}")
                
        if not results: return None
        
        # Summarize
        total = len(results)
        wins = len([r for r in results if r['is_correct']])
        win_rate = wins / total if total > 0 else 0
        avg_pnl = sum([r['pnl'] for r in results]) / total if total > 0 else 0
        
        return {
            'date': yesterday_str,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'details': results
        }

    def generate_report(self):
        """Generates the Markdown report."""
        files = self._get_todays_files()
        past_perf = self._analyze_past_performance()
        
        if not files and not past_perf:
            logger.warning("No data to report on.")
            return

        today_date = datetime.now().strftime("%Y-%m-%d")
        report_content = [f"# 🧪 Daily Research Report: {today_date}", ""]
        
        report_content.append("## 1. Executive Summary")
        report_content.append(f"- **Generated At**: {datetime.now().strftime('%H:%M:%S')}")
        report_content.append(f"- **Mode**: {self.mode.upper()}")
        report_content.append("")

        # --- Performance Section (Yesterday) ---
        if past_perf:
            report_content.append(f"## 2. 🔙 Performance Review ({past_perf['date']})")
            report_content.append(f"- **Win Rate**: {past_perf['win_rate']:.1%} ({len([x for x in past_perf['details'] if x['is_correct']])}/{len(past_perf['details'])})")
            report_content.append(f"- **Avg PnL**: {past_perf['avg_pnl']:.2%}")
            
            report_content.append("\n### 🔍 Failure Analysis (Wrong Predictions)")
            failures = [x for x in past_perf['details'] if not x['is_correct']]
            if failures:
                report_content.append("| Market | Signal | Conf | PnL | Result |")
                report_content.append("|--------|--------|------|-----|--------|")
                for fail in failures:
                    report_content.append(f"| {fail['market']} | {fail['signal']} | {fail['confidence']:.4f} | {fail['pnl']:.2%} | ❌ Fail |")
            else:
                report_content.append("🎉 No failures found! Perfect accuracy.")
            report_content.append("")
        else:
            report_content.append("## 2. 🔙 Performance Review")
            report_content.append("*No data found for yesterday to verify.*")
            report_content.append("")

        # --- Today's Predictions ---
        report_content.append("## 3. 🔭 Today's Predictions")
        for f in files:
            stats = self._analyze_csv(f)
            if not stats: continue

            report_content.append(f"### 📄 File: `{stats['filename']}`")
            report_content.append(f"- **Total Signals**: {stats['total_recs']} (Long: {stats['long_count']} / Short: {stats['short_count']})")
            report_content.append(f"- **Avg Confidence**: {stats['avg_confidence']:.4f}")
            
            report_content.append("\n**Top Picks:**")
            report_content.append("| Market | Signal | Confidence |")
            report_content.append("|--------|--------|------------|")
            for pick in stats['top_picks']:
                report_content.append(f"| {pick['market']} | {pick['signal']} | {pick['confidence']:.4f} |")
            report_content.append("")
            report_content.append("---")


        report_content.append("\n## 2. Model Diagnostics (Discussion)")
        report_content.append("- [ ] **Bias Check**: Is the Long/Short ratio balanced?")
        report_content.append("- [ ] **Confidence Check**: Are confidence scores within expected range (0.001~0.01)?")
        report_content.append("- [ ] **Action Item**: Review any outliers above.")

        # Save Report
        filename = f"Research_Log_{datetime.now().strftime('%Y%m%d')}.md"
        full_path = os.path.join(self.report_dir, filename)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        
        logger.info(f"✅ Research Report generated: {full_path}")
        return full_path

def run():
    reporter = ResearchReporter()
    reporter.generate_report()

if __name__ == "__main__":
    run()
