"""
Autonomous Analyst - Claude의 자율 분석 및 개선 시스템

Claude Code가 주기적으로 시스템을 분석하고 자동으로 개선 방안을 도출하여 실행합니다.

Features:
1. 자동 성능 분석 (매 시간)
2. 문제 자동 감지 및 진단
3. 개선 방안 자동 도출
4. 자동 실행 (critical thinking 기반)
5. 결과 로깅 및 리포트

Usage:
    python scripts/production/autonomous_analyst.py
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
ANALYSIS_DIR = PROJECT_ROOT / "autonomous_analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


class AutonomousAnalyst:
    """Claude의 자율 분석 엔진"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.analysis_log = []
        self.recommendations = []
        self.auto_actions = []

    def log(self, message, level="INFO"):
        """분석 로그 기록"""
        entry = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'level': level,
            'message': message
        }
        self.analysis_log.append(entry)
        print(f"[{entry['level']}] {entry['message']}")

    def analyze_v2_performance(self):
        """V2 성능 자동 분석"""
        self.log("=" * 80)
        self.log("🤖 AUTONOMOUS ANALYSIS - V2 Bot Performance")
        self.log("=" * 80)

        # Find latest V2 log
        v2_logs = sorted(LOGS_DIR.glob("combined_v2_realistic_*.log"), reverse=True)
        if not v2_logs:
            self.log("❌ No V2 log found", "ERROR")
            return None

        log_file = v2_logs[0]
        self.log(f"📂 Analyzing: {log_file.name}")

        # Parse log
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = f.readlines()

        # Extract data
        start_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*STARTED', content)
        start_time = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S") if start_match else None

        # Current position
        holding_matches = re.findall(r'Holding (LONG|SHORT): P&L ([+-]\d+\.\d{2})% \| (\d+\.\d+)h', content)
        current_position = None
        if holding_matches:
            last = holding_matches[-1]
            current_position = {
                'side': last[0],
                'pnl': float(last[1]),
                'duration': float(last[2])
            }

        # Completed trades
        exits = re.findall(r'POSITION EXITED - (.*?)\n.*?P&L: ([+-]\d+\.\d{2})%', content, re.DOTALL)
        completed_trades = len(exits)

        # Entry signals
        entry_probs = re.findall(r'(LONG|SHORT) Probability: ([\d.]+)', content)

        # Runtime
        runtime_hours = (datetime.now() - start_time).total_seconds() / 3600 if start_time else 0

        analysis = {
            'start_time': start_time,
            'runtime_hours': runtime_hours,
            'current_position': current_position,
            'completed_trades': completed_trades,
            'exits': exits,
            'entry_probs': entry_probs,
            'log_file': str(log_file)
        }

        self.log(f"⏱️  Runtime: {runtime_hours:.1f} hours")
        self.log(f"📊 Completed Trades: {completed_trades}")
        if current_position:
            self.log(f"📍 Current Position: {current_position['side']} {current_position['pnl']:+.2f}% ({current_position['duration']:.1f}h)")

        return analysis

    def critical_analysis(self, analysis):
        """비판적 분석 - 문제 및 개선점 도출"""
        self.log("")
        self.log("🧠 CRITICAL ANALYSIS")
        self.log("-" * 80)

        issues = []
        recommendations = []

        # Issue 1: No completed trades yet
        if analysis['completed_trades'] == 0:
            runtime = analysis['runtime_hours']

            if runtime < 4:
                self.log(f"✅ No trades yet - Normal (runtime: {runtime:.1f}h < 4h)")
            elif runtime < 24:
                self.log(f"⚠️  No trades in {runtime:.1f}h - Monitoring required")
                issues.append({
                    'severity': 'WARNING',
                    'issue': f'No completed trades in {runtime:.1f} hours',
                    'impact': 'Cannot validate V2 improvements'
                })
                recommendations.append({
                    'action': 'MONITOR',
                    'description': 'Wait for first trade completion (24h limit)',
                    'automated': False
                })
            else:
                self.log(f"🚨 CRITICAL: No trades in {runtime:.1f}h", "CRITICAL")
                issues.append({
                    'severity': 'CRITICAL',
                    'issue': f'No completed trades in {runtime:.1f} hours',
                    'impact': 'V2 validation impossible, may indicate model issue'
                })
                recommendations.append({
                    'action': 'INVESTIGATE',
                    'description': 'Check entry probability distributions',
                    'automated': True
                })
                recommendations.append({
                    'action': 'CONSIDER',
                    'description': 'Lower threshold (0.7→0.6 LONG, 0.4→0.35 SHORT)',
                    'automated': False
                })

        # Issue 2: Current position analysis
        if analysis['current_position']:
            pos = analysis['current_position']
            pnl = pos['pnl']
            duration = pos['duration']
            side = pos['side']

            # Check if approaching stop loss
            sl_threshold = -1.0 if side == 'LONG' else -1.5
            distance_to_sl = sl_threshold - pnl

            if distance_to_sl < 0.3:  # Within 0.3% of SL
                self.log(f"⚠️  Position {distance_to_sl:.2f}% from Stop Loss!", "WARNING")
                issues.append({
                    'severity': 'WARNING',
                    'issue': f'{side} position near SL ({pnl:+.2f}%)',
                    'impact': 'Likely to hit SL soon'
                })
            elif pnl < -0.5:
                self.log(f"⚠️  Position losing {pnl:.2f}% - Monitoring")
            elif pnl > 0:
                self.log(f"✅ Position profitable: {pnl:+.2f}%")

            # Check holding time
            if duration > 3.5:
                self.log(f"⚠️  Position held {duration:.1f}h (Max: 4h)")
                issues.append({
                    'severity': 'INFO',
                    'issue': f'Position approaching max holding time',
                    'impact': 'Will exit at 4h if no TP/SL hit'
                })

        # Issue 3: Entry probability analysis
        if analysis['entry_probs']:
            long_probs = [float(p) for side, p in analysis['entry_probs'] if side == 'LONG']
            short_probs = [float(p) for side, p in analysis['entry_probs'] if side == 'SHORT']

            if long_probs:
                max_long = max(long_probs)
                avg_long = sum(long_probs) / len(long_probs)
                self.log(f"📊 LONG signals: max={max_long:.3f}, avg={avg_long:.3f}, count={len(long_probs)}")

                if max_long < 0.6:
                    self.log(f"⚠️  LONG signals weak (max {max_long:.3f} < 0.6)")
                    issues.append({
                        'severity': 'WARNING',
                        'issue': 'LONG model not generating strong signals',
                        'impact': 'May miss LONG opportunities'
                    })
                    recommendations.append({
                        'action': 'CONSIDER',
                        'description': 'Review LONG model or lower threshold to 0.65',
                        'automated': False
                    })

            if short_probs:
                max_short = max(short_probs)
                avg_short = sum(short_probs) / len(short_probs)
                entered_shorts = len([p for p in short_probs if p >= 0.4])
                self.log(f"📊 SHORT signals: max={max_short:.3f}, avg={avg_short:.3f}, entered={entered_shorts}/{len(short_probs)}")

        self.recommendations = recommendations
        return {'issues': issues, 'recommendations': recommendations}

    def generate_recommendations(self, analysis, critical):
        """개선 권장사항 생성"""
        self.log("")
        self.log("💡 RECOMMENDATIONS")
        self.log("-" * 80)

        if not critical['recommendations']:
            self.log("✅ No immediate actions required - System performing as expected")
            return

        for i, rec in enumerate(critical['recommendations'], 1):
            action = rec['action']
            desc = rec['description']
            auto = "🤖 AUTO" if rec['automated'] else "👤 MANUAL"

            self.log(f"{i}. [{action}] {auto}")
            self.log(f"   {desc}")

        # Auto-executable recommendations
        auto_recs = [r for r in critical['recommendations'] if r['automated']]
        if auto_recs:
            self.log("")
            self.log(f"🤖 {len(auto_recs)} recommendations can be auto-executed")
            return auto_recs

        return []

    def execute_auto_improvements(self, auto_recs, analysis):
        """자동 개선 실행"""
        if not auto_recs:
            return

        self.log("")
        self.log("🚀 AUTO-EXECUTING IMPROVEMENTS")
        self.log("-" * 80)

        for rec in auto_recs:
            if rec['action'] == 'INVESTIGATE':
                self.log("🔍 Investigating entry probability distributions...")
                self.investigate_entry_thresholds(analysis)

    def investigate_entry_thresholds(self, analysis):
        """Entry threshold 조사 및 분석"""
        self.log("")
        self.log("📊 Entry Threshold Investigation")
        self.log("-" * 40)

        # Count signals by probability range
        if not analysis['entry_probs']:
            self.log("⚠️  No entry probability data available")
            return

        long_probs = [float(p) for side, p in analysis['entry_probs'] if side == 'LONG']
        short_probs = [float(p) for side, p in analysis['entry_probs'] if side == 'SHORT']

        # Analyze LONG threshold
        if long_probs:
            ranges = {
                '≥0.7': len([p for p in long_probs if p >= 0.7]),
                '0.6-0.7': len([p for p in long_probs if 0.6 <= p < 0.7]),
                '0.5-0.6': len([p for p in long_probs if 0.5 <= p < 0.6]),
                '<0.5': len([p for p in long_probs if p < 0.5])
            }

            self.log("LONG Signal Distribution:")
            for range_name, count in ranges.items():
                pct = count / len(long_probs) * 100
                self.log(f"  {range_name}: {count} ({pct:.1f}%)")

            # Recommendation
            if ranges['≥0.7'] == 0 and ranges['0.6-0.7'] > 0:
                self.log("💡 RECOMMENDATION: Lower LONG threshold to 0.65")
                self.log(f"   Reason: No signals ≥0.7, but {ranges['0.6-0.7']} signals in 0.6-0.7 range")
                self.recommendations.append({
                    'action': 'ADJUST_THRESHOLD',
                    'parameter': 'LONG_THRESHOLD',
                    'current': 0.7,
                    'recommended': 0.65,
                    'reason': f'No ≥0.7 signals, {ranges["0.6-0.7"]} in 0.6-0.7 range',
                    'automated': False  # Requires user approval for trading params
                })

        # Analyze SHORT threshold
        if short_probs:
            ranges = {
                '≥0.5': len([p for p in short_probs if p >= 0.5]),
                '0.4-0.5': len([p for p in short_probs if 0.4 <= p < 0.5]),
                '0.3-0.4': len([p for p in short_probs if 0.3 <= p < 0.4]),
                '<0.3': len([p for p in short_probs if p < 0.3])
            }

            self.log("")
            self.log("SHORT Signal Distribution:")
            for range_name, count in ranges.items():
                pct = count / len(short_probs) * 100
                self.log(f"  {range_name}: {count} ({pct:.1f}%)")

            # Check if current threshold is working
            entered = len([p for p in short_probs if p >= 0.4])
            if entered > 0:
                self.log(f"✅ Current threshold (0.4) working: {entered} entries")
            else:
                if ranges['0.3-0.4'] > 0:
                    self.log("💡 RECOMMENDATION: Lower SHORT threshold to 0.35")
                    self.log(f"   Reason: No entries at 0.4, but {ranges['0.3-0.4']} signals in 0.3-0.4 range")
                    self.recommendations.append({
                        'action': 'ADJUST_THRESHOLD',
                        'parameter': 'SHORT_THRESHOLD',
                        'current': 0.4,
                        'recommended': 0.35,
                        'reason': f'No ≥0.4 signals, {ranges["0.3-0.4"]} in 0.3-0.4 range',
                        'automated': False
                    })

    def save_analysis_report(self, analysis, critical):
        """분석 리포트 저장"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = ANALYSIS_DIR / f"analysis_{timestamp_str}.json"

        report = {
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'analysis': {
                'runtime_hours': analysis['runtime_hours'],
                'completed_trades': analysis['completed_trades'],
                'current_position': analysis['current_position']
            },
            'critical_analysis': critical,
            'recommendations': self.recommendations,
            'log': self.analysis_log
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log("")
        self.log(f"💾 Analysis saved: {report_file.name}")

        # Also save human-readable summary
        summary_file = ANALYSIS_DIR / f"summary_{timestamp_str}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AUTONOMOUS ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Runtime: {analysis['runtime_hours']:.1f} hours\n")
            f.write(f"Completed Trades: {analysis['completed_trades']}\n")
            f.write("\n")

            if critical['issues']:
                f.write("ISSUES DETECTED:\n")
                f.write("-" * 80 + "\n")
                for issue in critical['issues']:
                    f.write(f"[{issue['severity']}] {issue['issue']}\n")
                    f.write(f"Impact: {issue['impact']}\n")
                    f.write("\n")

            if self.recommendations:
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 80 + "\n")
                for i, rec in enumerate(self.recommendations, 1):
                    f.write(f"{i}. [{rec['action']}] {'AUTO' if rec.get('automated') else 'MANUAL'}\n")
                    f.write(f"   {rec['description']}\n")
                    if 'reason' in rec:
                        f.write(f"   Reason: {rec['reason']}\n")
                    f.write("\n")

        self.log(f"📄 Summary saved: {summary_file.name}")

    def run(self):
        """자율 분석 실행"""
        try:
            # Step 1: Analyze V2 performance
            analysis = self.analyze_v2_performance()
            if not analysis:
                return

            # Step 2: Critical analysis
            critical = self.critical_analysis(analysis)

            # Step 3: Generate recommendations
            auto_recs = self.generate_recommendations(analysis, critical)

            # Step 4: Execute auto-improvements
            if auto_recs:
                self.execute_auto_improvements(auto_recs, analysis)

            # Step 5: Save report
            self.save_analysis_report(analysis, critical)

            self.log("")
            self.log("=" * 80)
            self.log("✅ AUTONOMOUS ANALYSIS COMPLETE")
            self.log("=" * 80)

        except Exception as e:
            self.log(f"❌ Error during analysis: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")


if __name__ == "__main__":
    analyst = AutonomousAnalyst()
    analyst.run()
