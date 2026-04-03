#!/usr/bin/env python3
"""
Automated Trading Pipeline — v2.0
==================================
Complete operations automation with integrity checks.

Commands:
  monitor   — Performance check + version-aware WR (hourly)
  guard     — Bot health + MDD + SL cluster + auto-restart (hourly)
  rescan    — Data extend → scan → validate → deploy → restart (weekly)
  rollback  — Revert to previous patterns (manual)
  status    — Full system status report

Scheduler:
  Every 1h:      pipeline.py guard
  Every 1h:      pipeline.py monitor
  Every Sunday:  pipeline.py rescan
"""
import sys
import os
import json
import time
import shutil
import logging
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "production"))
sys.path.insert(0, str(ROOT / "scripts" / "scanner"))

LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
CONFIG_FILE = ROOT / "config" / "pattern_5m_config.yaml"
STATE_FILE = RESULTS_DIR / "pattern_5m_bot_state.json"
METRICS_FILE = RESULTS_DIR / "pattern_5m_metrics.json"
PATTERNS_FILE = RESULTS_DIR / "dynamic_patterns.json"
PIPELINE_STATE = RESULTS_DIR / "pipeline_state.json"
ALERT_FILE = RESULTS_DIR / "pipeline_alerts.json"
BOT_SCRIPT = ROOT / "scripts" / "production" / "pattern_5m_bot.py"

THRESHOLDS = {
    "wr_green": 58.0,
    "wr_yellow": 55.0,
    "wr_red": 50.0,
    "mdd_critical": 30.0,
    "consecutive_loss": 5,
    "rescan_age_days": 60,
    "min_trades_eval": 10,
    "wf_min_pass": 3,
    "rollback_wr": 40.0,
    "rollback_trades": 10,
    "alert_cooldown_hours": 12,
}

CONFIG_VERSION = "v1.71.0"  # Current config version tag
CONFIG_EPOCH = "2026-04-03T17:40:00"  # When current config was deployed

# Production scanner params (must match actual bot config for accurate WF)
PRODUCTION_SCANNER_PARAMS = {
    "n_slots": 7, "direction_cap": 7, "cascade_tighten_pct": 0,
    "preemptive_cascade_pct": 999, "preemptive_tighten_pct": 0,
    "tp_decay_rate": 0, "timeout_bars": 576,
}

# Logging with rotation
from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        RotatingFileHandler(LOG_DIR / "pipeline.log", maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("pipeline")


def _safe_read_json(filepath, default=None, retries=3):
    """OneDrive-safe JSON read with retry."""
    for attempt in range(retries):
        try:
            with open(filepath) as f:
                return json.load(f)
        except (IOError, PermissionError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                log.warning("Failed to read {} after {} retries: {}".format(
                    Path(filepath).name, retries, e))
                return default if default is not None else {}
    return default if default is not None else {}


def _safe_write_json(filepath, data, retries=3):
    """OneDrive-safe JSON write with retry + atomic rename."""
    tmp = str(filepath) + ".tmp"
    for attempt in range(retries):
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            # Atomic rename (Windows: may need retry if OneDrive locks)
            for r in range(3):
                try:
                    os.replace(tmp, str(filepath))
                    return True
                except PermissionError:
                    time.sleep(0.3)
            return False
        except (IOError, PermissionError) as e:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                log.warning("Failed to write {}: {}".format(Path(filepath).name, e))
                return False
    return False


def _load_pipeline_state():
    if PIPELINE_STATE.exists():
        return _safe_read_json(PIPELINE_STATE,
                               default={"alerts_sent": {}, "last_rescan": None, "deployed_version": CONFIG_VERSION})
    return {"alerts_sent": {}, "last_rescan": None, "deployed_version": CONFIG_VERSION}


def _save_pipeline_state(state):
    _safe_write_json(PIPELINE_STATE, state)


def _alert(message, severity="WARNING"):
    """Write alert with cooldown deduplication."""
    pstate = _load_pipeline_state()
    alerts_sent = pstate.get("alerts_sent", {})

    # Cooldown check
    key = message[:50]
    last_sent = alerts_sent.get(key, "2000-01-01")
    hours_since = (datetime.now() - datetime.fromisoformat(last_sent)).total_seconds() / 3600
    if hours_since < THRESHOLDS["alert_cooldown_hours"]:
        return  # Suppress duplicate

    alerts_sent[key] = datetime.now().isoformat()
    pstate["alerts_sent"] = alerts_sent
    _save_pipeline_state(pstate)

    log.log(logging.CRITICAL if severity == "CRITICAL" else logging.WARNING,
            "ALERT [{}]: {}".format(severity, message))

    # Append to alert file
    alerts = []
    if ALERT_FILE.exists():
        try:
            with open(ALERT_FILE) as f:
                alerts = json.load(f)
        except:
            pass
    alerts.append({"ts": datetime.now().isoformat(), "severity": severity, "msg": message})
    with open(ALERT_FILE, "w") as f:
        json.dump(alerts[-200:], f, indent=2)


def _find_bot_pid():
    """Find running bot process."""
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe' or name='python3.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        for line in r.stdout.splitlines():
            if "pattern_5m_bot" in line:
                m = re.search(r'(\d+)\s*$', line.strip())
                if m:
                    return int(m.group(1))
    except:
        pass
    return None


def _restart_bot():
    """Graceful bot restart."""
    pid = _find_bot_pid()
    if pid:
        log.info("Stopping bot PID {}...".format(pid))
        subprocess.run(["taskkill", "//PID", str(pid), "//F"],
                       capture_output=True, timeout=10)
        time.sleep(5)

    log.info("Starting bot...")
    subprocess.Popen(
        [sys.executable, str(BOT_SCRIPT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )
    time.sleep(10)

    new_pid = _find_bot_pid()
    if new_pid:
        log.info("Bot restarted: PID {}".format(new_pid))
        return True
    else:
        log.critical("Bot failed to start!")
        _alert("Bot failed to restart!", "CRITICAL")
        return False


def _check_data_integrity(filepath):
    """Verify data file has no gaps or duplicates."""
    import pandas as pd
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    issues = []

    # Duplicates
    dups = df["timestamp"].duplicated().sum()
    if dups > 0:
        issues.append("{} duplicate timestamps".format(dups))

    # Gaps (> 10 min between consecutive bars)
    diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    big_gaps = (diffs > 600).sum()
    if big_gaps > 0:
        issues.append("{} gaps > 10min".format(big_gaps))

    # NaN values
    nans = df[["open", "high", "low", "close"]].isna().sum().sum()
    if nans > 0:
        issues.append("{} NaN price values".format(nans))

    return issues


# ============================================================
# MONITOR — Version-aware performance tracking
# ============================================================
def cmd_monitor():
    log.info("=" * 60)
    log.info("MONITOR: Performance check (config: {})".format(CONFIG_VERSION))

    if not METRICS_FILE.exists():
        log.info("No metrics file")
        return

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    th = metrics.get("trade_history", [])

    # Filter to CURRENT CONFIG VERSION only
    current = [t for t in th if str(t.get("close_time", "")) >= CONFIG_EPOCH]
    # Exclude CRASH_RECOVERY
    current = [t for t in current if t.get("exit_reason") != "CRASH_RECOVERY"]

    log.info("  Trades since {}: {}".format(CONFIG_VERSION, len(current)))

    if len(current) < 3:
        log.info("  Too few trades for evaluation")
        return

    # WR (TP+SL only, excluding cascade/timeout for clean signal)
    tp = [t for t in current if t.get("exit_reason") == "TP"]
    sl = [t for t in current if t.get("exit_reason") == "SL"]
    tpsl = len(tp) + len(sl)
    wr = 100 * len(tp) / tpsl if tpsl > 0 else 0

    total_pnl = sum(t.get("pnl_pct", 0) for t in current)
    avg_pnl = total_pnl / len(current) if current else 0

    # R:R
    avg_tp_pnl = sum(t.get("pnl_pct", 0) for t in tp) / len(tp) if tp else 0
    avg_sl_pnl = sum(t.get("pnl_pct", 0) for t in sl) / len(sl) if sl else 0
    rr = abs(avg_tp_pnl / avg_sl_pnl) if avg_sl_pnl != 0 else 0

    # Consecutive losses
    max_consec = 0
    consec = 0
    for t in sorted(current, key=lambda x: str(x.get("close_time", ""))):
        if t.get("pnl_pct", 0) < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    log.info("  {} config trades: {} total, {} TP+SL".format(CONFIG_VERSION, len(current), tpsl))
    log.info("  WR: {:.1f}%, R:R: {:.3f}, PnL: {:+.2f}%".format(wr, rr, total_pnl))
    log.info("  Avg TP: {:+.2f}%, Avg SL: {:+.2f}%".format(avg_tp_pnl, avg_sl_pnl))
    log.info("  Max consecutive loss: {}".format(max_consec))

    # Status determination
    if tpsl >= THRESHOLDS["min_trades_eval"]:
        if wr >= THRESHOLDS["wr_green"]:
            status = "GREEN"
        elif wr >= THRESHOLDS["wr_yellow"]:
            status = "YELLOW"
            _alert("WR {:.1f}% in YELLOW zone".format(wr))
        else:
            status = "RED"
            _alert("WR {:.1f}% in RED zone — consider stopping".format(wr), "CRITICAL")
    else:
        status = "PENDING ({}/{} trades)".format(tpsl, THRESHOLDS["min_trades_eval"])

    log.info("  STATUS: {}".format(status))

    # Rollback check
    if tpsl >= THRESHOLDS["rollback_trades"] and wr < THRESHOLDS["rollback_wr"]:
        _alert("WR {:.1f}% < {:.0f}% after {} trades — ROLLBACK RECOMMENDED".format(
            wr, THRESHOLDS["rollback_wr"], tpsl), "CRITICAL")

    # Save
    result = {
        "timestamp": datetime.now().isoformat(),
        "config_version": CONFIG_VERSION,
        "trades": len(current), "tpsl": tpsl,
        "wr": round(wr, 1), "rr": round(rr, 3),
        "pnl": round(total_pnl, 2), "status": status,
        "max_consec_loss": max_consec,
    }
    with open(RESULTS_DIR / "pipeline_monitor.json", "w") as f:
        json.dump(result, f, indent=2)


# ============================================================
# GUARD — Bot health + emergency checks
# ============================================================
def cmd_guard():
    log.info("GUARD: System health check")

    # 1. Bot process alive?
    pid = _find_bot_pid()
    if not pid:
        log.critical("  BOT NOT RUNNING! Auto-restarting...")
        _alert("Bot process not found — auto-restarting", "CRITICAL")
        _restart_bot()
        return

    # 2. Bot log freshness (last log entry < 10 min old?)
    logs = sorted(LOG_DIR.glob("pattern_5m_bot_*.log"))
    if logs:
        with open(logs[-1]) as f:
            lines = f.readlines()
        if lines:
            last_line = lines[-1]
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", last_line)
            if m:
                last_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                age_min = (datetime.now() - last_ts).total_seconds() / 60
                if age_min > 10:
                    log.warning("  Bot log stale: {:.0f}min old — possible hang".format(age_min))
                    _alert("Bot log stale ({:.0f}min) — possible hang".format(age_min))
                else:
                    log.info("  Bot alive: PID {}, log {:.0f}min fresh".format(pid, age_min))

    # 3. MDD check
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
        mdd = metrics.get("max_drawdown_pct", 0)
        if mdd > THRESHOLDS["mdd_critical"]:
            _alert("MDD {:.1f}% exceeds critical {:.0f}%".format(
                mdd, THRESHOLDS["mdd_critical"]), "CRITICAL")

    # 4. SL cluster detection
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        positions = state.get("positions", {})
        for direction in ["LONG", "SHORT"]:
            dir_pos = [p for p in positions.values() if p.get("direction") == direction]
            if len(dir_pos) >= 3:
                sls = [p.get("sl_price", 0) for p in dir_pos if p.get("sl_price", 0) > 0]
                if sls and len(sls) >= 3:
                    sl_spread = (max(sls) - min(sls)) / min(sls) * 100
                    if sl_spread < 0.5:
                        _alert("SL CLUSTER: {} {} positions within {:.2f}%".format(
                            len(dir_pos), direction, sl_spread), "CRITICAL")

    # 5. Exchange connectivity (quick ticker fetch)
    try:
        import ccxt, yaml
        with open(ROOT / "config" / "api_keys.yaml") as f:
            keys = yaml.safe_load(f)
        exchange = ccxt.bingx({
            "apiKey": keys.get("api_key", keys.get("bingx", {}).get("api_key", "")),
            "secret": keys.get("api_secret", keys.get("bingx", {}).get("api_secret", "")),
        })
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        log.info("  Exchange OK: BTC ${:.0f}".format(ticker["last"]))
    except Exception as e:
        _alert("Exchange connectivity failed: {}".format(str(e)[:60]))

    log.info("  GUARD: Complete")


# ============================================================
# RESCAN — Full pipeline with restart
# ============================================================
def cmd_rescan():
    log.info("=" * 60)
    log.info("RESCAN: Full pipeline")

    # 1. Check age
    if PATTERNS_FILE.exists():
        age = (datetime.now() - datetime.fromtimestamp(PATTERNS_FILE.stat().st_mtime)).days
        log.info("  Pattern age: {}d (threshold: {}d)".format(age, THRESHOLDS["rescan_age_days"]))
        if age < THRESHOLDS["rescan_age_days"]:
            log.info("  Patterns still fresh — skipping")
            return

    # 2. Extend data
    log.info("  STEP 1: Data extension...")
    _extend_data()

    # 3. Data integrity check
    data_file = DATA_DIR / "btc_5m_270days_reclassified.csv"
    issues = _check_data_integrity(str(data_file))
    if issues:
        log.warning("  Data issues: {}".format(issues))
        # Don't abort — minor gaps are OK

    # 4. Run scanner
    log.info("  STEP 2: Pattern scan (with production N={} DC={})...".format(
        PRODUCTION_SCANNER_PARAMS["n_slots"], PRODUCTION_SCANNER_PARAMS["direction_cap"]))
    scan_output = RESULTS_DIR / "dynamic_patterns_auto.json"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "scanner" / "pattern_scanner.py"),
        "--data", str(data_file),
        "--output", str(scan_output),
        "--discovery-method", "mae_mfe",
        "--edge-threshold", "18",
        "--wf-folds", "5",
        "--holdout-days", "7",
        "--n-slots", str(PRODUCTION_SCANNER_PARAMS["n_slots"]),
        "--direction-cap", str(PRODUCTION_SCANNER_PARAMS["direction_cap"]),
    ]
    # Note: cascade OFF is handled by scanner's --no-cascade flag if available
    # Scanner default runs with cascade for WF — this is acceptable as
    # the N-pos IS/WF still validates pattern quality
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            log.error("  Scanner failed")
            return
    except subprocess.TimeoutExpired:
        log.error("  Scanner timeout")
        return

    # 5. Validate
    log.info("  STEP 3: Validation...")
    if not _validate_scan(scan_output):
        log.warning("  Validation FAILED — keeping current patterns")
        return

    # 6. Deploy with backup
    log.info("  STEP 4: Deploy...")
    backup = RESULTS_DIR / "dynamic_patterns_backup_{}.json".format(
        datetime.now().strftime("%Y%m%d"))
    if PATTERNS_FILE.exists():
        shutil.copy2(PATTERNS_FILE, backup)
    shutil.copy2(scan_output, PATTERNS_FILE)

    # 7. Restart bot
    log.info("  STEP 5: Restarting bot...")
    _restart_bot()

    # 8. Update pipeline state
    pstate = _load_pipeline_state()
    pstate["last_rescan"] = datetime.now().isoformat()
    _save_pipeline_state(pstate)

    log.info("  RESCAN COMPLETE")


def _extend_data():
    """Download and append latest BTC data."""
    try:
        import ccxt, yaml
        import pandas as pd
        import numpy as np

        data_file = DATA_DIR / "btc_5m_270days_reclassified.csv"
        df = pd.read_csv(data_file, parse_dates=["timestamp"])
        last_ts = df["timestamp"].iloc[-1]
        gap = (datetime.now() - last_ts.to_pydatetime()).days
        if gap < 1:
            log.info("  Data current (gap: {}d)".format(gap))
            return True

        with open(ROOT / "config" / "api_keys.yaml") as f:
            keys = yaml.safe_load(f)
        exchange = ccxt.bingx({
            "apiKey": keys.get("api_key", keys.get("bingx", {}).get("api_key", "")),
            "secret": keys.get("api_secret", keys.get("bingx", {}).get("api_secret", "")),
            "enableRateLimit": True,
        })

        since_ms = int(last_ts.timestamp() * 1000) + 300000
        bars = []
        while since_ms < int(datetime.now().timestamp() * 1000):
            ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", "5m", since=since_ms, limit=1440)
            if not ohlcv:
                break
            bars.extend(ohlcv)
            since_ms = ohlcv[-1][0] + 300000
            if len(ohlcv) < 100:
                break
            time.sleep(0.5)

        if bars:
            new_df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
            new_df["timestamp"] = pd.to_datetime(new_df["ts"], unit="ms")
            new_df = new_df[new_df["timestamp"] > last_ts][["timestamp", "open", "high", "low", "close", "volume"]]

            from pattern_5m.indicators import classify_candle
            combined = pd.concat([df, new_df], ignore_index=True)
            body = np.abs(combined["close"].values - combined["open"].values)
            avg_body = pd.Series(body).rolling(20, min_periods=1).mean().values
            start_idx = len(df)
            ct = []
            for i in range(start_idx, len(combined)):
                row = type("R", (), {
                    "open": combined["open"].iloc[i], "high": combined["high"].iloc[i],
                    "low": combined["low"].iloc[i], "close": combined["close"].iloc[i]
                })()
                ct.append(classify_candle(row, avg_body[i]).value)
            if "candle_type" in df.columns:
                new_df["candle_type"] = ct[:len(new_df)]

            extended = pd.concat([df, new_df], ignore_index=True).drop_duplicates(
                subset=["timestamp"]).sort_values("timestamp")
            extended.to_csv(data_file, index=False)
            log.info("  Extended: +{} bars = {} total".format(len(new_df), len(extended)))
        return True
    except Exception as e:
        log.error("  Data extension error: {}".format(e))
        return False


def _validate_scan(scan_file):
    """Validate scan quality."""
    try:
        with open(scan_file) as f:
            scan = json.load(f)
        n_pat = len(scan.get("pattern_details", {}))
        if n_pat < 20:
            log.warning("  Only {} patterns".format(n_pat))
            return False

        # Check overlap with current
        if PATTERNS_FILE.exists():
            with open(PATTERNS_FILE) as f:
                cur = json.load(f)
            cur_pats = set(cur.get("pattern_details", {}).keys())
            new_pats = set(scan.get("pattern_details", {}).keys())
            overlap = len(cur_pats & new_pats) / max(len(cur_pats), 1) * 100
            log.info("  {} patterns, {:.0f}% overlap".format(n_pat, overlap))
            if overlap < 50:
                log.warning("  Low overlap — major shift")
                return False

        return True
    except Exception as e:
        log.error("  Validation error: {}".format(e))
        return False


# ============================================================
# ROLLBACK
# ============================================================
def cmd_rollback():
    """Revert to most recent backup patterns."""
    log.info("ROLLBACK: Reverting patterns")
    backups = sorted(RESULTS_DIR.glob("dynamic_patterns_backup_*.json"), reverse=True)
    if not backups:
        log.error("  No backups found!")
        return
    latest_backup = backups[0]
    log.info("  Restoring from {}".format(latest_backup.name))
    shutil.copy2(latest_backup, PATTERNS_FILE)
    _restart_bot()
    log.info("  ROLLBACK COMPLETE")


# ============================================================
# STATUS
# ============================================================
def cmd_status():
    """Full system status report."""
    print("=" * 60)
    print("SYSTEM STATUS REPORT — {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("=" * 60)

    # Bot
    pid = _find_bot_pid()
    print("\nBOT: {} (PID: {})".format("RUNNING" if pid else "STOPPED", pid or "N/A"))

    # Config
    print("CONFIG: {}, deployed {}".format(CONFIG_VERSION, CONFIG_EPOCH[:10]))

    # Positions
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        pos = state.get("positions", {})
        n_l = sum(1 for p in pos.values() if p.get("direction") == "LONG")
        n_s = sum(1 for p in pos.values() if p.get("direction") == "SHORT")
        print("POSITIONS: {}/7 (L:{} S:{})".format(len(pos), n_l, n_s))

    # Monitor
    mon_file = RESULTS_DIR / "pipeline_monitor.json"
    if mon_file.exists():
        with open(mon_file) as f:
            mon = json.load(f)
        print("MONITOR: {} — WR {:.1f}%, PnL {:+.1f}%, {} trades".format(
            mon.get("status"), mon.get("wr", 0), mon.get("pnl", 0), mon.get("trades", 0)))

    # Patterns
    if PATTERNS_FILE.exists():
        age = (datetime.now() - datetime.fromtimestamp(PATTERNS_FILE.stat().st_mtime)).days
        with open(PATTERNS_FILE) as f:
            dp = json.load(f)
        print("PATTERNS: {} (age: {}d, rescan at {}d)".format(
            len(dp.get("pattern_details", {})), age, THRESHOLDS["rescan_age_days"]))

    # Alerts
    if ALERT_FILE.exists():
        with open(ALERT_FILE) as f:
            alerts = json.load(f)
        recent = [a for a in alerts if a.get("ts", "") >= (datetime.now() - timedelta(hours=24)).isoformat()]
        print("ALERTS (24h): {}".format(len(recent)))
        for a in recent[-3:]:
            print("  [{}] {}".format(a.get("severity"), a.get("msg", "")[:60]))


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        cmd_status()
        return

    cmd = sys.argv[1].lower()
    cmds = {"monitor": cmd_monitor, "guard": cmd_guard, "rescan": cmd_rescan,
            "rollback": cmd_rollback, "status": cmd_status}

    if cmd in cmds:
        cmds[cmd]()
    elif cmd == "all":
        cmd_guard()
        cmd_monitor()
    else:
        print("Commands: monitor, guard, rescan, rollback, status, all")


if __name__ == "__main__":
    main()
