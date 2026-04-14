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
DEPLOY_MANIFEST = RESULTS_DIR / "deploy_manifest.json"
BOT_SCRIPT = ROOT / "scripts" / "production" / "pattern_5m_bot.py"

def _get_symbol_config():
    """Get symbol/data paths from config (extensible to multi-asset)."""
    cfg = _load_config()
    symbol_raw = cfg.get("symbol", "BTC-USDT")  # e.g., "BTC-USDT"
    # Convert to CCXT format
    base = symbol_raw.split("-")[0]  # "BTC"
    ccxt_symbol = "{}/USDT:USDT".format(base)
    # Data file (convention: data/{base}_5m_*.csv)
    data_files = sorted(DATA_DIR.glob("{}_5m_*.csv".format(base.lower())))
    data_file = data_files[-1] if data_files else DATA_DIR / "btc_5m_270days_reclassified.csv"
    return {"symbol_raw": symbol_raw, "ccxt_symbol": ccxt_symbol,
            "base": base, "data_file": data_file}

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

_config_cache = {}

def _load_config():
    """Load production config with caching (single source of truth)."""
    if not _config_cache:
        import yaml
        with open(CONFIG_FILE) as f:
            _config_cache.update(yaml.safe_load(f))
    return _config_cache

def _get_production_params():
    """Extract scanner-relevant params from production config."""
    cfg = _load_config()
    risk = cfg.get("risk", {})
    strategy = cfg.get("strategy", {})
    cascade = risk.get("cascade_sl_tightening", {})
    pre = risk.get("preemptive_cascade", {})
    return {
        "n_slots": cfg.get("max_positions", 7),
        "direction_cap": strategy.get("direction_cap", 7),
        "cascade_tighten_pct": cascade.get("tighten_pct", 0) if cascade.get("enabled") else 0,
        "preemptive_cascade_pct": pre.get("unrealized_loss_pct", 999) if pre.get("enabled") else 999,
        "preemptive_tighten_pct": pre.get("tighten_pct", 0) if pre.get("enabled") else 0,
        "tp_decay_rate": strategy.get("tp_decay", {}).get("decay_rate", 0) if strategy.get("tp_decay", {}).get("enabled") else 0,
        "timeout_bars": strategy.get("timeout_bars", 576),
    }

def _get_config_version():
    """Get version from deploy manifest (single source of truth)."""
    if DEPLOY_MANIFEST.exists():
        m = _safe_read_json(DEPLOY_MANIFEST, {})
        return m.get("version", "unknown")
    return "unknown"

def _get_config_epoch():
    """Get deploy timestamp from manifest."""
    if DEPLOY_MANIFEST.exists():
        m = _safe_read_json(DEPLOY_MANIFEST, {})
        return m.get("deployed_at", "2026-04-03T17:40:00")
    return "2026-04-03T17:40:00"

def _update_manifest(version, description, patterns_info=None):
    """Update deploy manifest on config/pattern change."""
    manifest = _safe_read_json(DEPLOY_MANIFEST, {"history": []})
    cfg = _load_config()

    # Record history
    history = manifest.get("history", [])
    history.append({
        "version": manifest.get("version", "unknown"),
        "date": manifest.get("deployed_at", "")[:10],
        "changes": manifest.get("description", ""),
    })

    manifest.update({
        "version": version,
        "deployed_at": datetime.now().isoformat(),
        "description": description,
        "config_snapshot": {
            "max_positions": cfg.get("max_positions"),
            "direction_cap": cfg.get("strategy", {}).get("direction_cap"),
            "leverage": cfg.get("leverage"),
            "timeout_bars": cfg.get("strategy", {}).get("timeout_bars"),
            "cascade_sl_tightening_enabled": cfg.get("risk", {}).get("cascade_sl_tightening", {}).get("enabled"),
            "preemptive_cascade_enabled": cfg.get("risk", {}).get("preemptive_cascade", {}).get("enabled"),
            "tp_mode": cfg.get("strategy", {}).get("tp_mode"),
        },
        "history": history[-20:],  # Keep last 20
    })

    if patterns_info:
        manifest["patterns"] = patterns_info

    _safe_write_json(DEPLOY_MANIFEST, manifest)

# Derived at runtime (no hardcoding)
PRODUCTION_SCANNER_PARAMS = _get_production_params()

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
                               default={"alerts_sent": {}, "last_rescan": None, "deployed_version": _get_config_version()})
    return {"alerts_sent": {}, "last_rescan": None, "deployed_version": _get_config_version()}


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
    """Graceful bot restart: SIGINT first, force kill only as fallback."""
    pid = _find_bot_pid()
    if pid:
        # Step 1: Graceful stop (CTRL_BREAK → triggers bot's shutdown handler)
        log.info("Stopping bot PID {} (graceful)...".format(pid))
        try:
            import signal
            os.kill(pid, signal.CTRL_BREAK_EVENT)
            # Wait up to 15s for graceful shutdown
            for _ in range(15):
                time.sleep(1)
                if not _find_bot_pid():
                    log.info("  Bot stopped gracefully")
                    break
            else:
                # Step 2: Force kill if graceful failed
                log.warning("  Graceful stop timeout — force killing")
                subprocess.run(["taskkill", "//PID", str(pid), "//F"],
                               capture_output=True, timeout=10)
                time.sleep(3)
        except (OSError, PermissionError):
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
def _collect_ops_metrics():
    """Collect operational metrics regardless of trade count."""
    ops = {}
    if STATE_FILE.exists():
        state = _safe_read_json(STATE_FILE, {})
        positions = state.get("positions", {})
        n_long = sum(1 for p in positions.values() if p.get("direction") == "LONG")
        n_short = len(positions) - n_long
        cfg = _load_config()
        max_pos = cfg.get("max_positions", 7)
        ops["positions"] = len(positions)
        ops["max_positions"] = max_pos
        ops["lockout"] = len(positions) >= max_pos
        ops["direction_balance"] = "L:{} S:{}".format(n_long, n_short)
        ops["imbalance_pct"] = round(abs(n_long - n_short) / max(len(positions), 1) * 100)

    logs = sorted(LOG_DIR.glob("pattern_5m_bot_*.log"))
    if logs:
        try:
            with open(logs[-1]) as f:
                bot_lines = f.readlines()
            recent_lines = [l for l in bot_lines if l >= _get_config_epoch()[:10]]
            sigs = sum(1 for l in recent_lines if "Signal detected" in l)
            entries = sum(1 for l in recent_lines if "Position opened" in l)
            ops["signals"] = sigs
            ops["entries"] = entries
            ops["capture_rate"] = round(100 * entries / max(sigs, 1), 1)
        except:
            pass
    return ops


def cmd_monitor():
    log.info("=" * 60)
    log.info("MONITOR: Performance check (config: {})".format(_get_config_version()))

    if not METRICS_FILE.exists():
        log.info("No metrics file")
        return

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    th = metrics.get("trade_history", [])

    # Filter to CURRENT CONFIG VERSION only
    current = [t for t in th if str(t.get("close_time", "")) >= _get_config_epoch()]
    # Exclude CRASH_RECOVERY
    current = [t for t in current if t.get("exit_reason") != "CRASH_RECOVERY"]

    log.info("  Trades since {}: {}".format(_get_config_version(), len(current)))

    # Always collect operational metrics (even with 0 trades)
    ops = _collect_ops_metrics()
    log.info("  Ops: {}".format(ops))

    if len(current) < 3:
        log.info("  Too few trades for evaluation")
        result = {
            "timestamp": datetime.now().isoformat(),
            "config_version": _get_config_version(),
            "trades": len(current), "tpsl": 0, "wr": 0, "rr": 0,
            "pnl": 0, "status": "PENDING",
            "max_consec_loss": 0, "operational": ops,
        }
        _safe_write_json(RESULTS_DIR / "pipeline_monitor.json", result)
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

    log.info("  {} config trades: {} total, {} TP+SL".format(_get_config_version(), len(current), tpsl))
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

    # Operational metrics (extracted to function above)
    ops = _collect_ops_metrics()

    log.info("  Ops: {}".format(ops))

    # Save
    result = {
        "timestamp": datetime.now().isoformat(),
        "config_version": _get_config_version(),
        "trades": len(current), "tpsl": tpsl,
        "wr": round(wr, 1), "rr": round(rr, 3),
        "pnl": round(total_pnl, 2), "status": status,
        "max_consec_loss": max_consec,
        "operational": ops,
    }
    _safe_write_json(RESULTS_DIR / "pipeline_monitor.json", result)


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
        sym = _get_symbol_config()
        ticker = exchange.fetch_ticker(sym["ccxt_symbol"])
        log.info("  Exchange OK: {} ${:.0f}".format(sym["base"], ticker["last"]))
    except Exception as e:
        _alert("Exchange connectivity failed: {}".format(str(e)[:60]))

    # 6. Bot log error trend
    logs = sorted(LOG_DIR.glob("pattern_5m_bot_*.log"))
    if logs:
        try:
            with open(logs[-1]) as f:
                bot_lines = f.readlines()
            errors_1h = sum(1 for l in bot_lines[-720:]  # ~1h of 5s lines
                           if "ERROR" in l or "CRITICAL" in l)
            if errors_1h > 10:
                _alert("Bot log: {} errors in last hour".format(errors_1h))
        except:
            pass

    # Always save pipeline state (ensures cooldown tracking works)
    pstate = _load_pipeline_state()
    pstate["last_guard"] = datetime.now().isoformat()
    _save_pipeline_state(pstate)

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
    data_file = _get_symbol_config()["data_file"]
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

    # 8. Update manifest + pipeline state
    try:
        with open(scan_output) as f:
            scan_data = json.load(f)
        pat_details = scan_data.get("pattern_details", {})
        n_long = sum(1 for v in pat_details.values() if v.get("direction") == "LONG")
        n_short = len(pat_details) - n_long
        import pandas as pd
        data_file = _get_symbol_config()["data_file"]
        data_days = len(pd.read_csv(data_file)) // 288
        _update_manifest(
            version="auto_{}".format(datetime.now().strftime("%Y%m%d")),
            description="Auto rescan: {} patterns ({}L+{}S), {}d data".format(
                len(pat_details), n_long, n_short, data_days),
            patterns_info={"count": len(pat_details), "long": n_long,
                          "short": n_short, "data_days": data_days,
                          "scan_date": datetime.now().strftime("%Y-%m-%d")},
        )
    except Exception as e:
        log.warning("  Manifest update failed: {}".format(e))

    pstate = _load_pipeline_state()
    pstate["last_rescan"] = datetime.now().isoformat()
    _save_pipeline_state(pstate)

    log.info("  RESCAN COMPLETE")


def _extend_data():
    """Download and append latest data (symbol from config)."""
    try:
        import ccxt, yaml
        import pandas as pd
        import numpy as np

        data_file = _get_symbol_config()["data_file"]
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
            ohlcv = exchange.fetch_ohlcv(_get_symbol_config()["ccxt_symbol"], "5m", since=since_ms, limit=1440)
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
# DEPLOY — Atomic config change + restart + record
# ============================================================
def cmd_deploy():
    """Deploy a config change: record manifest + restart bot.
    Usage: pipeline.py deploy "v1.72.0" "description of changes"
    """
    if len(sys.argv) < 4:
        print("Usage: pipeline.py deploy <version> <description>")
        return

    version = sys.argv[2]
    description = sys.argv[3]

    log.info("DEPLOY: {} — {}".format(version, description))

    # 1. Record manifest (reads current config.yaml)
    _update_manifest(version, description)
    log.info("  Manifest updated")

    # 2. Restart bot to pick up any config changes
    _restart_bot()

    log.info("  DEPLOY COMPLETE: {}".format(version))


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
    print("CONFIG: {}, deployed {}".format(_get_config_version(), _get_config_epoch()[:10]))

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
            "rollback": cmd_rollback, "deploy": cmd_deploy, "status": cmd_status}

    if cmd in cmds:
        cmds[cmd]()
    elif cmd == "all":
        cmd_guard()
        cmd_monitor()
    else:
        print("Commands: monitor, guard, rescan, rollback, status, all")


if __name__ == "__main__":
    main()
