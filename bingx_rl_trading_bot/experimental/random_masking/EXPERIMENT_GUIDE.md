# Experiment Execution Guide

**Status**: Ready to Execute
**Last Updated**: 2025-01-08
**Estimated Total Time**: 2 weeks

## Quick Start (Copy-Paste Commands)

### Day 1-2: Data Collection
```bash
# Create necessary directories
mkdir -p data/raw logs

# Start data collection (background)
nohup python -m random_masking.data.collector \
    --symbols BTCUSDT ETHUSDT \
    --start 2022-01-01 \
    --end 2024-09-30 \
    --interval 1m \
    --output data/raw/crypto_candles.parquet \
    > logs/data_collection_$(date +%Y%m%d).log 2>&1 &

# Monitor progress
tail -f logs/data_collection_*.log
```

### Day 3-4: Baseline Training
```bash
python main.py --config configs/baseline_config.yaml
```

### Day 5-7: Ablation Study (CRITICAL)
```bash
# Automated ablation study
nohup python experiments/run_ablation.py \
    --data-path data/raw/crypto_candles.parquet \
    --output-dir results/ablation_study_$(date +%Y%m%d) \
    > logs/ablation_study_$(date +%Y%m%d).log 2>&1 &

# Monitor progress
tail -f logs/ablation_study_*.log
```

### Day 8: Statistical Validation
```bash
python experiments/statistical_test.py
```

## Expected Results

### Success Criteria (Minimum)
- ✅ Sharpe Ratio > Baseline + 10%
- ✅ p-value < 0.05
- ✅ Max Drawdown < 20%
- ✅ Win Rate > 52%

### Ideal Results
- 🎯 Sharpe Ratio > 2.0
- 🎯 Consistent across BTC/ETH
- 🎯 Robust across market regimes

### Production Criteria
- 🚀 6+ months paper trading
- 🚀 Sharpe Ratio > 3.0
- 🚀 Inference < 100ms

## Troubleshooting

### Issue: Data collection fails
```bash
# Check API connection
python -c "import ccxt; exchange = ccxt.binance(); print(exchange.fetch_ticker('BTC/USDT'))"

# Check disk space
df -h

# Check Python packages
pip list | grep -E "ccxt|pandas|numpy"
```

### Issue: GPU out of memory
```python
# Reduce batch size in config
# configs/baseline_config.yaml
data:
  batch_size: 32  # Reduce from 64

model:
  hidden_dim: 128  # Reduce from 256
```

### Issue: Training too slow
```bash
# Use mixed precision training (already enabled in configs)
# Check GPU utilization
watch -n 1 nvidia-smi

# Enable model compilation (PyTorch 2.0+)
# configs/baseline_config.yaml
hardware:
  compile_model: true
```

## Results Organization

```
results/
├── ablation_study_20250108/
│   ├── experiments/
│   │   ├── baseline_pure_forecasting_*/
│   │   ├── proposed_40_40_20_*/
│   │   ├── variant_infill_heavy_*/
│   │   └── variant_forecast_heavy_*/
│   ├── ablation_comparison.csv
│   ├── ablation_summary.json
│   └── ablation_study.log
└── statistical_tests/
    ├── bootstrap_results.json
    └── significance_tests.csv
```

## Next Steps After Experiments

### If Successful (p < 0.05, improvement > 10%)
1. Generate paper figures: `python scripts/generate_figures.py`
2. Cross-asset validation: Test on more crypto pairs
3. Hyperparameter optimization: `python experiments/hyperparam_search.py`
4. Paper writing: Start with template in `paper/`

### If Unsuccessful (p > 0.05 or no improvement)
1. Failure analysis: `python scripts/failure_analysis.py`
2. Try sequential curriculum
3. Adjust loss weights
4. Increase training epochs
5. Consider different auxiliary tasks

## Paper Submission Targets

If results are positive:
1. **NeurIPS 2025 Workshop** (Time Series + Deep Learning)
2. **ICML 2025 Workshop** (AutoML or Representation Learning)
3. **AAAI 2026 Main Conference**
4. **Journal of Machine Learning Research**

## Important Notes

- **Save checkpoints frequently**: Training can take 8-12 hours
- **Monitor GPU temperature**: Should stay < 80°C
- **Backup results immediately**: Copy to external storage
- **Document everything**: Keep detailed logs of all experiments
- **Version control**: Commit after each major milestone

## Contact & Support

If you encounter issues:
1. Check logs in `logs/` directory
2. Review error messages carefully
3. Consult README.md and SYSTEM_READY.md
4. Check GitHub issues (if repository is public)

---

**Good luck with your experiments! 🚀**
