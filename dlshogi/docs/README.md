# DeepLearningShogi Documentation

This directory contains general documentation for the DeepLearningShogi project.

## Files

### [optimization_ideas.md](optimization_ideas.md)
Research-backed optimization strategies for improving accuracy and inference speed.

**Contents:**
- Channel shuffle between blocks
- Fused depthwise-pointwise convolutions
- Quantization-aware training (QAT)
- Knowledge distillation
- Adaptive MLP expansion ratios
- Structured pruning
- References to recent papers (2025-2026)

**When to read:** When planning new experiments or optimizing existing models.

---

### [PROFILING_GUIDE.md](PROFILING_GUIDE.md)
Guide to profiling experiments and choosing the right profiling tool.

**Contents:**
- Comparison of profile.py vs profile_detailed.py vs profile_large_batch.py
- When to use each profiling script
- Standard workflow recommendations
- Tips for comparing experiments

**When to read:** Before profiling a new experiment or when unsure which tool to use.

---

## Experiment-Specific Documentation

Detailed profiling results and analysis for specific experiments are stored in:
```
dlshogi/experiments/expNNN_description/docs/
```

For example, exp015 documentation:
```
dlshogi/experiments/exp015_inceptionnext_depth10/docs/
├── PROFILE_RESULTS.md           # Initial profiling results
├── REALISTIC_BATCH_ANALYSIS.md  # Training (1024) vs Search (128) analysis
└── LARGE_BATCH_ANALYSIS.md      # Throughput testing (batch 1-4096)
```

---

## Documentation Structure

```
dlshogi/
├── docs/                                    # General documentation
│   ├── README.md                            # This file
│   ├── optimization_ideas.md                # Optimization strategies
│   └── PROFILING_GUIDE.md                   # Profiling guide
│
├── experiments/
│   ├── log.md                               # Experiment log (all experiments)
│   ├── README.md                            # Experiment creation guide
│   │
│   └── expNNN_description/
│       ├── docs/                            # Experiment-specific docs
│       │   ├── PROFILE_RESULTS.md           # Profiling results
│       │   └── ...                          # Other analysis docs
│       ├── model.py                         # Model implementation
│       ├── config.yaml                      # Configuration
│       ├── run.sh                           # Training script
│       ├── profile.py                       # Quick profiling
│       ├── profile_detailed.py              # Detailed profiling
│       └── profile_large_batch.py           # Throughput testing
│
├── InceptionNeXt.md                         # Architecture rationale
└── shogi_engine_architecture_recommendation.md  # Architecture recommendations
```

---

## Quick Links

### For New Experiments
1. Read: [experiments/README.md](../experiments/README.md) - How to create experiments
2. Read: [PROFILING_GUIDE.md](PROFILING_GUIDE.md) - How to profile
3. Reference: [optimization_ideas.md](optimization_ideas.md) - Optimization strategies

### For Optimization Work
1. Read: [optimization_ideas.md](optimization_ideas.md) - Research-backed strategies
2. Check: Experiment-specific docs in `experiments/expNNN/docs/`
3. Compare: Multiple experiment profiles

### For Production Deployment
1. Read: Experiment docs (e.g., `exp015/docs/REALISTIC_BATCH_ANALYSIS.md`)
2. Run: `profile_large_batch.py` for throughput testing
3. Reference: [PROFILING_GUIDE.md](PROFILING_GUIDE.md) for best practices

---

## Contributing Documentation

### Adding General Documentation
Place in `dlshogi/docs/`:
- Optimization strategies
- Architecture guides
- Best practices
- Tool comparisons

### Adding Experiment Documentation
Place in `dlshogi/experiments/expNNN_description/docs/`:
- Profiling results
- Performance analysis
- Experiment-specific findings
- Comparison with other experiments

### Naming Conventions
- `UPPERCASE.md` - Important reference documents
- `lowercase.md` - General guides and explanations
- `expNNN_*.md` - Experiment-specific documents

---

## Maintenance

### Keeping Documentation Updated
- Update `experiments/log.md` when completing experiments
- Add profiling results to experiment `docs/` folder
- Update this README when adding new general documentation
- Archive old optimization ideas when implemented

### Cleaning Up
- Remove duplicate or outdated documents
- Consolidate related documents when appropriate
- Keep experiment docs with their experiments
- Move general insights to `dlshogi/docs/`
