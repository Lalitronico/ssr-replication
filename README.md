# SSR Replication Materials

Replication data, code, and anchor library for:

> **Vera Pichardo, E. (2026).** Measuring Self-Rating Bias in LLM-Generated Survey Data: A Semantic Similarity Framework for Independent Scale Mapping. *arXiv preprint.*

## Repository Structure

```
data/results/          # Raw JSON results from all experiments (35 files)
scripts/
  calibration/         # TypeScript experiment scripts (14 files)
  analysis/            # Python analysis scripts (circularity statistics)
src/
  ssr-engine.ts        # SSR framework implementation
  scale-anchors.ts     # 15 anchor families (75 naturalistic statements)
  embedding-service.ts # Voyage AI / OpenAI embedding integration
figures/data/          # CSV data underlying paper figures
```

## Experiments

| Experiment | Script | Data Files |
|---|---|---|
| Pilot calibration (N=17) | `calibration-test.ts` | `calibration-*.json` |
| Normalization study | `normalization-experiment.ts` | `normalization-experiment-*.json` |
| Ablation (anchor quality) | `ablation-experiment-v2.ts` | `ablation-v2-*.json` |
| Cross-validation (N=69) | `cross-validation.ts` | `cross-validation-*.json` |
| LLM baseline comparison | `llm-baseline-experiment.ts` | `llm-baseline-*.json` |
| Circularity test (N=345) | `self-rating-experiment.ts` | `circularity-*.json`, `generated-texts-*.json`, `llm-selfratings-*.json`, `ssr-selfratings-*.json` |
| Multi-model embeddings | `multi-model-embeddings.ts` | `multi-model-embeddings-*.json` |
| Cross-model control (GPT-4o) | `cross-model-circularity.ts` | `crossmodel-*.json` |
| No-question baseline | `no-question-baseline.ts` | `no-question-baseline-*.json` |
| Statistical analysis | `analyze-circularity.py` | (reads circularity JSONs) |

## Key Results

- **SSR accuracy**: 65-67% exact match, 91% within +/-1 (Voyage AI, N=69)
- **Cross-provider**: 77% exact match (OpenAI text-embedding-3-small)
- **LLM baselines**: Claude Haiku 4.5: 87%, GPT-4o: 83%
- **Anchor quality effect**: +29 pp from naturalistic over formal anchors
- **Variance compression**: 4-fold in LLM rating (sigma^2 = 0.21 vs 0.87 for SSR)
- **Cross-model ratio**: 0.93 (compression is a general LLM property)

## Requirements

- Node.js 18+ and TypeScript for experiment scripts
- Python 3.10+ with scipy, numpy, pandas for statistical analysis
- API keys: `VOYAGE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Citation

```bibtex
@misc{verapichardo2026ssr,
  author        = {Vera Pichardo, Eduardo},
  title         = {Measuring Self-Rating Bias in {LLM}-Generated Survey Data:
                   A Semantic Similarity Framework for Independent Scale Mapping},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL},
}
```

## License

MIT
