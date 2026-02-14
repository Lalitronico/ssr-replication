/**
 * Normalization Experiment: find the optimal similarity normalization + temperature
 * combination for embedding-based Likert mapping.
 *
 * Tests: none/minmax/zscore normalization × multiple temperatures
 * against hand-labeled ground truth responses.
 *
 * Usage: npx tsx research/calibration/normalization-experiment.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService, type SimilarityNormalization } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import type { SurveyQuestion } from "../../src/lib/ssr-engine";
import * as fs from "fs";
import * as path from "path";

// ─── Questions ───────────────────────────────────────────────────

const SATISFACTION_Q: SurveyQuestion = {
  id: "sat", type: "likert", text: "How satisfied are you with the checkout experience?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very dissatisfied", high: "Very satisfied" },
};

const LIKELIHOOD_Q: SurveyQuestion = {
  id: "lik", type: "likert", text: "How likely are you to purchase again?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very unlikely", high: "Very likely" },
};

const AGREEMENT_Q: SurveyQuestion = {
  id: "agr", type: "likert", text: "I am satisfied with the overall quality of this product.",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Strongly disagree", high: "Strongly agree" },
};

// ─── Ground truth responses ──────────────────────────────────────

interface TestCase {
  question: SurveyQuestion;
  text: string;
  expected: number;
  label: string;
}

const TEST_CASES: TestCase[] = [
  // Satisfaction — clear signals
  { question: SATISFACTION_Q, expected: 1, label: "SAT-1",
    text: "Absolutely awful. The page froze three times, I lost my payment info, and had to start over. Worst checkout I've ever used." },
  { question: SATISFACTION_Q, expected: 2, label: "SAT-2",
    text: "Not a good experience. The layout was confusing and I couldn't find the shipping options easily. Took me 15 minutes." },
  { question: SATISFACTION_Q, expected: 3, label: "SAT-3",
    text: "It was okay. Nothing terrible but nothing great either. Just a standard checkout process." },
  { question: SATISFACTION_Q, expected: 4, label: "SAT-4",
    text: "Pretty good overall. The process was mostly smooth and I liked the saved payment option. Minor hiccup with the address form." },
  { question: SATISFACTION_Q, expected: 5, label: "SAT-5",
    text: "Excellent checkout experience! Everything was fast, clear, and intuitive. One-click payment worked perfectly." },

  // Satisfaction — subtle/mixed signals
  { question: SATISFACTION_Q, expected: 2, label: "SAT-2b",
    text: "I managed to complete my order but it was frustrating. Several steps felt unnecessary and the coupon code didn't work right." },
  { question: SATISFACTION_Q, expected: 4, label: "SAT-4b",
    text: "Good experience for the most part. Quick and easy, though the mobile layout could use some work." },

  // Likelihood — clear signals
  { question: LIKELIHOOD_Q, expected: 1, label: "LIK-1",
    text: "No way. After that experience I'm done with this store. I'll shop elsewhere from now on." },
  { question: LIKELIHOOD_Q, expected: 2, label: "LIK-2",
    text: "Probably not. The experience wasn't great and I have other options. Maybe if they have a really good sale." },
  { question: LIKELIHOOD_Q, expected: 3, label: "LIK-3",
    text: "Maybe. It depends on whether they have what I need and if the price is right. I don't have strong feelings either way." },
  { question: LIKELIHOOD_Q, expected: 4, label: "LIK-4",
    text: "Yeah, I'll probably order again. They had decent prices and good selection. Pretty convenient overall." },
  { question: LIKELIHOOD_Q, expected: 5, label: "LIK-5",
    text: "Definitely! I already have items in my wishlist. Great selection and prices, I'll be back within the week." },

  // Agreement — clear signals
  { question: AGREEMENT_Q, expected: 1, label: "AGR-1",
    text: "I completely disagree. The product quality has been terrible. It broke after two uses and the materials feel cheap." },
  { question: AGREEMENT_Q, expected: 2, label: "AGR-2",
    text: "I don't really agree with that. The quality is below what I expected for the price. There are some issues." },
  { question: AGREEMENT_Q, expected: 3, label: "AGR-3",
    text: "I'm kind of neutral on this. The quality is acceptable but nothing special. It does what it's supposed to do." },
  { question: AGREEMENT_Q, expected: 4, label: "AGR-4",
    text: "I mostly agree. The product quality is good, well-made and reliable. A few minor things could be improved." },
  { question: AGREEMENT_Q, expected: 5, label: "AGR-5",
    text: "Absolutely agree. Outstanding quality in every way. Best product I've used in this category, exceeded all expectations." },
];

// ─── Experiment configurations ───────────────────────────────────

interface ExperimentConfig {
  normalization: SimilarityNormalization;
  temperature: number;
}

const CONFIGS: ExperimentConfig[] = [
  // No normalization (raw cosine sims ~0.75-0.83)
  { normalization: "none", temperature: 0.01 },
  { normalization: "none", temperature: 0.005 },
  { normalization: "none", temperature: 0.001 },

  // Min-max normalization (stretches to [0, 1])
  { normalization: "minmax", temperature: 1.0 },
  { normalization: "minmax", temperature: 0.5 },
  { normalization: "minmax", temperature: 0.2 },
  { normalization: "minmax", temperature: 0.1 },
  { normalization: "minmax", temperature: 0.05 },
  { normalization: "minmax", temperature: 0.01 },

  // Z-score normalization (center + scale)
  { normalization: "zscore", temperature: 1.0 },
  { normalization: "zscore", temperature: 0.5 },
  { normalization: "zscore", temperature: 0.2 },
  { normalization: "zscore", temperature: 0.1 },
];

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const service = getEmbeddingService();
  if (!service) { console.error("VOYAGE_API_KEY not set"); process.exit(1); }

  console.log("=".repeat(70));
  console.log("NORMALIZATION EXPERIMENT");
  console.log("=".repeat(70));
  console.log(`Test cases: ${TEST_CASES.length}`);
  console.log(`Configurations: ${CONFIGS.length}`);

  // ─── Step 1: Embed all anchors ─────────────────────────────────

  const questionSets = [
    { question: SATISFACTION_Q, label: "satisfaction" },
    { question: LIKELIHOOD_Q, label: "likelihood" },
    { question: AGREEMENT_Q, label: "agreement" },
  ];

  const anchorData = new Map<string, { vecs: number[][]; anchors: ReturnType<typeof resolveAnchors> }>();

  console.log("\nEmbedding anchors...");
  for (const qs of questionSets) {
    const anchors = resolveAnchors(qs.question);
    console.log(`  ${qs.label}: ${anchors.anchors.length} anchors (semantic: ${anchors.semantic})`);
    await new Promise(r => setTimeout(r, 1500));
    const vecs = await service.embed(anchors.anchors);
    anchorData.set(qs.question.id, { vecs, anchors });
  }

  // ─── Step 2: Embed all test responses ──────────────────────────

  console.log("\nEmbedding test responses...");
  const responseVecs = new Map<string, number[]>();

  for (const tc of TEST_CASES) {
    await new Promise(r => setTimeout(r, 1500));
    const [vec] = await service.embed([tc.text]);
    responseVecs.set(tc.label, vec);
    console.log(`  ${tc.label}: done`);
  }

  // ─── Step 3: Compute raw similarities ──────────────────────────

  console.log("\nComputing raw similarities...");
  const rawSims = new Map<string, number[]>();

  for (const tc of TEST_CASES) {
    const respVec = responseVecs.get(tc.label)!;
    const { vecs } = anchorData.get(tc.question.id)!;
    const sims = service.computeSimilarities(respVec, vecs);
    rawSims.set(tc.label, sims);

    const min = Math.min(...sims);
    const max = Math.max(...sims);
    const spread = max - min;
    console.log(`  ${tc.label}: [${sims.map(s => s.toFixed(4)).join(", ")}] spread=${spread.toFixed(4)}`);
  }

  // ─── Step 4: Test all configurations ───────────────────────────

  console.log("\n" + "=".repeat(70));
  console.log("RESULTS");
  console.log("=".repeat(70));

  interface ConfigResult {
    config: ExperimentConfig;
    exact: number;
    withinOne: number;
    mae: number;
    meanConfidence: number;
    details: Array<{ label: string; expected: number; predicted: number; confidence: number }>;
  }

  const results: ConfigResult[] = [];

  for (const cfg of CONFIGS) {
    let exact = 0;
    let withinOne = 0;
    const errors: number[] = [];
    const confidences: number[] = [];
    const details: ConfigResult["details"] = [];

    for (const tc of TEST_CASES) {
      const sims = rawSims.get(tc.label)!;
      const { anchors } = anchorData.get(tc.question.id)!;

      const dist = service.similaritiesToDistribution(sims, cfg.temperature, cfg.normalization);
      const rating = service.distributionToRating(dist, anchors.scaleMin);
      const clamped = Math.min(Math.max(rating, anchors.scaleMin), anchors.scaleMax);

      // Confidence from distribution
      const maxP = Math.max(...dist);
      const entropy = -dist.reduce((h, p) => p > 0 ? h + p * Math.log2(p) : h, 0);
      const maxEntropy = Math.log2(dist.length);
      const confidence = maxEntropy > 0 ? Math.round((1 - entropy / maxEntropy) * 100) / 100 : 0;

      const err = Math.abs(clamped - tc.expected);
      errors.push(err);
      confidences.push(confidence);
      details.push({ label: tc.label, expected: tc.expected, predicted: clamped, confidence });

      if (err === 0) exact++;
      if (err <= 1) withinOne++;
    }

    const mae = errors.reduce((a, b) => a + b, 0) / errors.length;
    const meanConf = confidences.reduce((a, b) => a + b, 0) / confidences.length;

    results.push({
      config: cfg,
      exact,
      withinOne,
      mae,
      meanConfidence: Math.round(meanConf * 100) / 100,
      details,
    });
  }

  // ─── Step 5: Summary table ─────────────────────────────────────

  console.log("\n" + "─".repeat(90));
  console.log(
    `${"Normalization".padEnd(14)} ${"τ".padEnd(8)} ${"Exact".padEnd(12)} ${"Within±1".padEnd(12)} ${"MAE".padEnd(10)} ${"Confidence".padEnd(12)}`
  );
  console.log("─".repeat(90));

  // Sort by exact matches (desc), then MAE (asc)
  const sorted = [...results].sort((a, b) => {
    if (b.exact !== a.exact) return b.exact - a.exact;
    return a.mae - b.mae;
  });

  for (const r of sorted) {
    const exactPct = `${r.exact}/${TEST_CASES.length} (${Math.round(r.exact / TEST_CASES.length * 100)}%)`;
    const w1Pct = `${r.withinOne}/${TEST_CASES.length} (${Math.round(r.withinOne / TEST_CASES.length * 100)}%)`;
    const marker = r === sorted[0] ? " ★ BEST" : "";
    console.log(
      `${r.config.normalization.padEnd(14)} ${r.config.temperature.toFixed(3).padEnd(8)} ${exactPct.padEnd(12)} ${w1Pct.padEnd(12)} ${r.mae.toFixed(2).padEnd(10)} ${r.meanConfidence.toFixed(2).padEnd(12)}${marker}`
    );
  }

  // ─── Step 6: Best config detail ────────────────────────────────

  const best = sorted[0];
  console.log("\n" + "=".repeat(70));
  console.log(`BEST: normalization=${best.config.normalization}, τ=${best.config.temperature}`);
  console.log("=".repeat(70));
  console.log(
    `${"Label".padEnd(8)} ${"Expected".padEnd(10)} ${"Predicted".padEnd(10)} ${"Confidence".padEnd(12)} ${"Match"}`
  );
  console.log("─".repeat(55));

  for (const d of best.details) {
    const err = d.predicted - d.expected;
    const match = err === 0 ? "exact" : Math.abs(err) <= 1 ? "±1" : `ERR(${err > 0 ? "+" : ""}${err})`;
    console.log(
      `${d.label.padEnd(8)} ${d.expected.toString().padEnd(10)} ${d.predicted.toString().padEnd(10)} ${d.confidence.toFixed(2).padEnd(12)} ${match}`
    );
  }

  // ─── Step 7: Save results ──────────────────────────────────────

  const outputPath = path.join(__dirname, "..", "data", `normalization-experiment-${Date.now()}.json`);
  const output = {
    timestamp: new Date().toISOString(),
    testCases: TEST_CASES.length,
    configurations: CONFIGS.length,
    rawSimilarities: Object.fromEntries(rawSims),
    results: sorted.map(r => ({
      normalization: r.config.normalization,
      temperature: r.config.temperature,
      exact: r.exact,
      exactPct: Math.round(r.exact / TEST_CASES.length * 100),
      withinOne: r.withinOne,
      withinOnePct: Math.round(r.withinOne / TEST_CASES.length * 100),
      mae: r.mae,
      meanConfidence: r.meanConfidence,
      details: r.details,
    })),
    recommendation: {
      normalization: best.config.normalization,
      temperature: best.config.temperature,
      exact: best.exact,
      withinOne: best.withinOne,
      mae: best.mae,
    },
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nResults saved to: ${outputPath}`);
}

main().catch(console.error);
