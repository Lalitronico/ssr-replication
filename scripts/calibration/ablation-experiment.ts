/**
 * Ablation experiment: test the effect of question-contextualized anchors (H2)
 * and asymmetric embedding (H3) on rating accuracy.
 *
 * Tests 4 conditions in a 2×2 factorial design:
 *   A: baseline (current best — plain anchors, symmetric doc/doc)
 *   B: H2 only (question-contextualized anchors, symmetric doc/doc)
 *   C: H3 only (plain anchors, asymmetric doc/query)
 *   D: H2+H3 (question-contextualized anchors + asymmetric doc/query)
 *
 * Usage: npx tsx research/calibration/ablation-experiment.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService } from "../../src/lib/embedding-service";
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

// ─── Ground truth ────────────────────────────────────────────────

interface TestCase {
  question: SurveyQuestion;
  text: string;
  expected: number;
  label: string;
}

const TEST_CASES: TestCase[] = [
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
  { question: SATISFACTION_Q, expected: 2, label: "SAT-2b",
    text: "I managed to complete my order but it was frustrating. Several steps felt unnecessary and the coupon code didn't work right." },
  { question: SATISFACTION_Q, expected: 4, label: "SAT-4b",
    text: "Good experience for the most part. Quick and easy, though the mobile layout could use some work." },
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

// ─── Experiment conditions ───────────────────────────────────────

interface Condition {
  name: string;
  contextualized: boolean;  // H2: prepend question to anchors
  asymmetric: boolean;      // H3: use query input_type for responses
}

const CONDITIONS: Condition[] = [
  { name: "A: baseline",     contextualized: false, asymmetric: false },
  { name: "B: H2 (context)", contextualized: true,  asymmetric: false },
  { name: "C: H3 (asymm)",   contextualized: false, asymmetric: true },
  { name: "D: H2+H3 (both)", contextualized: true,  asymmetric: true },
];

// Best temperature/normalization from previous experiment
const TEMPERATURE = 0.2;
const NORMALIZATION = "minmax" as const;

// ─── Helpers ─────────────────────────────────────────────────────

function contextualizeAnchors(anchors: string[], questionText: string): string[] {
  return anchors.map(a => `Question: "${questionText}" Answer: ${a}`);
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const service = getEmbeddingService();
  if (!service) { console.error("VOYAGE_API_KEY not set"); process.exit(1); }

  console.log("=".repeat(75));
  console.log("ABLATION EXPERIMENT — H2 (contextualized anchors) × H3 (asymmetric embed)");
  console.log("=".repeat(75));
  console.log(`Config: normalization=${NORMALIZATION}, τ=${TEMPERATURE}`);
  console.log(`Test cases: ${TEST_CASES.length}, Conditions: ${CONDITIONS.length}\n`);

  // ─── Resolve anchors for each question ─────────────────────

  const questions = [
    { question: SATISFACTION_Q, label: "satisfaction" },
    { question: LIKELIHOOD_Q, label: "likelihood" },
    { question: AGREEMENT_Q, label: "agreement" },
  ];

  // ─── Pre-embed all anchor variants ─────────────────────────

  // We need 4 variants per question set:
  // plain/doc, contextualized/doc, plain/doc (reused), contextualized/doc
  const anchorVecsMap = new Map<string, number[][]>();

  for (const qs of questions) {
    const anchors = resolveAnchors(qs.question);
    const plainTexts = anchors.anchors;
    const ctxTexts = contextualizeAnchors(anchors.anchors, qs.question.text);

    console.log(`Embedding anchors for ${qs.label}...`);
    console.log(`  Plain: "${plainTexts[0].slice(0, 60)}..."`);
    console.log(`  Ctx:   "${ctxTexts[0].slice(0, 60)}..."`);

    // Embed plain anchors (as document)
    await new Promise(r => setTimeout(r, 1500));
    const plainVecs = await service.embed(plainTexts, "document");
    anchorVecsMap.set(`${qs.question.id}-plain`, plainVecs);

    // Embed contextualized anchors (as document)
    await new Promise(r => setTimeout(r, 1500));
    const ctxVecs = await service.embed(ctxTexts, "document");
    anchorVecsMap.set(`${qs.question.id}-ctx`, ctxVecs);
  }

  // ─── Pre-embed all response variants ───────────────────────

  console.log("\nEmbedding test responses...");
  const responseVecsDoc = new Map<string, number[]>();
  const responseVecsQuery = new Map<string, number[]>();

  for (const tc of TEST_CASES) {
    // Embed as document (symmetric)
    await new Promise(r => setTimeout(r, 1500));
    const [docVec] = await service.embed([tc.text], "document");
    responseVecsDoc.set(tc.label, docVec);

    // Embed as query (asymmetric)
    await new Promise(r => setTimeout(r, 1500));
    const [queryVec] = await service.embed([tc.text], "query");
    responseVecsQuery.set(tc.label, queryVec);

    console.log(`  ${tc.label}: done (doc + query)`);
  }

  // ─── Test all conditions ───────────────────────────────────

  console.log("\n" + "=".repeat(75));
  console.log("RESULTS");
  console.log("=".repeat(75));

  interface ConditionResult {
    condition: Condition;
    exact: number;
    withinOne: number;
    mae: number;
    meanConfidence: number;
    details: Array<{ label: string; expected: number; predicted: number; confidence: number; sims: number[] }>;
  }

  const results: ConditionResult[] = [];

  for (const cond of CONDITIONS) {
    let exact = 0;
    let withinOne = 0;
    const errors: number[] = [];
    const confidences: number[] = [];
    const details: ConditionResult["details"] = [];

    for (const tc of TEST_CASES) {
      const anchors = resolveAnchors(tc.question);
      const anchorKey = `${tc.question.id}-${cond.contextualized ? "ctx" : "plain"}`;
      const anchorVecs = anchorVecsMap.get(anchorKey)!;

      const respVec = cond.asymmetric
        ? responseVecsQuery.get(tc.label)!
        : responseVecsDoc.get(tc.label)!;

      const sims = service.computeSimilarities(respVec, anchorVecs);
      const dist = service.similaritiesToDistribution(sims, TEMPERATURE, NORMALIZATION);
      const rating = service.distributionToRating(dist, anchors.scaleMin);
      const clamped = Math.min(Math.max(rating, anchors.scaleMin), anchors.scaleMax);

      const entropy = -dist.reduce((h, p) => p > 0 ? h + p * Math.log2(p) : h, 0);
      const maxEntropy = Math.log2(dist.length);
      const confidence = maxEntropy > 0 ? Math.round((1 - entropy / maxEntropy) * 100) / 100 : 0;

      const err = Math.abs(clamped - tc.expected);
      errors.push(err);
      confidences.push(confidence);
      details.push({ label: tc.label, expected: tc.expected, predicted: clamped, confidence, sims });

      if (err === 0) exact++;
      if (err <= 1) withinOne++;
    }

    const mae = errors.reduce((a, b) => a + b, 0) / errors.length;
    const meanConf = confidences.reduce((a, b) => a + b, 0) / confidences.length;

    results.push({
      condition: cond,
      exact,
      withinOne,
      mae,
      meanConfidence: Math.round(meanConf * 100) / 100,
      details,
    });
  }

  // ─── Summary table ─────────────────────────────────────────

  console.log("\n" + "─".repeat(85));
  console.log(
    `${"Condition".padEnd(22)} ${"Exact".padEnd(14)} ${"Within±1".padEnd(14)} ${"MAE".padEnd(10)} ${"Confidence".padEnd(12)}`
  );
  console.log("─".repeat(85));

  for (const r of results) {
    const exactPct = `${r.exact}/${TEST_CASES.length} (${Math.round(r.exact / TEST_CASES.length * 100)}%)`;
    const w1Pct = `${r.withinOne}/${TEST_CASES.length} (${Math.round(r.withinOne / TEST_CASES.length * 100)}%)`;
    console.log(
      `${r.condition.name.padEnd(22)} ${exactPct.padEnd(14)} ${w1Pct.padEnd(14)} ${r.mae.toFixed(2).padEnd(10)} ${r.meanConfidence.toFixed(2).padEnd(12)}`
    );
  }

  // ─── Best condition detail ─────────────────────────────────

  const best = results.reduce((a, b) => {
    if (b.exact !== a.exact) return b.exact > a.exact ? b : a;
    if (b.withinOne !== a.withinOne) return b.withinOne > a.withinOne ? b : a;
    return b.mae < a.mae ? b : a;
  });

  console.log("\n" + "=".repeat(75));
  console.log(`BEST: ${best.condition.name}`);
  console.log("=".repeat(75));

  console.log(
    `${"Label".padEnd(8)} ${"Exp".padEnd(5)} ${"Pred".padEnd(6)} ${"Conf".padEnd(8)} ${"Match".padEnd(10)} Raw Sims`
  );
  console.log("─".repeat(75));

  for (const d of best.details) {
    const err = d.predicted - d.expected;
    const match = err === 0 ? "exact" : Math.abs(err) <= 1 ? "±1" : `ERR(${err > 0 ? "+" : ""}${err})`;
    const simsStr = d.sims.map(s => s.toFixed(3)).join(", ");
    console.log(
      `${d.label.padEnd(8)} ${d.expected.toString().padEnd(5)} ${d.predicted.toString().padEnd(6)} ${d.confidence.toFixed(2).padEnd(8)} ${match.padEnd(10)} [${simsStr}]`
    );
  }

  // ─── Per-condition detail (show errors only) ───────────────

  console.log("\n" + "=".repeat(75));
  console.log("ERRORS BY CONDITION (non-exact matches only)");
  console.log("=".repeat(75));

  for (const r of results) {
    const errCases = r.details.filter(d => d.predicted !== d.expected);
    if (errCases.length === 0) {
      console.log(`\n${r.condition.name}: NO ERRORS (perfect!)`);
      continue;
    }
    console.log(`\n${r.condition.name}: ${errCases.length} errors`);
    for (const d of errCases) {
      const err = d.predicted - d.expected;
      const sign = err > 0 ? "+" : "";
      console.log(`  ${d.label}: expected=${d.expected}, got=${d.predicted} (${sign}${err})`);
    }
  }

  // ─── Save results ──────────────────────────────────────────

  const outputPath = path.join(__dirname, "..", "data", `ablation-experiment-${Date.now()}.json`);
  const output = {
    timestamp: new Date().toISOString(),
    config: { temperature: TEMPERATURE, normalization: NORMALIZATION },
    testCases: TEST_CASES.length,
    conditions: results.map(r => ({
      name: r.condition.name,
      contextualized: r.condition.contextualized,
      asymmetric: r.condition.asymmetric,
      exact: r.exact,
      exactPct: Math.round(r.exact / TEST_CASES.length * 100),
      withinOne: r.withinOne,
      withinOnePct: Math.round(r.withinOne / TEST_CASES.length * 100),
      mae: r.mae,
      meanConfidence: r.meanConfidence,
      details: r.details,
    })),
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nResults saved to: ${outputPath}`);
}

main().catch(console.error);
