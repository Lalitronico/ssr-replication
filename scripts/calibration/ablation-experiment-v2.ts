/**
 * Ablation experiment v2: H2 (contextualized anchors) × H3 (asymmetric embed)
 *
 * Uses expanded ground-truth-v2 (62 test cases, 8 domains).
 * 2×2 factorial design:
 *   A: baseline (plain anchors, symmetric doc/doc)
 *   B: H2 only (question-contextualized anchors, symmetric doc/doc)
 *   C: H3 only (plain anchors, asymmetric doc/query)
 *   D: H2+H3 (question-contextualized anchors + asymmetric doc/query)
 *
 * Usage: npx tsx research/calibration/ablation-experiment-v2.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import { TEST_CASES, getDomains, QUESTIONS_BY_DOMAIN, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const TEMPERATURE = 0.2;
const NORMALIZATION = "minmax" as const;

// ─── Experiment conditions ───────────────────────────────────────

interface Condition {
  name: string;
  contextualized: boolean;
  asymmetric: boolean;
}

const CONDITIONS: Condition[] = [
  { name: "A: baseline",     contextualized: false, asymmetric: false },
  { name: "B: H2 (context)", contextualized: true,  asymmetric: false },
  { name: "C: H3 (asymm)",   contextualized: false, asymmetric: true },
  { name: "D: H2+H3 (both)", contextualized: true,  asymmetric: true },
];

// ─── Helpers ─────────────────────────────────────────────────────

function contextualizeAnchors(anchors: string[], questionText: string): string[] {
  return anchors.map(a => `Question: "${questionText}" Answer: ${a}`);
}

function delay(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const service = getEmbeddingService();
  if (!service) { console.error("VOYAGE_API_KEY not set"); process.exit(1); }

  const N = TEST_CASES.length;
  const domains = getDomains();

  console.log("=".repeat(80));
  console.log("ABLATION EXPERIMENT v2 — H2 × H3 (Expanded Test Set)");
  console.log("=".repeat(80));
  console.log(`Config: normalization=${NORMALIZATION}, τ=${TEMPERATURE}`);
  console.log(`Test cases: ${N}, Domains: ${domains.length} (${domains.join(", ")})`);
  console.log(`Conditions: ${CONDITIONS.length}\n`);

  // ─── Deduplicate questions ──────────────────────────────────

  const uniqueQuestions = new Map<string, { question: typeof TEST_CASES[0]["question"]; domain: string }>();
  for (const tc of TEST_CASES) {
    if (!uniqueQuestions.has(tc.question.id)) {
      uniqueQuestions.set(tc.question.id, { question: tc.question, domain: tc.domain });
    }
  }

  // ─── Pre-embed all anchor variants ─────────────────────────

  const anchorVecsMap = new Map<string, number[][]>();

  for (const [qId, { question, domain }] of uniqueQuestions) {
    const anchors = resolveAnchors(question);
    const plainTexts = anchors.anchors;
    const ctxTexts = contextualizeAnchors(anchors.anchors, question.text);

    console.log(`Embedding anchors for ${domain} (${anchors.semantic})...`);
    console.log(`  Plain[1]: "${plainTexts[0].slice(0, 60)}..."`);
    console.log(`  Ctx[1]:   "${ctxTexts[0].slice(0, 60)}..."`);

    await delay(22000);
    const plainVecs = await service.embed(plainTexts, "document");
    anchorVecsMap.set(`${qId}-plain`, plainVecs);

    await delay(22000);
    const ctxVecs = await service.embed(ctxTexts, "document");
    anchorVecsMap.set(`${qId}-ctx`, ctxVecs);
  }

  console.log(`\nEmbedded anchors for ${uniqueQuestions.size} question types (plain + contextualized).\n`);

  // ─── Pre-embed all response variants ───────────────────────

  console.log("Embedding test responses (doc + query)...");
  const responseVecsDoc = new Map<string, number[]>();
  const responseVecsQuery = new Map<string, number[]>();

  for (let i = 0; i < TEST_CASES.length; i++) {
    const tc = TEST_CASES[i];

    await delay(22000);
    const [docVec] = await service.embed([tc.text], "document");
    responseVecsDoc.set(tc.label, docVec);

    await delay(22000);
    const [queryVec] = await service.embed([tc.text], "query");
    responseVecsQuery.set(tc.label, queryVec);

    console.log(`  [${i + 1}/${N}] ${tc.label}: done`);
  }

  // ─── Test all conditions ───────────────────────────────────

  console.log("\n" + "=".repeat(80));
  console.log("RESULTS");
  console.log("=".repeat(80));

  interface ConditionResult {
    condition: Condition;
    exact: number;
    withinOne: number;
    mae: number;
    meanConfidence: number;
    details: Array<{ label: string; domain: string; difficulty: string; expected: number; predicted: number; confidence: number; sims: number[] }>;
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
      details.push({
        label: tc.label, domain: tc.domain, difficulty: tc.difficulty,
        expected: tc.expected, predicted: clamped, confidence, sims,
      });

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

  console.log("\n" + "─".repeat(90));
  console.log(
    `${"Condition".padEnd(22)} ${"Exact".padEnd(18)} ${"Within±1".padEnd(18)} ${"MAE".padEnd(10)} ${"Confidence".padEnd(12)}`
  );
  console.log("─".repeat(90));

  for (const r of results) {
    const exactPct = `${r.exact}/${N} (${Math.round(r.exact / N * 100)}%)`;
    const w1Pct = `${r.withinOne}/${N} (${Math.round(r.withinOne / N * 100)}%)`;
    console.log(
      `${r.condition.name.padEnd(22)} ${exactPct.padEnd(18)} ${w1Pct.padEnd(18)} ${r.mae.toFixed(2).padEnd(10)} ${r.meanConfidence.toFixed(2).padEnd(12)}`
    );
  }

  // ─── Best condition detail ─────────────────────────────────

  const best = results.reduce((a, b) => {
    if (b.exact !== a.exact) return b.exact > a.exact ? b : a;
    if (b.withinOne !== a.withinOne) return b.withinOne > a.withinOne ? b : a;
    return b.mae < a.mae ? b : a;
  });

  console.log("\n" + "=".repeat(80));
  console.log(`BEST: ${best.condition.name}`);
  console.log("=".repeat(80));

  // Per-domain breakdown for best condition
  console.log("\nPer-domain breakdown:");
  console.log(`  ${"Domain".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(16)} ${"±1".padEnd(16)} ${"MAE".padEnd(8)}`);
  console.log("  " + "─".repeat(60));

  for (const domain of domains) {
    const dCases = best.details.filter(d => d.domain === domain);
    const dExact = dCases.filter(d => d.predicted === d.expected).length;
    const dW1 = dCases.filter(d => Math.abs(d.predicted - d.expected) <= 1).length;
    const dMAE = dCases.reduce((s, d) => s + Math.abs(d.predicted - d.expected), 0) / dCases.length;
    console.log(
      `  ${domain.padEnd(18)} ${dCases.length.toString().padEnd(5)} ` +
      `${dExact}/${dCases.length} (${Math.round(dExact / dCases.length * 100)}%)`.padEnd(16) + " " +
      `${dW1}/${dCases.length} (${Math.round(dW1 / dCases.length * 100)}%)`.padEnd(16) + " " +
      dMAE.toFixed(2)
    );
  }

  // Per-difficulty breakdown
  console.log("\nPer-difficulty breakdown:");
  for (const diff of ["clear", "subtle", "edge"] as const) {
    const dCases = best.details.filter(d => d.difficulty === diff);
    const dExact = dCases.filter(d => d.predicted === d.expected).length;
    console.log(`  ${diff.padEnd(8)} ${dExact}/${dCases.length} exact (${Math.round(dExact / dCases.length * 100)}%)`);
  }

  // ─── Errors detail ─────────────────────────────────────────

  console.log("\n" + "=".repeat(80));
  console.log("ERRORS BY CONDITION");
  console.log("=".repeat(80));

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
      console.log(`  ${d.label.padEnd(8)} [${d.domain}/${d.difficulty}] exp=${d.expected} got=${d.predicted} (${sign}${err})`);
    }
  }

  // ─── Save results ──────────────────────────────────────────

  const outputPath = path.join(__dirname, "..", "data", `ablation-v2-${Date.now()}.json`);
  const output = {
    timestamp: new Date().toISOString(),
    version: "v2",
    config: { temperature: TEMPERATURE, normalization: NORMALIZATION },
    testCases: N,
    domains: domains,
    conditions: results.map(r => ({
      name: r.condition.name,
      contextualized: r.condition.contextualized,
      asymmetric: r.condition.asymmetric,
      exact: r.exact,
      exactPct: Math.round(r.exact / N * 100),
      withinOne: r.withinOne,
      withinOnePct: Math.round(r.withinOne / N * 100),
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
