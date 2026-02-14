/**
 * SSR Engine v2 — Cross-Validation
 *
 * Two validation strategies to test generalization:
 *
 * 1. Leave-one-domain-out CV (8 folds)
 *    - For each domain, calibrate τ on the other 7 domains, test on held-out domain
 *    - Tests whether optimal temperature generalizes across semantic families
 *
 * 2. Random 70/30 bootstrap (100 repeats)
 *    - Randomly split into 70% train / 30% test
 *    - For each split, find best τ on train, evaluate on test
 *    - Produces 95% confidence intervals for exact match, within±1, MAE
 *
 * Usage: npx tsx research/calibration/cross-validation.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService, type SimilarityNormalization } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import { TEST_CASES, getDomains, getTestCasesByDomain, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const NORMALIZATION: SimilarityNormalization = "minmax";
const CANDIDATE_TEMPS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5];
const BOOTSTRAP_REPEATS = 100;
const BOOTSTRAP_TRAIN_RATIO = 0.7;
const SEED = 42;

// ─── Simple seeded PRNG (Mulberry32) ─────────────────────────────

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const shuffled = [...arr];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// ─── Embedding Cache ─────────────────────────────────────────────

interface EmbeddedTestCase extends TestCase {
  responseVec: number[];
  anchorVecs: number[];
  scaleMin: number;
  scaleMax: number;
}

async function embedAllTestCases(): Promise<EmbeddedTestCase[]> {
  const service = getEmbeddingService();
  if (!service) throw new Error("VOYAGE_API_KEY not set");

  console.log("Pre-embedding all test cases and anchors...\n");

  // Embed anchors per question (deduplicated)
  const anchorCache = new Map<string, { vecs: number[][]; scaleMin: number; scaleMax: number }>();

  for (const tc of TEST_CASES) {
    const qId = tc.question.id;
    if (anchorCache.has(qId)) continue;

    const anchors = resolveAnchors(tc.question);
    console.log(`  Embedding anchors for ${tc.domain} (${anchors.semantic}, ${anchors.anchors.length} points)...`);
    await new Promise(r => setTimeout(r, 1500));
    const vecs = await service.embed(anchors.anchors, "document");
    anchorCache.set(qId, { vecs, scaleMin: anchors.scaleMin, scaleMax: anchors.scaleMax });
  }

  // Embed all responses (as query for asymmetric)
  const embedded: EmbeddedTestCase[] = [];

  for (const tc of TEST_CASES) {
    console.log(`  Embedding response: ${tc.label}...`);
    await new Promise(r => setTimeout(r, 1500));
    const [responseVec] = await service.embed([tc.text], "query");
    const { vecs, scaleMin, scaleMax } = anchorCache.get(tc.question.id)!;

    // Pre-compute raw similarities
    const rawSims = service.computeSimilarities(responseVec, vecs);

    embedded.push({
      ...tc,
      responseVec,
      anchorVecs: rawSims, // Store raw sims, not vecs (saves recomputation)
      scaleMin,
      scaleMax,
    });
  }

  console.log(`\nEmbedded ${embedded.length} test cases.\n`);
  return embedded;
}

// ─── Evaluation ──────────────────────────────────────────────────

interface EvalResult {
  exact: number;
  withinOne: number;
  mae: number;
  total: number;
  exactPct: number;
  withinOnePct: number;
}

function evaluate(cases: EmbeddedTestCase[], temperature: number): EvalResult {
  const service = getEmbeddingService()!;
  let exact = 0;
  let withinOne = 0;
  let totalError = 0;

  for (const tc of cases) {
    const dist = service.similaritiesToDistribution(tc.anchorVecs, temperature, NORMALIZATION);
    const rating = service.distributionToRating(dist, tc.scaleMin);
    const clamped = Math.min(Math.max(rating, tc.scaleMin), tc.scaleMax);

    const err = Math.abs(clamped - tc.expected);
    totalError += err;
    if (err === 0) exact++;
    if (err <= 1) withinOne++;
  }

  return {
    exact,
    withinOne,
    mae: totalError / cases.length,
    total: cases.length,
    exactPct: Math.round((exact / cases.length) * 100),
    withinOnePct: Math.round((withinOne / cases.length) * 100),
  };
}

function findBestTemp(trainCases: EmbeddedTestCase[]): { temp: number; result: EvalResult } {
  let bestTemp = CANDIDATE_TEMPS[0];
  let bestResult = evaluate(trainCases, bestTemp);

  for (const temp of CANDIDATE_TEMPS.slice(1)) {
    const result = evaluate(trainCases, temp);
    if (result.exact > bestResult.exact || (result.exact === bestResult.exact && result.mae < bestResult.mae)) {
      bestTemp = temp;
      bestResult = result;
    }
  }

  return { temp: bestTemp, result: bestResult };
}

// ─── Leave-One-Domain-Out CV ─────────────────────────────────────

interface LODOResult {
  heldOutDomain: string;
  calibratedTemp: number;
  trainResult: EvalResult;
  testResult: EvalResult;
}

function runLODO(embedded: EmbeddedTestCase[]): LODOResult[] {
  console.log("=" .repeat(75));
  console.log("LEAVE-ONE-DOMAIN-OUT CROSS-VALIDATION (8 folds)");
  console.log("=" .repeat(75));

  const domains = getDomains();
  const results: LODOResult[] = [];

  for (const holdout of domains) {
    const trainCases = embedded.filter(tc => tc.domain !== holdout);
    const testCases = embedded.filter(tc => tc.domain === holdout);

    const { temp, result: trainResult } = findBestTemp(trainCases);
    const testResult = evaluate(testCases, temp);

    results.push({
      heldOutDomain: holdout,
      calibratedTemp: temp,
      trainResult,
      testResult,
    });

    console.log(
      `  [${holdout.padEnd(16)}] τ=${temp.toFixed(2)} | ` +
      `Train: ${trainResult.exactPct}% exact | ` +
      `Test: ${testResult.exact}/${testResult.total} exact (${testResult.exactPct}%), ` +
      `${testResult.withinOne}/${testResult.total} ±1 (${testResult.withinOnePct}%), ` +
      `MAE=${testResult.mae.toFixed(2)}`
    );
  }

  // Aggregate
  const totalTestExact = results.reduce((s, r) => s + r.testResult.exact, 0);
  const totalTestWithinOne = results.reduce((s, r) => s + r.testResult.withinOne, 0);
  const totalTestCases = results.reduce((s, r) => s + r.testResult.total, 0);
  const totalTestMAE = results.reduce((s, r) => s + r.testResult.mae * r.testResult.total, 0) / totalTestCases;

  console.log("─".repeat(75));
  console.log(
    `  AGGREGATE:  Exact: ${totalTestExact}/${totalTestCases} (${Math.round(totalTestExact / totalTestCases * 100)}%) | ` +
    `Within ±1: ${totalTestWithinOne}/${totalTestCases} (${Math.round(totalTestWithinOne / totalTestCases * 100)}%) | ` +
    `MAE: ${totalTestMAE.toFixed(2)}`
  );

  // Temperature stability
  const temps = results.map(r => r.calibratedTemp);
  const meanTemp = temps.reduce((a, b) => a + b, 0) / temps.length;
  const stdTemp = Math.sqrt(temps.reduce((s, t) => s + (t - meanTemp) ** 2, 0) / temps.length);
  console.log(`  Temperature stability: mean=${meanTemp.toFixed(2)}, std=${stdTemp.toFixed(3)}`);
  console.log();

  return results;
}

// ─── Bootstrap 70/30 CV ──────────────────────────────────────────

interface BootstrapResult {
  exactPcts: number[];
  withinOnePcts: number[];
  maes: number[];
  temps: number[];
}

function runBootstrap(embedded: EmbeddedTestCase[]): BootstrapResult {
  console.log("=" .repeat(75));
  console.log(`RANDOM 70/30 BOOTSTRAP (${BOOTSTRAP_REPEATS} repeats, seed=${SEED})`);
  console.log("=" .repeat(75));

  const rng = mulberry32(SEED);
  const exactPcts: number[] = [];
  const withinOnePcts: number[] = [];
  const maes: number[] = [];
  const temps: number[] = [];

  for (let i = 0; i < BOOTSTRAP_REPEATS; i++) {
    const shuffled = shuffle(embedded, rng);
    const splitIdx = Math.floor(shuffled.length * BOOTSTRAP_TRAIN_RATIO);
    const train = shuffled.slice(0, splitIdx);
    const test = shuffled.slice(splitIdx);

    const { temp } = findBestTemp(train);
    const testResult = evaluate(test, temp);

    exactPcts.push(testResult.exactPct);
    withinOnePcts.push(testResult.withinOnePct);
    maes.push(testResult.mae);
    temps.push(temp);

    if ((i + 1) % 25 === 0) {
      console.log(`  Completed ${i + 1}/${BOOTSTRAP_REPEATS} iterations...`);
    }
  }

  // Compute 95% CI
  const sortedExact = [...exactPcts].sort((a, b) => a - b);
  const sortedW1 = [...withinOnePcts].sort((a, b) => a - b);
  const sortedMAE = [...maes].sort((a, b) => a - b);
  const sortedTemps = [...temps].sort((a, b) => a - b);

  const lo = Math.floor(BOOTSTRAP_REPEATS * 0.025);
  const hi = Math.floor(BOOTSTRAP_REPEATS * 0.975);
  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = (arr: number[]) => {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  };

  console.log();
  console.log("  Metric              Mean ± SD          95% CI");
  console.log("  " + "─".repeat(60));
  console.log(`  Exact match (%)     ${mean(exactPcts).toFixed(1)} ± ${std(exactPcts).toFixed(1)}          [${sortedExact[lo]}, ${sortedExact[hi]}]`);
  console.log(`  Within ±1 (%)       ${mean(withinOnePcts).toFixed(1)} ± ${std(withinOnePcts).toFixed(1)}          [${sortedW1[lo]}, ${sortedW1[hi]}]`);
  console.log(`  MAE                 ${mean(maes).toFixed(3)} ± ${std(maes).toFixed(3)}        [${sortedMAE[lo].toFixed(3)}, ${sortedMAE[hi].toFixed(3)}]`);
  console.log(`  Calibrated τ        ${mean(temps).toFixed(3)} ± ${std(temps).toFixed(3)}        [${sortedTemps[lo].toFixed(2)}, ${sortedTemps[hi].toFixed(2)}]`);
  console.log();

  return { exactPcts, withinOnePcts, maes, temps };
}

// ─── Global evaluation (no CV) ───────────────────────────────────

function runGlobalEval(embedded: EmbeddedTestCase[]): void {
  console.log("=" .repeat(75));
  console.log("GLOBAL EVALUATION (all 62 cases, all temperatures)");
  console.log("=" .repeat(75));

  console.log(`\n  ${"τ".padEnd(8)} ${"Exact".padEnd(16)} ${"Within ±1".padEnd(16)} ${"MAE".padEnd(10)}`);
  console.log("  " + "─".repeat(50));

  for (const temp of CANDIDATE_TEMPS) {
    const result = evaluate(embedded, temp);
    const marker = temp === 0.2 ? "  ← default" : "";
    console.log(
      `  ${temp.toFixed(2).padEnd(8)} ` +
      `${result.exact}/${result.total} (${result.exactPct}%)`.padEnd(16) + " " +
      `${result.withinOne}/${result.total} (${result.withinOnePct}%)`.padEnd(16) + " " +
      `${result.mae.toFixed(2)}${marker}`
    );
  }
  console.log();
}

// ─── Per-domain breakdown ────────────────────────────────────────

function runPerDomainBreakdown(embedded: EmbeddedTestCase[], temperature: number): void {
  console.log("=" .repeat(75));
  console.log(`PER-DOMAIN BREAKDOWN (τ=${temperature})`);
  console.log("=" .repeat(75));

  const domains = getDomains();
  console.log(`\n  ${"Domain".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"±1".padEnd(14)} ${"MAE".padEnd(8)}`);
  console.log("  " + "─".repeat(55));

  for (const domain of domains) {
    const cases = embedded.filter(tc => tc.domain === domain);
    const result = evaluate(cases, temperature);
    console.log(
      `  ${domain.padEnd(18)} ${cases.length.toString().padEnd(5)} ` +
      `${result.exact}/${result.total} (${result.exactPct}%)`.padEnd(14) + " " +
      `${result.withinOne}/${result.total} (${result.withinOnePct}%)`.padEnd(14) + " " +
      `${result.mae.toFixed(2)}`
    );
  }

  // By difficulty
  console.log();
  console.log(`  ${"Difficulty".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"±1".padEnd(14)} ${"MAE".padEnd(8)}`);
  console.log("  " + "─".repeat(55));

  for (const diff of ["clear", "subtle", "edge"] as const) {
    const cases = embedded.filter(tc => tc.difficulty === diff);
    const result = evaluate(cases, temperature);
    console.log(
      `  ${diff.padEnd(18)} ${cases.length.toString().padEnd(5)} ` +
      `${result.exact}/${result.total} (${result.exactPct}%)`.padEnd(14) + " " +
      `${result.withinOne}/${result.total} (${result.withinOnePct}%)`.padEnd(14) + " " +
      `${result.mae.toFixed(2)}`
    );
  }
  console.log();
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  console.log("=".repeat(75));
  console.log("SSR ENGINE v2 — CROSS-VALIDATION");
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Test cases: ${TEST_CASES.length}, Domains: ${getDomains().length}`);
  console.log(`Normalization: ${NORMALIZATION}, Candidate temps: [${CANDIDATE_TEMPS.join(", ")}]`);
  console.log("=".repeat(75));
  console.log();

  // Step 1: Embed everything (expensive, do once)
  const embedded = await embedAllTestCases();

  // Step 2: Global evaluation
  runGlobalEval(embedded);

  // Step 3: Per-domain breakdown at default temp
  runPerDomainBreakdown(embedded, 0.2);

  // Step 4: Leave-one-domain-out CV
  const lodoResults = runLODO(embedded);

  // Step 5: Bootstrap
  const bootstrapResults = runBootstrap(embedded);

  // Step 6: Save all results
  const outputPath = path.join(__dirname, "..", "data", `cross-validation-${Date.now()}.json`);
  const output = {
    timestamp: new Date().toISOString(),
    config: {
      normalization: NORMALIZATION,
      candidateTemps: CANDIDATE_TEMPS,
      bootstrapRepeats: BOOTSTRAP_REPEATS,
      bootstrapTrainRatio: BOOTSTRAP_TRAIN_RATIO,
      seed: SEED,
    },
    testSetSize: TEST_CASES.length,
    domains: getDomains(),
    lodo: lodoResults.map(r => ({
      heldOutDomain: r.heldOutDomain,
      calibratedTemp: r.calibratedTemp,
      trainExactPct: r.trainResult.exactPct,
      testExact: r.testResult.exact,
      testTotal: r.testResult.total,
      testExactPct: r.testResult.exactPct,
      testWithinOnePct: r.testResult.withinOnePct,
      testMAE: r.testResult.mae,
    })),
    bootstrap: {
      exactPct: {
        mean: bootstrapResults.exactPcts.reduce((a, b) => a + b, 0) / BOOTSTRAP_REPEATS,
        ci95: [
          [...bootstrapResults.exactPcts].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.025)],
          [...bootstrapResults.exactPcts].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.975)],
        ],
      },
      withinOnePct: {
        mean: bootstrapResults.withinOnePcts.reduce((a, b) => a + b, 0) / BOOTSTRAP_REPEATS,
        ci95: [
          [...bootstrapResults.withinOnePcts].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.025)],
          [...bootstrapResults.withinOnePcts].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.975)],
        ],
      },
      mae: {
        mean: bootstrapResults.maes.reduce((a, b) => a + b, 0) / BOOTSTRAP_REPEATS,
        ci95: [
          [...bootstrapResults.maes].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.025)],
          [...bootstrapResults.maes].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.975)],
        ],
      },
      temperature: {
        mean: bootstrapResults.temps.reduce((a, b) => a + b, 0) / BOOTSTRAP_REPEATS,
        ci95: [
          [...bootstrapResults.temps].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.025)],
          [...bootstrapResults.temps].sort((a, b) => a - b)[Math.floor(BOOTSTRAP_REPEATS * 0.975)],
        ],
      },
    },
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`Results saved to: ${outputPath}`);
}

main().catch(console.error);
