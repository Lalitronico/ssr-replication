/**
 * Multi-Model Embedding Comparison for SSR Engine v2
 *
 * Compares SSR accuracy across embedding providers on the 69-case benchmark:
 *   1. Voyage 3.5-lite (asymmetric: anchors=document, response=query)
 *   2. Voyage 3.5-lite (symmetric: anchors=document, response=document)
 *   3. OpenAI text-embedding-3-small (symmetric: no input_type support)
 *
 * Tests multiple softmax temperatures and produces per-domain breakdowns.
 * All embeddings are computed upfront (batch) to minimize API calls.
 *
 * Usage: npx tsx research/calibration/multi-model-embeddings.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { resolveAnchors, type ScaleAnchorSet } from "../../src/lib/scale-anchors";
import { TEST_CASES, getDomains, getTestCasesByDomain, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const TEMPERATURES = [0.05, 0.10, 0.15, 0.20, 0.25];
const NORMALIZATION = "minmax" as const;

interface ProviderConfig {
  name: string;
  shortName: string;
  mode: "asymmetric" | "symmetric";
}

const PROVIDERS: ProviderConfig[] = [
  { name: "Voyage 3.5-lite", shortName: "voyage-asymm", mode: "asymmetric" },
  { name: "Voyage 3.5-lite", shortName: "voyage-symm", mode: "symmetric" },
  { name: "OpenAI emb-3-sm", shortName: "openai-symm", mode: "symmetric" },
];

// ─── Math Functions (local, no class dependency) ─────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return dot / denom;
}

function normalizeSimilarities(sims: number[], method: "minmax"): number[] {
  if (sims.length <= 1) return [...sims];

  const min = Math.min(...sims);
  const max = Math.max(...sims);
  const range = max - min;
  if (range === 0) return sims.map(() => 1 / sims.length);
  return sims.map((s) => (s - min) / range);
}

function similaritiesToDistribution(sims: number[], temperature: number): number[] {
  const normalized = normalizeSimilarities(sims, "minmax");

  // Temperature scaling
  const scaled = normalized.map((s) => s / temperature);

  // Softmax with numerical stability (subtract max)
  const maxVal = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - maxVal));
  const sumExps = exps.reduce((a, b) => a + b, 0);

  return exps.map((e) => e / sumExps);
}

function distributionToRating(dist: number[], scaleMin: number): number {
  let weightedSum = 0;
  for (let i = 0; i < dist.length; i++) {
    weightedSum += dist[i] * (scaleMin + i);
  }
  return Math.round(weightedSum);
}

// ─── API Helpers ─────────────────────────────────────────────────

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

interface EmbeddingResponse {
  data: Array<{ embedding: number[]; index: number }>;
  usage: { total_tokens: number };
}

/**
 * Embed texts via Voyage AI API.
 * Rate limit: 3 RPM free tier -> 500ms between calls, 22s backoff on 429.
 */
async function embedVoyage(
  texts: string[],
  inputType: "document" | "query",
  apiKey: string,
  maxRetries: number = 3
): Promise<number[][]> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fetch("https://api.voyageai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "voyage-3.5-lite",
        input: texts,
        input_type: inputType,
      }),
    });

    if (response.ok) {
      const data = (await response.json()) as EmbeddingResponse;
      return data.data.map((d) => d.embedding);
    }

    if (response.status === 429 && attempt < maxRetries) {
      const waitMs = 22000 * Math.pow(2, attempt);
      console.log(`  Voyage 429 rate limited, waiting ${Math.round(waitMs / 1000)}s (attempt ${attempt + 1}/${maxRetries})...`);
      await delay(waitMs);
      continue;
    }

    const errorText = await response.text();
    throw new Error(`Voyage API error ${response.status}: ${errorText}`);
  }
  throw new Error("Voyage API: max retries exceeded");
}

/**
 * Embed texts via OpenAI API.
 * No input_type parameter (always symmetric). Returns 1536-dim vectors.
 */
async function embedOpenAI(
  texts: string[],
  apiKey: string,
  maxRetries: number = 3
): Promise<number[][]> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "text-embedding-3-small",
        input: texts,
      }),
    });

    if (response.ok) {
      const data = (await response.json()) as EmbeddingResponse;
      // Sort by index to ensure order matches input
      const sorted = data.data.sort((a, b) => a.index - b.index);
      return sorted.map((d) => d.embedding);
    }

    if (response.status === 429 && attempt < maxRetries) {
      const waitMs = 5000 * Math.pow(2, attempt);
      console.log(`  OpenAI 429 rate limited, waiting ${Math.round(waitMs / 1000)}s (attempt ${attempt + 1}/${maxRetries})...`);
      await delay(waitMs);
      continue;
    }

    const errorText = await response.text();
    throw new Error(`OpenAI API error ${response.status}: ${errorText}`);
  }
  throw new Error("OpenAI API: max retries exceeded");
}

// ─── Embedded Test Data ──────────────────────────────────────────

interface AnchorData {
  anchorSet: ScaleAnchorSet;
  texts: string[];
}

interface EmbeddedAnchorsByProvider {
  /** Map<questionId, anchorVecs[][]> */
  anchorVecs: Map<string, number[][]>;
}

interface EmbeddedResponsesByProvider {
  /** Map<testCaseLabel, responseVec[]> */
  responseVecs: Map<string, number[]>;
}

interface ProviderEmbeddings {
  config: ProviderConfig;
  anchors: Map<string, number[][]>;
  responses: Map<string, number[]>;
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

interface CasePrediction {
  label: string;
  domain: string;
  difficulty: string;
  expected: number;
  predicted: number;
  error: number;
  rawSims: number[];
}

function evaluateProvider(
  provider: ProviderEmbeddings,
  anchorDataMap: Map<string, AnchorData>,
  temperature: number
): { result: EvalResult; predictions: CasePrediction[] } {
  let exact = 0;
  let withinOne = 0;
  let totalError = 0;
  const predictions: CasePrediction[] = [];

  for (const tc of TEST_CASES) {
    const anchorVecs = provider.anchors.get(tc.question.id)!;
    const responseVec = provider.responses.get(tc.label)!;
    const anchorData = anchorDataMap.get(tc.question.id)!;

    // Compute cosine similarities
    const sims = anchorVecs.map((av) => cosineSimilarity(responseVec, av));

    // Convert to distribution and rating
    const dist = similaritiesToDistribution(sims, temperature);
    const rating = distributionToRating(dist, anchorData.anchorSet.scaleMin);
    const clamped = Math.min(Math.max(rating, anchorData.anchorSet.scaleMin), anchorData.anchorSet.scaleMax);

    const err = Math.abs(clamped - tc.expected);
    totalError += err;
    if (err === 0) exact++;
    if (err <= 1) withinOne++;

    predictions.push({
      label: tc.label,
      domain: tc.domain,
      difficulty: tc.difficulty,
      expected: tc.expected,
      predicted: clamped,
      error: err,
      rawSims: sims,
    });
  }

  return {
    result: {
      exact,
      withinOne,
      mae: totalError / TEST_CASES.length,
      total: TEST_CASES.length,
      exactPct: Math.round((exact / TEST_CASES.length) * 100),
      withinOnePct: Math.round((withinOne / TEST_CASES.length) * 100),
    },
    predictions,
  };
}

// ─── Embedding Phase ─────────────────────────────────────────────

async function embedAllAnchorsAndResponses(): Promise<{
  providerEmbeddings: ProviderEmbeddings[];
  anchorDataMap: Map<string, AnchorData>;
}> {
  const voyageKey = process.env.VOYAGE_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  if (!voyageKey) throw new Error("VOYAGE_API_KEY not set in .env.local");
  if (!openaiKey) throw new Error("OPENAI_API_KEY not set in .env.local");

  // ─── Step 1: Resolve anchors for each unique question ──────

  const anchorDataMap = new Map<string, AnchorData>();

  for (const tc of TEST_CASES) {
    if (anchorDataMap.has(tc.question.id)) continue;
    const anchorSet = resolveAnchors(tc.question);
    anchorDataMap.set(tc.question.id, { anchorSet, texts: anchorSet.anchors });
  }

  const uniqueQuestionIds = [...anchorDataMap.keys()];
  console.log(`Resolved anchors for ${uniqueQuestionIds.length} unique questions.\n`);

  // ─── Step 2: Collect all unique texts for batch embedding ──

  // Anchor texts (per question)
  const allAnchorTexts: string[] = [];
  const anchorOffsets: Map<string, { start: number; count: number }> = new Map();
  for (const qId of uniqueQuestionIds) {
    const data = anchorDataMap.get(qId)!;
    anchorOffsets.set(qId, { start: allAnchorTexts.length, count: data.texts.length });
    allAnchorTexts.push(...data.texts);
  }

  // Response texts (all 69 test cases)
  const allResponseTexts = TEST_CASES.map((tc) => tc.text);

  console.log(`Total anchor texts to embed: ${allAnchorTexts.length}`);
  console.log(`Total response texts to embed: ${allResponseTexts.length}`);
  console.log();

  // ─── Step 3: Embed with Voyage (3 calls: anchors-doc, responses-doc, responses-query) ──

  console.log("── Voyage AI Embeddings ──────────────────────────────────");

  console.log("  Embedding anchors (document mode)...");
  const voyageAnchorVecs = await embedVoyage(allAnchorTexts, "document", voyageKey);
  await delay(500);

  console.log("  Embedding responses (document mode, symmetric)...");
  const voyageResponseVecsDoc = await embedVoyage(allResponseTexts, "document", voyageKey);
  await delay(500);

  console.log("  Embedding responses (query mode, asymmetric)...");
  const voyageResponseVecsQuery = await embedVoyage(allResponseTexts, "query", voyageKey);
  await delay(500);

  console.log(`  Done. Voyage dims: ${voyageAnchorVecs[0].length}\n`);

  // ─── Step 4: Embed with OpenAI (2 calls: anchors, responses) ──

  console.log("── OpenAI Embeddings ────────────────────────────────────");

  console.log("  Embedding anchors...");
  const openaiAnchorVecs = await embedOpenAI(allAnchorTexts, openaiKey);
  await delay(200);

  console.log("  Embedding responses...");
  const openaiResponseVecs = await embedOpenAI(allResponseTexts, openaiKey);
  await delay(200);

  console.log(`  Done. OpenAI dims: ${openaiAnchorVecs[0].length}\n`);

  // ─── Step 5: Organize into per-provider maps ──────────────

  function buildAnchorMap(vecs: number[][]): Map<string, number[][]> {
    const map = new Map<string, number[][]>();
    for (const qId of uniqueQuestionIds) {
      const { start, count } = anchorOffsets.get(qId)!;
      map.set(qId, vecs.slice(start, start + count));
    }
    return map;
  }

  function buildResponseMap(vecs: number[][]): Map<string, number[]> {
    const map = new Map<string, number[]>();
    for (let i = 0; i < TEST_CASES.length; i++) {
      map.set(TEST_CASES[i].label, vecs[i]);
    }
    return map;
  }

  const providerEmbeddings: ProviderEmbeddings[] = [
    {
      config: PROVIDERS[0], // Voyage asymmetric
      anchors: buildAnchorMap(voyageAnchorVecs),
      responses: buildResponseMap(voyageResponseVecsQuery),
    },
    {
      config: PROVIDERS[1], // Voyage symmetric
      anchors: buildAnchorMap(voyageAnchorVecs),
      responses: buildResponseMap(voyageResponseVecsDoc),
    },
    {
      config: PROVIDERS[2], // OpenAI symmetric
      anchors: buildAnchorMap(openaiAnchorVecs),
      responses: buildResponseMap(openaiResponseVecs),
    },
  ];

  return { providerEmbeddings, anchorDataMap };
}

// ─── Reporting ───────────────────────────────────────────────────

interface GridResult {
  provider: ProviderConfig;
  temperature: number;
  result: EvalResult;
  predictions: CasePrediction[];
}

function printComparisonTable(grid: GridResult[]): void {
  console.log("=".repeat(85));
  console.log("COMPARISON TABLE — All Providers x Temperatures");
  console.log("=".repeat(85));
  console.log();

  const header =
    `${"Provider".padEnd(20)}| ${"Mode".padEnd(12)}| ${"t".padEnd(6)}| ${"Exact".padEnd(14)}| ${"+-1".padEnd(14)}| ${"MAE".padEnd(6)}`;
  console.log(header);
  console.log("-".repeat(85));

  for (const g of grid) {
    const exact = `${g.result.exact}/${g.result.total} (${g.result.exactPct}%)`;
    const w1 = `${g.result.withinOne}/${g.result.total} (${g.result.withinOnePct}%)`;
    console.log(
      `${g.provider.name.padEnd(20)}| ${g.provider.mode.padEnd(12)}| ${g.temperature.toFixed(2).padEnd(6)}| ${exact.padEnd(14)}| ${w1.padEnd(14)}| ${g.result.mae.toFixed(2).padEnd(6)}`
    );
  }
  console.log();
}

function printPerDomainBreakdown(
  grid: GridResult[],
  anchorDataMap: Map<string, AnchorData>
): void {
  console.log("=".repeat(85));
  console.log("PER-DOMAIN BREAKDOWN — Best Temperature per Provider");
  console.log("=".repeat(85));
  console.log();

  // Find best temperature for each provider (by exact match, then MAE as tiebreak)
  const providerBest = new Map<string, GridResult>();
  for (const g of grid) {
    const key = g.provider.shortName;
    const current = providerBest.get(key);
    if (
      !current ||
      g.result.exact > current.result.exact ||
      (g.result.exact === current.result.exact && g.result.mae < current.result.mae)
    ) {
      providerBest.set(key, g);
    }
  }

  const domains = getDomains();

  for (const [provKey, best] of providerBest) {
    console.log(`── ${best.provider.name} (${best.provider.mode}, t=${best.temperature.toFixed(2)}) ──`);
    console.log(
      `   Overall: ${best.result.exact}/${best.result.total} exact (${best.result.exactPct}%), ` +
      `${best.result.withinOne}/${best.result.total} +-1 (${best.result.withinOnePct}%), ` +
      `MAE=${best.result.mae.toFixed(2)}`
    );
    console.log();

    console.log(`   ${"Domain".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"+-1".padEnd(14)} ${"MAE".padEnd(8)}`);
    console.log("   " + "-".repeat(60));

    for (const domain of domains) {
      const domainPreds = best.predictions.filter((p) => p.domain === domain);
      const n = domainPreds.length;
      const exact = domainPreds.filter((p) => p.error === 0).length;
      const w1 = domainPreds.filter((p) => p.error <= 1).length;
      const mae = domainPreds.reduce((s, p) => s + p.error, 0) / n;

      console.log(
        `   ${domain.padEnd(18)} ${n.toString().padEnd(5)} ` +
        `${exact}/${n} (${Math.round((exact / n) * 100)}%)`.padEnd(14) + " " +
        `${w1}/${n} (${Math.round((w1 / n) * 100)}%)`.padEnd(14) + " " +
        mae.toFixed(2)
      );
    }

    // By difficulty
    console.log();
    console.log(`   ${"Difficulty".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"+-1".padEnd(14)} ${"MAE".padEnd(8)}`);
    console.log("   " + "-".repeat(60));

    for (const diff of ["clear", "subtle", "edge"] as const) {
      const diffPreds = best.predictions.filter((p) => p.difficulty === diff);
      const n = diffPreds.length;
      const exact = diffPreds.filter((p) => p.error === 0).length;
      const w1 = diffPreds.filter((p) => p.error <= 1).length;
      const mae = diffPreds.reduce((s, p) => s + p.error, 0) / n;

      console.log(
        `   ${diff.padEnd(18)} ${n.toString().padEnd(5)} ` +
        `${exact}/${n} (${Math.round((exact / n) * 100)}%)`.padEnd(14) + " " +
        `${w1}/${n} (${Math.round((w1 / n) * 100)}%)`.padEnd(14) + " " +
        mae.toFixed(2)
      );
    }

    console.log();
  }
}

function printErrorComparison(grid: GridResult[]): void {
  // Find best temp per provider
  const providerBest = new Map<string, GridResult>();
  for (const g of grid) {
    const key = g.provider.shortName;
    const current = providerBest.get(key);
    if (
      !current ||
      g.result.exact > current.result.exact ||
      (g.result.exact === current.result.exact && g.result.mae < current.result.mae)
    ) {
      providerBest.set(key, g);
    }
  }

  console.log("=".repeat(85));
  console.log("ERROR COMPARISON — Cases where providers disagree");
  console.log("=".repeat(85));
  console.log();

  const bestResults = [...providerBest.values()];

  // Find cases where at least one provider got it wrong
  const allLabels = TEST_CASES.map((tc) => tc.label);
  let disagreements = 0;

  for (const label of allLabels) {
    const preds = bestResults.map((br) => {
      const p = br.predictions.find((pred) => pred.label === label)!;
      return { provider: br.provider.shortName, predicted: p.predicted, error: p.error };
    });

    const hasError = preds.some((p) => p.error > 0);
    const allSame = preds.every((p) => p.predicted === preds[0].predicted);

    if (hasError && !allSame) {
      disagreements++;
      const tc = TEST_CASES.find((t) => t.label === label)!;
      const predStrs = preds.map((p) => `${p.provider}=${p.predicted}${p.error === 0 ? " OK" : ""}`).join(", ");
      console.log(`  ${tc.label.padEnd(8)} [${tc.domain}/${tc.difficulty}] expected=${tc.expected} | ${predStrs}`);
    }
  }

  if (disagreements === 0) {
    console.log("  No disagreements found (all providers agree on every case).");
  } else {
    console.log(`\n  Total disagreements: ${disagreements}`);
  }
  console.log();
}

// ─── Save Results ────────────────────────────────────────────────

function saveResults(
  grid: GridResult[],
  anchorDataMap: Map<string, AnchorData>
): string {
  const timestamp = Date.now();
  const outputPath = path.join(__dirname, "..", "data", `multi-model-embeddings-${timestamp}.json`);

  const output = {
    timestamp: new Date().toISOString(),
    experiment: "multi-model-embedding-comparison",
    config: {
      temperatures: TEMPERATURES,
      normalization: NORMALIZATION,
      providers: PROVIDERS.map((p) => ({ name: p.name, shortName: p.shortName, mode: p.mode })),
      testCases: TEST_CASES.length,
      domains: getDomains(),
    },
    results: grid.map((g) => ({
      provider: g.provider.shortName,
      providerName: g.provider.name,
      mode: g.provider.mode,
      temperature: g.temperature,
      exact: g.result.exact,
      exactPct: g.result.exactPct,
      withinOne: g.result.withinOne,
      withinOnePct: g.result.withinOnePct,
      mae: g.result.mae,
      total: g.result.total,
      predictions: g.predictions.map((p) => ({
        label: p.label,
        domain: p.domain,
        difficulty: p.difficulty,
        expected: p.expected,
        predicted: p.predicted,
        error: p.error,
        rawSims: p.rawSims,
      })),
    })),
  };

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  return outputPath;
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  console.log("=".repeat(85));
  console.log("SSR ENGINE v2 — MULTI-MODEL EMBEDDING COMPARISON");
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Test cases: ${TEST_CASES.length}, Domains: ${getDomains().length}`);
  console.log(`Normalization: ${NORMALIZATION}`);
  console.log(`Temperatures: [${TEMPERATURES.join(", ")}]`);
  console.log(`Providers: ${PROVIDERS.map((p) => `${p.name} (${p.mode})`).join(", ")}`);
  console.log("=".repeat(85));
  console.log();

  // Step 1: Embed everything (expensive, do once)
  const { providerEmbeddings, anchorDataMap } = await embedAllAnchorsAndResponses();

  // Step 2: Evaluate all provider x temperature combinations
  console.log("Evaluating all configurations...\n");

  const grid: GridResult[] = [];

  for (const pe of providerEmbeddings) {
    for (const temp of TEMPERATURES) {
      const { result, predictions } = evaluateProvider(pe, anchorDataMap, temp);
      grid.push({ provider: pe.config, temperature: temp, result, predictions });
    }
  }

  // Step 3: Print comparison table
  printComparisonTable(grid);

  // Step 4: Per-domain breakdown for best temperature per provider
  printPerDomainBreakdown(grid, anchorDataMap);

  // Step 5: Error comparison
  printErrorComparison(grid);

  // Step 6: Save full results
  const outputPath = saveResults(grid, anchorDataMap);
  console.log(`Results saved to: ${outputPath}`);
}

main().catch(console.error);
