/**
 * LLM Baseline Experiment — Direct LLM Rating (Circular Approach)
 *
 * Demonstrates the circularity problem: the same model that generates text
 * also rates it. Uses Claude Haiku 4.5 with the exact prompt from
 * ssr-engine.ts:mapToLikertViaLLM to rate all 62 ground-truth test cases.
 *
 * This provides the baseline comparison for the SSR paper:
 *   - LLM rates text it could have generated (maximum circularity)
 *   - Compare accuracy against SSR embedding-based approach
 *
 * Usage:
 *   npx tsx research/calibration/llm-baseline-experiment.ts as-deployed
 *   npx tsx research/calibration/llm-baseline-experiment.ts domain-aware
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import Anthropic from "@anthropic-ai/sdk";
import { TEST_CASES, getDomains, getTestCasesByDomain, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const MODEL = "claude-haiku-4-5-20251001";
const TEMPERATURE = 0; // Deterministic output for reproducibility
const MAX_RETRIES = 3;
const DELAY_BETWEEN_CALLS_MS = 500; // Rate limiting

type Variant = "as-deployed" | "domain-aware";
const VARIANT: Variant = (process.argv[2] || "as-deployed") as Variant;
if (!["as-deployed", "domain-aware"].includes(VARIANT)) {
  console.error(`ERROR: Invalid variant "${process.argv[2]}". Use "as-deployed" or "domain-aware".`);
  process.exit(1);
}

// ─── Types ───────────────────────────────────────────────────────

interface LLMRatingResult {
  label: string;
  domain: string;
  difficulty: string;
  expected: number;
  predicted: number;
  confidence: number;
  error: number;
  exact: boolean;
  withinOne: boolean;
}

interface ExperimentOutput {
  timestamp: string;
  model: string;
  temperature: number;
  variant: Variant;
  testSetSize: number;
  domains: string[];
  prompt: string;
  global: {
    exact: number;
    exactPct: number;
    withinOne: number;
    withinOnePct: number;
    mae: number;
    total: number;
  };
  perDomain: Record<string, {
    exact: number;
    exactPct: number;
    withinOne: number;
    withinOnePct: number;
    mae: number;
    total: number;
  }>;
  perDifficulty: Record<string, {
    exact: number;
    exactPct: number;
    withinOne: number;
    withinOnePct: number;
    mae: number;
    total: number;
  }>;
  results: LLMRatingResult[];
}

// ─── LLM Rating (replicates ssr-engine.ts:mapToLikertViaLLM) ────

function buildPrompt(tc: TestCase, variant: Variant): string {
  const scaleMin = tc.question.scaleMin ?? 1;
  const scaleMax = tc.question.scaleMax ?? 5;

  // Scale description depends on variant
  let scaleDescription: string;
  if (tc.question.type === "nps") {
    scaleDescription = "This is an NPS (Net Promoter Score) question. 0-6 = Detractor, 7-8 = Passive, 9-10 = Promoter.";
  } else if (variant === "domain-aware") {
    // Use the question's actual scale anchors
    const lowLabel = tc.question.scaleAnchors?.low || "Strongly Disagree/Very Negative";
    const highLabel = tc.question.scaleAnchors?.high || "Strongly Agree/Very Positive";
    scaleDescription = `This is a Likert scale where ${scaleMin} = ${lowLabel} and ${scaleMax} = ${highLabel}.`;
  } else {
    // as-deployed: generic labels (replicates production ssr-engine.ts before fix)
    scaleDescription = `This is a Likert scale where ${scaleMin} = Strongly Disagree/Very Negative and ${scaleMax} = Strongly Agree/Very Positive.`;
  }

  return `Analyze the following survey response and determine the appropriate rating.

Question: "${tc.question.text}"

Response: "${tc.text}"

Scale: ${scaleMin} to ${scaleMax}
${scaleDescription}

IMPORTANT: If the response explicitly mentions a specific number (e.g. "I'd give it a 7", "maybe a 4"), use that number as the rating. Only infer from sentiment if no explicit number is stated.

Based on the sentiment and content of the response, provide:
1. A rating from ${scaleMin} to ${scaleMax}
2. Your confidence level (0.0 to 1.0)

Respond ONLY in this exact JSON format:
{"rating": <number>, "confidence": <number>}`;
}

async function rateSingleCase(
  anthropic: Anthropic,
  tc: TestCase
): Promise<LLMRatingResult> {
  const prompt = buildPrompt(tc, VARIANT);
  const scaleMin = tc.question.scaleMin ?? 1;
  const scaleMax = tc.question.scaleMax ?? 5;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await anthropic.messages.create({
        model: MODEL,
        max_tokens: 100,
        temperature: TEMPERATURE,
        messages: [{ role: "user", content: prompt }],
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const rating = Math.min(Math.max(Math.round(parsed.rating), scaleMin), scaleMax);
        const confidence = Math.min(Math.max(parsed.confidence || 0.7, 0), 1);
        const error = Math.abs(rating - tc.expected);

        return {
          label: tc.label,
          domain: tc.domain,
          difficulty: tc.difficulty,
          expected: tc.expected,
          predicted: rating,
          confidence,
          error,
          exact: error === 0,
          withinOne: error <= 1,
        };
      }
    } catch (err) {
      console.error(`  Attempt ${attempt + 1} failed for ${tc.label}:`, err);
      await delay(2000 * (attempt + 1));
    }
  }

  // Fallback: midpoint (same as ssr-engine.ts fallback)
  const midRating = Math.round((scaleMin + scaleMax) / 2);
  const error = Math.abs(midRating - tc.expected);
  return {
    label: tc.label,
    domain: tc.domain,
    difficulty: tc.difficulty,
    expected: tc.expected,
    predicted: midRating,
    confidence: 0.5,
    error,
    exact: error === 0,
    withinOne: error <= 1,
  };
}

// ─── Helpers ─────────────────────────────────────────────────────

function delay(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

function computeMetrics(results: LLMRatingResult[]) {
  const total = results.length;
  const exact = results.filter(r => r.exact).length;
  const withinOne = results.filter(r => r.withinOne).length;
  const mae = results.reduce((sum, r) => sum + r.error, 0) / total;

  return {
    exact,
    exactPct: Math.round((exact / total) * 100),
    withinOne,
    withinOnePct: Math.round((withinOne / total) * 100),
    mae: parseFloat(mae.toFixed(2)),
    total,
  };
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ERROR: ANTHROPIC_API_KEY not set in .env.local");
    process.exit(1);
  }

  const anthropic = new Anthropic({ apiKey });

  console.log("=".repeat(75));
  console.log("LLM BASELINE EXPERIMENT — Direct LLM Rating (Circular Approach)");
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Model: ${MODEL}`);
  console.log(`Temperature: ${TEMPERATURE}`);
  console.log(`Variant: ${VARIANT}`);
  console.log(`Test cases: ${TEST_CASES.length}, Domains: ${getDomains().length}`);
  console.log("=".repeat(75));
  console.log();

  const samplePrompt = buildPrompt(TEST_CASES[0], VARIANT);
  console.log("Sample prompt (first test case):");
  console.log("─".repeat(60));
  console.log(samplePrompt);
  console.log("─".repeat(60));
  console.log();

  // Run all test cases
  const results: LLMRatingResult[] = [];

  for (let i = 0; i < TEST_CASES.length; i++) {
    const tc = TEST_CASES[i];
    console.log(`  [${i + 1}/${TEST_CASES.length}] Rating ${tc.label} (expected: ${tc.expected})...`);

    const result = await rateSingleCase(anthropic, tc);
    results.push(result);

    const marker = result.exact ? "EXACT" : result.withinOne ? "±1" : `MISS (off by ${result.error})`;
    console.log(`    → predicted: ${result.predicted}, confidence: ${result.confidence.toFixed(2)} [${marker}]`);

    // Rate limiting
    if (i < TEST_CASES.length - 1) {
      await delay(DELAY_BETWEEN_CALLS_MS);
    }
  }

  console.log();

  // ─── Global metrics ─────────────────────────────────────────
  const global = computeMetrics(results);
  console.log("=".repeat(75));
  console.log("GLOBAL RESULTS");
  console.log("=".repeat(75));
  console.log(`  Exact match: ${global.exact}/${global.total} (${global.exactPct}%)`);
  console.log(`  Within ±1:   ${global.withinOne}/${global.total} (${global.withinOnePct}%)`);
  console.log(`  MAE:         ${global.mae}`);
  console.log();

  // ─── Per-domain breakdown ───────────────────────────────────
  const perDomain: Record<string, ReturnType<typeof computeMetrics>> = {};
  console.log("=".repeat(75));
  console.log("PER-DOMAIN BREAKDOWN");
  console.log("=".repeat(75));
  console.log(`  ${"Domain".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"±1".padEnd(14)} ${"MAE".padEnd(8)}`);
  console.log("  " + "─".repeat(55));

  for (const domain of getDomains()) {
    const domainResults = results.filter(r => r.domain === domain);
    const metrics = computeMetrics(domainResults);
    perDomain[domain] = metrics;
    console.log(
      `  ${domain.padEnd(18)} ${metrics.total.toString().padEnd(5)} ` +
      `${metrics.exact}/${metrics.total} (${metrics.exactPct}%)`.padEnd(14) + " " +
      `${metrics.withinOne}/${metrics.total} (${metrics.withinOnePct}%)`.padEnd(14) + " " +
      `${metrics.mae}`
    );
  }
  console.log();

  // ─── Per-difficulty breakdown ───────────────────────────────
  const perDifficulty: Record<string, ReturnType<typeof computeMetrics>> = {};
  console.log("=".repeat(75));
  console.log("PER-DIFFICULTY BREAKDOWN");
  console.log("=".repeat(75));
  console.log(`  ${"Difficulty".padEnd(18)} ${"N".padEnd(5)} ${"Exact".padEnd(14)} ${"±1".padEnd(14)} ${"MAE".padEnd(8)}`);
  console.log("  " + "─".repeat(55));

  for (const diff of ["clear", "subtle", "edge"] as const) {
    const diffResults = results.filter(r => r.difficulty === diff);
    const metrics = computeMetrics(diffResults);
    perDifficulty[diff] = metrics;
    console.log(
      `  ${diff.padEnd(18)} ${metrics.total.toString().padEnd(5)} ` +
      `${metrics.exact}/${metrics.total} (${metrics.exactPct}%)`.padEnd(14) + " " +
      `${metrics.withinOne}/${metrics.total} (${metrics.withinOnePct}%)`.padEnd(14) + " " +
      `${metrics.mae}`
    );
  }
  console.log();

  // ─── Error analysis ─────────────────────────────────────────
  const misses = results.filter(r => !r.exact);
  if (misses.length > 0) {
    console.log("=".repeat(75));
    console.log(`MISSES (${misses.length} cases)`);
    console.log("=".repeat(75));
    for (const m of misses) {
      const dir = m.predicted > m.expected ? "↑" : "↓";
      console.log(`  ${m.label.padEnd(8)} expected=${m.expected} predicted=${m.predicted} (${dir}${m.error}) [${m.difficulty}]`);
    }
    console.log();
  }

  // ─── Save results ───────────────────────────────────────────
  const output: ExperimentOutput = {
    timestamp: new Date().toISOString(),
    model: MODEL,
    temperature: TEMPERATURE,
    variant: VARIANT,
    testSetSize: TEST_CASES.length,
    domains: getDomains(),
    prompt: VARIANT === "as-deployed"
      ? "Exact mapToLikertViaLLM prompt from ssr-engine.ts (generic labels)"
      : "Domain-aware prompt with question-specific scale anchors",
    global,
    perDomain,
    perDifficulty,
    results,
  };

  const outputPath = path.join(__dirname, "..", "data", `llm-baseline-${VARIANT}-${Date.now()}.json`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`Results saved to: ${outputPath}`);

  // ─── Summary for paper ──────────────────────────────────────
  console.log();
  console.log("=".repeat(75));
  console.log("PAPER SUMMARY");
  console.log("=".repeat(75));
  console.log(`  LLM Baseline (${MODEL}): ${global.exactPct}% exact, ${global.withinOnePct}% ±1, MAE=${global.mae}`);
  console.log(`  SSR (embedding-based):   65% exact, 92% ±1, MAE=0.44`);
  console.log(`  Difference:              ${global.exactPct > 65 ? "+" : ""}${global.exactPct - 65}pp exact, ${global.withinOnePct > 92 ? "+" : ""}${global.withinOnePct - 92}pp ±1`);
  console.log();
}

main().catch(console.error);
