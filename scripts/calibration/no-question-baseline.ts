/**
 * Control Condition Experiment — LLM Baseline WITHOUT Question Text
 *
 * Tests whether the LLM baseline's accuracy advantage over SSR comes from
 * having access to the question text (information asymmetry) rather than
 * superior rating ability. SSR maps response → scale via embeddings without
 * seeing the question. This experiment strips the question from the LLM prompt
 * to create an equal-information comparison.
 *
 * Design:
 *   - 69 test cases from ground-truth-v2
 *   - 2 models: Claude Haiku 4.5, GPT-4o
 *   - 2 conditions per model: with-question (standard), no-question (control)
 *   - Total: 4 configurations × 69 = 276 API calls
 *
 * Usage:
 *   npx tsx research/calibration/no-question-baseline.ts
 *
 * Environment variables required (.env.local):
 *   ANTHROPIC_API_KEY  — for Claude Haiku 4.5
 *   OPENAI_API_KEY     — for GPT-4o
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import Anthropic from "@anthropic-ai/sdk";
import { TEST_CASES, getDomains, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const CLAUDE_MODEL = "claude-haiku-4-5-20251001";
const GPT_MODEL = "gpt-4o";
const TEMPERATURE = 0;
const MAX_RETRIES = 3;
const CLAUDE_DELAY_MS = 500;
const GPT_DELAY_MS = 300;

type Condition = "with-question" | "no-question";
type ModelFamily = "claude" | "gpt";

interface ModelConfig {
  name: string;
  family: ModelFamily;
  modelId: string;
  delayMs: number;
}

const MODEL_CONFIGS: ModelConfig[] = [
  { name: "Claude Haiku 4.5", family: "claude", modelId: CLAUDE_MODEL, delayMs: CLAUDE_DELAY_MS },
  { name: "GPT-4o", family: "gpt", modelId: GPT_MODEL, delayMs: GPT_DELAY_MS },
];

const CONDITIONS: Condition[] = ["with-question", "no-question"];

// ─── Types ───────────────────────────────────────────────────────

interface RatingResult {
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

interface MetricsOutput {
  exact: number;
  exactPct: number;
  withinOne: number;
  withinOnePct: number;
  mae: number;
  total: number;
  meanSignedError: number;
  sdError: number;
}

interface ConfigResult {
  model: string;
  modelId: string;
  condition: Condition;
  global: MetricsOutput;
  perDomain: Record<string, MetricsOutput>;
  results: RatingResult[];
}

// ─── Prompts ─────────────────────────────────────────────────────

function buildPrompt(tc: TestCase, condition: Condition): string {
  const scaleMin = tc.question.scaleMin ?? 1;
  const scaleMax = tc.question.scaleMax ?? 5;
  const lowLabel = tc.question.scaleAnchors?.low || "Strongly Disagree/Very Negative";
  const highLabel = tc.question.scaleAnchors?.high || "Strongly Agree/Very Positive";

  if (condition === "with-question") {
    // Standard prompt (same as multi-model-baseline, domain-aware variant)
    return `Analyze the following survey response and determine the appropriate rating.

Question: "${tc.question.text}"

Response: "${tc.text}"

Scale: ${scaleMin} to ${scaleMax}
This is a Likert scale where ${scaleMin} = ${lowLabel} and ${scaleMax} = ${highLabel}.

IMPORTANT: If the response explicitly mentions a specific number (e.g. "I'd give it a 7", "maybe a 4"), use that number as the rating. Only infer from sentiment if no explicit number is stated.

Based on the sentiment and content of the response, provide:
1. A rating from ${scaleMin} to ${scaleMax}
2. Your confidence level (0.0 to 1.0)

Respond ONLY in this exact JSON format:
{"rating": <number>, "confidence": <number>}`;
  }

  // No-question prompt: strips the question entirely, provides only response + scale
  return `Analyze the following text and determine the appropriate rating based on its sentiment.

Text: "${tc.text}"

Scale: ${scaleMin} to ${scaleMax}
This is a Likert scale where ${scaleMin} = ${lowLabel} and ${scaleMax} = ${highLabel}.

IMPORTANT: If the text explicitly mentions a specific number (e.g. "I'd give it a 7", "maybe a 4"), use that number as the rating. Only infer from sentiment if no explicit number is stated.

Based on the sentiment and content of the text, provide:
1. A rating from ${scaleMin} to ${scaleMax}
2. Your confidence level (0.0 to 1.0)

Respond ONLY in this exact JSON format:
{"rating": <number>, "confidence": <number>}`;
}

// ─── API Callers ─────────────────────────────────────────────────

async function callClaude(anthropic: Anthropic, prompt: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 100,
    temperature: TEMPERATURE,
    messages: [{ role: "user", content: prompt }],
  });
  const textBlock = response.content.find((block) => block.type === "text");
  return textBlock?.type === "text" ? textBlock.text : "";
}

async function callGPT(apiKey: string, prompt: string): Promise<string> {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: GPT_MODEL,
      messages: [{ role: "user", content: prompt }],
      temperature: TEMPERATURE,
      max_tokens: 100,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${errorBody}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "";
}

// ─── Rating Logic ────────────────────────────────────────────────

async function rateSingleCase(
  tc: TestCase,
  condition: Condition,
  modelConfig: ModelConfig,
  anthropic: Anthropic | null,
  openaiKey: string | null
): Promise<RatingResult> {
  const prompt = buildPrompt(tc, condition);
  const scaleMin = tc.question.scaleMin ?? 1;
  const scaleMax = tc.question.scaleMax ?? 5;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      let text: string;
      if (modelConfig.family === "claude") {
        if (!anthropic) throw new Error("Anthropic client not initialized");
        text = await callClaude(anthropic, prompt);
      } else {
        if (!openaiKey) throw new Error("OpenAI API key not available");
        text = await callGPT(openaiKey, prompt);
      }

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
      console.error(`  Attempt ${attempt + 1}: No JSON in response for ${tc.label} (${modelConfig.name}/${condition})`);
    } catch (err) {
      console.error(`  Attempt ${attempt + 1} failed for ${tc.label} (${modelConfig.name}/${condition}):`, err);
    }
    await delay(2000 * (attempt + 1));
  }

  // Fallback: midpoint
  const midRating = Math.round((scaleMin + scaleMax) / 2);
  const error = Math.abs(midRating - tc.expected);
  console.warn(`  FALLBACK: midpoint (${midRating}) for ${tc.label} (${modelConfig.name}/${condition})`);
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
  return new Promise((r) => setTimeout(r, ms));
}

function computeMetrics(results: RatingResult[]): MetricsOutput {
  const total = results.length;
  if (total === 0) {
    return { exact: 0, exactPct: 0, withinOne: 0, withinOnePct: 0, mae: 0, total: 0, meanSignedError: 0, sdError: 0 };
  }
  const exact = results.filter((r) => r.exact).length;
  const withinOne = results.filter((r) => r.withinOne).length;
  const errors = results.map((r) => r.predicted - r.expected);
  const absErrors = errors.map(Math.abs);
  const mae = absErrors.reduce((a, b) => a + b, 0) / total;
  const meanSigned = errors.reduce((a, b) => a + b, 0) / total;
  const variance = errors.reduce((sum, e) => sum + (e - meanSigned) ** 2, 0) / total;

  return {
    exact,
    exactPct: Math.round((exact / total) * 100),
    withinOne,
    withinOnePct: Math.round((withinOne / total) * 100),
    mae: parseFloat(mae.toFixed(2)),
    total,
    meanSignedError: parseFloat(meanSigned.toFixed(3)),
    sdError: parseFloat(Math.sqrt(variance).toFixed(3)),
  };
}

function pad(str: string, len: number): string {
  return str.length >= len ? str : str + " ".repeat(len - str.length);
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  if (!anthropicKey) { console.error("ERROR: ANTHROPIC_API_KEY not set"); process.exit(1); }
  if (!openaiKey) { console.error("ERROR: OPENAI_API_KEY not set"); process.exit(1); }

  const anthropic = new Anthropic({ apiKey: anthropicKey });

  console.log("=".repeat(80));
  console.log("CONTROL CONDITION EXPERIMENT — LLM Baseline Without Question Text");
  console.log("=".repeat(80));
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Models: ${CLAUDE_MODEL}, ${GPT_MODEL}`);
  console.log(`Temperature: ${TEMPERATURE} (deterministic)`);
  console.log(`Test cases: ${TEST_CASES.length}, Domains: ${getDomains().length}`);
  console.log(`Conditions: with-question (standard), no-question (SSR-equivalent information)`);
  console.log(`Total API calls: ${MODEL_CONFIGS.length * CONDITIONS.length * TEST_CASES.length}`);
  console.log("=".repeat(80));

  // Show both prompts for comparison
  console.log("\nSample prompt WITH question (standard):");
  console.log("-".repeat(60));
  console.log(buildPrompt(TEST_CASES[0], "with-question"));
  console.log("-".repeat(60));
  console.log("\nSample prompt WITHOUT question (control):");
  console.log("-".repeat(60));
  console.log(buildPrompt(TEST_CASES[0], "no-question"));
  console.log("-".repeat(60));
  console.log();

  // ─── Run all configurations ─────────────────────────────────
  const allConfigs: ConfigResult[] = [];

  for (const modelConfig of MODEL_CONFIGS) {
    for (const condition of CONDITIONS) {
      console.log("=".repeat(80));
      console.log(`RUNNING: ${modelConfig.name} + ${condition}`);
      console.log("=".repeat(80));

      const results: RatingResult[] = [];

      for (let i = 0; i < TEST_CASES.length; i++) {
        const tc = TEST_CASES[i];
        console.log(`  [${i + 1}/${TEST_CASES.length}] ${tc.label} (expected: ${tc.expected})...`);

        const result = await rateSingleCase(
          tc,
          condition,
          modelConfig,
          modelConfig.family === "claude" ? anthropic : null,
          modelConfig.family === "gpt" ? openaiKey : null
        );
        results.push(result);

        const marker = result.exact ? "EXACT" : result.withinOne ? "+/-1" : `MISS (off by ${result.error})`;
        console.log(`    -> predicted: ${result.predicted}, conf: ${result.confidence.toFixed(2)} [${marker}]`);

        if (i < TEST_CASES.length - 1) await delay(modelConfig.delayMs);
      }

      const global = computeMetrics(results);
      const perDomain: Record<string, MetricsOutput> = {};
      for (const domain of getDomains()) {
        perDomain[domain] = computeMetrics(results.filter((r) => r.domain === domain));
      }

      allConfigs.push({ model: modelConfig.name, modelId: modelConfig.modelId, condition, global, perDomain, results });

      console.log(`\n  => ${modelConfig.name} (${condition}): ${global.exactPct}% exact, ${global.withinOnePct}% +/-1, MAE=${global.mae}\n`);
    }
  }

  // ─── Summary Table ─────────────────────────────────────────────
  console.log("\n" + "=".repeat(80));
  console.log("SUMMARY — INFORMATION ASYMMETRY ANALYSIS");
  console.log("=".repeat(80));
  console.log();

  const header = pad("Model", 20) + pad("Condition", 16) + pad("Exact", 16) + pad("+/-1", 16) + pad("MAE", 8);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const cfg of allConfigs) {
    const g = cfg.global;
    console.log(
      pad(cfg.model, 20) +
        pad(cfg.condition, 16) +
        pad(`${g.exact}/${g.total} (${g.exactPct}%)`, 16) +
        pad(`${g.withinOne}/${g.total} (${g.withinOnePct}%)`, 16) +
        pad(g.mae.toFixed(2), 8)
    );
  }

  // Add SSR reference line
  console.log("-".repeat(header.length));
  console.log(pad("SSR (Voyage)", 20) + pad("no-question*", 16) + pad("45/69 (65%)", 16) + pad("63/69 (91%)", 16) + pad("0.43", 8));
  console.log("  * SSR never sees the question text — comparable to no-question condition");
  console.log();

  // ─── Information Asymmetry Delta ───────────────────────────────
  console.log("=".repeat(80));
  console.log("INFORMATION ASYMMETRY DELTA (with-question minus no-question)");
  console.log("=".repeat(80));
  console.log();

  for (const modelConfig of MODEL_CONFIGS) {
    const withQ = allConfigs.find((c) => c.model === modelConfig.name && c.condition === "with-question");
    const noQ = allConfigs.find((c) => c.model === modelConfig.name && c.condition === "no-question");
    if (!withQ || !noQ) continue;

    const exactDelta = withQ.global.exactPct - noQ.global.exactPct;
    const w1Delta = withQ.global.withinOnePct - noQ.global.withinOnePct;
    const maeDelta = noQ.global.mae - withQ.global.mae; // Positive = question helps

    console.log(`  ${modelConfig.name}:`);
    console.log(`    With question:    ${withQ.global.exactPct}% exact, ${withQ.global.withinOnePct}% +/-1, MAE=${withQ.global.mae}`);
    console.log(`    Without question: ${noQ.global.exactPct}% exact, ${noQ.global.withinOnePct}% +/-1, MAE=${noQ.global.mae}`);
    console.log(`    Delta:            ${exactDelta >= 0 ? "+" : ""}${exactDelta}pp exact, ${w1Delta >= 0 ? "+" : ""}${w1Delta}pp +/-1, ${maeDelta >= 0 ? "+" : ""}${maeDelta.toFixed(2)} MAE`);
    console.log(`    => Question text accounts for ${Math.abs(exactDelta)}pp of baseline accuracy advantage`);
    console.log();
  }

  // ─── Per-Domain: No-Question vs SSR ────────────────────────────
  for (const modelConfig of MODEL_CONFIGS) {
    const noQ = allConfigs.find((c) => c.model === modelConfig.name && c.condition === "no-question");
    if (!noQ) continue;

    console.log("=".repeat(80));
    console.log(`PER-DOMAIN: ${modelConfig.name} (no-question) vs SSR`);
    console.log("=".repeat(80));

    const domHeader = pad("Domain", 20) + pad("No-Q Exact", 14) + pad("SSR Exact", 14) + pad("Delta", 10);
    console.log(domHeader);
    console.log("-".repeat(domHeader.length));

    // SSR per-domain from cross-validation (approximate from paper)
    const ssrPerDomain: Record<string, number> = {
      satisfaction: 90,
      purchase_intent: 88,
      agreement: 75,
      value: 75,
      likelihood: 63,
      ease: 56,
      trust: 44,
      importance: 33,
    };

    for (const domain of getDomains()) {
      const m = noQ.perDomain[domain];
      if (!m) continue;
      const ssrPct = ssrPerDomain[domain] || 0;
      const delta = m.exactPct - ssrPct;
      console.log(
        pad(domain, 20) +
          pad(`${m.exactPct}%`, 14) +
          pad(`${ssrPct}%`, 14) +
          pad(`${delta >= 0 ? "+" : ""}${delta}pp`, 10)
      );
    }
    console.log();
  }

  // ─── Save Results ──────────────────────────────────────────────
  const output = {
    timestamp: new Date().toISOString(),
    description: "Control condition: LLM baseline with and without question text, 69-case benchmark",
    temperature: TEMPERATURE,
    testSetSize: TEST_CASES.length,
    domains: getDomains(),
    configs: allConfigs,
    analysis: {
      purpose: "Quantify information asymmetry between LLM baseline (sees question) and SSR (does not see question)",
      hypothesis: "Stripping the question from the LLM prompt should reduce its accuracy toward SSR levels",
      ssrReference: { exactPct: 65, withinOnePct: 91, mae: 0.43 },
    },
  };

  const outputPath = path.join(__dirname, "..", "data", `no-question-baseline-${Date.now()}.json`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nResults saved to: ${outputPath}`);
}

main().catch(console.error);
