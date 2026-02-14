/**
 * Multi-Model Baseline Experiment — Cross-Model LLM Rating Comparison
 *
 * Compares rating accuracy across model families (Claude Haiku 4.5 vs GPT-4o)
 * on the 69-case ground truth benchmark. Each model is tested with two prompt
 * variants (as-deployed generic labels vs domain-aware anchors) for a total
 * of 4 configurations.
 *
 * This addresses a key validity question for the SSR paper:
 *   - Is LLM rating accuracy model-dependent?
 *   - Does prompt engineering (domain-aware anchors) help equally across models?
 *   - Do different model families exhibit similar error patterns?
 *
 * Usage:
 *   npx tsx research/calibration/multi-model-baseline.ts
 *
 * Environment variables required (.env.local):
 *   ANTHROPIC_API_KEY  — for Claude Haiku 4.5
 *   OPENAI_API_KEY     — for GPT-4o
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import Anthropic from "@anthropic-ai/sdk";
import { TEST_CASES, getDomains, getTestCasesByDomain, type TestCase } from "./ground-truth-v2";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const CLAUDE_MODEL = "claude-haiku-4-5-20251001";
const GPT_MODEL = "gpt-4o";
const TEMPERATURE = 0; // Deterministic output for reproducibility
const MAX_RETRIES = 3;
const CLAUDE_DELAY_MS = 500; // Rate limiting for Claude
const GPT_DELAY_MS = 300; // Rate limiting for OpenAI

type Variant = "as-deployed" | "domain-aware";
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

const VARIANTS: Variant[] = ["as-deployed", "domain-aware"];

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

interface ConfigResult {
  model: string;
  modelId: string;
  variant: Variant;
  global: MetricsOutput;
  perDomain: Record<string, MetricsOutput>;
  perDifficulty: Record<string, MetricsOutput>;
  results: LLMRatingResult[];
}

interface MetricsOutput {
  exact: number;
  exactPct: number;
  withinOne: number;
  withinOnePct: number;
  mae: number;
  total: number;
}

interface ExperimentOutput {
  timestamp: string;
  description: string;
  temperature: number;
  testSetSize: number;
  domains: string[];
  configs: ConfigResult[];
}

// ─── Prompt (exact copy from llm-baseline-experiment.ts) ─────────

function buildPrompt(tc: TestCase, variant: Variant): string {
  const scaleMin = tc.question.scaleMin ?? 1;
  const scaleMax = tc.question.scaleMax ?? 5;

  let scaleDescription: string;
  if (tc.question.type === "nps") {
    scaleDescription = "This is an NPS (Net Promoter Score) question. 0-6 = Detractor, 7-8 = Passive, 9-10 = Promoter.";
  } else if (variant === "domain-aware") {
    const lowLabel = tc.question.scaleAnchors?.low || "Strongly Disagree/Very Negative";
    const highLabel = tc.question.scaleAnchors?.high || "Strongly Agree/Very Positive";
    scaleDescription = `This is a Likert scale where ${scaleMin} = ${lowLabel} and ${scaleMax} = ${highLabel}.`;
  } else {
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

// ─── API Callers ─────────────────────────────────────────────────

async function callClaude(anthropic: Anthropic, prompt: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 100,
    temperature: TEMPERATURE,
    messages: [{ role: "user", content: prompt }],
  });

  const textBlock = response.content.find((block) => block.type === "text");
  const text = textBlock?.type === "text" ? textBlock.text : "";
  return text;
}

async function callGPT(apiKey: string, prompt: string): Promise<string> {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
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
  variant: Variant,
  modelConfig: ModelConfig,
  anthropic: Anthropic | null,
  openaiKey: string | null
): Promise<LLMRatingResult> {
  const prompt = buildPrompt(tc, variant);
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

      // JSON not found in response — retry
      console.error(`  Attempt ${attempt + 1}: No JSON found in response for ${tc.label} (${modelConfig.name})`);
    } catch (err) {
      console.error(`  Attempt ${attempt + 1} failed for ${tc.label} (${modelConfig.name}):`, err);
    }

    // Exponential backoff between retries
    await delay(2000 * (attempt + 1));
  }

  // Fallback: midpoint rating
  const midRating = Math.round((scaleMin + scaleMax) / 2);
  const error = Math.abs(midRating - tc.expected);
  console.warn(`  FALLBACK: Using midpoint (${midRating}) for ${tc.label} (${modelConfig.name})`);

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

function computeMetrics(results: LLMRatingResult[]): MetricsOutput {
  const total = results.length;
  if (total === 0) {
    return { exact: 0, exactPct: 0, withinOne: 0, withinOnePct: 0, mae: 0, total: 0 };
  }
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

function padRight(str: string, len: number): string {
  return str.length >= len ? str : str + " ".repeat(len - str.length);
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  // Validate API keys
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  if (!anthropicKey) {
    console.error("ERROR: ANTHROPIC_API_KEY not set in .env.local");
    process.exit(1);
  }
  if (!openaiKey) {
    console.error("ERROR: OPENAI_API_KEY not set in .env.local");
    process.exit(1);
  }

  const anthropic = new Anthropic({ apiKey: anthropicKey });

  console.log("=".repeat(80));
  console.log("MULTI-MODEL BASELINE EXPERIMENT");
  console.log("Cross-Model LLM Rating Comparison (Claude Haiku 4.5 vs GPT-4o)");
  console.log("=".repeat(80));
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Models: ${CLAUDE_MODEL}, ${GPT_MODEL}`);
  console.log(`Temperature: ${TEMPERATURE} (deterministic)`);
  console.log(`Variants: as-deployed (generic labels), domain-aware (question-specific anchors)`);
  console.log(`Test cases: ${TEST_CASES.length}, Domains: ${getDomains().length}`);
  console.log(`Configurations: ${MODEL_CONFIGS.length} models x ${VARIANTS.length} variants = ${MODEL_CONFIGS.length * VARIANTS.length} runs`);
  console.log("=".repeat(80));
  console.log();

  // Show sample prompt
  const samplePrompt = buildPrompt(TEST_CASES[0], "as-deployed");
  console.log("Sample prompt (first test case, as-deployed):");
  console.log("-".repeat(60));
  console.log(samplePrompt);
  console.log("-".repeat(60));
  console.log();

  // ─── Run all 4 configurations ─────────────────────────────────
  const allConfigs: ConfigResult[] = [];

  for (const modelConfig of MODEL_CONFIGS) {
    for (const variant of VARIANTS) {
      console.log("=".repeat(80));
      console.log(`RUNNING: ${modelConfig.name} + ${variant}`);
      console.log("=".repeat(80));

      const results: LLMRatingResult[] = [];

      for (let i = 0; i < TEST_CASES.length; i++) {
        const tc = TEST_CASES[i];
        console.log(`  [${i + 1}/${TEST_CASES.length}] Rating ${tc.label} (expected: ${tc.expected})...`);

        const result = await rateSingleCase(
          tc,
          variant,
          modelConfig,
          modelConfig.family === "claude" ? anthropic : null,
          modelConfig.family === "gpt" ? openaiKey : null
        );
        results.push(result);

        const marker = result.exact ? "EXACT" : result.withinOne ? "+/-1" : `MISS (off by ${result.error})`;
        console.log(`    -> predicted: ${result.predicted}, confidence: ${result.confidence.toFixed(2)} [${marker}]`);

        // Rate limiting between calls
        if (i < TEST_CASES.length - 1) {
          await delay(modelConfig.delayMs);
        }
      }

      // Compute metrics for this configuration
      const global = computeMetrics(results);

      const perDomain: Record<string, MetricsOutput> = {};
      for (const domain of getDomains()) {
        const domainResults = results.filter(r => r.domain === domain);
        perDomain[domain] = computeMetrics(domainResults);
      }

      const perDifficulty: Record<string, MetricsOutput> = {};
      for (const diff of ["clear", "subtle", "edge"]) {
        const diffResults = results.filter(r => r.difficulty === diff);
        perDifficulty[diff] = computeMetrics(diffResults);
      }

      allConfigs.push({
        model: modelConfig.name,
        modelId: modelConfig.modelId,
        variant,
        global,
        perDomain,
        perDifficulty,
        results,
      });

      console.log();
      console.log(`  => ${modelConfig.name} (${variant}): ${global.exactPct}% exact, ${global.withinOnePct}% +/-1, MAE=${global.mae}`);
      console.log();
    }
  }

  // ─── Summary Table ─────────────────────────────────────────────
  console.log();
  console.log("=".repeat(80));
  console.log("SUMMARY TABLE");
  console.log("=".repeat(80));
  console.log();

  const colModel = 20;
  const colVariant = 14;
  const colExact = 16;
  const colWithin = 16;
  const colMAE = 8;

  const header =
    padRight("Model", colModel) +
    padRight("Variant", colVariant) +
    padRight("Exact", colExact) +
    padRight("+/-1", colWithin) +
    padRight("MAE", colMAE);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const cfg of allConfigs) {
    const g = cfg.global;
    const row =
      padRight(cfg.model, colModel) +
      padRight(cfg.variant, colVariant) +
      padRight(`${g.exact}/${g.total} (${g.exactPct}%)`, colExact) +
      padRight(`${g.withinOne}/${g.total} (${g.withinOnePct}%)`, colWithin) +
      padRight(g.mae.toFixed(2), colMAE);
    console.log(row);
  }
  console.log();

  // ─── Per-Domain Breakdown (as-deployed variant for each model) ─
  for (const modelConfig of MODEL_CONFIGS) {
    const cfg = allConfigs.find(c => c.model === modelConfig.name && c.variant === "as-deployed");
    if (!cfg) continue;

    console.log("=".repeat(80));
    console.log(`PER-DOMAIN BREAKDOWN: ${modelConfig.name} (as-deployed)`);
    console.log("=".repeat(80));

    const domHeader =
      padRight("Domain", 20) +
      padRight("N", 5) +
      padRight("Exact", 16) +
      padRight("+/-1", 16) +
      padRight("MAE", 8);
    console.log(domHeader);
    console.log("-".repeat(domHeader.length));

    for (const domain of getDomains()) {
      const m = cfg.perDomain[domain];
      if (!m) continue;
      const row =
        padRight(domain, 20) +
        padRight(m.total.toString(), 5) +
        padRight(`${m.exact}/${m.total} (${m.exactPct}%)`, 16) +
        padRight(`${m.withinOne}/${m.total} (${m.withinOnePct}%)`, 16) +
        padRight(m.mae.toFixed(2), 8);
      console.log(row);
    }
    console.log();
  }

  // ─── Misses for Each Model ─────────────────────────────────────
  for (const modelConfig of MODEL_CONFIGS) {
    // Show misses for as-deployed variant
    const cfg = allConfigs.find(c => c.model === modelConfig.name && c.variant === "as-deployed");
    if (!cfg) continue;

    const misses = cfg.results.filter(r => !r.exact);

    console.log("=".repeat(80));
    console.log(`MISSES: ${modelConfig.name} (as-deployed) — ${misses.length} cases`);
    console.log("=".repeat(80));

    if (misses.length === 0) {
      console.log("  No misses!");
    } else {
      for (const m of misses) {
        const dir = m.predicted > m.expected ? "+" : "-";
        console.log(
          `  ${padRight(m.label, 10)} expected=${m.expected} predicted=${m.predicted} (${dir}${m.error}) [${m.difficulty}]`
        );
      }
    }
    console.log();

    // Also show domain-aware misses
    const cfgDA = allConfigs.find(c => c.model === modelConfig.name && c.variant === "domain-aware");
    if (!cfgDA) continue;

    const missesDA = cfgDA.results.filter(r => !r.exact);

    console.log("=".repeat(80));
    console.log(`MISSES: ${modelConfig.name} (domain-aware) — ${missesDA.length} cases`);
    console.log("=".repeat(80));

    if (missesDA.length === 0) {
      console.log("  No misses!");
    } else {
      for (const m of missesDA) {
        const dir = m.predicted > m.expected ? "+" : "-";
        console.log(
          `  ${padRight(m.label, 10)} expected=${m.expected} predicted=${m.predicted} (${dir}${m.error}) [${m.difficulty}]`
        );
      }
    }
    console.log();
  }

  // ─── Cross-Model Agreement Analysis ────────────────────────────
  console.log("=".repeat(80));
  console.log("CROSS-MODEL AGREEMENT (as-deployed)");
  console.log("=".repeat(80));

  const claudeAD = allConfigs.find(c => c.model === "Claude Haiku 4.5" && c.variant === "as-deployed");
  const gptAD = allConfigs.find(c => c.model === "GPT-4o" && c.variant === "as-deployed");

  if (claudeAD && gptAD) {
    let agree = 0;
    let bothCorrect = 0;
    let onlyClaudeCorrect = 0;
    let onlyGPTCorrect = 0;
    let neitherCorrect = 0;

    for (let i = 0; i < TEST_CASES.length; i++) {
      const cResult = claudeAD.results[i];
      const gResult = gptAD.results[i];

      if (cResult.predicted === gResult.predicted) agree++;
      if (cResult.exact && gResult.exact) bothCorrect++;
      else if (cResult.exact && !gResult.exact) onlyClaudeCorrect++;
      else if (!cResult.exact && gResult.exact) onlyGPTCorrect++;
      else neitherCorrect++;
    }

    console.log(`  Same prediction:     ${agree}/${TEST_CASES.length} (${Math.round((agree / TEST_CASES.length) * 100)}%)`);
    console.log(`  Both correct:        ${bothCorrect}/${TEST_CASES.length}`);
    console.log(`  Only Claude correct: ${onlyClaudeCorrect}/${TEST_CASES.length}`);
    console.log(`  Only GPT-4o correct: ${onlyGPTCorrect}/${TEST_CASES.length}`);
    console.log(`  Neither correct:     ${neitherCorrect}/${TEST_CASES.length}`);

    // Cases where models disagree and one is wrong
    console.log();
    console.log("  Disagreements (different prediction):");
    console.log("  " + "-".repeat(70));
    let disagreementCount = 0;
    for (let i = 0; i < TEST_CASES.length; i++) {
      const cResult = claudeAD.results[i];
      const gResult = gptAD.results[i];

      if (cResult.predicted !== gResult.predicted) {
        disagreementCount++;
        const cMark = cResult.exact ? "OK" : "MISS";
        const gMark = gResult.exact ? "OK" : "MISS";
        console.log(
          `  ${padRight(cResult.label, 10)} expected=${cResult.expected}  ` +
          `Claude=${cResult.predicted}(${cMark})  GPT=${gResult.predicted}(${gMark})`
        );
      }
    }
    if (disagreementCount === 0) {
      console.log("  No disagreements!");
    }
  }
  console.log();

  // ─── Save Results ──────────────────────────────────────────────
  const output: ExperimentOutput = {
    timestamp: new Date().toISOString(),
    description: "Multi-model baseline: Claude Haiku 4.5 vs GPT-4o, 2 variants each, 69-case benchmark",
    temperature: TEMPERATURE,
    testSetSize: TEST_CASES.length,
    domains: getDomains(),
    configs: allConfigs,
  };

  const outputPath = path.join(__dirname, "..", "data", `multi-model-baseline-${Date.now()}.json`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`Results saved to: ${outputPath}`);

  // ─── Paper Summary ─────────────────────────────────────────────
  console.log();
  console.log("=".repeat(80));
  console.log("PAPER SUMMARY");
  console.log("=".repeat(80));

  for (const cfg of allConfigs) {
    console.log(`  ${padRight(cfg.model, 20)} (${padRight(cfg.variant + ")", 14)} ${cfg.global.exactPct}% exact, ${cfg.global.withinOnePct}% +/-1, MAE=${cfg.global.mae}`);
  }

  console.log();
  console.log("  SSR (embedding-based):                      65% exact, 91% +/-1, MAE=0.43");
  console.log();

  // Delta vs SSR
  for (const cfg of allConfigs) {
    const exactDelta = cfg.global.exactPct - 65;
    const w1Delta = cfg.global.withinOnePct - 91;
    const maeDelta = cfg.global.mae - 0.43;
    console.log(
      `  Delta vs SSR (${padRight(cfg.model + " " + cfg.variant + ")", 38)} ` +
      `${exactDelta >= 0 ? "+" : ""}${exactDelta}pp exact, ` +
      `${w1Delta >= 0 ? "+" : ""}${w1Delta}pp +/-1, ` +
      `${maeDelta >= 0 ? "+" : ""}${maeDelta.toFixed(2)} MAE`
    );
  }
  console.log();
}

main().catch(console.error);
