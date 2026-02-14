/**
 * Self-Rating Circularity Experiment
 *
 * Tests whether LLM self-rating (same model generates + rates text) produces
 * measurably different results than SSR (embedding-based independent rating).
 *
 * Design: 5 personas × 69 test cases = 345 generated texts
 * Each text rated by both LLM (circular) and SSR (independent).
 *
 * 4 phases with checkpointing:
 *   Phase 1: Generate 345 texts via Claude Haiku 4.5 (temp=0.7)
 *   Phase 2: LLM self-rating via Claude Haiku 4.5 (temp=0, as-deployed prompt)
 *   Phase 3: SSR rating via Voyage AI embeddings (τ=0.15, minmax, asymmetric)
 *   Phase 4: Compute metrics and save results
 *
 * Usage:
 *   npx tsx research/calibration/self-rating-experiment.ts [phase]
 *   Phases: all (default), generate, llm-rate, ssr-rate, metrics
 *
 * Pre-registration date: 2026-02-09
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import Anthropic from "@anthropic-ai/sdk";
import { TEST_CASES, getDomains, type TestCase } from "./ground-truth-v2";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import { VoyageEmbeddingService } from "../../src/lib/embedding-service";
import { normalizedEntropy } from "../../src/lib/statistics";
import * as fs from "fs";
import * as path from "path";

// ─── Configuration ───────────────────────────────────────────────

const GENERATION_MODEL = "claude-haiku-4-5-20251001";
const RATING_MODEL = "claude-haiku-4-5-20251001";
const GENERATION_TEMPERATURE = 0.7;
const RATING_TEMPERATURE = 0;
const SSR_TAU = 0.15; // LODO-optimized (not default 0.2)
const SSR_NORMALIZATION = "minmax" as const;

const DELAY_BETWEEN_LLM_CALLS_MS = 500;
const MAX_RETRIES = 3;
const PRE_REGISTRATION_DATE = "2026-02-09";

// ─── Personas ────────────────────────────────────────────────────

const PERSONAS = [
  {
    id: "detailed_professional",
    name: "Maria",
    description: "35-year-old marketing manager. Analytical and thorough. Uses specific details and examples. Writes 3-4 well-structured sentences.",
    style: "detailed and specific, uses concrete examples",
  },
  {
    id: "blunt_retiree",
    name: "James",
    description: "58-year-old retired engineer. Direct and no-nonsense. Gets to the point quickly. Writes 1-2 short, blunt sentences.",
    style: "brief and direct, no hedging, matter-of-fact",
  },
  {
    id: "hedging_student",
    name: "Alex",
    description: "24-year-old graduate student. Overthinks, uses qualifiers like 'kind of', 'I guess', 'sort of'. Often sees both sides. Writes 2-3 sentences with qualifications.",
    style: "hedging and qualified, uses 'kind of', 'I guess', sees both sides",
  },
  {
    id: "busy_parent",
    name: "Linda",
    description: "42-year-old nurse and mother of two. Practical, time-constrained, values efficiency. Writes 1-2 practical sentences.",
    style: "practical and concise, focused on function over form",
  },
  {
    id: "casual_young",
    name: "Ryan",
    description: "21-year-old barista and college student. Casual, uses informal language, emotionally expressive. Writes 1-3 casual sentences.",
    style: "informal and casual, emotionally expressive, uses colloquial language",
  },
];

// ─── Types ───────────────────────────────────────────────────────

interface GeneratedText {
  testCaseLabel: string;
  domain: string;
  difficulty: string;
  targetRating: number;
  personaId: string;
  personaName: string;
  questionText: string;
  generatedText: string;
}

interface ExperimentCase {
  testCaseLabel: string;
  domain: string;
  difficulty: string;
  targetRating: number;
  personaId: string;
  personaName: string;
  questionText: string;
  generatedText: string;
  llmRating: number;
  llmConfidence: number;
  ssrRating: number;
  ssrConfidence: number;
  ssrDistribution: number[];
  ssrRawSimilarities: number[];
}

interface AggregatedCase {
  testCaseLabel: string;
  domain: string;
  difficulty: string;
  targetRating: number;
  llmModalRating: number;
  ssrModalRating: number;
  llmExactMatch: boolean;
  ssrExactMatch: boolean;
}

interface ConditionMetrics {
  n: number;
  exactMatch: number;
  exactMatchPct: number;
  withinOne: number;
  withinOnePct: number;
  mae: number;
  meanSignedError: number;
  sdSignedError: number;
}

interface ExperimentOutput {
  metadata: {
    timestamp: string;
    preRegistrationDate: string;
    config: {
      generationModel: string;
      ratingModel: string;
      generationTemperature: number;
      ratingTemperature: number;
      ssrConfig: {
        embeddingModel: string;
        normalization: string;
        tau: number;
        asymmetric: boolean;
      };
      personas: typeof PERSONAS;
    };
  };
  cases: ExperimentCase[];
  summary: {
    conditionC: ConditionMetrics;
    conditionD: ConditionMetrics;
    aggregatedByCase: AggregatedCase[];
    perPersona: Record<string, { llmExact: number; ssrExact: number; meanDivergence: number }>;
    perDomain: Record<string, { llmExact: number; ssrExact: number; meanDivergence: number }>;
  };
}

// ─── Helpers ─────────────────────────────────────────────────────

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function getDataDir(): string {
  return path.join(__dirname, "..", "data");
}

function findLatestFile(prefix: string): string | null {
  const dir = getDataDir();
  if (!fs.existsSync(dir)) return null;
  const files = fs.readdirSync(dir)
    .filter((f) => f.startsWith(prefix) && !f.includes("-partial-") && f.endsWith(".json"))
    .sort()
    .reverse();
  return files.length > 0 ? path.join(dir, files[0]) : null;
}

function getSentimentGuide(tc: TestCase, targetRating: number): string {
  const low = tc.question.scaleAnchors?.low || "Very negative";
  const high = tc.question.scaleAnchors?.high || "Very positive";
  const guides: Record<number, string> = {
    1: `${low}. You had a terrible experience and feel strongly negative about this.`,
    2: `Leaning toward ${low.toLowerCase()}. You're disappointed, below your expectations.`,
    3: `Neutral, right in the middle. Not particularly ${low.toLowerCase()} or ${high.toLowerCase()}.`,
    4: `Leaning toward ${high.toLowerCase()}. Generally positive with minor reservations.`,
    5: `${high}. You feel strongly positive and are very pleased about this.`,
  };
  return guides[targetRating];
}

// ─── Phase 1: Text Generation ────────────────────────────────────

async function phase1Generate(anthropic: Anthropic): Promise<GeneratedText[]> {
  // Check for existing checkpoint
  const existing = findLatestFile("generated-texts-");
  if (existing) {
    console.log(`\n  Checkpoint found: ${path.basename(existing)}`);
    console.log("  Loading existing generated texts (skip Phase 1)...");
    return JSON.parse(fs.readFileSync(existing, "utf-8"));
  }

  console.log("\n  Generating 345 texts (69 cases × 5 personas)...");
  const results: GeneratedText[] = [];
  let count = 0;
  const total = TEST_CASES.length * PERSONAS.length;

  for (const tc of TEST_CASES) {
    for (const persona of PERSONAS) {
      count++;
      const sentimentGuide = getSentimentGuide(tc, tc.expected);

      const systemPrompt = `You are ${persona.name}, a ${persona.description}\n\nYour communication style is ${persona.style}.\n\nRULES:\n- Write 1-4 sentences in your natural voice\n- Do NOT mention any numbers, scales, or ratings\n- Do NOT start with filler words (Well, Hmm, Oh, So, Honestly)\n- Do NOT describe yourself or your demographics\n- Do NOT use the phrase "As a..." or "Being a..."`;

      const userPrompt = `A survey asks: "${tc.question.text}"\n\nYour genuine feeling about this is: ${sentimentGuide}\n\nWrite a natural response expressing this sentiment in your own words.`;

      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
          const response = await anthropic.messages.create({
            model: GENERATION_MODEL,
            max_tokens: 300,
            temperature: GENERATION_TEMPERATURE,
            system: systemPrompt,
            messages: [{ role: "user", content: userPrompt }],
          });

          const textBlock = response.content.find((b) => b.type === "text");
          const text = textBlock?.type === "text" ? textBlock.text.trim() : "";

          if (text.length < 10) {
            console.log(`    WARNING: Short text for ${tc.label}/${persona.id}, retrying...`);
            await delay(1000);
            continue;
          }

          results.push({
            testCaseLabel: tc.label,
            domain: tc.domain,
            difficulty: tc.difficulty,
            targetRating: tc.expected,
            personaId: persona.id,
            personaName: persona.name,
            questionText: tc.question.text,
            generatedText: text,
          });

          if (count % 50 === 0 || count === total) {
            console.log(`    [${count}/${total}] ${tc.label}/${persona.id} ✓`);
          }
          break;
        } catch (err) {
          console.error(`    Attempt ${attempt + 1} failed for ${tc.label}/${persona.id}:`, err);
          await delay(2000 * (attempt + 1));
          if (attempt === MAX_RETRIES - 1) {
            // Push empty text as fallback — will be flagged in quality check
            results.push({
              testCaseLabel: tc.label,
              domain: tc.domain,
              difficulty: tc.difficulty,
              targetRating: tc.expected,
              personaId: persona.id,
              personaName: persona.name,
              questionText: tc.question.text,
              generatedText: "[GENERATION FAILED]",
            });
          }
        }
      }

      await delay(DELAY_BETWEEN_LLM_CALLS_MS);
    }
  }

  // Save checkpoint
  const outputPath = path.join(getDataDir(), `generated-texts-${Date.now()}.json`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`  Saved ${results.length} generated texts to: ${path.basename(outputPath)}`);
  return results;
}

// ─── Phase 2: LLM Self-Rating ────────────────────────────────────

function buildRatingPrompt(questionText: string, responseText: string): string {
  // Replicates the as-deployed prompt from llm-baseline-experiment.ts
  return `Analyze the following survey response and determine the appropriate rating.

Question: "${questionText}"

Response: "${responseText}"

Scale: 1 to 5
This is a Likert scale where 1 = Strongly Disagree/Very Negative and 5 = Strongly Agree/Very Positive.

IMPORTANT: If the response explicitly mentions a specific number (e.g. "I'd give it a 7", "maybe a 4"), use that number as the rating. Only infer from sentiment if no explicit number is stated.

Based on the sentiment and content of the response, provide:
1. A rating from 1 to 5
2. Your confidence level (0.0 to 1.0)

Respond ONLY in this exact JSON format:
{"rating": <number>, "confidence": <number>}`;
}

async function phase2LLMRate(
  anthropic: Anthropic,
  generatedTexts: GeneratedText[]
): Promise<Array<{ llmRating: number; llmConfidence: number }>> {
  // Check for existing checkpoint
  const existing = findLatestFile("llm-selfratings-");
  if (existing) {
    console.log(`\n  Checkpoint found: ${path.basename(existing)}`);
    console.log("  Loading existing LLM ratings (skip Phase 2)...");
    return JSON.parse(fs.readFileSync(existing, "utf-8"));
  }

  console.log(`\n  LLM self-rating ${generatedTexts.length} texts...`);
  const results: Array<{ llmRating: number; llmConfidence: number }> = [];

  for (let i = 0; i < generatedTexts.length; i++) {
    const gt = generatedTexts[i];
    const prompt = buildRatingPrompt(gt.questionText, gt.generatedText);

    let rating = 3;
    let confidence = 0.5;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const response = await anthropic.messages.create({
          model: RATING_MODEL,
          max_tokens: 100,
          temperature: RATING_TEMPERATURE,
          messages: [{ role: "user", content: prompt }],
        });

        const textBlock = response.content.find((b) => b.type === "text");
        const text = textBlock?.type === "text" ? textBlock.text : "";
        const jsonMatch = text.match(/\{[\s\S]*\}/);

        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          rating = Math.min(Math.max(Math.round(parsed.rating), 1), 5);
          confidence = Math.min(Math.max(parsed.confidence || 0.7, 0), 1);
          break;
        }
      } catch (err) {
        console.error(`    Attempt ${attempt + 1} failed for [${i}]:`, err);
        await delay(2000 * (attempt + 1));
      }
    }

    results.push({ llmRating: rating, llmConfidence: confidence });

    if ((i + 1) % 50 === 0 || i === generatedTexts.length - 1) {
      const exact = rating === gt.targetRating ? "EXACT" : `off by ${Math.abs(rating - gt.targetRating)}`;
      console.log(`    [${i + 1}/${generatedTexts.length}] ${gt.testCaseLabel}/${gt.personaId}: predicted=${rating}, target=${gt.targetRating} [${exact}]`);
    }

    if (i < generatedTexts.length - 1) {
      await delay(DELAY_BETWEEN_LLM_CALLS_MS);
    }
  }

  // Save checkpoint
  const outputPath = path.join(getDataDir(), `llm-selfratings-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`  Saved ${results.length} LLM ratings to: ${path.basename(outputPath)}`);
  return results;
}

// ─── Phase 3: SSR Rating ─────────────────────────────────────────

async function phase3SSRRate(
  generatedTexts: GeneratedText[]
): Promise<Array<{ ssrRating: number; ssrConfidence: number; ssrDistribution: number[]; ssrRawSimilarities: number[] }>> {
  // Check for existing checkpoint
  const existing = findLatestFile("ssr-selfratings-");
  if (existing) {
    console.log(`\n  Checkpoint found: ${path.basename(existing)}`);
    console.log("  Loading existing SSR ratings (skip Phase 3)...");
    return JSON.parse(fs.readFileSync(existing, "utf-8"));
  }

  const voyageKey = process.env.VOYAGE_API_KEY;
  if (!voyageKey) {
    console.error("ERROR: VOYAGE_API_KEY not set in .env.local");
    process.exit(1);
  }

  const embeddingService = new VoyageEmbeddingService(voyageKey);

  // Build a map from domain → question (to resolve anchors)
  const questionByDomain = new Map<string, TestCase["question"]>();
  for (const tc of TEST_CASES) {
    if (!questionByDomain.has(tc.domain)) {
      questionByDomain.set(tc.domain, tc.question);
    }
  }

  console.log(`\n  SSR rating ${generatedTexts.length} texts (τ=${SSR_TAU}, ${SSR_NORMALIZATION})...`);
  console.log("  NOTE: Voyage free tier ~3 RPM. This phase takes ~2 hours.");

  const results: Array<{ ssrRating: number; ssrConfidence: number; ssrDistribution: number[]; ssrRawSimilarities: number[] }> = [];

  for (let i = 0; i < generatedTexts.length; i++) {
    const gt = generatedTexts[i];
    const question = questionByDomain.get(gt.domain)!;
    const anchorSet = resolveAnchors(question);
    const cacheKey = `${anchorSet.semantic}-${anchorSet.anchors.length}`;

    try {
      const result = await embeddingService.mapResponseToScale(
        gt.generatedText,
        anchorSet.anchors,
        cacheKey,
        question.scaleMin ?? 1,
        question.scaleMax ?? 5,
        SSR_TAU,
        SSR_NORMALIZATION
      );

      results.push({
        ssrRating: result.rating,
        ssrConfidence: result.confidence,
        ssrDistribution: result.distribution,
        ssrRawSimilarities: result.rawSimilarities || [],
      });
    } catch (err) {
      console.error(`    SSR failed for [${i}] ${gt.testCaseLabel}/${gt.personaId}:`, err);
      // Fallback: midpoint
      results.push({
        ssrRating: 3,
        ssrConfidence: 0,
        ssrDistribution: [0.2, 0.2, 0.2, 0.2, 0.2],
        ssrRawSimilarities: [],
      });
    }

    if ((i + 1) % 10 === 0 || i === generatedTexts.length - 1) {
      const last = results[results.length - 1];
      const exact = last.ssrRating === gt.targetRating ? "EXACT" : `off by ${Math.abs(last.ssrRating - gt.targetRating)}`;
      console.log(`    [${i + 1}/${generatedTexts.length}] ${gt.testCaseLabel}/${gt.personaId}: ssr=${last.ssrRating}, target=${gt.targetRating} [${exact}] (conf=${last.ssrConfidence})`);
    }

    // Save intermediate checkpoint every 50 items
    if ((i + 1) % 50 === 0) {
      const partialPath = path.join(getDataDir(), `ssr-selfratings-partial-${Date.now()}.json`);
      fs.writeFileSync(partialPath, JSON.stringify(results, null, 2));
      console.log(`    Partial checkpoint saved (${results.length} items)`);
    }
  }

  // Save final checkpoint
  const outputPath = path.join(getDataDir(), `ssr-selfratings-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`  Saved ${results.length} SSR ratings to: ${path.basename(outputPath)}`);
  return results;
}

// ─── Phase 4: Compute Metrics ────────────────────────────────────

function computeConditionMetrics(
  cases: ExperimentCase[],
  getRating: (c: ExperimentCase) => number
): ConditionMetrics {
  const n = cases.length;
  const errors = cases.map((c) => getRating(c) - c.targetRating);
  const absErrors = errors.map(Math.abs);
  const exact = absErrors.filter((e) => e === 0).length;
  const withinOne = absErrors.filter((e) => e <= 1).length;
  const mae = absErrors.reduce((a, b) => a + b, 0) / n;
  const meanSigned = errors.reduce((a, b) => a + b, 0) / n;
  const variance = errors.reduce((sum, e) => sum + (e - meanSigned) ** 2, 0) / n;

  return {
    n,
    exactMatch: exact,
    exactMatchPct: Math.round((exact / n) * 100),
    withinOne,
    withinOnePct: Math.round((withinOne / n) * 100),
    mae: parseFloat(mae.toFixed(3)),
    meanSignedError: parseFloat(meanSigned.toFixed(3)),
    sdSignedError: parseFloat(Math.sqrt(variance).toFixed(3)),
  };
}

function computeModalRating(ratings: number[], target: number): number {
  // Count occurrences of each rating
  const counts = new Map<number, number>();
  for (const r of ratings) {
    counts.set(r, (counts.get(r) || 0) + 1);
  }

  // Find max count
  const maxCount = Math.max(...counts.values());
  const modes = [...counts.entries()].filter(([, c]) => c === maxCount).map(([r]) => r);

  // If single mode, return it
  if (modes.length === 1) return modes[0];

  // Tiebreak: closest to target
  modes.sort((a, b) => Math.abs(a - target) - Math.abs(b - target));
  return modes[0];
}

function phase4Metrics(
  generatedTexts: GeneratedText[],
  llmRatings: Array<{ llmRating: number; llmConfidence: number }>,
  ssrRatings: Array<{ ssrRating: number; ssrConfidence: number; ssrDistribution: number[]; ssrRawSimilarities: number[] }>
): ExperimentOutput {
  // Merge into cases
  const cases: ExperimentCase[] = generatedTexts.map((gt, i) => ({
    ...gt,
    llmRating: llmRatings[i].llmRating,
    llmConfidence: llmRatings[i].llmConfidence,
    ssrRating: ssrRatings[i].ssrRating,
    ssrConfidence: ssrRatings[i].ssrConfidence,
    ssrDistribution: ssrRatings[i].ssrDistribution,
    ssrRawSimilarities: ssrRatings[i].ssrRawSimilarities,
  }));

  // Condition C: LLM on generated text
  const conditionC = computeConditionMetrics(cases, (c) => c.llmRating);
  // Condition D: SSR on generated text
  const conditionD = computeConditionMetrics(cases, (c) => c.ssrRating);

  // Aggregate by case (modal rating across 5 personas)
  const caseLabels = [...new Set(cases.map((c) => c.testCaseLabel))];
  const aggregatedByCase: AggregatedCase[] = caseLabels.map((label) => {
    const casesForLabel = cases.filter((c) => c.testCaseLabel === label);
    const target = casesForLabel[0].targetRating;
    const domain = casesForLabel[0].domain;
    const difficulty = casesForLabel[0].difficulty;

    const llmRatingsForCase = casesForLabel.map((c) => c.llmRating);
    const ssrRatingsForCase = casesForLabel.map((c) => c.ssrRating);

    const llmModal = computeModalRating(llmRatingsForCase, target);
    const ssrModal = computeModalRating(ssrRatingsForCase, target);

    return {
      testCaseLabel: label,
      domain,
      difficulty,
      targetRating: target,
      llmModalRating: llmModal,
      ssrModalRating: ssrModal,
      llmExactMatch: llmModal === target,
      ssrExactMatch: ssrModal === target,
    };
  });

  // Per-persona breakdown
  const perPersona: Record<string, { llmExact: number; ssrExact: number; meanDivergence: number }> = {};
  for (const persona of PERSONAS) {
    const personaCases = cases.filter((c) => c.personaId === persona.id);
    const llmExact = personaCases.filter((c) => c.llmRating === c.targetRating).length;
    const ssrExact = personaCases.filter((c) => c.ssrRating === c.targetRating).length;
    const divergences = personaCases.map((c) => c.llmRating - c.ssrRating);
    const meanDiv = divergences.reduce((a, b) => a + b, 0) / divergences.length;

    perPersona[persona.id] = {
      llmExact: Math.round((llmExact / personaCases.length) * 100),
      ssrExact: Math.round((ssrExact / personaCases.length) * 100),
      meanDivergence: parseFloat(meanDiv.toFixed(3)),
    };
  }

  // Per-domain breakdown
  const perDomain: Record<string, { llmExact: number; ssrExact: number; meanDivergence: number }> = {};
  for (const domain of getDomains()) {
    const domainCases = cases.filter((c) => c.domain === domain);
    const llmExact = domainCases.filter((c) => c.llmRating === c.targetRating).length;
    const ssrExact = domainCases.filter((c) => c.ssrRating === c.targetRating).length;
    const divergences = domainCases.map((c) => c.llmRating - c.ssrRating);
    const meanDiv = divergences.reduce((a, b) => a + b, 0) / divergences.length;

    perDomain[domain] = {
      llmExact: Math.round((llmExact / domainCases.length) * 100),
      ssrExact: Math.round((ssrExact / domainCases.length) * 100),
      meanDivergence: parseFloat(meanDiv.toFixed(3)),
    };
  }

  return {
    metadata: {
      timestamp: new Date().toISOString(),
      preRegistrationDate: PRE_REGISTRATION_DATE,
      config: {
        generationModel: GENERATION_MODEL,
        ratingModel: RATING_MODEL,
        generationTemperature: GENERATION_TEMPERATURE,
        ratingTemperature: RATING_TEMPERATURE,
        ssrConfig: {
          embeddingModel: "voyage-3.5-lite",
          normalization: SSR_NORMALIZATION,
          tau: SSR_TAU,
          asymmetric: true,
        },
        personas: PERSONAS,
      },
    },
    cases,
    summary: {
      conditionC,
      conditionD,
      aggregatedByCase,
      perPersona,
      perDomain,
    },
  };
}

// ─── Console Output ──────────────────────────────────────────────

function printResults(output: ExperimentOutput): void {
  const { conditionC, conditionD, aggregatedByCase, perPersona, perDomain } = output.summary;

  console.log("\n" + "=".repeat(75));
  console.log("SELF-RATING CIRCULARITY EXPERIMENT — RESULTS");
  console.log("=".repeat(75));

  // 2×2 comparison table
  console.log("\n  2×2 Design (all values = exact match %):");
  console.log("  " + "─".repeat(60));
  console.log(`  ${"".padEnd(20)} ${"Human text".padEnd(18)} ${"Generated text".padEnd(18)}`);
  console.log("  " + "─".repeat(60));
  console.log(`  ${"LLM rating".padEnd(20)} ${"87% (Cond A)".padEnd(18)} ${conditionC.exactMatchPct}% (Cond C)`);
  console.log(`  ${"SSR rating".padEnd(20)} ${"65% (Cond B)".padEnd(18)} ${conditionD.exactMatchPct}% (Cond D)`);
  console.log("  " + "─".repeat(60));

  // Detailed metrics
  console.log("\n  Condition C (LLM self-rating, N=345):");
  console.log(`    Exact match: ${conditionC.exactMatch}/${conditionC.n} (${conditionC.exactMatchPct}%)`);
  console.log(`    Within ±1:   ${conditionC.withinOne}/${conditionC.n} (${conditionC.withinOnePct}%)`);
  console.log(`    MAE:         ${conditionC.mae}`);
  console.log(`    Mean error:  ${conditionC.meanSignedError} (SD=${conditionC.sdSignedError})`);

  console.log("\n  Condition D (SSR independent, N=345):");
  console.log(`    Exact match: ${conditionD.exactMatch}/${conditionD.n} (${conditionD.exactMatchPct}%)`);
  console.log(`    Within ±1:   ${conditionD.withinOne}/${conditionD.n} (${conditionD.withinOnePct}%)`);
  console.log(`    MAE:         ${conditionD.mae}`);
  console.log(`    Mean error:  ${conditionD.meanSignedError} (SD=${conditionD.sdSignedError})`);

  // Aggregated (modal across personas, N=69)
  const aggLLMExact = aggregatedByCase.filter((c) => c.llmExactMatch).length;
  const aggSSRExact = aggregatedByCase.filter((c) => c.ssrExactMatch).length;
  console.log(`\n  Aggregated (modal rating, N=69):`);
  console.log(`    LLM: ${aggLLMExact}/69 (${Math.round((aggLLMExact / 69) * 100)}%)`);
  console.log(`    SSR: ${aggSSRExact}/69 (${Math.round((aggSSRExact / 69) * 100)}%)`);

  // Per-persona
  console.log("\n  Per-Persona Breakdown:");
  console.log(`  ${"Persona".padEnd(25)} ${"LLM exact%".padEnd(12)} ${"SSR exact%".padEnd(12)} ${"Mean div".padEnd(10)}`);
  console.log("  " + "─".repeat(55));
  for (const [pid, metrics] of Object.entries(perPersona)) {
    console.log(`  ${pid.padEnd(25)} ${(metrics.llmExact + "%").padEnd(12)} ${(metrics.ssrExact + "%").padEnd(12)} ${metrics.meanDivergence}`);
  }

  // Per-domain
  console.log("\n  Per-Domain Breakdown:");
  console.log(`  ${"Domain".padEnd(20)} ${"LLM exact%".padEnd(12)} ${"SSR exact%".padEnd(12)} ${"Mean div".padEnd(10)}`);
  console.log("  " + "─".repeat(50));
  for (const [domain, metrics] of Object.entries(perDomain)) {
    console.log(`  ${domain.padEnd(20)} ${(metrics.llmExact + "%").padEnd(12)} ${(metrics.ssrExact + "%").padEnd(12)} ${metrics.meanDivergence}`);
  }

  console.log("\n" + "=".repeat(75));
  console.log("  Run analyze-circularity.py for statistical tests (Phase 5).");
  console.log("=".repeat(75));
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const phase = process.argv[2] || "all";
  const validPhases = ["all", "generate", "llm-rate", "ssr-rate", "metrics"];
  if (!validPhases.includes(phase)) {
    console.error(`Invalid phase "${phase}". Use: ${validPhases.join(", ")}`);
    process.exit(1);
  }

  console.log("=".repeat(75));
  console.log("SELF-RATING CIRCULARITY EXPERIMENT");
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`Pre-registration: ${PRE_REGISTRATION_DATE}`);
  console.log(`Phase: ${phase}`);
  console.log(`Test cases: ${TEST_CASES.length}, Personas: ${PERSONAS.length}`);
  console.log(`Total texts: ${TEST_CASES.length * PERSONAS.length}`);
  console.log("=".repeat(75));

  // Verify API keys
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  const voyageKey = process.env.VOYAGE_API_KEY;

  if (!anthropicKey && (phase === "all" || phase === "generate" || phase === "llm-rate")) {
    console.error("ERROR: ANTHROPIC_API_KEY not set in .env.local");
    process.exit(1);
  }
  if (!voyageKey && (phase === "all" || phase === "ssr-rate")) {
    console.error("ERROR: VOYAGE_API_KEY not set in .env.local");
    process.exit(1);
  }

  const anthropic = anthropicKey ? new Anthropic({ apiKey: anthropicKey }) : null;

  // ─── Phase 1: Generate texts ──────────────────────────────────
  let generatedTexts: GeneratedText[];

  if (phase === "all" || phase === "generate") {
    console.log("\n── PHASE 1: Text Generation ──────────────────────────");
    generatedTexts = await phase1Generate(anthropic!);
    console.log(`  Total generated: ${generatedTexts.length}`);

    if (phase === "generate") {
      console.log("\n  Phase 1 complete. Review texts, then run with 'llm-rate'.");
      return;
    }
  } else {
    // Load from checkpoint
    const existing = findLatestFile("generated-texts-");
    if (!existing) {
      console.error("ERROR: No generated texts found. Run 'generate' phase first.");
      process.exit(1);
    }
    generatedTexts = JSON.parse(fs.readFileSync(existing, "utf-8"));
    console.log(`\n  Loaded ${generatedTexts.length} generated texts from checkpoint.`);
  }

  // ─── Phase 2: LLM Self-Rating ─────────────────────────────────
  let llmRatings: Array<{ llmRating: number; llmConfidence: number }>;

  if (phase === "all" || phase === "llm-rate") {
    console.log("\n── PHASE 2: LLM Self-Rating ─────────────────────────");
    llmRatings = await phase2LLMRate(anthropic!, generatedTexts);

    if (phase === "llm-rate") {
      console.log("\n  Phase 2 complete. Run with 'ssr-rate' next.");
      return;
    }
  } else if (phase !== "metrics") {
    const existing = findLatestFile("llm-selfratings-");
    if (!existing) {
      console.error("ERROR: No LLM ratings found. Run 'llm-rate' phase first.");
      process.exit(1);
    }
    llmRatings = JSON.parse(fs.readFileSync(existing, "utf-8"));
    console.log(`  Loaded ${llmRatings.length} LLM ratings from checkpoint.`);
  } else {
    const existing = findLatestFile("llm-selfratings-");
    if (!existing) {
      console.error("ERROR: No LLM ratings found. Run 'llm-rate' phase first.");
      process.exit(1);
    }
    llmRatings = JSON.parse(fs.readFileSync(existing, "utf-8"));
  }

  // ─── Phase 3: SSR Rating ──────────────────────────────────────
  let ssrRatings: Array<{ ssrRating: number; ssrConfidence: number; ssrDistribution: number[]; ssrRawSimilarities: number[] }>;

  if (phase === "all" || phase === "ssr-rate") {
    console.log("\n── PHASE 3: SSR Rating ──────────────────────────────");
    ssrRatings = await phase3SSRRate(generatedTexts);

    if (phase === "ssr-rate") {
      console.log("\n  Phase 3 complete. Run with 'metrics' next.");
      return;
    }
  } else {
    const existing = findLatestFile("ssr-selfratings-");
    if (!existing) {
      console.error("ERROR: No SSR ratings found. Run 'ssr-rate' phase first.");
      process.exit(1);
    }
    ssrRatings = JSON.parse(fs.readFileSync(existing, "utf-8"));
  }

  // ─── Phase 4: Compute Metrics ─────────────────────────────────
  console.log("\n── PHASE 4: Compute Metrics ─────────────────────────");
  const output = phase4Metrics(generatedTexts, llmRatings, ssrRatings);

  // Save full results
  const outputPath = path.join(getDataDir(), `circularity-results-${Date.now()}.json`);
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`  Results saved to: ${path.basename(outputPath)}`);

  // Print summary
  printResults(output);
}

main().catch(console.error);
