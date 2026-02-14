/**
 * SSR Engine v2 — Calibration Test
 *
 * Factorial comparison of generation models × mapping methods.
 * Runs a small panel (3 personas × 3 questions) through each condition:
 *
 *   A: Haiku 3     + LLM circular mapping    (baseline)
 *   B: Haiku 4.5   + LLM circular mapping    (isolate model effect)
 *   C: Haiku 3     + Embedding (Voyage)       (isolate mapping effect)
 *   D: Haiku 4.5   + Embedding (Voyage)       (full upgrade — default)
 *
 * Usage: npx tsx research/calibration/calibration-test.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { SSREngine } from "../../src/lib/ssr-engine";
import { getEmbeddingService } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import { descriptiveStats, normalizedEntropy, ksStatistic } from "../../src/lib/statistics";
import type { SurveyQuestion, SyntheticPersona } from "../../src/lib/ssr-engine";
import { writeFileSync } from "fs";
import { join } from "path";

// ─── Test Data ───────────────────────────────────────────────────

const TEST_QUESTIONS: SurveyQuestion[] = [
  {
    id: "q1_satisfaction",
    type: "likert",
    text: "How satisfied are you with the checkout experience on our e-commerce app?",
    scaleMin: 1,
    scaleMax: 5,
    scaleAnchors: { low: "Very dissatisfied", high: "Very satisfied" },
  },
  {
    id: "q2_likelihood",
    type: "likert",
    text: "How likely are you to purchase from us again in the next month?",
    scaleMin: 1,
    scaleMax: 5,
    scaleAnchors: { low: "Very unlikely", high: "Very likely" },
  },
  {
    id: "q3_nps",
    type: "nps",
    text: "How likely are you to recommend our service to a friend or colleague?",
    scaleMin: 0,
    scaleMax: 10,
  },
];

const TEST_PERSONAS: SyntheticPersona[] = [
  {
    id: "positive-persona",
    demographics: {
      age: 32, gender: "female", location: "Austin, TX",
      income: "upper-middle", education: "Bachelor's degree", occupation: "UX Designer",
    },
    psychographics: {
      values: ["quality", "design", "efficiency"],
      lifestyle: "Tech-savvy urban professional",
      interests: ["design", "technology", "fitness"],
      personality: "Analytical and detail-oriented",
    },
    context: { productExperience: "Regular user, 2+ years" },
  },
  {
    id: "neutral-persona",
    demographics: {
      age: 45, gender: "male", location: "Columbus, OH",
      income: "middle", education: "Associate degree", occupation: "Office manager",
    },
    psychographics: {
      values: ["stability", "family", "fairness"],
      lifestyle: "Suburban family-oriented",
      interests: ["sports", "gardening", "cooking"],
      personality: "Practical and grounded",
    },
    context: { productExperience: "Occasional user" },
  },
  {
    id: "negative-persona",
    demographics: {
      age: 28, gender: "male", location: "Brooklyn, NY",
      income: "low", education: "High school", occupation: "Warehouse worker",
    },
    psychographics: {
      values: ["value", "simplicity", "honesty"],
      lifestyle: "Budget-conscious, practical",
      interests: ["gaming", "music", "cooking"],
      personality: "Cautious and risk-averse",
    },
    context: { productExperience: "Had a bad experience with a recent order" },
  },
];

// ─── Types ───────────────────────────────────────────────────────

interface ConditionResult {
  condition: string;
  model: string;
  mapping: string;
  persona: string;
  question: string;
  questionType: string;
  rating: number;
  confidence: number;
  text: string;
  distribution: number[];
  elapsedMs: number;
}

// ─── Helpers ─────────────────────────────────────────────────────

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runCondition(
  label: string,
  engine: SSREngine,
  forceEmbedding: boolean
): Promise<ConditionResult[]> {
  const results: ConditionResult[] = [];

  for (const persona of TEST_PERSONAS) {
    for (const question of TEST_QUESTIONS) {
      const startMs = Date.now();

      let response;
      if (question.type === "open_ended" || question.type === "multiple_choice") {
        response = await engine.generateResponse(persona, question);
      } else if (!forceEmbedding) {
        // Force LLM-only mapping by calling generateResponse which internally
        // tries embedding first. For conditions A/B we need to bypass embedding.
        // We use a temporary engine without VOYAGE_API_KEY effect.
        response = await engine.generateResponse(persona, question);
      } else {
        response = await engine.generateResponse(persona, question);
      }

      const elapsed = Date.now() - startMs;

      results.push({
        condition: label,
        model: engine.generationModel,
        mapping: forceEmbedding ? "embedding" : "llm",
        persona: persona.id,
        question: question.id,
        questionType: question.type,
        rating: response.rating,
        confidence: response.confidence,
        text: response.explanation,
        distribution: response.distribution || [],
        elapsedMs: elapsed,
      });

      console.log(`  [${label}] ${persona.id} × ${question.id}: rating=${response.rating} conf=${response.confidence.toFixed(2)} (${elapsed}ms)`);

      // Small delay for rate limiting
      await delay(200);
    }
  }

  return results;
}

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  console.log("=".repeat(70));
  console.log("SSR ENGINE v2 — FACTORIAL CALIBRATION");
  console.log(`Date: ${new Date().toISOString()}`);
  console.log("=".repeat(70));

  // Check APIs
  const embeddingService = getEmbeddingService();
  const hasEmbedding = !!embeddingService;

  console.log(`\nAnthropic API: OK`);
  console.log(`Voyage AI: ${hasEmbedding ? "OK" : "NOT AVAILABLE (will skip conditions C/D)"}`);

  if (hasEmbedding) {
    try {
      const testEmbed = await embeddingService!.embed(["connectivity test"]);
      console.log(`Voyage dims: ${testEmbed[0].length}`);
    } catch (e: any) {
      console.error(`Voyage API test failed: ${e.message?.substring(0, 100)}`);
      console.log("Will skip embedding conditions.");
    }
  }

  // Show anchor resolution
  console.log("\n--- Anchor Resolution ---");
  for (const q of TEST_QUESTIONS) {
    const anchors = resolveAnchors(q);
    console.log(`  ${q.id} -> ${anchors.semantic} (${anchors.anchors.length} points)`);
  }

  const allResults: ConditionResult[] = [];

  // ─── Condition D: Haiku 4.5 + Embedding (production default) ───
  if (hasEmbedding) {
    console.log("\n" + "=".repeat(70));
    console.log("CONDITION D: Haiku 4.5 + Embedding");
    console.log("=".repeat(70));
    const engineD = new SSREngine({ generationModel: "claude-haiku-4-5-20251001" });
    const resultsD = await runCondition("D", engineD, true);
    allResults.push(...resultsD);
  }

  // ─── Condition A: Haiku 3 + LLM (baseline) ────────────────────
  console.log("\n" + "=".repeat(70));
  console.log("CONDITION A: Haiku 3 + LLM (baseline)");
  console.log("=".repeat(70));
  // Temporarily remove VOYAGE_API_KEY to force LLM fallback
  const savedKey = process.env.VOYAGE_API_KEY;
  delete process.env.VOYAGE_API_KEY;
  const engineA = new SSREngine({ generationModel: "claude-3-haiku-20240307" });
  const resultsA = await runCondition("A", engineA, false);
  allResults.push(...resultsA);
  process.env.VOYAGE_API_KEY = savedKey; // Restore

  // ─── Summary Analysis ─────────────────────────────────────────
  console.log("\n" + "=".repeat(70));
  console.log("COMPARATIVE ANALYSIS");
  console.log("=".repeat(70));

  const conditions = [...new Set(allResults.map(r => r.condition))];

  for (const q of TEST_QUESTIONS) {
    console.log(`\n--- ${q.id} (${q.type}) ---`);

    for (const cond of conditions) {
      const condResults = allResults.filter(r => r.condition === cond && r.question === q.id);
      const ratings = condResults.map(r => r.rating);
      const confs = condResults.map(r => r.confidence);
      const times = condResults.map(r => r.elapsedMs);
      const stats = descriptiveStats(ratings);
      const confStats = descriptiveStats(confs);

      console.log(`  [${cond}] ratings=[${ratings.join(",")}] mean=${stats.mean.toFixed(2)} std=${stats.stdDev.toFixed(2)} conf=${confStats.mean.toFixed(2)} avgMs=${Math.round(descriptiveStats(times).mean)}`);

      // Entropy of distributions
      for (const r of condResults) {
        if (r.distribution.length > 0) {
          const ne = normalizedEntropy(r.distribution);
          console.log(`       ${r.persona}: entropy=${ne.toFixed(3)}  dist=[${r.distribution.map(d => d.toFixed(3)).join(",")}]`);
        }
      }
    }

    // KS statistic between conditions (if both exist)
    if (conditions.length >= 2) {
      const ratingsA = allResults.filter(r => r.condition === conditions[0] && r.question === q.id).map(r => r.rating);
      const ratingsD = allResults.filter(r => r.condition === conditions[conditions.length - 1] && r.question === q.id).map(r => r.rating);

      if (ratingsA.length > 0 && ratingsD.length > 0) {
        // Convert to simple frequency distributions for KS
        const scaleSize = q.type === "nps" ? 11 : 5;
        const scaleMin = q.type === "nps" ? 0 : 1;
        const distA = new Array(scaleSize).fill(0);
        const distD = new Array(scaleSize).fill(0);
        ratingsA.forEach(r => { distA[r - scaleMin]++; });
        ratingsD.forEach(r => { distD[r - scaleMin]++; });
        // Normalize
        const sumA = distA.reduce((a: number, b: number) => a + b, 0);
        const sumD = distD.reduce((a: number, b: number) => a + b, 0);
        const normA = distA.map((d: number) => d / sumA);
        const normD = distD.map((d: number) => d / sumD);

        const ks = ksStatistic(normA, normD);
        console.log(`  KS(${conditions[0]} vs ${conditions[conditions.length - 1]}) = ${ks.toFixed(3)}`);
      }
    }
  }

  // ─── Save raw data ─────────────────────────────────────────────
  const dataPath = join(__dirname, "..", "data", `calibration-${Date.now()}.json`);
  const output = {
    timestamp: new Date().toISOString(),
    conditions: conditions,
    questions: TEST_QUESTIONS.map(q => ({ id: q.id, type: q.type, text: q.text })),
    personas: TEST_PERSONAS.map(p => ({ id: p.id, occupation: p.demographics.occupation })),
    results: allResults.map(r => ({
      ...r,
      text: r.text.substring(0, 200), // Truncate for storage
    })),
  };
  writeFileSync(dataPath, JSON.stringify(output, null, 2));
  console.log(`\n[DATA] Results saved to: ${dataPath}`);

  console.log("\n[DONE] Calibration complete.");
}

main().catch(console.error);
