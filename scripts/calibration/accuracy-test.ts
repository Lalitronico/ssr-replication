/**
 * Accuracy test: compare embedding ratings vs expected ground truth.
 * Tests both the old (raw softmax) and new (min-max normalized) approaches.
 *
 * Usage: npx tsx research/calibration/accuracy-test.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import type { SurveyQuestion } from "../../src/lib/ssr-engine";

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

// ─── Ground truth responses ──────────────────────────────────────

interface TestCase {
  question: SurveyQuestion;
  text: string;
  expected: number;  // 1-5
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
  { question: LIKELIHOOD_Q, expected: 3, label: "LIK-3",
    text: "Maybe. It depends on whether they have what I need and if the price is right. I don't have strong feelings either way." },
  { question: LIKELIHOOD_Q, expected: 5, label: "LIK-5",
    text: "Definitely! I already have items in my wishlist. Great selection and prices, I'll be back within the week." },
];

// ─── Approaches to compare ───────────────────────────────────────

const APPROACHES = [
  { name: "raw τ=0.01",    normalization: "none" as const,   temperature: 0.01 },
  { name: "minmax τ=0.5",  normalization: "minmax" as const, temperature: 0.5 },
  { name: "minmax τ=0.2",  normalization: "minmax" as const, temperature: 0.2 },
  { name: "minmax τ=0.1",  normalization: "minmax" as const, temperature: 0.1 },
  { name: "minmax τ=0.05", normalization: "minmax" as const, temperature: 0.05 },
];

// ─── Main ────────────────────────────────────────────────────────

async function main() {
  const service = getEmbeddingService();
  if (!service) { console.error("VOYAGE_API_KEY not set"); process.exit(1); }

  console.log("=".repeat(80));
  console.log("EMBEDDING ACCURACY TEST — Raw vs Min-Max Normalization");
  console.log("=".repeat(80));

  // Pre-embed all anchors
  const satAnchors = resolveAnchors(SATISFACTION_Q);
  const likAnchors = resolveAnchors(LIKELIHOOD_Q);

  console.log("\nEmbedding anchors...");
  const satVecs = await service.embed(satAnchors.anchors);
  await new Promise(r => setTimeout(r, 1500));
  const likVecs = await service.embed(likAnchors.anchors);

  const anchorMap: Record<string, { vecs: number[][]; anchors: typeof satAnchors }> = {
    sat: { vecs: satVecs, anchors: satAnchors },
    lik: { vecs: likVecs, anchors: likAnchors },
  };

  // Build header
  const header = `${"Label".padEnd(8)} ${"Exp".padEnd(5)} ` +
    APPROACHES.map(a => a.name.padEnd(15)).join(" ") + " Best Match";

  console.log("\n" + "─".repeat(header.length));
  console.log(header);
  console.log("─".repeat(header.length));

  // Per-approach accumulators
  const stats = APPROACHES.map(() => ({ exact: 0, withinOne: 0, errors: [] as number[] }));

  for (const tc of TEST_CASES) {
    await new Promise(r => setTimeout(r, 1500));
    const [respVec] = await service.embed([tc.text]);

    const { vecs, anchors } = anchorMap[tc.question.id];
    const sims = service.computeSimilarities(respVec, vecs);

    const ratings: number[] = [];
    for (let i = 0; i < APPROACHES.length; i++) {
      const a = APPROACHES[i];
      const dist = service.similaritiesToDistribution(sims, a.temperature, a.normalization);
      const rating = service.distributionToRating(dist, anchors.scaleMin);
      const clamped = Math.min(Math.max(rating, anchors.scaleMin), anchors.scaleMax);
      ratings.push(clamped);

      const err = Math.abs(clamped - tc.expected);
      stats[i].errors.push(err);
      if (err === 0) stats[i].exact++;
      if (err <= 1) stats[i].withinOne++;
    }

    // Find best approach for this case
    const bestIdx = ratings.reduce((bi, r, i) => {
      const errI = Math.abs(r - tc.expected);
      const errBest = Math.abs(ratings[bi] - tc.expected);
      return errI < errBest ? i : bi;
    }, 0);
    const bestErr = Math.abs(ratings[bestIdx] - tc.expected);
    const bestMatch = bestErr === 0 ? "exact" : bestErr <= 1 ? "±1" : `ERR(${bestErr})`;

    const ratingsStr = ratings.map((r, i) => {
      const err = Math.abs(r - tc.expected);
      const sym = err === 0 ? "✓" : err <= 1 ? "~" : "✗";
      return `${r} ${sym}`.padEnd(15);
    }).join(" ");

    console.log(`${tc.label.padEnd(8)} ${tc.expected.toString().padEnd(5)} ${ratingsStr} ${bestMatch}`);
  }

  // Summary
  console.log("\n" + "=".repeat(80));
  console.log("SUMMARY");
  console.log("=".repeat(80));

  console.log(`\n${"Approach".padEnd(18)} ${"Exact".padEnd(15)} ${"Within±1".padEnd(15)} ${"MAE".padEnd(10)}`);
  console.log("─".repeat(58));

  for (let i = 0; i < APPROACHES.length; i++) {
    const s = stats[i];
    const mae = s.errors.reduce((a, b) => a + b, 0) / s.errors.length;
    const exactPct = `${s.exact}/${TEST_CASES.length} (${Math.round(s.exact / TEST_CASES.length * 100)}%)`;
    const w1Pct = `${s.withinOne}/${TEST_CASES.length} (${Math.round(s.withinOne / TEST_CASES.length * 100)}%)`;
    console.log(`${APPROACHES[i].name.padEnd(18)} ${exactPct.padEnd(15)} ${w1Pct.padEnd(15)} ${mae.toFixed(2)}`);
  }
}

main().catch(console.error);
