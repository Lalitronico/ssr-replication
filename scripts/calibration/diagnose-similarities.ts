/**
 * Diagnostic: inspect raw cosine similarities between response text and anchors.
 * Shows the effect of different normalization + temperature combinations.
 *
 * Usage: npx tsx research/calibration/diagnose-similarities.ts
 */

import { config } from "dotenv";
config({ path: ".env.local" });

import { getEmbeddingService } from "../../src/lib/embedding-service";
import { resolveAnchors } from "../../src/lib/scale-anchors";
import type { SurveyQuestion } from "../../src/lib/ssr-engine";

const QUESTION: SurveyQuestion = {
  id: "q1",
  type: "likert",
  text: "How satisfied are you with the checkout experience?",
  scaleMin: 1,
  scaleMax: 5,
  scaleAnchors: { low: "Very dissatisfied", high: "Very satisfied" },
};

// Responses with clear sentiment for validation
const TEST_RESPONSES = [
  { label: "VERY NEGATIVE", text: "Terrible. The checkout kept crashing and I lost my cart twice. Never again." },
  { label: "NEGATIVE",      text: "Not great. It was confusing and took way too long to complete." },
  { label: "NEUTRAL",       text: "It was fine. Nothing special but it worked." },
  { label: "POSITIVE",      text: "Pretty smooth experience, found what I needed quickly." },
  { label: "VERY POSITIVE", text: "Loved it! Super fast checkout, everything was intuitive and easy." },
];

async function main() {
  const service = getEmbeddingService();
  if (!service) {
    console.error("VOYAGE_API_KEY not set");
    process.exit(1);
  }

  const anchors = resolveAnchors(QUESTION);
  console.log("Anchors:");
  anchors.anchors.forEach((a, i) => console.log(`  [${i + 1}] ${a}`));

  // Embed anchors
  console.log("\nEmbedding anchors...");
  const anchorVecs = await service.embed(anchors.anchors);
  console.log(`  Done. ${anchorVecs.length} vectors × ${anchorVecs[0].length} dims`);

  // Embed responses one at a time (respecting rate limits)
  for (const resp of TEST_RESPONSES) {
    console.log(`\n--- ${resp.label}: "${resp.text}" ---`);

    await new Promise(r => setTimeout(r, 1500)); // rate limit buffer
    const [respVec] = await service.embed([resp.text]);

    const sims = service.computeSimilarities(respVec, anchorVecs);
    console.log(`  Raw cosine sims: [${sims.map(s => s.toFixed(4)).join(", ")}]`);
    console.log(`  Range: ${Math.min(...sims).toFixed(4)} – ${Math.max(...sims).toFixed(4)}  (spread: ${(Math.max(...sims) - Math.min(...sims)).toFixed(4)})`);

    // Show raw softmax at different temperatures
    console.log("\n  [Raw (no normalization)]");
    for (const tau of [1.0, 0.1, 0.05, 0.01]) {
      const dist = service.similaritiesToDistribution(sims, tau, "none");
      const rating = service.distributionToRating(dist, 1);
      const maxP = Math.max(...dist);
      console.log(`  τ=${tau.toFixed(2)} → dist=[${dist.map(d => d.toFixed(3)).join(",")}] rating=${rating} peak=${maxP.toFixed(3)}`);
    }

    // Show min-max normalized softmax
    console.log("\n  [Min-max normalization]");
    const normed = service.normalizeSimilarities(sims, "minmax");
    console.log(`  Normalized sims: [${normed.map(s => s.toFixed(4)).join(", ")}]`);
    for (const tau of [1.0, 0.5, 0.2, 0.1, 0.05]) {
      const dist = service.similaritiesToDistribution(sims, tau, "minmax");
      const rating = service.distributionToRating(dist, 1);
      const maxP = Math.max(...dist);
      console.log(`  τ=${tau.toFixed(2)} → dist=[${dist.map(d => d.toFixed(3)).join(",")}] rating=${rating} peak=${maxP.toFixed(3)}`);
    }
  }
}

main().catch(console.error);
