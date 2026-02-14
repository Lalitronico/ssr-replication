/**
 * Voyage AI Embedding Service for SSR Engine v2.
 *
 * Provides embedding-based Likert mapping that replaces the circular
 * LLM-based approach. Uses cosine similarity between response embeddings
 * and scale anchor embeddings to produce ratings and distributions.
 *
 * - Model: voyage-3.5-lite ($0.02/1M tokens, 1024 dimensions)
 * - API: native fetch, no npm dependency
 * - Batching: up to 128 texts per request
 * - Caching: anchor embeddings cached in-memory across warm invocations
 */

import { normalizedEntropy } from "./statistics";

// ─── Types ───────────────────────────────────────────────────────

export type SimilarityNormalization = "none" | "minmax" | "zscore";

export interface EmbeddingMappingResult {
  rating: number;
  distribution: number[];
  confidence: number;
  rawSimilarities?: number[];
  mappingMethod: "embedding-v1";
}

interface VoyageResponse {
  data: Array<{ embedding: number[] }>;
  usage: { total_tokens: number };
}

// ─── Constants ───────────────────────────────────────────────────

const VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings";
const VOYAGE_MODEL = "voyage-3.5-lite";
const MAX_BATCH_SIZE = 128;

// ─── Embedding Service ──────────────────────────────────────────

export class VoyageEmbeddingService {
  private apiKey: string;
  private anchorCache = new Map<string, number[][]>();

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  /**
   * Embed one or more texts via Voyage API.
   * Handles batching for arrays larger than 128.
   * @param inputType - "document" for anchors/reference texts, "query" for response texts (asymmetric retrieval)
   */
  async embed(texts: string[], inputType: "document" | "query" = "document"): Promise<number[][]> {
    if (texts.length === 0) return [];

    const allEmbeddings: number[][] = [];

    // Process in batches of MAX_BATCH_SIZE
    for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
      const batch = texts.slice(i, i + MAX_BATCH_SIZE);
      const data = await this.fetchWithRetry(batch, 3, inputType);
      allEmbeddings.push(...data.data.map((d) => d.embedding));
    }

    return allEmbeddings;
  }

  /**
   * Fetch with retry + exponential backoff for 429 rate limits.
   * Waits 22s on first 429 (respects ~3 RPM), then doubles.
   */
  private async fetchWithRetry(
    batch: string[],
    maxRetries: number = 3,
    inputType: "document" | "query" = "document"
  ): Promise<VoyageResponse> {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const response = await fetch(VOYAGE_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: VOYAGE_MODEL,
          input: batch,
          input_type: inputType,
        }),
      });

      if (response.ok) {
        return (await response.json()) as VoyageResponse;
      }

      if (response.status === 429 && attempt < maxRetries) {
        const waitMs = 22000 * Math.pow(2, attempt); // 22s, 44s, 88s
        console.log(`Voyage 429 rate limited, waiting ${Math.round(waitMs / 1000)}s (attempt ${attempt + 1}/${maxRetries})...`);
        await new Promise(resolve => setTimeout(resolve, waitMs));
        continue;
      }

      const errorText = await response.text();
      throw new Error(`Voyage API error ${response.status}: ${errorText}`);
    }
    throw new Error("Voyage API: max retries exceeded");
  }

  /**
   * Cosine similarity between two vectors.
   * dot(a,b) / (||a|| * ||b||)
   */
  cosineSimilarity(a: number[], b: number[]): number {
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

  /**
   * Compute cosine similarities between a response vector and each anchor vector.
   */
  computeSimilarities(
    responseVec: number[],
    anchorVecs: number[][]
  ): number[] {
    return anchorVecs.map((anchor) => this.cosineSimilarity(responseVec, anchor));
  }

  /**
   * Normalize raw cosine similarities before softmax.
   *
   * - "none": use raw similarities (requires very low τ due to compressed range)
   * - "minmax": stretch [min, max] → [0, 1] — recommended for embedding-based mapping
   * - "zscore": center and scale by standard deviation
   */
  normalizeSimilarities(
    similarities: number[],
    method: SimilarityNormalization = "none"
  ): number[] {
    if (method === "none" || similarities.length <= 1) return [...similarities];

    if (method === "minmax") {
      const min = Math.min(...similarities);
      const max = Math.max(...similarities);
      const range = max - min;
      if (range === 0) return similarities.map(() => 1 / similarities.length);
      return similarities.map((s) => (s - min) / range);
    }

    if (method === "zscore") {
      const mean = similarities.reduce((a, b) => a + b, 0) / similarities.length;
      const variance =
        similarities.reduce((sum, s) => sum + (s - mean) ** 2, 0) /
        similarities.length;
      const std = Math.sqrt(variance);
      if (std === 0) return similarities.map(() => 1 / similarities.length);
      return similarities.map((s) => (s - mean) / std);
    }

    return [...similarities];
  }

  /**
   * Convert cosine similarities to a probability distribution via softmax.
   *
   * @param similarities - Raw cosine similarity scores
   * @param temperature - Controls peakedness (lower = more decisive, higher = more spread)
   * @param normalization - Normalize similarities before softmax (default: "minmax")
   */
  similaritiesToDistribution(
    similarities: number[],
    temperature: number = 0.2,
    normalization: SimilarityNormalization = "minmax"
  ): number[] {
    // Normalize first
    const normalized = this.normalizeSimilarities(similarities, normalization);

    // Apply temperature scaling
    const scaled = normalized.map((s) => s / temperature);

    // Softmax with numerical stability (subtract max)
    const maxVal = Math.max(...scaled);
    const exps = scaled.map((s) => Math.exp(s - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);

    return exps.map((e) => e / sumExps);
  }

  /**
   * Convert a probability distribution to a single rating (weighted mean, rounded).
   * @param distribution - Probability distribution (sums to ~1)
   * @param scaleMin - Minimum scale value (e.g. 1 for Likert, 0 for NPS)
   */
  distributionToRating(distribution: number[], scaleMin: number): number {
    let weightedSum = 0;
    for (let i = 0; i < distribution.length; i++) {
      weightedSum += distribution[i] * (scaleMin + i);
    }
    return Math.round(weightedSum);
  }

  /**
   * Get or compute anchor embeddings with in-memory caching.
   * Key should be unique per scale type + size (e.g. "satisfaction-5").
   */
  async getOrComputeAnchorEmbeddings(
    cacheKey: string,
    anchorTexts: string[]
  ): Promise<number[][]> {
    const cached = this.anchorCache.get(cacheKey);
    if (cached) return cached;

    const embeddings = await this.embed(anchorTexts, "document");
    this.anchorCache.set(cacheKey, embeddings);
    return embeddings;
  }

  /**
   * Full pipeline: map a text response to a scale rating using embeddings.
   *
   * Uses asymmetric embedding: anchors as "document", response as "query".
   * This leverages Voyage's retrieval optimization for better differentiation.
   *
   * @param text - The generated text response
   * @param anchorTexts - Anchor statements for each scale point
   * @param anchorCacheKey - Cache key for anchor embeddings
   * @param scaleMin - Minimum scale value
   * @param scaleMax - Maximum scale value
   * @param temperature - Softmax temperature (default 0.2, used with minmax normalization)
   * @param normalization - Similarity normalization method (default "minmax")
   */
  async mapResponseToScale(
    text: string,
    anchorTexts: string[],
    anchorCacheKey: string,
    scaleMin: number,
    scaleMax: number,
    temperature: number = 0.2,
    normalization: SimilarityNormalization = "minmax"
  ): Promise<EmbeddingMappingResult> {
    // Get anchor embeddings (cached after first call for this scale type)
    const anchorVecs = await this.getOrComputeAnchorEmbeddings(
      anchorCacheKey,
      anchorTexts
    );

    // Embed the response text (as "query" for asymmetric retrieval)
    const [responseVec] = await this.embed([text], "query");

    // Compute raw similarities
    const similarities = this.computeSimilarities(responseVec, anchorVecs);

    // Convert to distribution (normalize + softmax)
    const distribution = this.similaritiesToDistribution(
      similarities,
      temperature,
      normalization
    );

    // Compute rating
    const rating = this.distributionToRating(distribution, scaleMin);

    // Clamp to valid range
    const clampedRating = Math.min(Math.max(rating, scaleMin), scaleMax);

    // Confidence: 1 - normalized entropy (delta dist → 1.0, uniform → 0.0)
    const confidence = Math.round((1 - normalizedEntropy(distribution)) * 100) / 100;

    return {
      rating: clampedRating,
      distribution,
      confidence,
      rawSimilarities: similarities,
      mappingMethod: "embedding-v1",
    };
  }
}

// ─── Singleton ───────────────────────────────────────────────────

let embeddingService: VoyageEmbeddingService | null = null;

/**
 * Get the singleton embedding service instance.
 * Returns null if VOYAGE_API_KEY is not set (enables graceful fallback to LLM mapping).
 */
export function getEmbeddingService(): VoyageEmbeddingService | null {
  if (embeddingService) return embeddingService;

  const apiKey = process.env.VOYAGE_API_KEY;
  if (!apiKey) return null;

  embeddingService = new VoyageEmbeddingService(apiKey);
  return embeddingService;
}
