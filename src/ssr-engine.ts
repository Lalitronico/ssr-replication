/**
 * SSR Engine - Semantic Similarity Rating
 *
 * Implementation based on the methodology from arXiv 2510.08338
 * This engine generates synthetic survey responses using LLMs
 * and maps them to Likert scales using embedding similarity.
 */

import Anthropic from "@anthropic-ai/sdk";
import { urlToBase64 } from "./storage";
import { getEmbeddingService, type SimilarityNormalization } from "./embedding-service";
import { resolveAnchors, getScaleContextString } from "./scale-anchors";

// Configuration for timeouts and parallelization
const ANTHROPIC_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes for long simulations
const PARALLEL_PERSONAS = 4; // Process 4 personas in parallel (respects 40 req/min with ~2 calls per persona)

// Types
export interface SyntheticPersona {
  id: string;
  demographics: {
    age: number;
    gender: "male" | "female" | "non-binary" | "other";
    location: string;
    income: string;
    education: string;
    occupation: string;
  };
  psychographics: {
    values: string[];
    lifestyle: string;
    interests: string[];
    personality: string;
  };
  context: {
    industry?: string;
    role?: string;
    productExperience?: string;
    brandAffinity?: string[];
  };
}

// Condition for showing/hiding questions based on previous responses
export interface QuestionCondition {
  questionId: string;    // ID of the reference question
  operator: "equals" | "notEquals" | "greaterThan" | "lessThan" | "contains";
  value: string | number;
}

// Base survey question interface
export interface SurveyQuestion {
  id: string;
  type: "likert" | "nps" | "multiple_choice" | "ranking" | "open_ended" | "matrix" | "slider" | "image_rating" | "image_choice" | "image_comparison";
  text: string;
  options?: string[];
  scaleMin?: number;
  scaleMax?: number;
  scaleAnchors?: {
    low: string;
    high: string;
    labels?: string[];
  };
  // Conditional logic - if undefined, question is always shown
  showIf?: QuestionCondition;
  // Matrix question specific fields
  items?: string[];        // Items to rate: ["Price", "Quality", "Service"]
  scaleLabels?: string[];  // Labels for each scale point
  // Slider question specific fields
  min?: number;            // Minimum value (e.g., 0)
  max?: number;            // Maximum value (e.g., 100)
  step?: number;           // Step increment (e.g., 1)
  leftLabel?: string;      // Label for left/min end
  rightLabel?: string;     // Label for right/max end
  // Image question specific fields
  imageUrl?: string;       // Single image URL (for image_rating, image_choice)
  imageUrls?: string[];    // Multiple image URLs (for image_comparison)
  imageLabels?: string[];  // Labels for each image in comparison
  imagePrompt?: string;    // Custom evaluation prompt
  imageScaleMin?: number;  // Rating scale minimum (for image_rating)
  imageScaleMax?: number;  // Rating scale maximum (for image_rating)
  imageScaleLabels?: { low: string; high: string }; // Scale labels
}

// Matrix question response
export interface MatrixResponse {
  questionId: string;
  itemRatings: Record<string, number>;       // { "Price": 4, "Quality": 5 }
  itemExplanations: Record<string, string>;  // Explanations for each rating
  avgRating: number;
  confidence: number;
  rawTextResponse: string;
}

// Slider question response
export interface SliderResponse {
  questionId: string;
  rating: number;          // 0-100 (can be decimal)
  explanation: string;
  confidence: number;
  rawTextResponse: string;
  distribution?: number[]; // Histogram buckets
}

// Product/Service context for more relevant responses
export interface ProductContext {
  productName?: string;
  productDescription?: string;
  brandName?: string;
  industry?: string;
  productCategory?: string;
  customContextInstructions?: string;
}

export interface SSRResponse {
  questionId: string;
  rating: number;
  explanation: string;
  confidence: number;
  rawTextResponse: string;
  distribution?: number[];
  // Matrix question specific fields
  itemRatings?: Record<string, number>;
  itemExplanations?: Record<string, string>;
  avgRating?: number;
  // Image question specific fields
  visualAnalysis?: string;        // Description of what persona sees in image
  imagePreferences?: Record<string, number>; // Preference scores for comparison
  selectedImage?: string;         // Selected image label for comparison
  // Conditional logic - whether this question was skipped
  skipped?: boolean;
  skipReason?: string;
}

export interface SimulationResult {
  personaId: string;
  responses: SSRResponse[];
  metadata: {
    modelUsed: string;
    mappingMethod: string;
    embeddingModel: string;
    methodologyVersion: string;
    temperature: number;
    embeddingTemperature: number;
    similarityNormalization: string;
    timestamp: string;
    processingTimeMs: number;
  };
}

// Dead anchor constants removed in v2 — now in scale-anchors.ts

// Personality-based communication styles for natural response variation
const PERSONALITY_STYLES: Record<string, {
  tone: string;
  length: string;
  examplePhrases: string[];
}> = {
  "Analytical and detail-oriented": {
    tone: "logical, precise, and thorough",
    length: "2-3 sentences with specific details",
    examplePhrases: ["The main factor here is...", "Specifically, I noticed that...", "Comparing the options..."]
  },
  "Creative and imaginative": {
    tone: "expressive, colorful, and original",
    length: "2-3 sentences",
    examplePhrases: ["What really stands out is...", "It's refreshing to see...", "This feels like..."]
  },
  "Practical and grounded": {
    tone: "direct, no-nonsense, and utilitarian",
    length: "1-2 short sentences",
    examplePhrases: ["Works well.", "Gets the job done.", "Good value for money."]
  },
  "Outgoing and energetic": {
    tone: "friendly, upbeat, and social",
    length: "2-3 sentences",
    examplePhrases: ["It's nice that...", "My friends and I usually...", "That works for me because..."]
  },
  "Reserved and thoughtful": {
    tone: "measured, reflective, and considered",
    length: "1-2 sentences",
    examplePhrases: ["In my experience...", "I tend to prefer...", "After some thought..."]
  },
  "Optimistic and enthusiastic": {
    tone: "positive, hopeful, and warm",
    length: "2-3 sentences",
    examplePhrases: ["I appreciate that...", "One good thing is...", "It's nice that..."]
  },
  "Cautious and risk-averse": {
    tone: "careful, questioning, and hedged",
    length: "2-3 sentences",
    examplePhrases: ["I'm not entirely sure...", "It seems okay, but...", "I'd want to know more about..."]
  },
  "Spontaneous and adventurous": {
    tone: "casual, bold, and open",
    length: "1-2 sentences",
    examplePhrases: ["Why not?", "I'd try it.", "Sounds fun to me."]
  },
  "Organized and methodical": {
    tone: "structured, systematic, and clear",
    length: "2-3 sentences",
    examplePhrases: ["First of all...", "The key points are...", "To summarize..."]
  },
  "Flexible and adaptable": {
    tone: "balanced, open-minded, and moderate",
    length: "2-3 sentences",
    examplePhrases: ["It depends on...", "I can see both sides...", "Generally speaking..."]
  },
  "Ambitious and driven": {
    tone: "confident, goal-focused, and decisive",
    length: "2-3 sentences",
    examplePhrases: ["What matters most is...", "The bottom line is...", "I look for..."]
  },
  "Relaxed and easy-going": {
    tone: "laid-back, casual, and unbothered",
    length: "1-2 short sentences",
    examplePhrases: ["It's fine.", "No complaints.", "Works for me."]
  }
};

const DEFAULT_STYLE = {
  tone: "natural and conversational",
  length: "1-3 sentences",
  examplePhrases: [] as string[]
};

// Default model IDs
const DEFAULT_GENERATION_MODEL = "claude-haiku-4-5-20251001";
const LEGACY_GENERATION_MODEL = "claude-3-haiku-20240307";
const GENERATION_TEMPERATURE = 0.7;
const DEFAULT_EMBEDDING_TEMPERATURE = 0.2;
const DEFAULT_SIMILARITY_NORMALIZATION = "minmax" as const;
const METHODOLOGY_VERSION = "ssr-v2.0";

export interface SSREngineConfig {
  /** Generation model ID (default: claude-haiku-4-5-20251001, legacy: claude-3-haiku-20240307) */
  generationModel?: string;
  /** Softmax temperature for embedding-to-distribution mapping (default: 0.1) */
  embeddingTemperature?: number;
  /** Similarity normalization method (default: "minmax") */
  similarityNormalization?: SimilarityNormalization;
}

export class SSREngine {
  private anthropic: Anthropic;
  private requestCount: number = 0;
  private lastResetTime: number = Date.now();
  private readonly MAX_REQUESTS_PER_MINUTE = 40; // Stay under 50 limit
  private requestQueue: Promise<void> = Promise.resolve();
  private queueLock: boolean = false;
  readonly generationModel: string;
  readonly embeddingTemperature: number;
  readonly similarityNormalization: SimilarityNormalization;

  constructor(config?: SSREngineConfig) {
    this.generationModel = config?.generationModel || DEFAULT_GENERATION_MODEL;
    this.embeddingTemperature = config?.embeddingTemperature || DEFAULT_EMBEDDING_TEMPERATURE;
    this.similarityNormalization = config?.similarityNormalization || DEFAULT_SIMILARITY_NORMALIZATION;
    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      timeout: ANTHROPIC_TIMEOUT_MS, // 30 minute timeout for long simulations
    });
  }

  /**
   * Wait if we're approaching rate limit
   */
  private async checkRateLimit(): Promise<void> {
    const now = Date.now();
    const elapsed = now - this.lastResetTime;

    // Reset counter every minute
    if (elapsed >= 60000) {
      this.requestCount = 0;
      this.lastResetTime = now;
    }

    // If we're at the limit, wait until the next minute
    if (this.requestCount >= this.MAX_REQUESTS_PER_MINUTE) {
      const waitTime = 60000 - elapsed + 1000; // Wait until reset + 1 second buffer
      console.log(`Rate limit approaching, waiting ${waitTime}ms...`);
      await this.delay(waitTime);
      this.requestCount = 0;
      this.lastResetTime = Date.now();
    }

    this.requestCount++;
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get communication style based on personality type
   */
  private getPersonalityStyle(personality: string): typeof DEFAULT_STYLE {
    return PERSONALITY_STYLES[personality] || DEFAULT_STYLE;
  }

  /**
   * Make API call with retry logic
   */
  private async callWithRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3
  ): Promise<T> {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        await this.checkRateLimit();
        return await fn();
      } catch (error: unknown) {
        const isRateLimit = error instanceof Error &&
          (error.message.includes('429') || error.message.includes('rate_limit'));

        if (isRateLimit && attempt < maxRetries) {
          const waitTime = Math.pow(2, attempt) * 5000; // 5s, 10s, 20s
          console.log(`Rate limited, retrying in ${waitTime}ms (attempt ${attempt + 1}/${maxRetries})...`);
          await this.delay(waitTime);
          this.requestCount = 0; // Reset counter after waiting
          continue;
        }
        throw error;
      }
    }
    throw new Error('Max retries exceeded');
  }

  /**
   * Build a detailed persona prompt for the LLM
   */
  private buildPersonaPrompt(persona: SyntheticPersona, productContext?: ProductContext): string {
    const { demographics, psychographics, context } = persona;

    let prompt = `You are a survey respondent with the following characteristics:

DEMOGRAPHICS:
- Age: ${demographics.age} years old
- Gender: ${demographics.gender}
- Location: ${demographics.location}
- Income level: ${demographics.income}
- Education: ${demographics.education}
- Occupation: ${demographics.occupation}

PSYCHOGRAPHICS:
- Core values: ${psychographics.values.join(", ")}
- Lifestyle: ${psychographics.lifestyle}
- Interests: ${psychographics.interests.join(", ")}
- Personality: ${psychographics.personality}

${context.industry ? `CONTEXT:\n- Industry: ${context.industry}` : ""}
${context.role ? `- Role: ${context.role}` : ""}
${context.productExperience ? `- Product Experience: ${context.productExperience}` : ""}
${context.brandAffinity?.length ? `- Brand preferences: ${context.brandAffinity.join(", ")}` : ""}`;

    // Add product/service context if available
    if (productContext && (productContext.productName || productContext.productDescription)) {
      prompt += `\n\nPRODUCT/SERVICE BEING EVALUATED:`;
      if (productContext.brandName) {
        prompt += `\n- Brand: ${productContext.brandName}`;
      }
      if (productContext.productName) {
        prompt += `\n- Product/Service: ${productContext.productName}`;
      }
      if (productContext.industry) {
        prompt += `\n- Industry: ${productContext.industry}`;
      }
      if (productContext.productCategory) {
        prompt += `\n- Category: ${productContext.productCategory}`;
      }
      if (productContext.productDescription) {
        prompt += `\n- Description: ${productContext.productDescription}`;
      }
    }

    // Get personality-specific style
    const style = this.getPersonalityStyle(psychographics.personality);

    prompt += `\n\nRESPONSE GUIDELINES:
Your communication style is ${style.tone}. Keep responses ${style.length}.

CRITICAL - DO NOT:
- Start with "As a [age]-year-old..." or "Being a [occupation]..."
- Mention your age, gender, income, or location in responses
- Use the phrase "As someone who..."
- Give generic responses like "I think it's good/bad"
- Repeat information from your profile - it's context, not content
- Start with filler words: Hmm, Oh, Ooh, Well, Ah, Eh, So, Yeah, Honestly, Actually, Look
- Claim roles, expertise, or professional identities that contradict your education/occupation (e.g., if you have no formal education, don't call yourself "a busy professional" or "an expert")
- Use exaggerated marketing language: "absolutely love", "game-changer", "must-have", "incredible", "amazing", "fantastic", "blown away"
- Use vocabulary or jargon that doesn't match your education level

DO:
- Answer directly as if someone asked you in casual conversation
- Use specific details from your actual experience/perspective
- Show your personality through word choice and tone, not self-description
- It's fine to be brief, uncertain, or have mixed feelings
- Match your vocabulary to your education level and occupation
${style.examplePhrases.length > 0 ? `- Example phrases that fit your style: "${style.examplePhrases.join('", "')}"` : ''}`;

    // Add custom context instructions if provided
    if (productContext?.customContextInstructions) {
      prompt += `\n\nADDITIONAL CONTEXT:\n${productContext.customContextInstructions}`;
    }

    return prompt;
  }

  /**
   * Strip filler-word prefixes that the model sometimes generates despite instructions.
   * E.g. "Hmm, I think..." → "I think..."
   */
  private stripFillerPrefix(text: string): string {
    return text.replace(
      /^(Hmm|Oh|Ooh|Well|Ah|Eh|So|Yeah|Honestly|Actually|Look)[,.\s!]+/i,
      ""
    ).trim();
  }

  /**
   * Generate a text response from the LLM
   */
  private async generateTextResponse(
    persona: SyntheticPersona,
    question: SurveyQuestion,
    productContext?: ProductContext
  ): Promise<string> {
    const personaPrompt = this.buildPersonaPrompt(persona, productContext);
    const style = this.getPersonalityStyle(persona.psychographics.personality);

    // Add scale context for Likert/NPS so text aligns with scale semantics
    const scaleContext =
      question.type === "likert" || question.type === "nps"
        ? `\n${getScaleContextString(question)}`
        : "";

    const response = await this.callWithRetry(async () => {
      return this.anthropic.messages.create({
        model: this.generationModel,
        max_tokens: 500,
        temperature: GENERATION_TEMPERATURE,
        system: personaPrompt,
        messages: [
          {
            role: "user",
            content: `Survey question: "${question.text}"${scaleContext}

Respond in ${style.length}. Be ${style.tone}. Give a direct answer — no filler words, no self-introduction. Keep it grounded in your actual life situation.`,
          },
        ],
      });
    });

    const textBlock = response.content.find((block) => block.type === "text");
    const raw = textBlock?.type === "text" ? textBlock.text : "";
    return this.stripFillerPrefix(raw);
  }

  /**
   * Map text response to Likert/NPS rating.
   *
   * v2: Uses embedding-based cosine similarity (independent measurement)
   * Fallback: LLM-based mapping if VOYAGE_API_KEY not set
   */
  async mapToLikert(
    textResponse: string,
    question: SurveyQuestion
  ): Promise<{ rating: number; distribution: number[]; confidence: number; mappingMethod: string }> {
    const scaleMax = question.type === "nps" ? 10 : (question.scaleMax || 5);
    const scaleMin = question.scaleMin || (question.type === "nps" ? 0 : 1);

    // Try embedding-based mapping first
    const embeddingService = getEmbeddingService();
    if (embeddingService) {
      try {
        const anchorSet = resolveAnchors(question);
        const cacheKey = `${anchorSet.semantic}-${anchorSet.anchors.length}`;

        const result = await embeddingService.mapResponseToScale(
          textResponse,
          anchorSet.anchors,
          cacheKey,
          scaleMin,
          scaleMax,
          this.embeddingTemperature,
          this.similarityNormalization
        );

        return result;
      } catch (error) {
        console.error("Embedding mapping failed, falling back to LLM:", error);
      }
    }

    // Fallback: LLM-based mapping (circular, but works without Voyage API key)
    return this.mapToLikertViaLLM(textResponse, question, scaleMin, scaleMax);
  }

  /**
   * Fallback LLM-based Likert mapping (v1 circular approach).
   * Used when VOYAGE_API_KEY is not set or embedding call fails.
   */
  private async mapToLikertViaLLM(
    textResponse: string,
    question: SurveyQuestion,
    scaleMin: number,
    scaleMax: number
  ): Promise<{ rating: number; distribution: number[]; confidence: number; mappingMethod: string }> {
    const prompt = `Analyze the following survey response and determine the appropriate rating.

Question: "${question.text}"

Response: "${textResponse}"

Scale: ${scaleMin} to ${scaleMax}
${question.type === "nps"
  ? "This is an NPS (Net Promoter Score) question. 0-6 = Detractor, 7-8 = Passive, 9-10 = Promoter."
  : `This is a Likert scale where ${scaleMin} = ${question.scaleAnchors?.low || "Strongly Disagree/Very Negative"} and ${scaleMax} = ${question.scaleAnchors?.high || "Strongly Agree/Very Positive"}.`}

IMPORTANT: If the response explicitly mentions a specific number (e.g. "I'd give it a 7", "maybe a 4"), use that number as the rating. Only infer from sentiment if no explicit number is stated.

Based on the sentiment and content of the response, provide:
1. A rating from ${scaleMin} to ${scaleMax}
2. Your confidence level (0.0 to 1.0)

Respond ONLY in this exact JSON format:
{"rating": <number>, "confidence": <number>}`;

    try {
      const response = await this.callWithRetry(async () => {
        return this.anthropic.messages.create({
          model: this.generationModel,
          max_tokens: 100,
          messages: [{ role: "user", content: prompt }],
        });
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      // Parse the JSON response
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const rating = Math.min(Math.max(parsed.rating, scaleMin), scaleMax);
        const confidence = Math.min(Math.max(parsed.confidence || 0.7, 0), 1);

        // Generate a simple distribution centered on the rating
        const distribution = this.generateDistribution(rating, scaleMin, scaleMax, confidence);

        return { rating, distribution, confidence, mappingMethod: "llm-fallback" };
      }
    } catch (error) {
      console.error("Error mapping to Likert via LLM:", error);
    }

    // Fallback: return middle rating
    const midRating = Math.round((scaleMin + scaleMax) / 2);
    return {
      rating: midRating,
      distribution: this.generateDistribution(midRating, scaleMin, scaleMax, 0.5),
      confidence: 0.5,
      mappingMethod: "llm-fallback",
    };
  }

  /**
   * Generate a probability distribution centered on a rating
   */
  private generateDistribution(
    rating: number,
    scaleMin: number,
    scaleMax: number,
    confidence: number
  ): number[] {
    const size = scaleMax - scaleMin + 1;
    const distribution = new Array(size).fill(0);
    const index = rating - scaleMin;

    // Higher confidence = more concentrated distribution
    const spread = 1 - confidence;

    for (let i = 0; i < size; i++) {
      const distance = Math.abs(i - index);
      distribution[i] = Math.exp(-distance * (2 + confidence * 3));
    }

    // Normalize
    const sum = distribution.reduce((a, b) => a + b, 0);
    return distribution.map(d => d / sum);
  }

  /**
   * Generate a complete response for a single question
   */
  async generateResponse(
    persona: SyntheticPersona,
    question: SurveyQuestion,
    productContext?: ProductContext
  ): Promise<SSRResponse> {
    // For open-ended questions, just return the text
    if (question.type === "open_ended") {
      const textResponse = await this.generateTextResponse(persona, question, productContext);
      return {
        questionId: question.id,
        rating: 0,
        explanation: textResponse,
        confidence: 1,
        rawTextResponse: textResponse,
      };
    }

    // For multiple choice, use direct selection
    if (question.type === "multiple_choice" && question.options) {
      const options = question.options; // TypeScript narrowing
      const response = await this.callWithRetry(async () => {
        return this.anthropic.messages.create({
          model: this.generationModel,
          max_tokens: 200,
          temperature: GENERATION_TEMPERATURE,
          system: this.buildPersonaPrompt(persona, productContext),
          messages: [
            {
              role: "user",
              content: `"${question.text}"

Options:
${options.map((o, i) => `${i + 1}. ${o}`).join("\n")}

Pick the option that fits you best and briefly explain why. Start your response with the option number.`,
            },
          ],
        });
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      // Extract the option number from the beginning of the response
      const match = text.match(/^(\d+)|option\s*(\d+)|choose\s*(\d+)|pick\s*(\d+)/i);
      const rating = match ? parseInt(match[1] || match[2] || match[3] || match[4]) : 1;

      return {
        questionId: question.id,
        rating,
        explanation: text,
        confidence: 0.9,
        rawTextResponse: text,
      };
    }

    // For matrix questions - rate multiple items on the same scale
    if (question.type === "matrix" && question.items && question.items.length > 0) {
      return this.generateMatrixResponse(persona, question, productContext);
    }

    // For slider questions - continuous scale (0-100)
    if (question.type === "slider") {
      return this.generateSliderResponse(persona, question, productContext);
    }

    // For image questions - use Claude Vision
    if (question.type === "image_rating" || question.type === "image_choice" || question.type === "image_comparison") {
      return this.generateImageResponse(persona, question, productContext);
    }

    // For Likert/NPS scales, use SSR methodology (v2: embedding-based)
    const textResponse = await this.generateTextResponse(persona, question, productContext);
    const { rating, distribution, confidence } = await this.mapToLikert(
      textResponse,
      question
    );

    return {
      questionId: question.id,
      rating,
      explanation: textResponse,
      confidence,
      rawTextResponse: textResponse,
      distribution,
    };
  }

  /**
   * Generate matrix question response - rate multiple items in a single LLM call
   */
  private async generateMatrixResponse(
    persona: SyntheticPersona,
    question: SurveyQuestion,
    productContext?: ProductContext
  ): Promise<SSRResponse> {
    const items = question.items || [];
    const scaleMin = question.scaleMin || 1;
    const scaleMax = question.scaleMax || 5;
    const scaleLabels = question.scaleLabels || [];

    const personaPrompt = this.buildPersonaPrompt(persona, productContext);

    // Build scale description
    let scaleDescription = `Scale: ${scaleMin} to ${scaleMax}`;
    if (scaleLabels.length > 0) {
      scaleDescription += ` (${scaleLabels.join(", ")})`;
    }

    const prompt = `Rate each of the following items based on the question.

Question: "${question.text}"

Items to rate:
${items.map((item, i) => `${i + 1}. ${item}`).join("\n")}

${scaleDescription}

For each item, provide a rating and a brief reason (1 sentence).

Respond ONLY as JSON in this exact format:
{
  "ratings": [
    {"item": "item name", "rating": <number ${scaleMin}-${scaleMax}>, "reason": "brief explanation"}
  ]
}`;

    try {
      const response = await this.callWithRetry(async () => {
        return this.anthropic.messages.create({
          model: this.generationModel,
          max_tokens: 800,
          temperature: GENERATION_TEMPERATURE,
          system: personaPrompt,
          messages: [{ role: "user", content: prompt }],
        });
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      // Parse JSON response
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const itemRatings: Record<string, number> = {};
        const itemExplanations: Record<string, string> = {};

        if (parsed.ratings && Array.isArray(parsed.ratings)) {
          for (const r of parsed.ratings) {
            const rating = Math.min(Math.max(Number(r.rating) || scaleMin, scaleMin), scaleMax);
            itemRatings[r.item] = rating;
            itemExplanations[r.item] = r.reason || "";
          }
        }

        // Calculate average rating
        const ratings = Object.values(itemRatings);
        const avgRating = ratings.length > 0
          ? Math.round((ratings.reduce((a, b) => a + b, 0) / ratings.length) * 100) / 100
          : (scaleMin + scaleMax) / 2;

        return {
          questionId: question.id,
          rating: avgRating,
          explanation: Object.entries(itemExplanations).map(([k, v]) => `${k}: ${v}`).join("; "),
          confidence: 0.85,
          rawTextResponse: text,
          itemRatings,
          itemExplanations,
          avgRating,
        };
      }
    } catch (error) {
      console.error("Error generating matrix response:", error);
    }

    // Fallback: return middle ratings for all items
    const midRating = Math.round((scaleMin + scaleMax) / 2);
    const itemRatings: Record<string, number> = {};
    const itemExplanations: Record<string, string> = {};
    items.forEach(item => {
      itemRatings[item] = midRating;
      itemExplanations[item] = "Unable to generate response";
    });

    return {
      questionId: question.id,
      rating: midRating,
      explanation: "Fallback response",
      confidence: 0.5,
      rawTextResponse: "",
      itemRatings,
      itemExplanations,
      avgRating: midRating,
    };
  }

  /**
   * Generate slider question response - continuous scale
   */
  private async generateSliderResponse(
    persona: SyntheticPersona,
    question: SurveyQuestion,
    productContext?: ProductContext
  ): Promise<SSRResponse> {
    const min = question.min ?? 0;
    const max = question.max ?? 100;
    const leftLabel = question.leftLabel || "Minimum";
    const rightLabel = question.rightLabel || "Maximum";

    // First generate a natural text response
    const textResponse = await this.generateTextResponse(persona, question, productContext);

    // Then analyze and map to continuous scale
    const prompt = `Analyze this response and provide a precise numeric score.

Question: "${question.text}"
Scale: ${min} (${leftLabel}) to ${max} (${rightLabel})
Response: "${textResponse}"

Based on the sentiment and content of the response, provide a precise numeric score.
The score can be any number between ${min} and ${max}, including decimals for precision.

Respond ONLY as JSON:
{"rating": <number>, "confidence": <0.0-1.0>}`;

    try {
      const response = await this.callWithRetry(async () => {
        return this.anthropic.messages.create({
          model: this.generationModel,
          max_tokens: 100,
          messages: [{ role: "user", content: prompt }],
        });
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        const rating = Math.min(Math.max(Number(parsed.rating) || (min + max) / 2, min), max);
        const confidence = Math.min(Math.max(parsed.confidence || 0.7, 0), 1);

        // Generate histogram distribution for visualization
        const distribution = this.generateHistogramDistribution(rating, min, max);

        return {
          questionId: question.id,
          rating: Math.round(rating * 10) / 10, // Round to 1 decimal place
          explanation: textResponse,
          confidence,
          rawTextResponse: textResponse,
          distribution,
        };
      }
    } catch (error) {
      console.error("Error generating slider response:", error);
    }

    // Fallback
    const midRating = (min + max) / 2;
    return {
      questionId: question.id,
      rating: midRating,
      explanation: textResponse,
      confidence: 0.5,
      rawTextResponse: textResponse,
      distribution: this.generateHistogramDistribution(midRating, min, max),
    };
  }

  /**
   * Generate histogram distribution for slider visualization
   * Returns 10 buckets representing the distribution
   */
  private generateHistogramDistribution(
    rating: number,
    min: number,
    max: number
  ): number[] {
    const buckets = 10;
    const distribution = new Array(buckets).fill(0);
    const range = max - min;
    const bucketSize = range / buckets;

    // Find which bucket the rating falls into
    const ratingBucket = Math.min(
      Math.floor((rating - min) / bucketSize),
      buckets - 1
    );

    // Create a bell curve centered on the rating bucket
    for (let i = 0; i < buckets; i++) {
      const distance = Math.abs(i - ratingBucket);
      distribution[i] = Math.exp(-distance * 0.5);
    }

    // Normalize
    const sum = distribution.reduce((a, b) => a + b, 0);
    return distribution.map(d => Math.round((d / sum) * 100));
  }

  /**
   * Generate response for image-based questions using Claude Vision
   */
  private async generateImageResponse(
    persona: SyntheticPersona,
    question: SurveyQuestion,
    productContext?: ProductContext
  ): Promise<SSRResponse> {
    const personaPrompt = this.buildPersonaPrompt(persona, productContext);
    const style = this.getPersonalityStyle(persona.psychographics.personality);

    try {
      // Prepare image content for the API
      const imageContent = await this.prepareImageContent(question);

      if (imageContent.length === 0) {
        // No valid images, return fallback
        return this.createFallbackImageResponse(question);
      }

      // Build the evaluation prompt based on question type
      const evaluationPrompt = this.buildImageEvaluationPrompt(question, style);

      // Call Claude Vision API
      const response = await this.callWithRetry(async () => {
        return this.anthropic.messages.create({
          model: "claude-sonnet-4-20250514", // Vision-capable model
          max_tokens: 800,
          system: personaPrompt,
          messages: [
            {
              role: "user",
              content: [
                ...imageContent,
                { type: "text", text: evaluationPrompt },
              ],
            },
          ],
        });
      });

      const textBlock = response.content.find((block) => block.type === "text");
      const text = textBlock?.type === "text" ? textBlock.text : "";

      // Parse the response based on question type
      return this.parseImageResponse(question, text);
    } catch (error) {
      console.error("Error generating image response:", error);
      return this.createFallbackImageResponse(question);
    }
  }

  /**
   * Prepare image content for Claude Vision API
   */
  private async prepareImageContent(
    question: SurveyQuestion
  ): Promise<Array<{ type: "image"; source: { type: "base64"; media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"; data: string } }>> {
    const imageContent: Array<{ type: "image"; source: { type: "base64"; media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"; data: string } }> = [];

    if (question.type === "image_rating" || question.type === "image_choice") {
      if (question.imageUrl) {
        try {
          const { data, mediaType } = await urlToBase64(question.imageUrl);
          imageContent.push({
            type: "image",
            source: {
              type: "base64",
              media_type: mediaType,
              data,
            },
          });
        } catch (error) {
          console.error("Failed to load image:", question.imageUrl, error);
        }
      }
    } else if (question.type === "image_comparison" && question.imageUrls) {
      for (const url of question.imageUrls) {
        try {
          const { data, mediaType } = await urlToBase64(url);
          imageContent.push({
            type: "image",
            source: {
              type: "base64",
              media_type: mediaType,
              data,
            },
          });
        } catch (error) {
          console.error("Failed to load image:", url, error);
        }
      }
    }

    return imageContent;
  }

  /**
   * Build evaluation prompt for image questions
   */
  private buildImageEvaluationPrompt(
    question: SurveyQuestion,
    style: { tone: string; length: string }
  ): string {
    const customPrompt = question.imagePrompt || "";

    if (question.type === "image_rating") {
      const scaleMin = question.imageScaleMin || 1;
      const scaleMax = question.imageScaleMax || 5;
      const lowLabel = question.imageScaleLabels?.low || "Very Poor";
      const highLabel = question.imageScaleLabels?.high || "Excellent";

      return `Question: "${question.text}"

${customPrompt ? `Additional context: ${customPrompt}\n` : ""}
Look at this image and provide your evaluation.

Rate the image from ${scaleMin} (${lowLabel}) to ${scaleMax} (${highLabel}).

Respond in ${style.length}. Be ${style.tone}.

Respond ONLY as JSON in this exact format:
{
  "rating": <number ${scaleMin}-${scaleMax}>,
  "visualAnalysis": "<brief description of what you see and your impression>",
  "explanation": "<why you gave this rating>",
  "confidence": <0.0-1.0>
}`;
    }

    if (question.type === "image_choice") {
      const options = question.options || [];
      return `Question: "${question.text}"

${customPrompt ? `Additional context: ${customPrompt}\n` : ""}
Look at this image and answer the question.

Options:
${options.map((o, i) => `${i + 1}. ${o}`).join("\n")}

Respond in ${style.length}. Be ${style.tone}.

Respond ONLY as JSON in this exact format:
{
  "rating": <option number 1-${options.length}>,
  "visualAnalysis": "<brief description of what you see>",
  "explanation": "<why you chose this option>",
  "confidence": <0.0-1.0>
}`;
    }

    if (question.type === "image_comparison") {
      const labels = question.imageLabels || question.imageUrls?.map((_, i) => `Image ${i + 1}`) || [];
      return `Question: "${question.text}"

${customPrompt ? `Additional context: ${customPrompt}\n` : ""}
Compare the ${labels.length} images shown above.

The images are labeled (in order):
${labels.map((label, i) => `${i + 1}. ${label}`).join("\n")}

Respond in ${style.length}. Be ${style.tone}.

Respond ONLY as JSON in this exact format:
{
  "selectedImage": "<the label of your preferred image>",
  "imagePreferences": {${labels.map(label => `"${label}": <score 1-10>`).join(", ")}},
  "visualAnalysis": "<brief comparison of the images>",
  "explanation": "<why you prefer your selected image>",
  "confidence": <0.0-1.0>
}`;
    }

    // Fallback
    return `Evaluate this image and provide your thoughts. Respond naturally.`;
  }

  /**
   * Parse image response from Claude
   */
  private parseImageResponse(
    question: SurveyQuestion,
    text: string
  ): SSRResponse {
    try {
      // Try to extract JSON from the response
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);

        if (question.type === "image_rating") {
          const scaleMin = question.imageScaleMin || 1;
          const scaleMax = question.imageScaleMax || 5;
          const rating = Math.min(Math.max(Number(parsed.rating) || 3, scaleMin), scaleMax);
          const confidence = Math.min(Math.max(parsed.confidence || 0.7, 0), 1);

          return {
            questionId: question.id,
            rating,
            explanation: parsed.explanation || text,
            confidence,
            rawTextResponse: text,
            visualAnalysis: parsed.visualAnalysis || "",
            distribution: this.generateDistribution(rating, scaleMin, scaleMax, confidence),
          };
        }

        if (question.type === "image_choice") {
          const rating = Math.min(Math.max(Number(parsed.rating) || 1, 1), question.options?.length || 4);

          return {
            questionId: question.id,
            rating,
            explanation: parsed.explanation || text,
            confidence: parsed.confidence || 0.8,
            rawTextResponse: text,
            visualAnalysis: parsed.visualAnalysis || "",
          };
        }

        if (question.type === "image_comparison") {
          // For comparison, rating is the index of selected image
          const labels = question.imageLabels || [];
          const selectedIndex = labels.indexOf(parsed.selectedImage) + 1;
          const rating = selectedIndex > 0 ? selectedIndex : 1;

          return {
            questionId: question.id,
            rating,
            explanation: parsed.explanation || text,
            confidence: parsed.confidence || 0.8,
            rawTextResponse: text,
            visualAnalysis: parsed.visualAnalysis || "",
            selectedImage: parsed.selectedImage || labels[0] || "Image 1",
            imagePreferences: parsed.imagePreferences || {},
          };
        }
      }
    } catch (error) {
      console.error("Error parsing image response:", error);
    }

    // Fallback parsing
    return this.createFallbackImageResponse(question, text);
  }

  /**
   * Create fallback response for image questions
   */
  private createFallbackImageResponse(
    question: SurveyQuestion,
    rawText: string = ""
  ): SSRResponse {
    const scaleMin = question.imageScaleMin || 1;
    const scaleMax = question.imageScaleMax || 5;
    const midRating = Math.round((scaleMin + scaleMax) / 2);

    return {
      questionId: question.id,
      rating: question.type === "image_comparison" ? 1 : midRating,
      explanation: rawText || "Unable to evaluate image",
      confidence: 0.5,
      rawTextResponse: rawText,
      visualAnalysis: "",
      distribution: question.type === "image_rating"
        ? this.generateDistribution(midRating, scaleMin, scaleMax, 0.5)
        : undefined,
      selectedImage: question.type === "image_comparison"
        ? (question.imageLabels?.[0] || "Image 1")
        : undefined,
    };
  }

  /**
   * Run a complete simulation for a persona
   * Handles conditional logic (showIf) for questions
   */
  async simulatePersona(
    persona: SyntheticPersona,
    questions: SurveyQuestion[],
    productContext?: ProductContext
  ): Promise<SimulationResult> {
    const startTime = Date.now();
    const responses: SSRResponse[] = [];
    const answeredQuestions = new Map<string, SSRResponse>();

    for (const question of questions) {
      // Evaluate conditional logic
      if (question.showIf) {
        const refResponse = answeredQuestions.get(question.showIf.questionId);
        if (!refResponse || !this.evaluateCondition(refResponse, question.showIf)) {
          // Skip this question - condition not met
          responses.push({
            questionId: question.id,
            rating: 0,
            explanation: "",
            confidence: 0,
            rawTextResponse: "",
            skipped: true,
            skipReason: `Condition not met: Question ${question.showIf.questionId} ${question.showIf.operator} ${question.showIf.value}`,
          });
          continue;
        }
      }

      const response = await this.generateResponse(persona, question, productContext);
      responses.push(response);
      answeredQuestions.set(question.id, response);
    }

    const embeddingAvailable = !!getEmbeddingService();
    return {
      personaId: persona.id,
      responses,
      metadata: {
        modelUsed: this.generationModel,
        mappingMethod: embeddingAvailable ? "embedding-v1" : "llm-fallback",
        embeddingModel: embeddingAvailable ? "voyage-3.5-lite" : "none",
        methodologyVersion: METHODOLOGY_VERSION,
        temperature: GENERATION_TEMPERATURE,
        embeddingTemperature: this.embeddingTemperature,
        similarityNormalization: this.similarityNormalization,
        timestamp: new Date().toISOString(),
        processingTimeMs: Date.now() - startTime,
      },
    };
  }

  /**
   * Evaluate a condition against a previous response
   */
  private evaluateCondition(response: SSRResponse, condition: QuestionCondition): boolean {
    const { operator, value } = condition;

    switch (operator) {
      case "equals":
        // Check numeric equality first
        if (typeof value === "number") {
          return response.rating === value;
        }
        // Check string equality (case-insensitive)
        return (
          response.rating === Number(value) ||
          response.explanation?.toLowerCase().includes(String(value).toLowerCase()) ||
          response.rawTextResponse?.toLowerCase().includes(String(value).toLowerCase())
        );

      case "notEquals":
        if (typeof value === "number") {
          return response.rating !== value;
        }
        return (
          response.rating !== Number(value) &&
          !response.explanation?.toLowerCase().includes(String(value).toLowerCase()) &&
          !response.rawTextResponse?.toLowerCase().includes(String(value).toLowerCase())
        );

      case "greaterThan":
        return (response.rating || 0) > Number(value);

      case "lessThan":
        return (response.rating || 0) < Number(value);

      case "contains":
        const searchValue = String(value).toLowerCase();
        return (
          response.explanation?.toLowerCase().includes(searchValue) ||
          response.rawTextResponse?.toLowerCase().includes(searchValue) ||
          false
        );

      default:
        return true;
    }
  }

  /**
   * Run simulation for multiple personas with controlled parallelization
   * Processes PARALLEL_PERSONAS personas at a time to respect rate limits while improving speed
   * @param shouldCancel - Optional async callback to check if simulation should be cancelled
   */
  async simulatePanel(
    personas: SyntheticPersona[],
    questions: SurveyQuestion[],
    productContext?: ProductContext,
    onProgress?: (current: number, total: number) => Promise<void>,
    shouldCancel?: () => Promise<boolean>
  ): Promise<{ results: SimulationResult[]; cancelled: boolean }> {
    const results: SimulationResult[] = new Array(personas.length);
    let completedCount = 0;
    let cancelled = false;

    // Process personas in batches for controlled parallelization
    const batchSize = PARALLEL_PERSONAS;
    const totalBatches = Math.ceil(personas.length / batchSize);

    console.log(`Starting parallel simulation: ${personas.length} personas, ${batchSize} parallel, ${totalBatches} batches`);

    for (let batchIndex = 0; batchIndex < totalBatches && !cancelled; batchIndex++) {
      // Check for cancellation before each batch
      if (shouldCancel) {
        cancelled = await shouldCancel();
        if (cancelled) {
          console.log(`Simulation cancelled at batch ${batchIndex + 1}/${totalBatches}`);
          // Filter out undefined entries from results
          return {
            results: results.filter((r): r is SimulationResult => r !== undefined),
            cancelled: true
          };
        }
      }

      const batchStart = batchIndex * batchSize;
      const batchEnd = Math.min(batchStart + batchSize, personas.length);
      const batchPersonas = personas.slice(batchStart, batchEnd);

      console.log(`Processing batch ${batchIndex + 1}/${totalBatches} (personas ${batchStart + 1}-${batchEnd})`);

      // Process batch in parallel
      const batchPromises = batchPersonas.map(async (persona, indexInBatch) => {
        const globalIndex = batchStart + indexInBatch;
        try {
          const result = await this.simulatePersona(persona, questions, productContext);
          results[globalIndex] = result;
          completedCount++;

          // Report progress after each persona completes
          if (onProgress) {
            await onProgress(completedCount, personas.length);
          }

          console.log(`Completed persona ${globalIndex + 1}/${personas.length}`);
          return result;
        } catch (error) {
          console.error(`Error processing persona ${globalIndex + 1}:`, error);
          // Create a fallback result with empty responses for failed personas
          const fallbackResult: SimulationResult = {
            personaId: persona.id,
            responses: questions.map(q => ({
              questionId: q.id,
              rating: 0,
              explanation: "Error generating response",
              confidence: 0,
              rawTextResponse: "",
              skipped: true,
              skipReason: error instanceof Error ? error.message : "Unknown error",
            })),
            metadata: {
              modelUsed: this.generationModel,
              mappingMethod: "error-fallback",
              embeddingModel: "none",
              methodologyVersion: METHODOLOGY_VERSION,
              temperature: GENERATION_TEMPERATURE,
              embeddingTemperature: this.embeddingTemperature,
              similarityNormalization: this.similarityNormalization,
              timestamp: new Date().toISOString(),
              processingTimeMs: 0,
            },
          };
          results[globalIndex] = fallbackResult;
          completedCount++;

          if (onProgress) {
            await onProgress(completedCount, personas.length);
          }

          return fallbackResult;
        }
      });

      // Wait for all personas in this batch to complete
      await Promise.all(batchPromises);

      // Add delay between batches to respect rate limits
      // With 4 parallel personas and ~2 API calls each = ~8 calls per batch
      // At 40 req/min limit, we can do ~5 batches per minute
      // Add 500ms delay between batches for safety margin
      if (batchIndex < totalBatches - 1) {
        await this.delay(500);
      }
    }

    // Filter out any undefined results (shouldn't happen but safety check)
    const finalResults = results.filter((r): r is SimulationResult => r !== undefined);

    console.log(`Simulation complete: ${finalResults.length}/${personas.length} personas processed`);

    return { results: finalResults, cancelled };
  }

  /**
   * Legacy sequential simulation method (kept for compatibility/debugging)
   */
  async simulatePanelSequential(
    personas: SyntheticPersona[],
    questions: SurveyQuestion[],
    productContext?: ProductContext,
    onProgress?: (current: number, total: number) => Promise<void>,
    shouldCancel?: () => Promise<boolean>
  ): Promise<{ results: SimulationResult[]; cancelled: boolean }> {
    const results: SimulationResult[] = [];

    // Process sequentially to respect rate limits
    for (let i = 0; i < personas.length; i++) {
      // Check for cancellation before processing each persona
      if (shouldCancel) {
        const cancelled = await shouldCancel();
        if (cancelled) {
          console.log(`Simulation cancelled at persona ${i + 1}/${personas.length}`);
          return { results, cancelled: true };
        }
      }

      console.log(`Processing persona ${i + 1}/${personas.length}...`);
      const result = await this.simulatePersona(personas[i], questions, productContext);
      results.push(result);

      // Report progress
      if (onProgress) {
        await onProgress(i + 1, personas.length);
      }

      // Add small delay between personas
      if (i < personas.length - 1) {
        await this.delay(500);
      }
    }

    return { results, cancelled: false };
  }
}

// Singleton instance
let ssrEngine: SSREngine | null = null;

export function getSSREngine(config?: SSREngineConfig): SSREngine {
  if (!ssrEngine) {
    ssrEngine = new SSREngine(config);
  }
  return ssrEngine;
}
