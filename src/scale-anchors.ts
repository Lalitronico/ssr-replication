/**
 * Scale Anchor System for SSR Engine v2.
 *
 * Provides semantically rich anchor statements for each scale point,
 * used as targets for embedding-based cosine similarity mapping.
 *
 * Resolution strategy (3 tiers):
 * 1. Predefined templates (~90%) — hand-written anchor statements per semantic family
 * 2. Label-based resolution (~9%) — maps scaleAnchors.low/high labels to a template
 * 3. Interpolation fallback (~1%) — generates anchors from custom low/high labels
 */

import type { SurveyQuestion } from "./ssr-engine";

// ─── Types ───────────────────────────────────────────────────────

export interface ScaleAnchorSet {
  semantic: string;            // e.g. "satisfaction", "agreement"
  anchors: string[];           // One statement per scale point, in order (low → high)
  scaleMin: number;
  scaleMax: number;
}

type AnchorTemplate = {
  anchors5: string[];    // 5-point version
  anchors7?: string[];   // 7-point version (only for agreement)
  anchors11?: string[];  // 11-point version (only for nps)
};

// ─── Predefined Anchor Templates ─────────────────────────────────

const TEMPLATES: Record<string, AnchorTemplate> = {
  agreement: {
    anchors5: [
      "Completely wrong. I strongly disagree, this doesn't match reality at all and I reject this statement.",
      "I don't think so. I disagree with this, there are problems and it doesn't seem right to me.",
      "Not sure either way. I could see arguments for and against, I don't have a strong opinion on this.",
      "I think so, mostly. I agree that this is about right, with maybe a few minor exceptions.",
      "Absolutely right. I completely agree, this is exactly correct and perfectly describes the situation.",
    ],
    anchors7: [
      "I strongly disagree. This is completely contrary to my views.",
      "I disagree. I don't think this is accurate.",
      "I slightly disagree. I have some reservations about this.",
      "I neither agree nor disagree. I'm neutral on this.",
      "I slightly agree. I'm somewhat inclined to agree.",
      "I agree. I think this is mostly correct.",
      "I strongly agree. This is absolutely right.",
    ],
  },

  satisfaction: {
    anchors5: [
      "Terrible. I was frustrated and angry the entire time, nothing worked right and I regret using this.",
      "Not great. I ran into problems and it was harder than it should have been, below what I expected.",
      "It was okay. Nothing special, not bad but not good either. Pretty average and forgettable.",
      "Pretty good. Things went smoothly for the most part, just a couple of small things that could be better.",
      "Excellent. Everything was smooth, fast, and easy. I'm really impressed and delighted with the result.",
    ],
  },

  likelihood: {
    anchors5: [
      "No, I definitely won't. I had a bad experience and I have no intention of doing this again. I'll go elsewhere.",
      "Probably not. It wasn't great and I have better options available. I'd need a strong reason to come back.",
      "Maybe, maybe not. It depends on the circumstances and what's available. I don't feel strongly either way.",
      "Yeah, I probably will. It was decent enough and fairly convenient. I'd come back if the opportunity arises.",
      "Absolutely, I definitely will. I had a great experience and I'm already looking forward to doing this again soon.",
    ],
  },

  ease: {
    anchors5: [
      "Incredibly frustrating. I kept getting stuck and couldn't figure out what to do next. It felt like fighting the system every step of the way.",
      "Kind of a hassle. I ran into a couple of confusing parts and had to stop and think more than I should have. Not the smoothest experience.",
      "It was fine, I guess. Not particularly hard but not effortless either. I got through it without too much trouble.",
      "Pretty straightforward. I figured things out quickly and only paused once or twice. Most of it just made sense right away.",
      "Couldn't have been simpler. Everything was obvious and intuitive, I breezed through the whole thing without a second thought.",
    ],
  },

  importance: {
    anchors5: [
      "Couldn't care less about this. It has zero impact on my life and I wouldn't even notice if it disappeared tomorrow.",
      "It's on my radar but barely. There are way bigger things I worry about, this is pretty low on my priority list.",
      "It matters somewhat. I'd pay attention to it but it's not something that keeps me up at night or changes my decisions.",
      "This is a big deal to me. I think about it regularly and it genuinely affects my choices and how I go about things.",
      "This is absolutely essential. I can't imagine doing without it, it's one of the most important things in my life right now.",
    ],
  },

  familiarity: {
    anchors5: [
      "Never heard of it. This is completely new to me, I have no idea what it is or what it does. Drawing a total blank.",
      "I've seen the name around but that's about it. I couldn't really tell you much about it beyond the most surface-level stuff.",
      "I know the basics. I've come across it enough to have a general sense of what it's about, but I'm no expert by any means.",
      "I know it pretty well. I've used it or dealt with it enough times that I'm comfortable with it and could explain it to someone.",
      "I know this inside and out. I use it all the time, I could talk about it in detail, and people come to me with questions about it.",
    ],
  },

  appeal: {
    anchors5: [
      "Hard pass. Nothing about this interests me at all. I scrolled right past it and wouldn't give it a second look.",
      "Eh, it's okay I guess. There's maybe one small thing that caught my eye, but overall it's not really for me.",
      "It's interesting enough. I can see why people would like it, and I might check it out if the timing was right.",
      "Yeah, this is really appealing. I'm drawn to it and I'd actively look into it more. It's got my attention for sure.",
      "I absolutely love this. I'm excited just thinking about it, this is exactly my kind of thing and I want it now.",
    ],
  },

  value: {
    anchors5: [
      "Total rip-off. I feel cheated paying this much for what you actually get. Completely overpriced and not worth a fraction of the cost.",
      "Not great value. It feels a bit expensive for what it is. I'd expect more features or better quality at this price point.",
      "Fair enough for the price. You get what you pay for, nothing more and nothing less. It's reasonable but not a bargain.",
      "Good value for the money. I feel like I'm getting a solid deal here. The quality and features justify the price and then some.",
      "Incredible deal. I can't believe how much you get for this price. It's an absolute steal and I'd happily pay more for it.",
    ],
  },

  trust: {
    anchors5: [
      "I don't trust this one bit. Something feels off and I'd be very careful sharing any personal information or relying on it for anything important.",
      "I'm a little skeptical. There are some red flags that make me uneasy. I'd proceed with caution and double-check things independently.",
      "I'm on the fence. I don't have a strong reason to trust or distrust it. I'd give it a chance but I'm keeping my guard up.",
      "I trust it for the most part. It seems reliable and legitimate. I'd feel comfortable using it without worrying too much.",
      "I trust this completely. It has a solid track record and I'd feel totally comfortable recommending it and relying on it for important things.",
    ],
  },

  quality: {
    anchors5: [
      "Terrible quality. It feels cheap, flimsy, and poorly made. I'm honestly surprised this passed any kind of quality check. Not acceptable at all.",
      "Below average quality. There are some noticeable issues like rough edges or inconsistent finish. It works but you can tell corners were cut.",
      "Decent quality overall. Nothing wrong with it but nothing that wows me either. It does what it's supposed to do, standard stuff.",
      "Really good quality. Well-made, solid feel, and clearly some thought went into the details. I'm impressed with the craftsmanship.",
      "Outstanding quality. Premium feel in every way, this is clearly best-in-class. The attention to detail and materials are top-notch.",
    ],
  },

  frequency: {
    anchors5: [
      "Never, not once. This just isn't something that's part of my routine at all. I can't remember the last time I did this, if ever.",
      "Hardly ever. Maybe once in a blue moon if the stars align. It's rare enough that it barely registers as something I do.",
      "Every now and then. I do it occasionally when the situation calls for it, but it's not a regular habit. Maybe a few times a month.",
      "Pretty regularly. It's part of my routine at this point. I do it multiple times a week and it feels natural and habitual.",
      "All the time, practically every day. I can't imagine not doing this, it's completely woven into my daily life at this point.",
    ],
  },

  uniqueness: {
    anchors5: [
      "Nothing special about this at all. It's basically a copy of what's already out there. I've seen this exact same thing a dozen times before.",
      "Mostly the same as other options. There might be a small twist here or there but nothing that really sets it apart in a meaningful way.",
      "It has some distinctive qualities. A few things about it feel fresh or different, though the core concept isn't entirely new.",
      "This really stands out from the crowd. It has a clear identity and does things differently enough that I'd remember it and point it out to others.",
      "Truly one of a kind. I've never seen anything quite like this before. It's so original and different that it feels like it created its own category.",
    ],
  },

  relevance: {
    anchors5: [
      "This has nothing to do with me. It's completely off-base for my situation and I can't see any connection to what I actually need.",
      "Tangentially related at best. There's a loose connection but it doesn't really address what I'm looking for in any meaningful way.",
      "Somewhat relevant. Parts of it apply to my situation and I can see some usefulness, but it's not exactly what I had in mind.",
      "Very relevant to what I need. It closely aligns with my situation and addresses most of the things I'm actually looking for.",
      "Spot on, exactly what I was looking for. This fits my needs perfectly and I feel like it was made specifically for my situation.",
    ],
  },

  impression: {
    anchors5: [
      "Awful first impression. Everything about it rubs me the wrong way and I came away feeling really negative. Not a fan at all.",
      "Not a great impression. A few things turned me off and I have some real concerns. I'd need convincing to give it another chance.",
      "My impression is neutral. I don't feel strongly one way or the other. It didn't wow me but it didn't put me off either.",
      "Good impression overall. I came away feeling positive and optimistic about it. There's a lot to like and I'm interested to see more.",
      "Absolutely fantastic impression. I'm blown away by everything I've seen. This exceeded all my expectations and I'm genuinely enthusiastic.",
    ],
  },

  purchase_intent: {
    anchors5: [
      "No chance I'm buying this. It's not for me at all and I'd never spend my money on it. I'd walk right past it every time.",
      "Probably not going to buy it. I'm not very interested and there are better ways I could spend my money. It'd take a lot to convince me.",
      "I might buy it, I might not. I'm on the fence. If the price was right or someone recommended it, I could see myself picking it up.",
      "Yeah, I'd probably buy this. It looks good and fits what I'm looking for. I can see myself pulling the trigger on this pretty soon.",
      "I'm buying this for sure. Take my money right now. This is exactly what I've been looking for and I don't need any more convincing.",
    ],
  },

  nps: {
    anchors5: [], // NPS uses 11-point only
    anchors11: [
      "I would never recommend this. Terrible experience. Score: 0 out of 10.",
      "I would strongly advise against this. Very poor experience. Score: 1 out of 10.",
      "I am very unlikely to recommend this. Poor experience. Score: 2 out of 10.",
      "I would probably not recommend this. Below average experience. Score: 3 out of 10.",
      "I would hesitate to recommend this. Mediocre experience. Score: 4 out of 10.",
      "I am neutral about recommending this. Average experience. Score: 5 out of 10.",
      "I might recommend this in some cases. Slightly above average. Score: 6 out of 10.",
      "I would likely recommend this. Good experience overall. Score: 7 out of 10.",
      "I would recommend this. Very good experience. Score: 8 out of 10.",
      "I would strongly recommend this. Excellent experience. Score: 9 out of 10.",
      "I would enthusiastically recommend this to everyone. Outstanding. Score: 10 out of 10.",
    ],
  },
};

// ─── Label → Semantic Lookup ─────────────────────────────────────
// Maps common anchor labels to their semantic family

const LABEL_TO_SEMANTIC: Record<string, string> = {
  // Satisfaction
  "very dissatisfied": "satisfaction",
  "very satisfied": "satisfaction",
  "extremely dissatisfied": "satisfaction",
  "extremely satisfied": "satisfaction",
  // Agreement
  "strongly disagree": "agreement",
  "strongly agree": "agreement",
  // Likelihood
  "very unlikely": "likelihood",
  "very likely": "likelihood",
  "extremely unlikely": "likelihood",
  "extremely likely": "likelihood",
  "not at all likely": "likelihood",
  // Ease
  "very difficult": "ease",
  "very easy": "ease",
  // Importance
  "not important": "importance",
  "not important at all": "importance",
  "very important": "importance",
  "extremely important": "importance",
  // Familiarity
  "not at all familiar": "familiarity",
  "extremely familiar": "familiarity",
  // Appeal
  "not at all appealing": "appeal",
  "extremely appealing": "appeal",
  "not appealing": "appeal",
  "very appealing": "appeal",
  // Value
  "very poor value": "value",
  "excellent value": "value",
  // Trust
  "not at all": "trust",
  "completely": "trust",
  // Quality
  "very poor": "quality",
  "excellent": "quality",
  // Impression
  "very negative": "impression",
  "very positive": "impression",
  // Purchase intent
  "definitely would not": "purchase_intent",
  "definitely would": "purchase_intent",
  // Uniqueness
  "not at all unique": "uniqueness",
  "extremely unique": "uniqueness",
  // Relevance
  "not at all relevant": "relevance",
  "extremely relevant": "relevance",
};

// ─── Public API ──────────────────────────────────────────────────

/**
 * Resolve anchor statements for a survey question.
 *
 * @param question - The survey question with type and scale info
 * @returns ScaleAnchorSet with one anchor statement per scale point
 */
export function resolveAnchors(question: SurveyQuestion): ScaleAnchorSet {
  const scaleMin = question.scaleMin ?? (question.type === "nps" ? 0 : 1);
  const scaleMax = question.scaleMax ?? (question.type === "nps" ? 10 : 5);
  const scaleSize = scaleMax - scaleMin + 1;

  // Tier 1: NPS always uses NPS template
  if (question.type === "nps") {
    const npsTemplate = TEMPLATES.nps.anchors11!;
    return {
      semantic: "nps",
      anchors: npsTemplate,
      scaleMin: 0,
      scaleMax: 10,
    };
  }

  // Tier 2: Try label-based resolution from scaleAnchors
  if (question.scaleAnchors) {
    const semantic = detectSemanticFromLabels(
      question.scaleAnchors.low,
      question.scaleAnchors.high
    );

    if (semantic && TEMPLATES[semantic]) {
      const anchors = getAnchorsForSize(semantic, scaleSize);
      if (anchors) {
        return { semantic, anchors, scaleMin, scaleMax };
      }
    }

    // Tier 3: Interpolation fallback — generate anchors from custom labels
    return {
      semantic: "custom",
      anchors: interpolateAnchors(
        question.scaleAnchors.low,
        question.scaleAnchors.high,
        scaleSize
      ),
      scaleMin,
      scaleMax,
    };
  }

  // Tier 1: Try to infer semantic from question text
  const inferred = inferSemanticFromText(question.text);
  if (inferred && TEMPLATES[inferred]) {
    const anchors = getAnchorsForSize(inferred, scaleSize);
    if (anchors) {
      return { semantic: inferred, anchors, scaleMin, scaleMax };
    }
  }

  // Default: agreement scale (most common Likert default)
  const defaultAnchors = getAnchorsForSize("agreement", scaleSize);
  return {
    semantic: "agreement",
    anchors: defaultAnchors || TEMPLATES.agreement.anchors5,
    scaleMin,
    scaleMax,
  };
}

/**
 * Get a scale context string for the generation prompt.
 * Tells the LLM what the scale means so text aligns with scale semantics.
 */
export function getScaleContextString(question: SurveyQuestion): string {
  const anchors = resolveAnchors(question);
  const min = anchors.scaleMin;
  const max = anchors.scaleMax;

  // Show min/max labels
  const lowLabel = anchors.anchors[0].split(".")[0]; // first sentence
  const highLabel = anchors.anchors[anchors.anchors.length - 1].split(".")[0];

  return `This uses a ${min}-${max} scale where ${min} means "${lowLabel}" and ${max} means "${highLabel}".`;
}

// ─── Internal Helpers ────────────────────────────────────────────

function detectSemanticFromLabels(low: string, high: string): string | null {
  const lowNorm = low.toLowerCase().trim();
  const highNorm = high.toLowerCase().trim();

  // Check low label
  const fromLow = LABEL_TO_SEMANTIC[lowNorm];
  if (fromLow) return fromLow;

  // Check high label
  const fromHigh = LABEL_TO_SEMANTIC[highNorm];
  if (fromHigh) return fromHigh;

  // Partial match — check if label contains key phrases
  for (const [pattern, semantic] of Object.entries(LABEL_TO_SEMANTIC)) {
    if (lowNorm.includes(pattern) || highNorm.includes(pattern)) {
      return semantic;
    }
  }

  return null;
}

function inferSemanticFromText(text: string): string | null {
  const lower = text.toLowerCase();

  if (lower.includes("satisfi")) return "satisfaction";
  if (lower.includes("agree") || lower.includes("disagree")) return "agreement";
  if (lower.includes("likely") || lower.includes("likelihood") || lower.includes("probability")) return "likelihood";
  if (lower.includes("easy") || lower.includes("difficult") || lower.includes("ease")) return "ease";
  if (lower.includes("important") || lower.includes("importance")) return "importance";
  if (lower.includes("familiar")) return "familiarity";
  if (lower.includes("appeal")) return "appeal";
  if (lower.includes("value") && lower.includes("money")) return "value";
  if (lower.includes("trust")) return "trust";
  if (lower.includes("quality")) return "quality";
  if (lower.includes("unique") || lower.includes("different")) return "uniqueness";
  if (lower.includes("relevant") || lower.includes("relevance")) return "relevance";
  if (lower.includes("impression") || lower.includes("overall")) return "impression";
  if (lower.includes("purchase") || lower.includes("buy")) return "purchase_intent";
  if (lower.includes("recommend")) return "nps";
  if (lower.includes("often") || lower.includes("frequently") || lower.includes("how often")) return "frequency";

  return null;
}

function getAnchorsForSize(semantic: string, size: number): string[] | null {
  const template = TEMPLATES[semantic];
  if (!template) return null;

  if (size === 11 && template.anchors11) return template.anchors11;
  if (size === 7 && template.anchors7) return template.anchors7;
  if (size === 5 && template.anchors5.length === 5) return template.anchors5;

  // Interpolate from 5-point to requested size
  if (template.anchors5.length === 5 && size > 0) {
    return interpolateFromTemplate(template.anchors5, size);
  }

  return null;
}

/**
 * Interpolate anchor statements from a 5-point template to arbitrary size.
 * Uses linear interpolation of indices to pick/blend from the source.
 */
function interpolateFromTemplate(source5: string[], targetSize: number): string[] {
  if (targetSize === 5) return source5;
  if (targetSize <= 1) return [source5[2]]; // middle

  const result: string[] = [];
  for (let i = 0; i < targetSize; i++) {
    const ratio = i / (targetSize - 1);
    const srcIdx = ratio * 4; // maps 0..1 to 0..4
    const nearestIdx = Math.round(srcIdx);
    result.push(source5[nearestIdx]);
  }
  return result;
}

/**
 * Generate anchor statements by interpolating between custom low/high labels.
 * Used as a last resort when no predefined template matches.
 */
function interpolateAnchors(low: string, high: string, size: number): string[] {
  if (size <= 1) return [`${low}. ${high}.`];
  if (size === 2) return [low + ".", high + "."];

  const intensities = [
    "very strongly",
    "strongly",
    "somewhat",
    "slightly",
    "neither",
    "slightly",
    "somewhat",
    "strongly",
    "very strongly",
  ];

  const result: string[] = [];
  for (let i = 0; i < size; i++) {
    const ratio = i / (size - 1);

    if (ratio < 0.2) {
      result.push(`${low}. I feel very ${low.toLowerCase().replace(/^(very |extremely )/, "")}.`);
    } else if (ratio < 0.4) {
      result.push(`Leaning toward ${low.toLowerCase()}. I somewhat feel this way.`);
    } else if (ratio < 0.6) {
      result.push(`Neither ${low.toLowerCase()} nor ${high.toLowerCase()}. I feel neutral.`);
    } else if (ratio < 0.8) {
      result.push(`Leaning toward ${high.toLowerCase()}. I somewhat feel this way.`);
    } else {
      result.push(`${high}. I feel very ${high.toLowerCase().replace(/^(very |extremely )/, "")}.`);
    }
  }
  return result;
}
