/**
 * SSR Engine v2 — Expanded Ground Truth Test Set (v2)
 *
 * 69 test cases across 8 semantic domains for calibration and cross-validation.
 * Each case: survey question + realistic response text + expected rating + difficulty.
 *
 * Difficulty levels:
 *   - clear: strong, unambiguous signal matching a single scale point
 *   - subtle: mixed signals or hedging that makes the "right" rating less obvious
 *   - edge: adversarial cases like negation, sarcasm, or near-boundary responses
 *
 * Domain coverage:
 *   satisfaction (10), likelihood (8), agreement (8), ease (9),
 *   importance (9), trust (9), value (8), purchase_intent (8)
 *   Total: 69
 */

import type { SurveyQuestion } from "../../src/lib/ssr-engine";

// ─── Types ───────────────────────────────────────────────────────

export interface TestCase {
  question: SurveyQuestion;
  text: string;
  expected: number;
  label: string;
  domain: string;
  difficulty: "clear" | "subtle" | "edge";
}

// ─── Questions ───────────────────────────────────────────────────

export const SATISFACTION_Q: SurveyQuestion = {
  id: "sat", type: "likert",
  text: "How satisfied are you with the checkout experience on our e-commerce app?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very dissatisfied", high: "Very satisfied" },
};

export const LIKELIHOOD_Q: SurveyQuestion = {
  id: "lik", type: "likert",
  text: "How likely are you to purchase from us again in the next month?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very unlikely", high: "Very likely" },
};

export const AGREEMENT_Q: SurveyQuestion = {
  id: "agr", type: "likert",
  text: "I am satisfied with the overall quality of this product.",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Strongly disagree", high: "Strongly agree" },
};

export const EASE_Q: SurveyQuestion = {
  id: "ease", type: "likert",
  text: "How easy was it to find the information you were looking for on our website?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very difficult", high: "Very easy" },
};

export const IMPORTANCE_Q: SurveyQuestion = {
  id: "imp", type: "likert",
  text: "How important is fast shipping when you decide where to shop online?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Not important at all", high: "Extremely important" },
};

export const TRUST_Q: SurveyQuestion = {
  id: "tru", type: "likert",
  text: "How much do you trust this brand to protect your personal data?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Not at all", high: "Completely" },
};

export const VALUE_Q: SurveyQuestion = {
  id: "val", type: "likert",
  text: "How would you rate the value for money of this product?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Very poor value", high: "Excellent value" },
};

export const PURCHASE_INTENT_Q: SurveyQuestion = {
  id: "pi", type: "likert",
  text: "How likely are you to purchase this product in the next 30 days?",
  scaleMin: 1, scaleMax: 5,
  scaleAnchors: { low: "Definitely would not", high: "Definitely would" },
};

// ─── All questions by domain ─────────────────────────────────────

export const QUESTIONS_BY_DOMAIN: Record<string, SurveyQuestion> = {
  satisfaction: SATISFACTION_Q,
  likelihood: LIKELIHOOD_Q,
  agreement: AGREEMENT_Q,
  ease: EASE_Q,
  importance: IMPORTANCE_Q,
  trust: TRUST_Q,
  value: VALUE_Q,
  purchase_intent: PURCHASE_INTENT_Q,
};

// ─── Test Cases ──────────────────────────────────────────────────

export const TEST_CASES: TestCase[] = [
  // ════════════════════════════════════════════════════════════════
  // SATISFACTION (10 cases: 5 clear + 3 subtle + 2 edge)
  // ════════════════════════════════════════════════════════════════

  { question: SATISFACTION_Q, expected: 1, label: "SAT-1", domain: "satisfaction", difficulty: "clear",
    text: "Absolutely awful. The page froze three times, I lost my payment info, and had to start over. Worst checkout I've ever used." },

  { question: SATISFACTION_Q, expected: 2, label: "SAT-2", domain: "satisfaction", difficulty: "clear",
    text: "Not a good experience. The layout was confusing and I couldn't find the shipping options easily. Took me 15 minutes." },

  { question: SATISFACTION_Q, expected: 3, label: "SAT-3", domain: "satisfaction", difficulty: "clear",
    text: "It was okay. Nothing terrible but nothing great either. Just a standard checkout process." },

  { question: SATISFACTION_Q, expected: 4, label: "SAT-4", domain: "satisfaction", difficulty: "clear",
    text: "Pretty good overall. The process was mostly smooth and I liked the saved payment option. Minor hiccup with the address form." },

  { question: SATISFACTION_Q, expected: 5, label: "SAT-5", domain: "satisfaction", difficulty: "clear",
    text: "Excellent checkout experience! Everything was fast, clear, and intuitive. One-click payment worked perfectly." },

  { question: SATISFACTION_Q, expected: 2, label: "SAT-2s", domain: "satisfaction", difficulty: "subtle",
    text: "I managed to complete my order but it was frustrating. Several steps felt unnecessary and the coupon code didn't work right." },

  { question: SATISFACTION_Q, expected: 4, label: "SAT-4s", domain: "satisfaction", difficulty: "subtle",
    text: "Good experience for the most part. Quick and easy, though the mobile layout could use some work." },

  { question: SATISFACTION_Q, expected: 3, label: "SAT-3s", domain: "satisfaction", difficulty: "subtle",
    text: "Some things worked well, like the search function, but the checkout itself was kind of clunky. Mixed bag really." },

  { question: SATISFACTION_Q, expected: 2, label: "SAT-2e", domain: "satisfaction", difficulty: "edge",
    text: "It wasn't terrible, I guess. I mean, nothing crashed. But I wouldn't call it good by any stretch of the imagination." },

  { question: SATISFACTION_Q, expected: 4, label: "SAT-4e", domain: "satisfaction", difficulty: "edge",
    text: "I don't have any complaints, which honestly says a lot these days. Everything just worked like it should." },

  // ════════════════════════════════════════════════════════════════
  // LIKELIHOOD (8 cases: 5 clear + 2 subtle + 1 edge)
  // ════════════════════════════════════════════════════════════════

  { question: LIKELIHOOD_Q, expected: 1, label: "LIK-1", domain: "likelihood", difficulty: "clear",
    text: "No way. After that experience I'm done with this store. I'll shop elsewhere from now on." },

  { question: LIKELIHOOD_Q, expected: 2, label: "LIK-2", domain: "likelihood", difficulty: "clear",
    text: "Probably not. The experience wasn't great and I have other options. Maybe if they have a really good sale." },

  { question: LIKELIHOOD_Q, expected: 3, label: "LIK-3", domain: "likelihood", difficulty: "clear",
    text: "Maybe. It depends on whether they have what I need and if the price is right. I don't have strong feelings either way." },

  { question: LIKELIHOOD_Q, expected: 4, label: "LIK-4", domain: "likelihood", difficulty: "clear",
    text: "Yeah, I'll probably order again. They had decent prices and good selection. Pretty convenient overall." },

  { question: LIKELIHOOD_Q, expected: 5, label: "LIK-5", domain: "likelihood", difficulty: "clear",
    text: "Definitely! I already have items in my wishlist. Great selection and prices, I'll be back within the week." },

  { question: LIKELIHOOD_Q, expected: 3, label: "LIK-3s", domain: "likelihood", difficulty: "subtle",
    text: "I liked some of their products but the shipping was slow. Could go either way honestly, it's not the only option I have." },

  { question: LIKELIHOOD_Q, expected: 4, label: "LIK-4s", domain: "likelihood", difficulty: "subtle",
    text: "The rewards program makes it worth coming back. Not that I'm super loyal, but the points add up and I already have an account." },

  { question: LIKELIHOOD_Q, expected: 2, label: "LIK-2e", domain: "likelihood", difficulty: "edge",
    text: "I wouldn't say never, but it's not like I'm rushing back. There'd have to be a pretty compelling reason for me to choose them over Amazon." },

  // ════════════════════════════════════════════════════════════════
  // AGREEMENT (8 cases: 5 clear + 2 subtle + 1 edge)
  // ════════════════════════════════════════════════════════════════

  { question: AGREEMENT_Q, expected: 1, label: "AGR-1", domain: "agreement", difficulty: "clear",
    text: "I completely disagree. The product quality has been terrible. It broke after two uses and the materials feel cheap." },

  { question: AGREEMENT_Q, expected: 2, label: "AGR-2", domain: "agreement", difficulty: "clear",
    text: "I don't really agree with that. The quality is below what I expected for the price. There are some issues." },

  { question: AGREEMENT_Q, expected: 3, label: "AGR-3", domain: "agreement", difficulty: "clear",
    text: "I'm kind of neutral on this. The quality is acceptable but nothing special. It does what it's supposed to do." },

  { question: AGREEMENT_Q, expected: 4, label: "AGR-4", domain: "agreement", difficulty: "clear",
    text: "I mostly agree. The product quality is good, well-made and reliable. A few minor things could be improved." },

  { question: AGREEMENT_Q, expected: 5, label: "AGR-5", domain: "agreement", difficulty: "clear",
    text: "Absolutely agree. Outstanding quality in every way. Best product I've used in this category, exceeded all expectations." },

  { question: AGREEMENT_Q, expected: 2, label: "AGR-2s", domain: "agreement", difficulty: "subtle",
    text: "The design is nice but the build quality lets it down. Looks good on the outside but feels fragile when you actually use it." },

  { question: AGREEMENT_Q, expected: 4, label: "AGR-4s", domain: "agreement", difficulty: "subtle",
    text: "For the price point, I think the quality is solid. Not luxury-level, but definitely better than most competitors I've tried." },

  { question: AGREEMENT_Q, expected: 3, label: "AGR-3e", domain: "agreement", difficulty: "edge",
    text: "It's not bad quality per se, but I wouldn't call it good either. I've seen better and I've seen worse. Right in the middle." },

  // ════════════════════════════════════════════════════════════════
  // EASE (9 cases: 5 clear + 2 subtle + 2 edge)
  // ════════════════════════════════════════════════════════════════

  { question: EASE_Q, expected: 1, label: "EASE-1", domain: "ease", difficulty: "clear",
    text: "I gave up after twenty minutes. The search barely worked, the categories were confusing, and the FAQ was useless. Total waste of time." },

  { question: EASE_Q, expected: 2, label: "EASE-2", domain: "ease", difficulty: "clear",
    text: "It took more effort than it should have. I had to dig through several pages and the navigation wasn't intuitive. I found what I needed eventually, but it was a pain." },

  { question: EASE_Q, expected: 3, label: "EASE-3", domain: "ease", difficulty: "clear",
    text: "It was manageable. I had to click around a bit but eventually found what I was looking for. Not the worst, not the best." },

  { question: EASE_Q, expected: 4, label: "EASE-4", domain: "ease", difficulty: "clear",
    text: "It was pretty straightforward. The navigation was clear and I found the product specs within a couple of clicks. No real issues." },

  { question: EASE_Q, expected: 5, label: "EASE-5", domain: "ease", difficulty: "clear",
    text: "Super easy. I typed what I needed in the search bar and it came right up on the first result. Took me like 30 seconds." },

  { question: EASE_Q, expected: 4, label: "EASE-4s", domain: "ease", difficulty: "subtle",
    text: "Pretty intuitive overall. The menu structure made sense and I only had to backtrack once to find the right section." },

  { question: EASE_Q, expected: 2, label: "EASE-2s", domain: "ease", difficulty: "subtle",
    text: "I ended up having to Google the answer because the site's own help section was so poorly organized. Not ideal when the info should be right there." },

  { question: EASE_Q, expected: 3, label: "EASE-3e", domain: "ease", difficulty: "edge",
    text: "The search was terrible but the category pages were actually well-organized. So it was hard and easy at the same time, depending on the approach." },

  { question: EASE_Q, expected: 4, label: "EASE-4e", domain: "ease", difficulty: "edge",
    text: "I wouldn't say it was difficult. Once I figured out where things were, it all flowed naturally. Just a small learning curve at the start." },

  // ════════════════════════════════════════════════════════════════
  // IMPORTANCE (9 cases: 5 clear + 2 subtle + 2 edge)
  // ════════════════════════════════════════════════════════════════

  { question: IMPORTANCE_Q, expected: 1, label: "IMP-1", domain: "importance", difficulty: "clear",
    text: "Shipping speed? Honestly couldn't care less. I plan ahead and I'm never in a rush. I'd rather save money than pay for fast shipping." },

  { question: IMPORTANCE_Q, expected: 2, label: "IMP-2", domain: "importance", difficulty: "clear",
    text: "It's not a huge factor for me. I notice when it's really slow but generally a week or so is fine. Other things like price and selection matter way more." },

  { question: IMPORTANCE_Q, expected: 3, label: "IMP-3", domain: "importance", difficulty: "clear",
    text: "It matters to some extent. I won't pay a premium for next-day shipping usually, but if two stores have similar prices I'd pick the faster one." },

  { question: IMPORTANCE_Q, expected: 4, label: "IMP-4", domain: "importance", difficulty: "clear",
    text: "Shipping speed is quite important to me. I check delivery estimates before placing every order and I usually go with the faster option even if it costs a bit more." },

  { question: IMPORTANCE_Q, expected: 5, label: "IMP-5", domain: "importance", difficulty: "clear",
    text: "This is make-or-break for me. If a store doesn't offer fast shipping, I'm going to Amazon. I'm paying for Prime specifically because I want things quickly." },

  { question: IMPORTANCE_Q, expected: 4, label: "IMP-4s", domain: "importance", difficulty: "subtle",
    text: "Definitely matters. I've cancelled orders before when shipping times were too long. I'm not obsessed with next-day delivery but 5+ business days is too much." },

  { question: IMPORTANCE_Q, expected: 3, label: "IMP-3s", domain: "importance", difficulty: "subtle",
    text: "For some purchases it's critical, like gifts or urgent replacements. For everyday stuff though, I'm pretty patient. Depends on the situation." },

  { question: IMPORTANCE_Q, expected: 2, label: "IMP-2e", domain: "importance", difficulty: "edge",
    text: "I used to care a lot about this but honestly, most things arrive in a few days anyway. It's become less of a differentiator since everyone's gotten faster." },

  { question: IMPORTANCE_Q, expected: 4, label: "IMP-4e", domain: "importance", difficulty: "edge",
    text: "I don't think about shipping speed until it's bad. When something takes two weeks, I'm furious. So I guess it's more important than I'd normally admit." },

  // ════════════════════════════════════════════════════════════════
  // TRUST (9 cases: 5 clear + 2 subtle + 2 edge)
  // ════════════════════════════════════════════════════════════════

  { question: TRUST_Q, expected: 1, label: "TRU-1", domain: "trust", difficulty: "clear",
    text: "No trust whatsoever. I've read about their data breaches and I would never put my credit card info on this site. They've proven they can't be trusted." },

  { question: TRUST_Q, expected: 2, label: "TRU-2", domain: "trust", difficulty: "clear",
    text: "I'm pretty skeptical. Their privacy policy is vague and they ask for way too much personal information. I only use prepaid cards when I buy from them." },

  { question: TRUST_Q, expected: 3, label: "TRU-3", domain: "trust", difficulty: "clear",
    text: "I guess I trust them as much as any other online company. No particular red flags but no special reason to feel safe either. Standard internet caution." },

  { question: TRUST_Q, expected: 4, label: "TRU-4", domain: "trust", difficulty: "clear",
    text: "I trust them with my information. They have solid security badges on their site, encrypted checkout, and I've ordered dozens of times without any problems." },

  { question: TRUST_Q, expected: 5, label: "TRU-5", domain: "trust", difficulty: "clear",
    text: "I completely trust them. They've been transparent about their security practices, they use two-factor auth, and I've never had a single issue in five years." },

  { question: TRUST_Q, expected: 4, label: "TRU-4s", domain: "trust", difficulty: "subtle",
    text: "I feel pretty safe with them. They're a well-known brand with good reviews and I've never heard of any security incidents. I use my real credit card without thinking twice." },

  { question: TRUST_Q, expected: 2, label: "TRU-2s", domain: "trust", difficulty: "subtle",
    text: "They keep sending me marketing emails I didn't sign up for, which makes me wonder what else they do with my data. Small thing, but it erodes confidence." },

  { question: TRUST_Q, expected: 3, label: "TRU-3e", domain: "trust", difficulty: "edge",
    text: "I trust their product quality but not necessarily their data practices. So it depends on what kind of trust we're talking about. For data? Middling." },

  { question: TRUST_Q, expected: 4, label: "TRU-4e", domain: "trust", difficulty: "edge",
    text: "I haven't had a reason not to trust them, which is about the best you can say about any company online these days. No complaints from me." },

  // ════════════════════════════════════════════════════════════════
  // VALUE (8 cases: 5 clear + 2 subtle + 1 edge)
  // ════════════════════════════════════════════════════════════════

  { question: VALUE_Q, expected: 1, label: "VAL-1", domain: "value", difficulty: "clear",
    text: "Complete waste of money. I paid $80 for something that looks and feels like it should cost $20. I feel genuinely ripped off." },

  { question: VALUE_Q, expected: 2, label: "VAL-2", domain: "value", difficulty: "clear",
    text: "Overpriced for what it is. The quality is mediocre and you can find comparable products for much less. I expected more at this price point." },

  { question: VALUE_Q, expected: 3, label: "VAL-3", domain: "value", difficulty: "clear",
    text: "The price is about right for what you get. It's not a steal but it's not overpriced either. Fair exchange of money for product." },

  { question: VALUE_Q, expected: 4, label: "VAL-4", domain: "value", difficulty: "clear",
    text: "Good deal overall. The build quality is impressive for this price range and it comes with useful accessories that would normally cost extra." },

  { question: VALUE_Q, expected: 5, label: "VAL-5", domain: "value", difficulty: "clear",
    text: "Unbelievable deal. The quality is way beyond what I expected at this price point. I would happily have paid twice as much." },

  { question: VALUE_Q, expected: 2, label: "VAL-2s", domain: "value", difficulty: "subtle",
    text: "It's okay but I've seen very similar products for 30% less. At this price I was expecting better materials or more features." },

  { question: VALUE_Q, expected: 4, label: "VAL-4s", domain: "value", difficulty: "subtle",
    text: "Considering the quality and how long it lasts, it's a smart purchase. Not the cheapest option but you definitely get what you pay for and then some." },

  { question: VALUE_Q, expected: 3, label: "VAL-3e", domain: "value", difficulty: "edge",
    text: "It's premium-priced and premium quality, so... is that good value? You get what you pay for but you're paying a lot. Hard to say." },

  // ════════════════════════════════════════════════════════════════
  // PURCHASE INTENT (8 cases: 5 clear + 2 subtle + 1 edge)
  // ════════════════════════════════════════════════════════════════

  { question: PURCHASE_INTENT_Q, expected: 1, label: "PI-1", domain: "purchase_intent", difficulty: "clear",
    text: "Absolutely not buying this. It doesn't solve any problem I have and the reviews are terrible. Hard pass, I'll save my money." },

  { question: PURCHASE_INTENT_Q, expected: 2, label: "PI-2", domain: "purchase_intent", difficulty: "clear",
    text: "I doubt I'll buy this. It looks decent but I really don't need another one and the price isn't low enough to tempt me into an impulse purchase." },

  { question: PURCHASE_INTENT_Q, expected: 3, label: "PI-3", domain: "purchase_intent", difficulty: "clear",
    text: "I'm going back and forth on this one. It looks useful but I'm not sure I need it right now. I might wait and see if it goes on sale." },

  { question: PURCHASE_INTENT_Q, expected: 4, label: "PI-4", domain: "purchase_intent", difficulty: "clear",
    text: "I'm leaning towards buying this. It fits my needs well and the reviews are encouraging. I'll most likely order it by the end of the week." },

  { question: PURCHASE_INTENT_Q, expected: 5, label: "PI-5", domain: "purchase_intent", difficulty: "clear",
    text: "Already added it to my cart. This is exactly what I've been researching for weeks and the reviews confirm it's the best option. Buying today." },

  { question: PURCHASE_INTENT_Q, expected: 4, label: "PI-4s", domain: "purchase_intent", difficulty: "subtle",
    text: "I'll probably pick this up next paycheck. I've been eyeing it for a while and the features look right for what I need. Just want to sleep on it first." },

  { question: PURCHASE_INTENT_Q, expected: 2, label: "PI-2s", domain: "purchase_intent", difficulty: "subtle",
    text: "It's interesting but I have a tight budget this month. Maybe down the road, but right now there are more important things to spend on." },

  { question: PURCHASE_INTENT_Q, expected: 3, label: "PI-3e", domain: "purchase_intent", difficulty: "edge",
    text: "If someone gave it to me I'd definitely use it, but I'm not sure I'd go out of my way to buy it myself. The need isn't urgent enough." },
];

// ─── Utility functions ───────────────────────────────────────────

export function getTestCasesByDomain(domain: string): TestCase[] {
  return TEST_CASES.filter(tc => tc.domain === domain);
}

export function getTestCasesByDifficulty(difficulty: TestCase["difficulty"]): TestCase[] {
  return TEST_CASES.filter(tc => tc.difficulty === difficulty);
}

export function getDomains(): string[] {
  return [...new Set(TEST_CASES.map(tc => tc.domain))];
}

// ─── Summary statistics ──────────────────────────────────────────

export function printTestSetSummary(): void {
  console.log("Ground Truth v2 — Test Set Summary");
  console.log("=".repeat(50));

  const domains = getDomains();
  let total = 0;
  for (const domain of domains) {
    const cases = getTestCasesByDomain(domain);
    const clear = cases.filter(c => c.difficulty === "clear").length;
    const subtle = cases.filter(c => c.difficulty === "subtle").length;
    const edge = cases.filter(c => c.difficulty === "edge").length;
    console.log(`  ${domain.padEnd(18)} ${cases.length} cases (${clear} clear, ${subtle} subtle, ${edge} edge)`);
    total += cases.length;
  }
  console.log("─".repeat(50));
  console.log(`  ${"TOTAL".padEnd(18)} ${total} cases`);

  // Rating distribution
  console.log("\nRating distribution:");
  for (let r = 1; r <= 5; r++) {
    const count = TEST_CASES.filter(tc => tc.expected === r).length;
    console.log(`  Rating ${r}: ${count} cases`);
  }
}
