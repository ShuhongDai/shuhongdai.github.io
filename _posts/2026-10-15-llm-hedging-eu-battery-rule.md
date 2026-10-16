---
layout: post
title: How Defensively Two LLMs Answered a TikTok Video on EU Battery Rules
date: 2026-10-15
description: Checking a TikTok video's claim about the EU Battery Regulation with Claude and ChatGPT, and what the difference between their answers says about hedging in LLMs.
tags: ["llm", "rlhf", "Defensive"]
tabs: true
# thumbnail: /assets/posts_img/2026-10-15/llm-hedging-thumb.png
toc:
  sidebar: left
---

## How This Started

I came across a short video on TikTok claiming that the EU had laid down an "iron rule": starting February 18, 2027, every phone sold on the EU market would have to comply with the new Battery Regulation passed in 2023. The clip was set against footage of the European Parliament, which gave it an air of authority.

I was thinking about buying a new iPhone, so this mattered to me directly. If the rule meant I could swap the battery myself, waiting might be worth it. I took a screenshot, asked an AI whether this was true, and used the occasion to run a small side-by-side comparison.

---

## The Test

I asked Claude the following questions in order, then sent the exact same prompts to ChatGPT:

1. I sent the screenshot and asked, "Do you know about this?"
2. I followed up: "So if I buy an iPhone after this date, I'll be able to replace the battery myself instead of paying Apple a lot for it?"
3. I asked it to look up how battery replacement works on the latest iPhone and how easy it is.
4. I asked whether models released after next year would be any different.

The second question was the one I cared about. It carries a hidden assumption: that the new rule will make iPhone batteries easier to replace yourself. I wanted to see whether the model would simply go along with it.

---

## Comparing the Two Answers

Neither answer contained an obvious factual error. The difference was whether the model caught the false premise in my question.

| | Claude | ChatGPT |
| --- | --- | --- |
| Opening line | "Not quite," rejecting the premise right away | "Yes, your understanding is basically correct" |
| Exemption for phones | Pointed out that an iPhone can keep its current design as long as it meets the battery-durability and water-resistance conditions | Not mentioned |
| Where the iPhone stands today | Explained that iPhones have already had to meet the EU's smartphone battery rules since 2025, so 2027 changes little | Implied Apple could no longer rely on complicated disassembly |
| Practical advice | Self-service repair, third-party repair shops, ways to extend battery life | You can buy compliant third-party batteries |
| A point the other missed | Noted that the video belongs to a type of claim tech media had already debunked | Noted that the regulation applies based on when a product is placed on the market, not when you buy it |

In the follow-ups, Claude found a teardown of the iPhone 18 Pro. The battery sits in a screwed-in tray and is easy to swap on its own, but you have to remove the glued-down display first, and in three of iFixit's four teardowns the frame cracked in the process. That detail is what finally made it clear to me that replacing the battery is nowhere near as simple as the video suggested.

---

## What I Took Away

I was happy with Claude here, mainly because it was willing to commit to an answer.

Every sentence in ChatGPT's answer was defensible on its own, but it was so full of "in principle," "not necessarily," and "is expected to" that I finished reading and still didn't know whether I should wait. I wasn't planning to do anything shady with the answer, and I wasn't going to blame it if it turned out to be wrong. Yet it kept leaving itself a way out, and the useful information got buried.

Claude put the conclusion in the first sentence and then laid out the conditions and exceptions. Where it was confident, it said so plainly. On things it couldn't know, like next year's iPhone, which hasn't been announced yet, it said clearly that it could only speculate. It saved its caution for the places that actually called for it.

This also went against an impression I'd had. Since Anthropic is known for its focus on safety and alignment, I expected Claude to be the more conservative of the two. After this, I no longer think being safe and being defensive are the same thing.

---

## Where Does the Hedging Come From?

Out of curiosity, I asked Claude whether ChatGPT's defensiveness comes from its system prompt or is baked in during pretraining. It said it had no view into OpenAI's internals and could only estimate based on what is publicly known about how these models are trained. Its guess was that the behavior **comes mostly from post-training rather than pretraining, with the system prompt playing a smaller role**.

Its reasoning went roughly like this:

- **Pretraining.** The model learns the distribution of an enormous amount of text. The internet contains both blunt statements and evasive ones, so pretraining alone shouldn't push a model much toward hedging.
- **Post-training (fine-tuning and RLHF).** This is where a model's "personality" gets shaped, and the scoring is asymmetric. Being wrong or overly assertive gets penalized; adding a few qualifiers rarely does. Over time, the model learns that hedging every sentence is a safe way to score well. Safety training that is tuned too aggressively can also spill over into a general habit of evasiveness.
- **System prompt.** It can adjust tone, but it works more like a light adjustment on top of an existing personality. A tendency that shows up across topics and is hard to shake by rephrasing the question looks more like something that settled in during training.

It added that this is ultimately a trade-off each company makes. A more defensive model makes fewer mistakes and draws fewer complaints, at the cost of information density and usefulness.

I then asked the obvious follow-up: if Claude is known for safety and alignment, why isn't it more defensive? Its answer was that safety and defensiveness are different things, and in some ways they pull in opposite directions. Declining to help doesn't make an answer safe, and a vague answer can lead someone to a bad decision. In its view, safety means drawing firm lines around the small number of things that are genuinely risky, while on everyday questions, being honest and useful is itself part of alignment. Honesty also includes calibration: saying it directly when it's sure, and pointing out where it isn't.

---

## Conclusion

I came away sure of two things. First, short-video headlines like this one are often half fact and half exaggeration. The regulation is real, but the conclusion drawn from it is inflated. Second, I don't want an AI that answers everything with "it depends." I want one that gives me the conclusion first and is clear about what it isn't sure of.

As for my iPhone, there's no reason to hold off until 2027 just for this rule.
