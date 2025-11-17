---
layout: post
title: A Measure of Range Compression on Different Board Textures
date: 2025-08-03 03:10:00
description: A mathematical approach to quantifying how different flop textures compress or expand pre-flop ranges in No-Limit Hold’em.
tags: ["Poker", "Probability", "Combinatorics", "Game Theory", "Information Theory"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-17/range-compression-thumb.png
toc:
  sidebar: left
---

## Introduction

I noticed a pattern that I had always felt intuitively but never formalized. Certain flops seemed to “collapse” the strategic possibilities of both players, making ranges narrower, more predictable, and structurally simpler. Others seemed to do the opposite, expanding the number of viable continuations.

This observation naturally led to a question:

**Is there a mathematical way to measure how much a flop compresses (or expands) a range?**

Poker players talk about “range advantage” and “nut advantage” frequently, but rarely about *range volume* what is the total weight of possible holdings consistent with rational play. Compressing this volume changes not only equities but also the informational structure of the hand.

In this post, I propose a formal measure of **range compression**, analyze its mathematical properties, and illustrate how different board textures affect the relationship between pre-flop and post-flop ranges.

---

## Range Volume

Let $$ R $$ be a player’s pre-flop range, represented as a set of weighted combinations:

$$
R = \{ (h_i, w_i) \}_{i=1}^N,
$$

where $$ h_i $$ is a specific starting hand and $$ w_i $$ is its probability weight.

Define the **range volume** as:

$$
V(R) = \sum_{i=1}^N w_i.
$$

For standard pre-flop ranges normalized to 100%, we simply have $$ V(R) = 1 $$. However, the concept becomes informative **after conditioning on a board**.

---

## Conditioning a Range on a Board

Let $$ B $$ be a flop (e.g., $$ \text{A♠ 7♦ 3♠} $$). Any starting hand inconsistent with $$ B $$ must be removed.

Define the conditioned range:

$$
R \mid B = \{ (h_i, w_i) : h_i \text{ does not conflict with } B \}.
$$

Then the post-flop range volume is:

$$
V(R \mid B) = \sum_{h_i \not\!\pitchfork\, B} w_i.
$$

This is the **raw volume**, reflecting how much of the pre-flop range survived the flop card removal.

---

## Range Compression Ratio

Now we define the central object of this article:

$$
\rho(B) = \frac{V(R \mid B)}{V(R)}.
$$

Because $$ V(R) = 1 $$ pre-flop, $$ \rho(B) = V(R \mid B) $$ directly measures how much of the range remains viable after the board is revealed.

Interpretation:

- If $$ \rho(B) \approx 1 $$ → **little compression**, wide continuation range  
- If $$ \rho(B) \ll 1 $$ → **high compression**, many hands eliminated  
- If two players have different $$ \rho $$, the one with higher $$ \rho $$ often holds **range advantage**


---

## Structural Decomposition of Compression

To understand what drives $$ \rho(B) $$, we decompose it into components.

### 1. Card-Removal Compression

This is the strict combinatorial effect:

$$
\rho_{\text{removal}}(B) 
= 
\frac{\text{# of surviving combos}}{\text{# of total combos}}.
$$

For example, removing an Ace from the deck eliminates 3 combinations of every Ax hand but leaves others untouched.

---

### 2. Strength-Based Compression

Beyond card removal, a board may render many holdings strategically non-viable.

To model this, define a viability indicator function:

$$
\chi(h_i, B) 
= 
\begin{cases}
1, & \text{if } h_i \text{ is playable on } B, \\
0, & \text{otherwise}.
\end{cases}
$$

Playability may be defined through:

- minimum equity threshold  
- minimum EV threshold  
- solver-derived continuation frequency  

Then define:

$$
\rho_{\text{strength}}(B) 
= 
\sum_i w_i \chi(h_i, B).
$$

---

### 3. Total Compression

Putting the two together:

$$
\rho(B) 
= 
\rho_{\text{removal}}(B) 
\times 
\rho_{\text{strength}}(B).
$$

This formula mirrors classical probability decompositions:

- structural elimination (card removal)  
- behavioral elimination (strategic folding)

The same structure appears in Bayesian conditioning.

---

## Examples

To illustrate how different boards shape the range, I ran simulations on a typical button-opening range against a big blind defend range.

Here are approximate compression ratios for representative flops:

```

Board           ρ(B)
A♠ K♦ 5♣        0.31
7♣ 8♣ 9♠        0.54
Q♥ 7♠ 2♦        0.62
3♦ 3♣ 3♠        0.95
T♠ J♠ Q♠        0.28

```

Interpretation:

- **A K 5 rainbow** highly compresses ranges: many hands are dead or dominated.  
- **777 or 333 boards** preserve most of the range volume: few hands are eliminated.  
- **T J Q monotone** compresses massively due to strong dominance and nuttiness.  
- **Q 7 2 rainbow** is one of the least compressive typical flops.

These results align well with expert intuition, but here they arise from formal volume computation.

---

## Information Compression

Volume alone does not capture how *uncertain* a player’s range remains.  A more refined measure is the **entropy** of the range:

$$
H(R) = - \sum_i w_i \log w_i.
$$

After conditioning on a board:

$$
H(R \mid B) 
= 
- \sum_{h_i \not\!\pitchfork\, B} \frac{w_i}{\rho(B)} 
\log 
\left(
\frac{w_i}{\rho(B)}
\right).
$$

Define information compression:

$$
\kappa(B) 
= 
\frac{H(R \mid B)}{H(R)}.
$$

Now we have two complementary measures:

- $$ \rho(B) $$ → *how many* hands survive  
- $$ \kappa(B) $$ → *how much uncertainty* survives  

A flop like A K 5 drastically reduces both and a flop like 3 3 3 hardly affects either.

---

## An Information-Theoretic Interpretation

Let the range be a probability distribution over combos. Let actions be random variables dependent on the board. Then the mutual information between range and action is:

$$
I(R; B) = H(R) - H(R \mid B).
$$

A highly compressive board has high $$ I(R; B) $$. This reframes board texture as an information revelation process:

- Dry boards reveal almost nothing.  
- A K x reveals a lot.  
- Coordinated connected boards reveal a different kind of structure (relative nuts density).

---

## Conclusion

This article introduced a mathematical framework for quantifying how different board textures compress pre-flop ranges. By defining:

- range volume $$ \rho(B) $$  
- information compression $$ \kappa(B) $$  
- and decomposing them into structural and strategic components  

we obtain a principled way to study the impact of board textures on strategic play.