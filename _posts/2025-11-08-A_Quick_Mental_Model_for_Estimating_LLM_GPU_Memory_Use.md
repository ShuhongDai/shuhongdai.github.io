---
layout: post
title: A Quick Mental Model for Estimating LLM GPU Memory Use
date: 2025-11-17 01:00:00
description: Before downloading a large model or spinning up a container, it’s useful to know whether an open-source LLM will actually fit on your GPU.
tags: ["LLMs", "GPU Memory"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-17/llm-memory-thumb.png
toc:
  sidebar: left
---

## Introduction

When browsing new open-source LLM releases, I often have a simple question in mind:  
**Will this model actually fit on my GPU?**

Sometimes the model page shows numbers like $$ 7\text{B} $$, $$ 14\text{B} $$, or $$ 70\text{B} $$, but that alone doesn’t immediately translate into how much memory the model needs once loaded and running. And when I only want a quick sanity check, I don’t want to:

- download tens of gigabytes of weights,  
- install a full environment,  
- start a runtime,  
- and only then discover that the model does not fit on the device at all.

For this kind of lightweight judgment, a rough mental model is far more helpful than an exact calculator. It doesn’t need to be accurate to the megabyte. It only needs to answer a practical question: **“Roughly fits?” or “Clearly too large?”**

This note summarizes the approximation that I use. It’s not a formal derivation. It’s simply a way to reason about LLM memory requirements quickly, in a way that works consistently across models.

---

## What Actually Occupies GPU Memory

For inference (not training), only a few components meaningfully consume GPU memory:

1. The model weights  
2. The key–value cache used during autoregressive generation  
3. Runtime overhead: intermediate buffers, framework allocations, small activations  

Optimizer state does not exist during inference, so the overall picture is simpler than training.

My routine is just:

- estimate weight memory,  
- add the KV cache,  
- apply a small safety margin.

That’s enough for a reliable first impression.

---

## From Parameters to VRAM

Model parameter counts are usually prominently displayed: $$ 7\text{B} $$, $$ 13\text{B} $$, $$ 34\text{B} $$, $$ 70\text{B} $$, and so on. Converting this into VRAM is straightforward once we remember how many bytes each parameter uses.

Typical cases:

- **FP16 / BF16:** $$ 2 $$ bytes per parameter  
- **FP32:** $$ 4 $$ bytes  
- **8-bit quantization:** about $$ 1 $$ byte  
- **4-bit quantization:** about $$ 0.5 $$ byte  

This is already enough for fast estimation:

- **7B FP16 model** → roughly $$ 7 \times 2 $$ GB ≈ **14 GB**  
- **13B FP16 model** → $$ 13 \times 2 $$ GB ≈ **26 GB**  
- **7B 4-bit model** → $$ 7 \times 0.5 $$ GB ≈ **3.5 GB**  

This accounts only for the parameters, not the KV cache. But it gives a solid baseline.

---

## KV Cache and Context Length

During generation, transformer decoders maintain a key–value cache for each attention layer. Its size scales with:

- number of layers,  
- hidden dimension,  
- context length.

The exact computation is more detailed, but for estimation purposes, only the magnitude matters. In practice: **KV cache often contributes hundreds of megabytes to several gigabytes**, depending on the context window.

For many modern $$ 6\text{B} $$–$$ 8\text{B} $$-scale models in FP16:

- every $$ 1\text{k} $$ tokens of context usually costs **a few hundred MB** of KV cache  
- a $$ 4\text{k} $$–$$ 8\text{k} $$ context easily adds **1–3 GB**  
- long-context models might require more  

This rough rule-of-thumb is accurate enough to determine whether a model with a particular context window fits on an average $$ 24\text{GB} $$–$$ 48\text{GB} $$ GPU.

---

## Putting the Pieces Together

Once the weight size and KV cache are roughly known, the total memory is just the sum plus a safety margin. In practice, I use a simple heuristic.

First, compute an approximate parameter memory from:

$$
\text{ParameterMemory} \approx \text{NumberOfParameters} \times \text{BytesPerParameter}
$$

Then add room for the KV cache, based on the context length. Finally, leave some headroom to account for framework overhead and fragmentation.

If I want to compress this into a single sentence I can recall mentally, it would be:

> **Take the parameter size, add a few gigabytes for KV cache, add a buffer, and that’s your practical VRAM requirement.**


---

## Additional Notes

A few architectural details can influence memory usage in practice, even when using the simple estimation model above:

1. **KV cache usage varies across architectures**
   Hidden sizes and layer counts differ between models. For example, even at similar 7B scales, newer efficient architectures such as **Qwen2.5** and **Mistral** typically use less KV memory per 1k tokens than earlier LLaMA-style models, and smaller models require even less. The “hundreds of MB per 1k tokens” rule still holds, but the exact amount can vary.

2. **A small amount of extra weight tensors exists**
   Beyond the main linear weights, models include embeddings, LayerNorm parameters, and other small tensors. These usually contribute under 5% of the total size, so the rough estimate remains valid, but the actual VRAM will be slightly higher than the simple parameter × bytes calculation.

3. **Quantized models may include additional metadata**
   Although 4-bit and 8-bit quantization can be estimated as 0.5 or 1 byte per parameter, many implementations store per-channel scales, zero-points, or other auxiliary data. This means the practical VRAM usage is often somewhat higher than the theoretical minimum.

These nuances don’t affect the overall direction of the estimate but matter when pushing GPU limits or optimizing for tight VRAM constraints.