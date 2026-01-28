---
layout: post
title: "Re-running an RL Experiment and Getting a Different Answer"
date: 2024-11-17 02:08:00
description: A  engineering reflection on why two RTX 4090 machines produced diverging RL curves despite identical code, seeds, and configurations. And what this reveals about RL’s numerical sensitivity.
tags: ["Reinforcement Learning", "CUDA", "Numerical Stability", "Reproducibility"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-16/replay-gpu-divergence-thumb.png
toc:
  sidebar: left
---

## Introduction

Not long ago, I tried to reproduce one of my RL experiments on a cloud server. The same code had run earlier on a local lab machine, and both hosts were equipped with **NVIDIA RTX 4090 GPUs**. The driver versions matched, the CUDA and PyTorch versions were identical, the environment dependencies mirrored each other, and every random seed was fixed. Under such conditions, the expectation was simple: the two training curves should overlap almost perfectly.

But this time, they didn’t. For the first few thousand steps, everything behaved exactly as expected. The lines overlapped so closely that they were visually indistinguishable. As training continued, however, a slight divergence appeared, barely noticeable at first, then increasingly persistent. Eventually, the two runs settled into significantly different behaviors. All high-level variables were controlled, and yet the divergence persisted.It wasn’t dramatic, but it was unmistakable. And it prompted me to re-examine some of the more fragile aspects of RL systems that often go unnoticed.


---

## Two Curves That Should Have Been One

The most striking part of this incident was how cleanly the divergence unfolded. During the early stage, the critic’s loss, the policy statistics, and the rewards from the environment aligned almost exactly between the two machines. The curves felt stable, even reassuring.  

Then the shift began. It wasn’t a sudden jump but a slow drift, like two lines that started parallel but eventually grew a small angle between them. Once the angle existed, the distance between the lines increased gradually and inevitably. What began as a tiny deviation eventually widened into a visible performance gap.

This kind of “quiet drift” is rare in supervised learning, but painfully common in RL, where feedback loops amplify small differences.

---

## Investigation

I didn’t start by suspecting the GPUs. Instead, I reviewed the usual suspects in a calm, methodical way:  

Whether the environment returned identical resets, whether the model initialization and random number streams matched, whether the replay buffer might have desynchronized the sampling order, whether the logging system affected timing, and whether the training loop had implicit branches that could influence execution order.

All these checks were quick to perform. Nothing at the framework or data-flow level explained the divergence. Which meant the issue had to be buried deeper than most RL bugs—deeper than Python, deeper than PyTorch, deeper than CUDA kernels invoked explicitly in code.

---

## Clue

The earliest measurable drift appeared not in the policy’s actions, but in the critic’s value estimates. That itself was a clue. The critic is often the most numerically sensitive component in many RL algorithms, and its outputs feed directly into the policy update. If two identical systems begin to disagree, the critic is a natural place to look.

The differences were tiny that barely outside floating-point noise but detectable under scrutiny. In the early stage of training, those deviations did not affect behavior, but they were already present. And because the critic affects everything downstream, even a small mismatch is enough for a long feedback loop to magnify.

---

## A Subtle Difference From Identical Hardware

Eventually, the real cause became clear:  
two GPUs of the same model can still produce slightly different floating-point results.

This sounds counterintuitive at first, but it’s neither rare nor mysterious. Several factors can produce such differences:

- Slight variations in how CUDA dispatches certain kernels  
- Minor differences introduced by fused multiply–add behavior  
- Driver-level optimizations that change arithmetic ordering  
- Subtle kernel selection differences across installations  

The scale of these discrepancies is extremely small on the order of the last few bits of a float. They are invisible in most workloads. In supervised learning, such noise is diluted by batching, averaging, and the absence of recurrent dependencies. In RL, however, the story is different. RL acts as an amplifier. A microscopic variance in a critic output can change a gradient, which changes a policy, which changes the data distribution, which changes the critic’s future inputs, and so on. Over tens of thousands of iterations, this accumulation can transform an imperceptible discrepancy into a meaningful behavioral difference.

At the heart of this phenomenon is the recursive nature of RL training. The critic’s estimation errors influence the policy update. The policy influences the trajectory of states and rewards. These, in turn, influence how the critic is updated. The loop continues, step after step.

This structure makes RL far more sensitive to numerical discrepancies than most other machine learning pipelines. A difference invisible at step 2,000 can become visible at step 50,000 simply because it is allowed to feed back into itself. This is not a bug; it is a property of RL.