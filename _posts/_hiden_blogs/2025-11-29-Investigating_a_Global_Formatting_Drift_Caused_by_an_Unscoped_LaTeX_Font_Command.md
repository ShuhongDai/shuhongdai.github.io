---
layout: post
title: Investigating a Global Formatting Drift Caused by an Unscoped LaTeX Font Command
date: 2025-11-29 14:00:00
description: A brief account of how an oversized section led to uncovering a hidden font-scope leak inside a KOMA-Script template.
tags: ["LaTeX", "KOMA-Script", "Debugging"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-17/llm-memory-thumb.png
toc:
  sidebar: left
---

## Introduction

I had been adapting a university that shall remain unnamed thesis template into a lightweight format for a proposal. Nothing exotic just a cover page, a short executive summary, and a few sections. The kind of task that should be uneventful. Yet somewhere along the way, LaTeX decided that portions of my document deserved to be set in billboard-sized fonts with the line spacing of a children’s picture book. Curiously, the *Summary* section looked perfectly ordinary, while *Motivation*, which was only a few lines below, expanded into an oversized landscape.

The discrepancy couldn’t be random. Whenever LaTeX behaves this unevenly, it usually indicates a shift in page style, a class-level environment, or some seemingly harmless command with global side effects. I kept that thought in mind as I began testing every assumption.

---

## Problem Statement
The problem appeared the moment I recompiled after reorganizing the early pages. The *Motivation* section ballooned in both type size and interline spacing. My first instinct was to check the obvious culprits: section formatting and page numbering. KOMA-Script, after all, is known for embedding layout logic in places one might not expect.

I examined the relevant lines:

```latex
\sectionfont{\color{blue}}
```

The earlier `\sectionfont{\fontsize{16}{15}\selectfont}` had already been removed, so nothing here should have forced a larger font. To verify, I replaced `\section{Motivation}` with a starred version. No change. The environment clearly had deeper roots.

That led me to the page-numbering suspicion. In KOMA-Script, switching between `roman` and `arabic` modes sometimes alters spacing rules. I tried relocating *Motivation* to the region after `\pagenumbering{arabic}`. The formatting snapped back to normal, which initially looked like evidence. But once I trimmed the document down further, that explanation began to feel insufficient.

---

## Re-examining the Early Pages

Once I stripped the project to its bones, only the cover page remained before the *Motivation* section. At that point, the pattern became unmistakable: whenever the cover page was included, *Motivation* inherited the exaggerated font and line spacing. Without the cover page, the section displayed correctly.

This suggested a scope leak originating from the cover code. Something there must have altered typography at a global level.

I returned to `CoverPage.tex` and inspected every line. The core layout was straightforward including some logos, a few horizontal spaces, and the title block. Yet buried inside that block sat the following line:

```latex
\fontsize{20}{50}\selectfont
```

Crucially, it lacked any bounding braces. In LaTeX, `\fontsize{...}\selectfont` is not self-contained; unless explicitly confined, it rewrites the current font settings for *all subsequent text*. In hindsight, it was inevitable that everything following the cover would adopt the same 20pt size and 50pt baseline skip.

Originally, the full thesis template wrapped its cover page inside a group, so the leaked settings never reached the rest of the document. In refactoring the template, I had unintentionally removed the very container that prevented this contamination.

To test that theory, I inserted a single line immediately after loading the cover:

```latex
\normalsize
```

Recompile. The *Motivation* section returned to its usual scale, and the spacing tightened to what one expects from an 11pt scrartcl document. That simple reset erased the entire anomaly.

For completeness, I enclosed the suspicious portion of the cover page within braces:

```latex
{
    \fontsize{20}{50}\selectfont
    ... cover title ...
}
```

This also resolved the issue. At that point, the cause was beyond doubt: the cover page had been bleeding its oversized font configuration directly into the main document.

The minimal fix is trivial, because you can wrap any `\fontsize` usage in braces or manually restore the normal font after the cover. But the episode is a reminder of how scope works in TeX. Group boundaries, not files or page breaks, define where local formatting ends. Removing a wrapper can silently transform code that once behaved safely into something more invasive. In this context, KOMA-Script’s page-numbering conventions were merely noise. The real mechanism was much simpler: a global font state set early in the document and never returned to baseline.

---