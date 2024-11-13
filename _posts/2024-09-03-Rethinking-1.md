---
layout: distill
title: "Rethinking an Olefin Oligomerization from Three Years Ago – Chapter 1: Problem Statement and Some Initial Thoughts from Back Then"
description: an example of a distill-style blog post and main elements
typograms: true
tikzjax: true
thumbnail:
category:  A Column on Rethinking a Mathematical Modeling Competition
tags: ["Machine Learning", "Modeling", "Algorithms"]
date: 2024-09-03
authors:
  - name: Shuhong Dai
    affiliations:
      name:  North China Electric Power University & AI Lab, CRRC Academy
  - name: Saiqin Mao
    affiliations:
      name:  Xiamen University
  - name:  Jun-jun Li
    affiliations:
      name: Shanghai Jiao Tong University & Shanghai AI Lab 


---



In September 2021, following an intensive two-month summer training program in mathematical modeling, I embarked on my third year as an electrical engineering undergraduate and prepared for the grueling 72-hour National Mathematical Modeling Contest. Yet, the most memorable part of this experience was not the 20 hours I lost wrestling with a complex telecommunications problem—one I barely understood at the time, labeled as Problem  A. Instead, it was the incident at the gym just before the contest began. Following my usual routine, I was midway through a shoulder lateral raise on my fourth set when I received an unexpected call from the dean of the mathematics faculty. He asked, rather bluntly, why I wasn’t preparing for the competition. Shortly after, I found myself hastily reporting to his office, where he proceeded to berate me in no uncertain terms.

```typograms
.------------------------------------------.
|.----------------------------------------.|
||      "https://www.mcm.edu.cn"          ||
|'----------------------------------------'|
| ________________________________________ |
||                                        ||
||   * Problem A                          ||
||   Shape Adjustment of a FAST           ||
||   Active Reflector                     ||
||                                        ||
||   * Problem B                          ||
||   Ethanol Coupling to Prepare          ||
||   C4 Olefins                           ||
||                                        ||
||   * Problem C                          ||
||   Raw Material Ordering and            ||
||   Transportation for Manufacturing     ||
||                                        ||
||   * Problem D                          ||
||   Online Optimization of               ||
||   Continuous Casting and Cutting       ||
||                                        ||
||   * Problem E                          ||
||   Identification of Traditional        ||
||   Chinese Medicinal Materials          ||
||                                        ||
|+----------------------------------------+|
.------------------------------------------.
```
I often recount this episode as a humorous anecdote to friends, one of those peculiar experiences that, in hindsight, makes for a good story. But to return to the topic at hand: recently, I’ve been coming across the term “computational chemistry” more frequently. My close friend, Carlos Zhou, was an undergraduate in this field before shifting to computer science for his master’s. As a complete outsider to the world of chemistry, I may have some grounding in scientific computing and a passing interest in the trending field of AI4Chemistry, keeping tabs on its applications in drug discovery and protein structure prediction. But when it comes to the intricacies of organic chemistry, I remain wholly ignorant.



Nevertheless, this hasn’t stopped me from recalling my choice during that September competition to pivot to Problem B—*Ethanol Coupling to Prepare C4 Olefins*—after abandoning Problem E. Looking back, I can’t quite remember why I chose this particular problem, especially with no background in organic chemistry to speak of. Perhaps I assumed, rather simplistically, that a mathematical modeling competition would primarily demand mathematical rigor, with little reliance on actual chemistry knowledge (an assumption that proved largely correct). Or perhaps I dismissed it as a straightforward data analysis exercise.

Even now, I’m uncertain whether I’d be capable of fully solving that problem today (though, at the time, I did win a first prize). Logically speaking, my current thinking and methodological skills are undoubtedly far superior to what they were back then. Yet I still remember the unease I felt when submitting the MD5 hash after finishing the competition, feeling that my work had fallen short. As I watch today’s luminaries repeatedly “rethink” familiar concepts in top-tier computer science conferences, I’m often anxious over my own lack of publications. But, knowing my limitations, I find myself confined to my zero-impact-factor blog, where I “rethink” this foundational problem in mathematical modeling. Perhaps one day, this too will become another amusing anecdote.

In short, I plan to launch this column, aiming to write four to six blog posts as part of a “rethinking” series. My goal is to revisit this problem, attempting to solve it again three years later, while sharing various related musings along the way. To start, I’ll present the original problem statement and dataset in Chinese, along with my English translation. Then, I’ll give a brief overview of the approach used in my award-winning paper from back then (though most of it has faded from memory—primarily some straightforward machine learning algorithms). Finally, I’ll enter the “rethinking” phase, where my present self takes on my younger, less experienced self—a playful exercise in intellectual overmatch. (In Chinese, there’s a phrase I like: “用前朝的剑斩本朝的官.” While the nuances are quite different here, I enjoy the irreverent tone.) Of course, you can also question this "rethinking" as lacking substance, meaning, novelty, feasibility... lacking everything. That's your freedom, and it's very likely everyone's consensus, haha.
