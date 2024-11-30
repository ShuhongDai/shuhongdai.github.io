---
layout: distill
title: "Rethinking an Olefin Oligomerization from Three Years Ago – Chapter 1: Problem Statement and Some Initial Thoughts from Back Then"
description:  Treating experimental conditions as single-variable parameters or ignoring “minor” byproducts that seemed irrelevant at the time. Were those assumptions valid? Did we truly grasp the core requirements of the problem?
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

toc:
  - name: Problem Statement
  - name: The Initial and Naive Problem-Solving Approach
  - name: Conclusion

---

## Problem Statement

### Introduction

My memories of the problem Ethanol Coupling to Prepare C4 Olefins aren’t particularly vivid. It wasn’t as baffling as Problem A, which I struggled to grasp, nor did it prompt the immediate abandonment that Problem E did. Yet, in its own subtle way, it piqued my curiosity. A problem rooted in chemistry, presented to participants like me—engineers and mathematicians with little to no background in the field—carried a distinct air of unfamiliarity and challenge. At the time, I knew next to nothing about terms like “ethanol coupling” or “C4 olefins.” All I had to rely on was a succinct problem statement and a dataset provided in the competition materials. Yet, something about the problem hinted at opportunities where mathematical modeling could shine—a chance to deliver results that didn’t necessarily require deep expertise in chemistry but instead relied on computational tools and creativity.The problem description seemed straightforward at first glance—a few paragraphs of explanation and a dataset. It required participants to construct a mathematical model, based on the provided experimental data, to optimize the conditions of an ethanol coupling reaction for maximum C4 olefin yield. On the surface, this sounded like a typical modeling task: input data, fit a model, validate results, and generate outputs. But as I dove deeper into the problem, the challenges began to reveal themselves.

Ethanol coupling is a process that involves complex reaction mechanisms and multiple byproducts. While the dataset was detailed, it was clearly insufficient to capture the full dynamics of the reaction. What we faced was a highly nonlinear optimization problem with unclear boundaries. To make matters worse, with no background in chemistry, my team and I had to rely on several “common sense” assumptions: for example, treating experimental conditions as single-variable parameters or ignoring “minor” byproducts that seemed irrelevant at the time. Were those assumptions valid? Did we truly grasp the core requirements of the problem? In the heat of competition, we didn’t ask these questions. Under the pressure of a tight timeline, our sole focus was on building something that looked “reasonable” and delivered results. This fast-paced approach ultimately won us an award—but I now wonder if it can stand up to scrutiny. Three years later, I approach this problem with a new perspective, which is why I’ve chosen to revisit it.

Below, I will present the original Chinese problem statement along with my own English translation. You can also visit [the National Undergraduate Mathematical Modeling Contest official website](https://www.mcm.edu.cn/) to access the original 2021 problem and dataset. The source files are also available via the links included in the problem statement for direct viewing.


### Chinese Version

**B 题 乙醇偶合制备 C4 烯烃**

C4 烯烃广泛应用于化工产品及医药的生产，乙醇是生产制备 C4 烯烃的原料。在制备过程中，催化剂组合（即：Co 负载量、Co/SiO2 和 HAP 装料比、乙醇浓度的组合）与温度对 C4 烯烃的选择性和 C4 烯烃收率将产生影响（名词解释见附录）。因此通过对催化剂组合设计，探索乙醇催化偶合制备 C4 烯烃的工艺条件具有非常重要的意义和价值。

​某化工实验室针对不同催化剂在不同温度下做了一系列实验，结果如附件 1 和附件 2 所示。请通过数学建模完成下列问题：

​(1) 对附件 1 中每种催化剂组合，分别研究乙醇转化率、C4 烯烃的选择性与温度的关系，并对附件 2 中 350 度时给定的催化剂组合在一次实验不同时间的测试结果进行分析。

​(2) 探讨不同催化剂组合及温度对乙醇转化率以及 C4 烯烃选择性大小的影响。

​(3) 如何选择催化剂组合与温度，使得在相同实验条件下 C4 烯烃收率尽可能高。若使温度低于 350 度，又如何选择催化剂组合与温度，使得 C4 烯烃收率尽可能高。

​(4) 如果允许再增加 5 次实验，应如何设计，并给出详细理由。

**附录：名词解释与附件说明**

**温度**：反应温度。

**选择性**：某一个产物在所有产物中的占比。

**时间**：催化剂在乙醇氛围下的反应时间，单位分钟（min）。

**Co 负载量**： Co 与 SiO2 的重量之比。例如，“Co 负载量为 1wt%”表示 Co与 SiO2 的重量之比为 1:100，记作“1wt%Co/SiO2”，依次类推。

**HAP**：一种催化剂载体，中文名称羟基磷灰石。

**Co /SiO2 和 HAP 装料比**：指 Co/SiO2 和 HAP 的质量比。例如附件 1 中编号为A14 的催化剂组合“33mg 1wt%Co/SiO2-67mg HAP-乙醇浓度 1.68ml/min”指Co/SiO2 和 HAP 质量比为 33mg：67mg 且乙醇按每分钟 1.68 毫升加入，依次类推。

**乙醇转化率**：单位时间内乙醇的单程转化率，其值为 100 % × (乙醇进气量-乙醇剩余量)/乙醇进气量。

**C4 烯烃收率**：其值为乙醇转化率 × C4 烯烃的选择性。

**附件 1**：性能数据表。表中乙烯、C4 烯烃、乙醛、碳数为 4-12 脂肪醇等均为反应的生成物；编号 A1~A14 的催化剂实验中使用装料方式 I，B1～B7 的催化剂实
验中使用装料方式 II。

**附件 2**：350 度时给定的某种催化剂组合的测试数据。


### English Version


**Problem B: Ethanol Coupling to Prepare C4 Olefins**

C4 olefins are widely used in the production of chemical products and pharmaceuticals. Ethanol is the raw material for the production of C4 olefins. During the preparation process, the catalyst combination (i.e., Co loading, Co/SiO₂ and HAP mass ratio, and ethanol concentration) and temperature affect the selectivity and yield of C4 olefins (definitions of terms are provided in the Appendix). Therefore, exploring the process conditions for ethanol catalytic coupling to prepare C4 olefins through catalyst design is of significant importance and value.

A chemical laboratory has conducted a series of experiments with different catalysts at various temperatures, and the results are shown in Appendix 1 and Appendix 2. Please use mathematical modeling to complete the following tasks:

1. **Analysis of Catalyst Combinations**:  
   For each catalyst combination in Appendix 1, study the relationship between ethanol conversion, the selectivity of C4 olefins, and temperature. Additionally, analyze the test results for the given catalyst combination at 350°C at different times in Appendix 2.

2. **Impact of Different Catalyst Combinations and Temperatures**:  
   Investigate how different catalyst combinations and temperatures affect ethanol conversion rate and the selectivity of C4 olefins.

3. **Optimization of Catalyst Combination and Temperature**:  
   Determine how to choose the catalyst combination and temperature to maximize C4 olefin yield under the same experimental conditions. If the temperature is kept below 350°C, how should the catalyst combination and temperature be selected to maximize C4 olefin yield?

4. **Design of Additional Experiments**:  
   If five more experiments can be added, how should these experiments be designed? Please provide a detailed explanation for your design.

**Appendix: Definitions and Data Description**

- **Temperature**: The reaction temperature.
- **Selectivity**: The proportion of a specific product relative to all products produced.
- **Time**: The reaction time of the catalyst under ethanol atmosphere, measured in minutes (min).
- **Co Loading**: The weight ratio of Co to SiO₂. For example, "1wt% Co loading" means the weight ratio of Co to SiO₂ is 1:100, denoted as "1wt% Co/SiO₂", and so on.
- **HAP**: A catalyst support material, Hydroxyapatite.
- **Co/SiO₂ and HAP Mass Ratio**: The mass ratio of Co/SiO₂ to HAP. For example, the catalyst combination "33mg 1wt% Co/SiO₂ - 67mg HAP - Ethanol feed rate 1.68ml/min" in Appendix 1 refers to a Co/SiO₂ to HAP mass ratio of 33mg:67mg, with ethanol added at 1.68 ml/min.
- **Ethanol Conversion Rate**: The single-pass ethanol conversion rate per unit time, calculated as:  
  
$$
  \text{Ethanol Conversion Rate} = 100\% \times \frac{\text{Ethanol Inlet Amount} - \text{Ethanol Remaining Amount}}{\text{Ethanol Inlet Amount}}
$$

- **C4 Olefin Yield**: The value is given by the ethanol conversion rate multiplied by the selectivity of C4 olefins.

**Appendix 1**: Performance Data Table. The table lists the reaction products, including ethene, C4 olefins, acetaldehyde, and fatty alcohols with carbon numbers ranging from 4 to 12. Catalyst experiments A1 to A14 use Loading Method I, while B1 to B7 use Loading Method II.

**Appendix 2**: Test data for a given catalyst combination at 350°C.


---

## The Initial and Naive Problem-Solving Approach

Looking back at our first attempt to tackle the problem of Ethanol Coupling to Prepare C4 Olefins, I realize how earnest yet simplistic our methods were. With little knowledge of organic chemistry, we relied heavily on mathematical modeling, believing that data analysis and statistical techniques could compensate for our lack of domain expertise. We began by organizing the experimental data, plotting graphs to visualize relationships between temperature, catalyst combinations, ethanol conversion rates, and C4 olefin selectivity. Our enthusiasm led us to fit linear and quadratic regression models, hoping to capture the trends we observed. We even introduced interaction terms to account for possible synergies between variables like temperature and catalyst composition.

As we delved deeper, our primary goal became optimizing the conditions to maximize the yield of C4 olefins, which we defined as the product of ethanol conversion rate and C4 olefin selectivity. We formulated an objective function using our regression models and attempted to solve it using nonlinear optimization methods. Constraints were set based on the experimental data, and we proposed additional experiments to fill gaps and validate our models. Suggestions included testing new temperature ranges, varying catalyst ratios, and conducting stability tests over longer durations.

Reflecting on our approach now, it’s clear that while we were methodical, we might have oversimplified a complex chemical problem. Our reliance on regression models assumed smooth, continuous relationships that may not fully capture the nuances of chemical reactions. We also treated the optimization purely as a mathematical exercise, overlooking practical considerations like catalyst feasibility and industrial applicability. Our proposed experiments were more about filling data gaps than strategically exploring the chemical space, perhaps due to our limited understanding of the underlying chemistry.

This retrospective isn’t about diminishing our past efforts but recognizing how much we’ve grown in problem-solving, especially across disciplines. The experience highlighted the importance of blending mathematical techniques with domain knowledge and being humble when venturing into unfamiliar fields. As I revisit this problem, I aim to integrate a deeper appreciation of the chemistry involved with advanced modeling techniques. Modern tools like machine learning algorithms tailored for chemical data or simulation software for reaction kinetics might offer deeper insights. My goal is not just to find a better solution but to develop a holistic approach that respects both the mathematical and chemical complexities.

---

### Conclusion

As I write this, I can’t help but notice how the mathematical optimization methods we used back then still seem relevant today. Yet, if I were to take my current understanding and attempt to re-solve the problem using cutting-edge approaches in AI optimization theory, I doubt I’d feel any more confident. Just the other day, I came across [a new paper](https://ojs.aaai.org/index.php/AAAI/article/view/29301) by Professor Dongdong Ge and his first batch of students at Shanghai University of Finance and Economics, published at AAAI, on large-scale MDP and optimization theory. It’s incredible work—especially considering it’s only Prof. Ge’s first year at SUFE—and it left me once again feeling like I know nothing at all.