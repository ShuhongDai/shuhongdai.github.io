---
layout: post
title: "Running dm-control on a Headless Server: A Complete Debugging Log"
date: 2025-11-15 21:12:00
description: A practical record of configuring dm-control with Mujoco on a headless Ubuntu server, covering rendering failures, version mismatches, and the final workable setup.
tags: [ "Reinforcement Learning", "Mujoco", "dm-control", "Rendering" ]
tabs: true
toc:
  sidebar: left
---

## Introduction

This post documents the process of configuring **dm-control** on a **headless Ubuntu server** for reinforcement learning experiments that require pixel-based observations. The goal was straightforward: load a task from the DeepMind Control Suite and render frames for a CNN-based policy. However, running dm-control without a graphical display consistently triggered a series of rendering failures.  

Rather than providing a tutorial, this article records the issues encountered, the different directions explored, and the final configuration that proved reliable. Hopefully, this log will help anyone attempting to run dm-control in a similar environment.

---

## Initial Attempts and Rendering Errors

The starting point was simply loading a task and calling `env.physics.render()`. On a local machine, this works immediately. On a headless server, the first result was an error referencing **EGL** and **OpenGL**:

```

AttributeError: 'NoneType' object has no attribute 'eglQueryString'

```

Further attempts produced warnings about missing `DISPLAY` variables:

```

X11: The DISPLAY environment variable is missing

```

These messages made it clear that dm-control was attempting to initialize rendering through OpenGL/EGL, both of which require a graphics context the server did not provide. Disabling the `DISPLAY` variable or forcing EGL did not resolve the issue; the underlying environment simply lacked the dependencies required for hardware-accelerated rendering.

---

## Investigating the Role of PyOpenGL

The repeated mentions of EGL led to checking whether PyOpenGL played a role. Removing PyOpenGL temporarily modified the error messages, but it was automatically reinstalled when upgrading or reinstalling dm-control and Mujoco. This indicated that avoiding the OpenGL backend entirely was not feasible through uninstalling PyOpenGL alone.

---

## Version Mismatch Between dm-control and Mujoco

At a later stage, entirely different errors appeared:

```

eq_active not found
flex_xvert0 not found

```

These errors are characteristic of **Mujoco internal structure mismatches**, suggesting that the installed versions of Mujoco and dm-control were not aligned. dm-control indexes Mujoco model structures by field name, and when they do not match, initialization fails even before rendering begins.

This confirmed that two separate issues existed:

1. Rendering backend not compatible with the server  
2. dm-control and Mujoco versions not compatible with each other  

Both needed to be addressed.

---

## Moving Toward a Software Rendering Approach

Given that neither EGL nor OpenGL would work reliably on the server, the next candidate was **OSMesa**, a pure software renderer. OSMesa performs all rendering on the CPU and does not require a graphical display or GPU drivers. This makes it suitable for cloud or containerized environments.

Ubuntu provides OSMesa through:

```

libosmesa6-dev

```

The key environment variable for Mujoco is:

```

MUJOCO_GL=osmesa

```

With this backend, Mujoco no longer attempts to initialize hardware-accelerated contexts.

---

## Identifying a Stable Mujoco + dm-control Combination

After testing multiple combinations, the following versions proved to be both compatible with each other and functional under OSMesa:

- **mujoco 2.3.3**  
- **dm-control 1.0.11**  
- Python 3.9  

This combination avoids the structure-indexing errors seen in newer pairings and also avoids invoking backends requiring OpenGL or EGL by default.

---

## Verifying the Configuration

With OSMesa enabled and the compatible versions installed, the following minimal script successfully produced a rendered frame:

```python
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["DISPLAY"] = ""

from dm_control import suite
import imageio

env = suite.load("cartpole", "swingup")
frame = env.physics.render(height=128, width=128, camera_id=0)

print("Frame shape:", frame.shape)
imageio.imwrite("soft_render.png", frame)
```

The output confirmed that rendering worked:

```
Frame shape: (128, 128, 3)
Saved soft_render.png
```

A few non-critical messages appear during interpreter shutdown, related to thread cleanup inside dm-control’s internal executor. These do not affect functionality.

---

## Final Working Setup

Summarizing the configuration that worked consistently:

* **Python**: 3.9
* **Mujoco**: 2.3.3
* **dm-control**: 1.0.11
* **System package**: `libosmesa6-dev`
* **Rendering backend**: `MUJOCO_GL=osmesa`

This setup uses CPU-based software rendering, avoids any dependency on GPU drivers or graphical displays, and is compatible with reinforcement-learning algorithms requiring pixel observations.

---

## Conclusion

Running dm-control on a headless server involves more than installing the library. Rendering backends, environment variables, and version compatibility all play important roles in determining whether Mujoco initializes correctly. After exploring several unsuccessful paths, the combination of **OSMesa rendering** and a **compatible pair of dm-control and Mujoco versions** proved to be a reliable solution.

By documenting this process, this post aims to provide a clear reference for others setting up dm-control in a similar environment.