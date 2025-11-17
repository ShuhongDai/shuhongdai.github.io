---
layout: post
title: Tracing the Root Cause of Missing GPUs in Docker Containers
date: 2025-08-20 01:02:00
description: A debugging record of why Docker refused to expose GPUs inside a container even though the host recognized them perfectly, and how every layer of the system contributed a small piece to the failure.
tags: ["Docker", "NVIDIA", "CUDA"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-21/docker-gpu-debug-thumb.png
toc:
  sidebar: left
---

## Introduction

The GPU on my host machine worked flawlessly. `nvidia-smi` showed four cleanly indexed devices. CUDA tests ran without complaint. Nothing looked suspicious. But inside Docker, those same GPUs simply vanished. The container insisted it had no GPU at all, even when launched with `--gpus all`. This was not an application-level issue. Problems like this never come from the code running inside the container. They come from a mismatch somewhere between Docker, the NVIDIA runtime, and the host’s driver stack. This post is a reconstruction of how I verified that assumption and worked my way through the layers until the system finally admitted what was wrong.

---

## The Initial Symptom

The first sign of trouble came from a simple test:

```bash
$ docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

The output didn’t show a GPU. It didn’t even show a driver mismatch. Instead I received:

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

This message is always unhelpful. It can mean anything from missing libraries to container runtime failures to a completely broken driver. But since `nvidia-smi` worked perfectly on the host, the problem had to be elsewhere.

I checked the host first, just to be certain.

```bash
$ nvidia-smi
Sun Aug 17 00:15:19 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54       Driver Version: 550.54       CUDA Version: 12.4     |
| GPU Name        Persistence-M| Bus-Id ... (normal output)                   |
+-----------------------------------------------------------------------------+
```

Everything here was healthy. That told me the problem was not hardware. It also told me the Docker container was not receiving the correct runtime environment.

---

## A Quick Look at the NVIDIA Container Toolkit

The next step was confirming that the NVIDIA container runtime existed on the host.

```bash
$ dpkg -l | grep -i nvidia-container
ii  nvidia-container-toolkit  1.16.2-1  amd64
```

The toolkit was installed, at least according to the package manager. But “installed” is not the same as “integrated.” I checked Docker’s runtime configuration:

```bash
$ cat /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

At first glance this looked reasonable but Docker’s configuration files often lie by omission. I restarted Docker anyway, hoping for the rare case where a restart solves a real problem.

```bash
$ sudo systemctl restart docker
```

It changed nothing.

---

## The Runtime That Didn’t Exist

I tried manually invoking the runtime:

```bash
$ which nvidia-container-runtime
/usr/bin/nvidia-container-runtime
```

It was present. But inside Docker, the container still couldn’t see GPUs. I suspected mismatch in library paths, so I inspected the toolkit’s log:

```bash
$ sudo journalctl -u nvidia-container-runtime
```

The log was silent with no errors and no warnings, which is often more suspicious than a screaming log file.

---

## Trying a Minimal Container

Sometimes it helps to remove everything unrelated. I tried an empty Alpine container with explicit runtime:

```bash
$ docker run --runtime=nvidia --rm nvidia/cuda:12.4-base nvidia-smi
```

The result was the same:

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

At this point I was certain the issue wasn’t in the image. The problem was in the host-to-container plumbing.

---

## nvidia-container-cli

I checked low-level diagnostics:

```bash
$ nvidia-container-cli info
```

This should list driver version, devices, and capabilities. Instead it printed:

```
ERRO[0000] could not load NVML: libnvidia-ml.so.1: cannot open shared object file: No such file or directory
```

The NVML library is part of the NVIDIA driver. If the toolkit couldn’t load it, that meant the toolkit’s library search paths did not match the actual driver installation. Which usually happens after a driver upgrade that leaves symlinks pointing to the wrong place.

I checked the library:

```bash
$ ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*
```

The files were indeed present, but I noticed they were under:

```
/usr/lib/x86_64-linux-gnu/nvidia/current/
```

while the toolkit expected:

```
/usr/lib/x86_64-linux-gnu/
```

This mismatch is easy to miss.

---

## The Missing Symlink

On many systems, `/usr/lib/x86_64-linux-gnu/` should contain symlinks to the actual driver libraries, but mine didn’t. I created the symlink manually:

```bash
$ sudo ln -s /usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-ml.so.1 \
             /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
```

Then I tried the diagnostic again:

```bash
$ nvidia-container-cli info
```

This time it printed a full report with devices, drivers, and capabilities listed correctly. That told me NVIDIA’s container toolkit could finally see the GPU.

---

## The Final Test

With everything aligned, I launched a fresh container:

```bash
$ docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```

The output looked normal.
