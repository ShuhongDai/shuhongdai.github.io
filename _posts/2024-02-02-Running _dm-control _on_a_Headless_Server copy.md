
---
layout: post
title: Using Local v2rayN Proxy for Cloud Servers via SSH Reverse Tunnel
date: 2024-02-02 00:38:10
description: A practical record of troubleshooting outbound network restrictions on Chinese cloud servers and enabling stable access to foreign academic resources.
tags: ["Networking", "Proxy", "SSH", "DevOps"]
tabs: true
thumbnail: /assets/posts_img/2024-02-02/thumbnail.png
toc:
  sidebar: left
---

## Introduction

In mainland China, even cloud servers often struggle to access foreign open-source or academic resources. Many people interpret this as “network censorship,” but in my view the situation is more nuanced. It is not necessarily the result of direct, targeted enforcement against technical resources. Instead, it resembles a systemic consequence of long-term authoritarian governance, where private network operators behave with extreme caution in order to avoid any possible regulatory risk.

Under such an environment—opaque rules, inconsistent enforcement, and heavy potential penalties—service providers tend to implement their own overly strict filtering. As a result, traffic to platforms like HuggingFace, GitHub, PyPI, and other purely technical services may be blocked or reset “just in case.” In practice, it becomes a case of *tying themselves up with their own rope*.

For developers and researchers, this means that even a cloud server intended for normal machine learning tasks may have difficulty accessing essential foreign resources. This post documents the issues I encountered and the final working solution.

---

## Attempt 1: Running a Proxy Directly on the Server

My first attempt was to deploy a proxy environment directly on the cloud server using sing-box or Xray, and to reuse my existing Shadowsocks 2022 (SS2022) nodes from my Windows machine.

This approach ran into multiple problems:

- Different implementations of SS2022 use different field names  
- The key format (base64) requirements vary by version  
- Certain fields (e.g., `secondary`, `psk`) are not supported in older releases  
- Some configurations pass syntax checks but fail during actual traffic forwarding  
- sing-box support for SS2022 differs significantly among releases  

Even after repeated adjustments, the connection remained unstable or unusable.

---

## Attempt 2: Letting the Server Reuse My Local v2rayN Proxy (Successful)

Since my local Windows environment with v2rayN worked reliably, I attempted to make the cloud server reuse **my desktop’s proxy** via **SSH reverse port forwarding**.

I established an SSH session from the cloud server:

```bash
ssh -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -R 0.0.0.0:10808:127.0.0.1:10808 \
    root@<server-address>
````

This command:

* Opens `10808` on the cloud server
* Forwards all traffic from that port
* Back through the SSH tunnel to my Windows machine
* Where v2rayN listens on local port `10808`

Testing the proxy:

```bash
curl --socks5 127.0.0.1:10808 https://ipinfo.io/ip
```

The returned IP was exactly the exit node of my v2rayN setup, confirming that the cloud server was successfully using my local proxy.

---

## Python Configuration

Once `curl` worked, I needed Python (especially `requests` and HuggingFace Hub) to use SOCKS5.

Python does not support SOCKS by default, so I installed:

```bash
pip install pysocks
```

Then set the environment variables:

```bash
export http_proxy=socks5h://127.0.0.1:10808
export https_proxy=socks5h://127.0.0.1:10808
export all_proxy=socks5h://127.0.0.1:10808
```

After this, both `requests.get()` and HuggingFace model downloads worked correctly.

---

## Discussion

It is important to note that deploying a proxy service directly on the server is not only feasible but is, in fact, the more standard, professional, and long-term maintainable approach. Whether using sing-box, Xray, Hysteria, or Tuic, one can build a fully independent outbound capability on the cloud server, which aligns better with the engineering practice of “managing your own network boundary.”

However, this approach typically involves multiple layers of complexity: protocol specifications, key formats, server–client version compatibility, differences in supported cipher suites, and firewall behavior. This is especially true for SS2022, which lacks unified documentation and consistent implementation across projects. As a result, several subtle issues may arise during configuration and require careful troubleshooting.

In contrast, reusing a local proxy through SSH reverse port forwarding serves as a fast, low-overhead, and almost configuration-free alternative. Its main advantage is that it works immediately and is not affected by the cloud provider’s network policies. The drawback, of course, is that it depends on the local machine being online, making it unsuitable as a long-term infrastructure solution.

Based on this distinction: for time-sensitive situations, such as urgently needing to download model weights from HuggingFace, reusing a local proxy is extremely convenient. But for long-term project environments, if the goal is to maintain an autonomous and stable outbound capability on the server, deploying a proper proxy service on the server remains the recommended final solution.