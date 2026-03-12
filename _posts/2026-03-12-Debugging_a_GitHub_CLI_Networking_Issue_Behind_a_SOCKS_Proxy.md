---
layout: post
title: Debugging a GitHub CLI Networking Issue Behind a SOCKS Proxy
date: 2026-03-12 12:09:24
description: A GitHub CLI networking issue caused by a SOCKS proxy misconfiguration on Windows, and how to diagnose it using netstat.
tags: ["github-cli", "proxy", "Networking"]
tabs: true
# thumbnail: /assets/posts_img/2025-11-17/llm-memory-thumb.png
toc:
  sidebar: left
---

## Migrating a Familiar GitHub Workflow from macOS to Windows

For the past several years, I have done most of my development work on macOS. In this environment, tools such as git, ssh, and the GitHub CLI (gh) integrate smoothly with the system shell. Creating repositories, cloning code, and managing pull requests from the terminal is nearly frictionless. The workflow has proven sufficiently simple and reliable that I rarely find myself thinking about the underlying network configuration.

Recently, I needed to reproduce that workflow on a Windows machine for the first time. My system routes external traffic through V2Ray because of the network environment I work in; this exposes a local proxy for applications requiring access to services like GitHub. However, while repository creation through the GitHub CLI worked without issues, cloning the repository from the terminal immediately failed with a network connection error.

---

## Debugging the Failure

When I attempted to clone the repository locally using:

```bash
gh repo clone <repository-name>
```

the command failed with a network error:

```bash
fatal: unable to access 'https://github.com/...':
Failed to connect to github.com port 443
```

This was puzzling for two reasons. First, GitHub was clearly reachable from my browser. Second, the GitHub CLI had just successfully created the repository through the GitHub API.

My first assumption was that the issue was related to proxy configuration. Since the system was using V2Ray, I tried setting the standard environment variables used by many CLI tools:

```bash
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
```

However, this attempt immediately produced another error:

```bash
proxyconnect tcp: dial tcp 127.0.0.1:7890: connectex: connection refused
```

This indicated that the terminal was indeed attempting to use the proxy, yet the port itself was not accepting connections. This confirmed that the terminal was attempting to use the proxy, but nothing was actually listening on that port. At this point, the question became straightforward. What exactly was the local proxy port that V2Ray had exposed?

To verify this, I inspected the active local ports using:

```bash
netstat -ano | findstr 127.0.0.1
```

The command returned a large number of entries, as many local processes communicate over the loopback interface. Most of these lines represented established connections between local processes, which were not relevant to identifying the proxy.

For example, entries such as:

```bash
TCP    127.0.0.1:1052    127.0.0.1:10808    ESTABLISHED
```

simply indicate that one local process is communicating with another over the loopback network. These connections don't tell us which port is actually acting as the proxy endpoint.

What I actually needed were ports in the `LISTENING` state. A port in this state indicates that a local service is actively exposing a network interface that other applications can connect to. Within the output, one entry stood out:

```
TCP    127.0.0.1:10808    0.0.0.0:0    LISTENING
```

The `LISTENING` state suggested that a service was accepting connections on this port, making it a strong candidate for the local proxy endpoint used by V2Ray. Therefore the proxy was SOCKS5 rather than HTTP.

---


## Conclusion

In the end, the issue had nothing to do with GitHub itself. The real cause was simply a mismatch between the proxy protocol and the way I had configured the CLI. One thing I find worth noting is that browsers and command-line tools treat proxies very differently. Browsers typically rely on system proxy settings, while tools like `git` or `gh` depend entirely on environment variables such as `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY`. Because of this difference, it is entirely possible for GitHub to work perfectly in a browser while failing in the terminal.

Another point is that the proxy protocol itself matters. In my case, the local proxy exposed by V2Ray was a SOCKS5 proxy, but my initial configuration assumed an HTTP proxy. The CLI was therefore attempting to communicate with the proxy using the wrong protocol.

Finally, I find it easy to confuse remote server ports with local proxy ports when working with these tools. Command-line tools should always connect to the local proxy endpoint, typically something like `127.0.0.1:<port>`, rather than the remote node port displayed in the proxy client.