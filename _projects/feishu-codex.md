---
layout: page
title: Feishu-Codex
description: A Feishu / Lark bot that connects chat messages to your local Codex CLI and sends progress plus final replies back to chat.
img: assets/img/project_preview/feishu-codex.png
img_no_responsive: true
importance: 1
category: open-source
status: Maintained
github: https://github.com/ShuhongDai/Feishu-Codex
---

Feishu-Codex is an open-source bridge between **Feishu / Lark** and your local **Codex CLI**.
It lets you talk to a Codex agent from chat, runs `codex exec --json` on your own machine, and sends tool progress together with the final response back to Feishu cards.

## What It Does

- Connects direct chats and group chats in Feishu / Lark to a local Codex workflow
- Supports session and workspace commands such as `/new`, `/resume`, `/model`, `/mode`, `/cd`, `/ls`, and `/ws`
- Restores local Codex sessions from `~/.codex/session_index.jsonl`
- Lists installed Codex Skills and configured MCP servers
- Passes image messages to Codex as local file paths for analysis
- Supports local bot notifications through `POST /send`

## Why It Is Interesting

This project adapts an earlier Claude-based Feishu bot workflow to **Codex CLI** while keeping the same chat-oriented interaction model.
In practice, it makes local agent workflows much easier to trigger from a messaging interface that is already part of daily collaboration.

## Links

- Repository: [ShuhongDai/Feishu-Codex](https://github.com/ShuhongDai/Feishu-Codex)
- Original upstream inspiration: [joewongjc/feishu-claude-code](https://github.com/joewongjc/feishu-claude-code)
