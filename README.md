# headroom

[![Built with Ona](https://ona.com/build-with-ona.svg)](https://app.ona.com/#https://github.com/Interested-Deving-1896/headroom) [![KDE Eco](https://img.shields.io/badge/KDE%20Eco-certified-brightgreen?logo=kde&logoColor=white&style=flat-square)](https://eco.kde.org/) [![Blue Angel](https://img.shields.io/badge/Blue%20Angel-DE--UZ%20215-0055a4?style=flat-square)](https://www.blauer-engel.de/en/certification/criteria)


<!-- AI:start:what-it-does -->
_Description pending._
<!-- AI:end:what-it-does -->

## Architecture

<!-- AI:start:architecture -->
_Architecture documentation pending._
<!-- AI:end:architecture -->

## Install


```bash
uv tool install --python 3.13 "headroom-ai[all]"  # CLI, isolated app env
pip install "headroom-ai[all]"                    # Python, everything — includes the CLI
npm install headroom-ai                           # TypeScript SDK (library only)
docker pull ghcr.io/headroomlabs-ai/headroom:latest
```

Granular extras: `[proxy]`, `[mcp]`, `[ml]` (Kompress-v2-base), `[code]`,
`[memory]`, `[vector]` (optional HNSW backend — needs a C++ toolchain, not in
`[all]`), `[relevance]`, `[image]`, `[agno]`, `[langchain]`, `[evals]`,
`[pytorch-mps]` (Apple-GPU memory-embedder offload — set
`HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`). Requires **Python 3.10+**.

> `[all]` covers the core stack but not the framework adapters. Install those
> separately: `pip install "headroom-ai[langchain]"`, and likewise `[agno]`,
> `[strands]`, `[anyllm]`, `[bedrock]`.

→ [Installation guide](https://docs.headroomlabs.ai/docs/installation) — Docker
tags, persistent service, PowerShell, devcontainers.

<details>
<summary><b>uv, pipx, and MCP clients that don't inherit your PATH</b></summary>

Prefer `uv tool install` for the CLI so the command lives in an isolated app
environment. On macOS, pass `--python 3.13` if your default `python3` is newer
than the current wheel set:

```bash
brew install python@3.13  # if 3.13 is not already available
uv tool install --python 3.13 "headroom-ai[all]"
uv tool update-shell      # if ~/.local/bin is not on PATH
headroom --version
```

Codex and other MCP clients often cannot inherit an interactive shell `PATH`.
Configure the absolute path returned by `command -v headroom`:

```toml
[mcp_servers.headroom]
command = "/Users/you/.local/bin/headroom"
args = ["mcp", "serve"]
```

`command = "headroom"` only works when the client starts with a `PATH` that
already includes the uv tool directory.

With pipx, choose the interpreter explicitly:

```bash
pipx install --python python3.13 "headroom-ai[all]"
```

Native wheels currently cover macOS Apple Silicon and Linux. On Intel macOS, use
the Docker-native install until native wheel support lands.

**CPU requirement (x86/x86_64).** The ONNX-backed features — Magika content
detection and embedding relevance — use a precompiled ONNX Runtime that needs
**AVX2**. On x86 hosts without AVX2 (some Docker/QEMU setups, older cloud VMs)
Headroom falls back to its non-ONNX paths — BM25 relevance, heuristic detection —
rather than crashing. `arm64` and Apple Silicon need no AVX2.

</details>

<details>
<summary><b>Updating</b></summary>

```bash
headroom update          # detects pip / pipx / uv tool and upgrades in place
headroom update --check  # report the latest release without upgrading
headroom update --pre    # include pre-releases
```

`headroom update` works out how Headroom was installed (pip/venv, `pip --user`,
pipx, uv tool) and runs the matching upgrade on macOS, Linux and Windows. For git
checkouts, editable installs, Docker images and externally-managed system Pythons
(PEP 668) it prints the correct manual step instead of guessing.

The proxy also prints a one-line "update available" notice at startup. It checks
PyPI at most once a day, in the background, and never blocks. Opt out with
`HEADROOM_UPDATE_CHECK=off`; it is also skipped in `--stateless` mode and CI.

</details>

<details>
<summary><b>Corporate networks and SSL inspection</b></summary>

If `pip install "headroom-ai[all]"` fails with `CERTIFICATE_VERIFY_FAILED`
(`unable to get local issuer certificate`), your network runs SSL inspection — a
MITM proxy presenting a company CA. The build backend (`maturin`) downloads
`rustup` over a connection your TLS stack does not trust. Install Rust first so
the build never fetches it:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && rustup default stable
# Windows
winget install Rustlang.Rustup && rustup default stable
```

Restart your shell, then install. A prebuilt wheel avoids the Rust build
entirely: `pip install --only-binary headroom-ai headroom-ai`. Wheels are
published for Windows (`win_amd64`), Linux (`x86_64` / `aarch64`) and macOS
(Apple Silicon and Intel), so those platforms never need a local Rust toolchain —
the Rust-first step above is only for the sdist fallback when no wheel matches.

Two runtime assets are fetched over TLS. If they are blocked, trust your
corporate CA through `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE`:

- **`cdn.pyke.io`** — the ONNX Runtime for the Rust core. Or pre-provide it with `ORT_STRATEGY=system` and `ORT_LIB_LOCATION=/path/to/onnxruntime`.
- **`huggingface.co`** — the `kompress-base` model. Pre-download it and run with `HF_HUB_OFFLINE=1`, or point `HF_ENDPOINT` at a trusted mirror.

Running with compression disabled (pure gateway) needs neither asset.

**Intel macOS: no prebuilt ONNX Runtime ([#941](https://github.com/headroomlabs-ai/headroom/issues/941)).**
`ort-sys` ships no prebuilt binary for `x86_64-apple-darwin`, so a source build
fails by default even outside a corporate proxy. Point it at a system runtime:

```bash
brew install onnxruntime
ORT_STRATEGY=system \
ORT_LIB_LOCATION="$(brew --prefix onnxruntime)/lib" \
ORT_PREFER_DYNAMIC_LINK=1 \
  pip install "headroom-ai[all]"

# ORT is dlopen'd at runtime too:
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
```

`ORT_LIB_LOCATION` must point at `lib/`, not the bare prefix, and
`ORT_PREFER_DYNAMIC_LINK=1` is required — without it `ORT_STRATEGY=system` still
attempts static linking, which the Homebrew keg does not provide.

**"Basic Constraints of CA cert not marked critical"** is a different failure. If
TLS fails with:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
Basic Constraints of CA cert not marked critical
```

then the corporate CA *is* found and trusted, and adding it to a CA bundle
changes nothing. Python 3.13 with OpenSSL 3.x enables `VERIFY_X509_STRICT` by
default, which enforces RFC 5280 §4.2.1.9: a CA cert's `basicConstraints` must be
marked critical. Inspection roots such as Zscaler set `CA:TRUE` without the
critical bit, so the chain is rejected.

`HEADROOM_TLS_STRICT=0` clears only the strict flag, from every TLS context
Headroom controls — the proxy's httpx upstream client and the
urllib3/`huggingface_hub` path used for model downloads. Chain validation,
signature, expiry and hostname checks all stay on.

```bash
HEADROOM_TLS_STRICT=0 headroom proxy --port 8787
```

The Rust core's ONNX download uses a separate TLS stack (rustls / OS trust store)
and is unaffected by `HEADROOM_TLS_STRICT`. On Windows the corporate root must be
in the **machine** certificate store — browsers already trust it there — or
pre-provision ONNX Runtime with `ORT_STRATEGY=system` to skip the download.

</details>

## Usage

<!-- Add usage examples here. This section is yours — the AI will not modify it. -->

## Configuration

<!-- Document configuration options here. This section is yours — the AI will not modify it. -->

## CI

<!-- AI:start:ci -->
_CI documentation pending._
<!-- AI:end:ci -->

## Mirror chain

<!-- AI:start:mirror-chain -->
This repo is maintained in [`Interested-Deving-1896/headroom`](https://github.com/Interested-Deving-1896/headroom) and mirrored through:

```
Interested-Deving-1896/headroom  ──►  OpenOS-Project-OSP/headroom  ──►  OpenOS-Project-Ecosystem-OOC/headroom
```

Changes flow downstream automatically via the hourly mirror chain in
[`fork-sync-all`](https://github.com/Interested-Deving-1896/fork-sync-all).
Direct commits to OSP or OOC are detected and opened as PRs back to `Interested-Deving-1896`.
<!-- AI:end:mirror-chain -->

## Contributors

<!-- AI:start:contributors -->
_Contributors pending._
<!-- AI:end:contributors -->

## Origins

<!-- AI:start:origins -->
_Original project — no upstream influences recorded._
<!-- AI:end:origins -->

## Resources

<!-- AI:start:resources -->
_No additional resource files found._
<!-- AI:end:resources -->

## Accessibility

<!-- AI:start:accessibility -->
This repo uses automated accessibility auditing via `check-accessibility.yml`.

Checks include: CODEOWNERS ownership coverage, README screen-reader compatibility,
WCAG 2.1 AA HTML compliance, audio overview (espeak-ng), and Braille output (liblouis).




Run the [Check Accessibility](https://github.com/Interested-Deving-1896/headroom/actions/workflows/check-accessibility.yml)
workflow to generate the first report and accessibility artifacts.
See [DOCS/accessibility.md](https://github.com/Interested-Deving-1896/headroom/blob/main/DOCS/accessibility.md) for the full reference.
<!-- AI:end:accessibility -->

## License

<!-- AI:start:license -->
[Apache-2.0](https://github.com/Interested-Deving-1896/headroom/blob/main/LICENSE) © 2026 [Interested-Deving-1896](https://github.com/Interested-Deving-1896)
<!-- AI:end:license -->
