[update-readmes]   Mode: rewrite — migrating to template structure...
# headroom

[![Built with Ona](https://ona.com/build-with-ona.svg)](https://app.ona.com/#https://github.com/Interested-Deving-1896/headroom)

<!-- AI:start:what-it-does -->
_Description pending._
<!-- AI:end:what-it-does -->

## Architecture

<!-- AI:start:architecture -->
_Architecture documentation pending._
<!-- AI:end:architecture -->

## Install


```bash
pip install "headroom-ai[all]"          # Python, everything — includes the `headroom` CLI
npm install headroom-ai                 # TypeScript SDK (library only — no `headroom` CLI)
docker pull ghcr.io/chopratejas/headroom:latest
```

Granular extras: `[proxy]`, `[mcp]`, `[ml]` (Kompress-v2-base), `[code]`, `[memory]`, `[vector]` (optional HNSW backend — needs a C++ toolchain, not in `[all]`), `[relevance]`, `[image]`, `[agno]`, `[langchain]`, `[evals]`, `[pytorch-mps]` (Apple-GPU memory-embedder offload — set `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`). Requires **Python 3.10+**.

> **Note**: `[all]` covers the core stack but excludes framework adapters. Install them separately: `pip install "headroom-ai[langchain]"` (also `[agno]`, `[strands]`, `[anyllm]`, `[bedrock]`).

Using `pipx`? Choose a supported interpreter explicitly:

```bash
pipx install --python python3.13 "headroom-ai[all]"
```

> **Pick 3.13 if you want dollar savings.** The dashboard's *Proxy $ Saved* tile prices compression with [LiteLLM](https://github.com/BerriAI/litellm), and LiteLLM can't be installed on Python 3.14+. On 3.14 token savings still track, but the dollar figure stays `$0.00`. If you already installed on 3.14, switch with `pipx reinstall headroom-ai --python python3.13` and restart the proxy.

→ [Installation guide](https://headroom-docs.vercel.app/docs/installation) — Docker tags, persistent service, PowerShell, devcontainers.

> **CPU requirement (x86/x86_64):** the ONNX-backed features — Magika content
> detection and embedding relevance — use a precompiled ONNX Runtime that needs
> **AVX2**. On x86 hosts without AVX2 (some Docker/QEMU setups and older cloud
> VMs) Headroom automatically falls back to its non-ONNX paths (BM25 relevance,
> heuristic detection) rather than crashing. `arm64`/Apple Silicon needs no AVX2.

### Updating

```bash
headroom update          # detects pip / pipx / uv tool and upgrades in place
headroom update --check  # report the latest release without upgrading
headroom update --pre    # include pre-releases
```

`headroom update` figures out how Headroom was installed (pip/venv, `pip --user`,
pipx, uv tool) and runs the matching upgrade across macOS, Linux, and Windows.
For git checkouts, editable installs, Docker images, and externally-managed
system Pythons (PEP 668) it prints the correct manual step instead of guessing.

The proxy also shows a one-line "update available" notice on startup. It checks
PyPI at most once a day, in the background, and never blocks. Opt out with
`HEADROOM_UPDATE_CHECK=off` (also skipped in `--stateless` mode and CI).

### Corporate / SSL-inspection environments

If `pip install "headroom-ai[all]"` fails with `CERTIFICATE_VERIFY_FAILED`
(`unable to get local issuer certificate`), your network uses **SSL inspection** — a MITM
proxy presenting a company-issued CA. The build backend (`maturin`) downloads `rustup` over a
connection your TLS stack doesn't trust. **Install Rust first** so the build doesn't fetch it:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && rustup default stable
# Windows
winget install Rustlang.Rustup && rustup default stable
```

Restart your shell, then `pip install "headroom-ai[all]"`. A prebuilt wheel avoids the Rust
build entirely where available: `pip install --only-binary headroom-ai headroom-ai`. Prebuilt
wheels are published for Windows (`win_amd64`), Linux (`x86_64` / `aarch64`), and macOS
(Apple Silicon and Intel), so installs on those platforms never need a local Rust toolchain — the
Rust-first dance above is only for the platform-independent sdist fallback when no wheel matches.

Two runtime assets are fetched over TLS; if they are blocked, trust your corporate CA via
`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE`:

- **`cdn.pyke.io`** — the ONNX Runtime for the Rust core. Alternatively pre-provide it with
  `ORT_STRATEGY=system` and `ORT_LIB_LOCATION=/path/to/onnxruntime`.
- **`huggingface.co`** — the `kompress-base` compression model. Pre-download it and run with
  `HF_HUB_OFFLINE=1`, or set `HF_ENDPOINT` to a trusted mirror.

Running with compression disabled (pure gateway) requires neither asset.

#### "Basic Constraints of CA cert not marked critical" (Python 3.13+ strict mode)

A **different** failure from the one above. If TLS fails with:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
Basic Constraints of CA cert not marked critical
```

then the corporate CA *is* found and trusted — adding it to a CA bundle changes nothing.
Python 3.13 + OpenSSL 3.x enable `VERIFY_X509_STRICT` by default, which enforces RFC 5280
§4.2.1.9: a CA cert's `basicConstraints` must be marked *critical*. Inspection roots like
Zscaler set `CA:TRUE` without the critical bit, so the chain is rejected.

Set **`HEADROOM_TLS_STRICT=0`** to clear *only* the strict flag from every TLS context
Headroom controls — the proxy's httpx upstream client **and** the urllib3/`huggingface_hub`
path used for model downloads. Chain validation, signature, expiry, and hostname checks all
stay on; this is strictly narrower than disabling verification.

```bash
HEADROOM_TLS_STRICT=0 headroom proxy --port 8787
```

The Rust core's ONNX download (`cdn.pyke.io`) uses a separate TLS stack (rustls / OS trust
store), unaffected by `HEADROOM_TLS_STRICT`. On Windows the corporate root must be in the
**machine** certificate store (browsers already trust it there); or pre-provision ONNX
Runtime with `ORT_STRATEGY=system` + `ORT_LIB_LOCATION=/path/to/onnxruntime` to skip the
download entirely.

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
_Original project — no upstream fork._
<!-- AI:end:origins -->

## Resources

<!-- AI:start:resources -->
_No additional resource files found._
<!-- AI:end:resources -->

## License

<!-- AI:start:license -->
[Apache-2.0](https://github.com/Interested-Deving-1896/headroom/blob/main/LICENSE) © 2026 [Interested-Deving-1896](https://github.com/Interested-Deving-1896)
<!-- AI:end:license -->
