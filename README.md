# ignite-tools

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.0.1-green.svg)](https://github.com/Artain-AI/ignite-tools/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Tools for working with text embeddings. Pick the right model for your data, find duplicates, classify, search by meaning. Runs on a laptop.

## What this is

You have text data. Support tickets, product listings, documents, commit messages, whatever. You want to do something useful with it: categorize it, find the duplicates, make it searchable, group similar items together.

These tools do that. They use embedding models under the hood, but you don't need to know what that means to get value out of them.

## Tools

| Tool | What it does | Status |
|---|---|---|
| `ignite-eval` | Figures out which embedding model works best for your specific data | Working |
| `ignite-read` | Shows you what your data looks like before you process it | Working |
| `ignite-explore` | Finds duplicates, natural groupings, outliers in your text | Planned |
| `ignite-classify` | Sorts text into categories without training a model | Planned |
| `ignite-index` | Makes your content searchable by meaning, not just keywords | Planned |

## Install

```bash
git clone https://github.com/Artain-AI/ignite-tools.git
cd ignite-tools
pip install -e .
```

Python 3.10 or newer.

## Quick start

### Find the right model for your data

```bash
ignite-eval your-data/ --yes
```

What happens:

1. Reads your data. Figures out the language, text length, domain.
2. Picks 3-4 models that make sense for your situation.
3. Downloads them, runs each one on your data.
4. Measures speed and quality.
5. Tells you which one to use, and why.

Output looks like this:

```
-- Result ---------------------------------------------------

  Recommendation: BGE-small
  Best balance of quality (AUC 0.72) and speed (1200 texts/sec).
  384-dimensional embeddings, 450 MB memory.
  Confidence: +++ (clear winner)

  Why not the others:
    - MiniLM-L12: lower quality (AUC 0.65)
    - E5-small: similar quality but slower

  Benchmarked on:
    Apple M2 Max, 32 GB RAM, Apple Silicon GPU
```

### Look at your data first

```bash
ignite-read your-data/ --yes
```

Shows file structure, text lengths, detected languages, topic distribution, label balance. Useful for sanity-checking before you run anything expensive.

## Configuration

One config file controls everything: `ignite-format.yaml`. It tells the tools where your data is, how to read it, and what each tool should do.

### Auto-detection

If you don't have a config, the tool creates one for you:

```bash
ignite-eval your-data/ --save-config ./ignite-format.yaml
```

This sniffs your data (format, fields, languages, structure) and writes a proposed config. Review it, edit if needed, then use it for all runs.

### Basic structure

```yaml
# Where the data is
storage:
  type: local
  path: ./your-data/
  recursive: true

# How to parse it
format:
  type: jsonl

# Where the text lives in each record
text:
  fields: [body, title]

# Optional: which field has category labels
labels:
  field: category
```

### Tool-specific settings

Each tool can have its own settings in the same file. Use the tool's name as the key:

```yaml
# Shared data reading (used by all tools)
storage: ...
text: ...
labels: ...

# ignite-read settings
ignite-read:
  sections: [corpus_stats, per_source, top_words]
  top_words:
    count: 30

# ignite-eval settings
ignite-eval:
  task: classify
  priority: quality
  constraints:
    max_size_mb: 500
```

You can also put tool settings in a separate file:

```yaml
# In ignite-format.yaml:
ignite-read: ./read-settings.yaml
ignite-eval: ./eval-settings.yaml
```

The tool loads its own block and ignores the others.

### Config discovery

The tools look for a config in this order:

1. `--config path` (explicit, always wins)
2. `./ignite-format.yaml` in the current directory
3. `ignite-format.yaml` next to the data
4. `~/.config/ignite-tools/ignite-format.yaml` (global default)
5. Auto-detect and propose (interactive)

Full config reference: [docs/format-config.md](docs/format-config.md)

## Data formats

- JSONL: one JSON object per line. Supports `.gz` and `.zst` compression.
- CSV/TSV: tabular. Header row expected.
- Plain text: one record per line, or one file per record.

Reads from local disk, S3 (`s3://bucket/path/`), or Azure Blob (`azure://account/container/`).

## How model selection works

The evaluator looks at three things:

1. Your data. Language, average text length, how many records, what domain the vocabulary suggests.
2. Your requirements (optional). What task you're doing (search, classification, clustering), whether you care more about speed or quality.
3. Your hardware. CPU, Apple Silicon GPU, or NVIDIA GPU.

Based on those three inputs, it picks models from a registry of 42 open-source options, runs them, and tells you which one performed best on your actual data. Not on a generic benchmark. On yours.

## When you outgrow your laptop

These tools run fine on a MacBook or a cheap cloud VM. At some point your data gets big enough that embedding takes too long:

- Under 10K texts: seconds.
- 10K to 100K: minutes.
- Over 1M: you probably want [IgniteMS](https://github.com/Artain-AI/ignite-ms), which does the same thing 100x faster on GPU hardware.

## License

Apache 2.0
