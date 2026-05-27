# `ignite-format.yaml` reference

The format config tells `ignite-*` tools where the user's data lives and how to read it. One file works for every tool in the suite (evaluator, explorer, classifier, indexer).

For the design rationale and module layout, see `read-layer.md`.

---

## Quick start

If you don't have a config yet, just point a tool at a folder. The tool will sniff it and propose one:

```bash
ignite-eval ./my-data/
# Sniffer prints what it sees, asks [y/n/save/edit].
# Choose 'save' to write ./ignite-format.yaml and edit before running.
```

For scripting, sniff once, save, then use the saved file:

```bash
ignite-eval ./my-data/ --save-config ./ignite-format.yaml
# review and edit ./ignite-format.yaml
ignite-eval --format-config ./ignite-format.yaml
```

## Minimal config

```yaml
storage:
  type: local
  path: ./data/
format:
  type: jsonl
text:
  fields: [text]
```

This reads every `.jsonl` file under `./data/`, treating each record's `text` field as the content to embed. Defaults handle everything else.

---

## Schema

### `storage`

```yaml
storage:
  type: local                 # local | s3 | azure
  path: ./data/               # path or URI
  recursive: true             # walk subdirs (default true)
  include: ["*.jsonl.gz"]     # glob list; default depends on format.type
  exclude: ["**/_archive/**"] # glob list; default empty
  cache_dir: ~/.cache/ignite-tools   # cloud sources only
  region: us-east-1           # s3 only
```

| Type | URI form | Auth |
|---|---|---|
| `local` | `./path` or absolute | n/a |
| `s3` | `s3://bucket/prefix/` | env vars, `~/.aws/credentials`, IMDS |
| `azure` | `azure://container/prefix/` | env vars, `az login`, managed identity |

Cloud credentials are never in this file.

Compression is auto-detected by suffix: `.gz`, `.zst`. Streaming, no whole-file loads.

### `format`

```yaml
format:
  type: jsonl                 # jsonl | csv | tsv | text
  encoding: utf-8

  # csv/tsv only:
  delimiter: ","
  has_header: true
  quote: "\""

  # text only:
  unit: line                  # line | file
```

`text` with `unit: file` reads each file as one record. Useful for RAG corpora where each document should be embedded whole. With `unit: line` (default), each non-empty line is a record.

### `text`

Two forms. **Simple** - single field path with fallbacks:

```yaml
text:
  fields: [body, content, message]
```

Each field is tried in order, first non-empty wins. Dotted paths work: `attributes.body`.

**Routed** - different rules per source:

```yaml
text:
  router_field: attributes.source
  routes:
    github:      { fields: [attributes.body, attributes.title] }
    reddit:      { fields: [attributes.body, attributes.title] }
    hackernews:  { fields: [attributes.text, attributes.title] }
    producthunt: { fields: [attributes.body, attributes.title] }
    _default:    { fields: [body, text, title] }
```

The `router_field` value selects a route. Records whose router value isn't in `routes` and which have no `_default` are dropped (counted in the read summary).

### `labels`

```yaml
labels:
  field: attributes.source
```

Optional. The evaluator's labeled-items mode keys off this. Other tools may use it for filtering or stratification.

### `id`

```yaml
id:
  field: attributes.id
```

Optional. If omitted, IDs are auto-assigned as `<relative_path>:<line_number>`.

### `normalize`

Applied in fixed order: lowercase → masks → collapse_whitespace → strip → trim.

```yaml
normalize:
  lowercase: true
  collapse_whitespace: true
  strip: true
  masks:
    - { pattern: "https?://\\S+",                  replacement: "<url>" }
    - { pattern: "\\b[0-9a-f]{40}\\b",             replacement: "<sha>" }
    - { pattern: "\\bv?\\d+\\.\\d+\\.\\d+\\b",     replacement: "<version>" }
    - { pattern: "#\\d+",                          replacement: "<ref>" }
  trim:
    max_chars: 256
    min_chars: 10
```

Each section is independently optional. Records that fall below `trim.min_chars` after normalization are dropped and counted.

### `filters`

```yaml
filters:
  time_field: timestamp
  time_from: "2025-01-01"
  time_to:   "2025-06-01"
  labels_include: [github, reddit]
  labels_exclude: []
```

All optional. Filters are applied before sampling counts toward the target.

### `sampling`

```yaml
sampling:
  mode: stratified            # full | head | random | stride | stratified | weighted
  total: 10000                # sample size; ignored for 'full'
  per_group: 2000             # stratified only
  group_field: attributes.source
  per_file_cap: 5000          # cap per file regardless of mode (optional)
  seed: 42
  stratified_unknown: drop    # drop | keep
  weights:                    # weighted only
    github: 0.5
    reddit: 0.3
    hackernews: 0.2
```

| Mode | What |
|---|---|
| `full` | Read everything. |
| `head` | First N. Fast, biased. |
| `random` | Uniform reservoir sample of N. |
| `stride` | Every k-th record (k = total_count / target). Avoids time-ordering bias. |
| `stratified` | Up to `per_group` records per group. Counts equalized across groups. |
| `weighted` | Per-group quotas from `weights`. |

If `sampling` is omitted entirely, the default is `full`.

---

## Examples

### Folder of JSONL files, single text field

```yaml
storage: { type: local, path: ./tickets/, recursive: true }
format:  { type: jsonl }
text:    { fields: [body, subject] }
labels:  { field: category }
sampling:{ mode: random, total: 5000 }
```

### Multi-source corpus with per-source routing

```yaml
storage:
  type: local
  path: ./events/
  recursive: true

format:
  type: jsonl

text:
  router_field: attributes.source
  routes:
    github:      { fields: [attributes.body, attributes.title] }
    reddit:      { fields: [attributes.body, attributes.title] }
    hackernews:  { fields: [attributes.text, attributes.title] }
    producthunt: { fields: [attributes.body, attributes.title] }

labels: { field: attributes.source }
id:     { field: attributes.id }

normalize:
  lowercase: true
  collapse_whitespace: true
  strip: true
  masks:
    - { pattern: "https?://\\S+",              replacement: "<url>" }
    - { pattern: "\\b[0-9a-f]{40}\\b",         replacement: "<sha>" }
    - { pattern: "\\bv?\\d+\\.\\d+\\.\\d+\\b", replacement: "<version>" }
  trim: { max_chars: 256, min_chars: 10 }

sampling:
  mode: stratified
  per_group: 2500
  group_field: attributes.source
```

### S3 source

```yaml
storage:
  type: s3
  path: s3://my-bucket/events/processed/
  region: us-east-1
  recursive: true

format: { type: jsonl }
text:   { fields: [body, title] }
```

Files are downloaded to `~/.cache/ignite-tools/<hash>/` on first read and reused on subsequent runs. `--no-cache` forces re-download.

### Azure Blob source

```yaml
storage:
  type: azure
  path: azure://my-container/events/

format: { type: jsonl }
text:   { fields: [body, title] }
```

Auth uses the standard Azure credential chain (env vars, `az login`, managed identity). Credentials never live in this file.

### CSV with labels

```yaml
storage: { type: local, path: ./products.csv }
format:
  type: csv
  has_header: true
text:   { fields: [description, name] }
labels: { field: category }
```

### Plain-text RAG corpus, one document per file

```yaml
storage: { type: local, path: ./docs/, recursive: true, include: ["*.md", "*.txt"] }
format:
  type: text
  unit: file
```

For `text` with `unit: file`, the `text` block's field rules don't apply. Each file becomes one record; ID defaults to its relative path; label is null unless set elsewhere.

### Plain-text corpus, one record per line

```yaml
storage: { type: local, path: ./logs/, include: ["*.log"] }
format:
  type: text
  unit: line
```

Each non-empty line is a record. ID defaults to `<relative_path>:<line_number>`.

---

## CLI overrides

A few common knobs are also CLI flags. Flags override config values.

| Flag | Overrides |
|---|---|
| `--format-config PATH` | Skip sniffer, use this config |
| `--path PATH` | `storage.path` |
| `--recursive` / `--no-recursive` | `storage.recursive` |
| `--sample N` | `sampling.total` |
| `--sample-mode MODE` | `sampling.mode` |
| `--seed N` | `sampling.seed` |
| `--yes` | Accept sniffer proposal non-interactively |
| `--save-config PATH` | Sniff, save to file, exit without running |
| `--strict` | Per-record errors are fatal instead of counted |
| `--cache-dir PATH` | `storage.cache_dir` |
| `--no-cache` | Force re-download of cloud sources |

Everything else (text rules, routing, normalization, filters) is config-only by design. The schema is too rich for command-line flags, and reproducibility benefits from one source of truth on disk.
