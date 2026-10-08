---
name: "code-cleanup"
description: "Finds and safely removes dead/unused code and guides behavior-preserving refactors. Invoke when user asks to prune unused code, delete dead code, or refactor legacy modules."
---

# Code Cleanup & Safe Refactoring

A disciplined, reversible workflow for finding dead/unused code and removing it without changing behavior, plus guardrails for behavior-preserving refactoring of legacy modules.

## When to Use

Invoke when the user asks to: clean up / prune / remove unused or dead code, find unreachable functions/classes/files/exports/dependencies, slim down a codebase, or safely refactor a legacy module. Do not invoke for adding features or fixing a specific bug.

## Hard Safety Rules

- **Never delete from a single static-tool signal.** Static analysis has false positives: dynamic dispatch, reflection/`getattr`, decorators, plugin/registry entry points, string-based references, serialized names, framework callbacks.
- **Preserve behavior during deletion.** A cleanup commit must not alter runtime behavior. Keep cleanup, refactoring, and feature changes in separate commits.
- **Version-control gate.** Start from a clean working tree (commit or stash existing changes). Commit after each green step so every removal is reversible.
- **Tests are the safety net.** Establish a green test/lint/typecheck baseline first. Code without tests is legacy code: add characterization tests before changing it.
- **Scope discipline.** Handle one module or candidate set at a time. Stop when the requested scope ends; do not opportunistically rewrite unrelated code (avoid scope creep).

## Workflow

### Phase 1 — Inventory

- Detect languages, build system, and dependency manifests; map source roots and test roots.
- Identify third-party/generated directories to exclude (virtualenvs, `node_modules`, build/dist output, vendored code).
- Enumerate every entry point: console scripts, CLI modules, HTTP routes, plugin/registry hooks, package `__init__` exports, framework and CI configuration, templates, and data files that may reference symbols by name.

### Phase 2 — Candidate Discovery (collect multiple signals)

- Python: `vulture` (use `--min-confidence` and `--sort-by-size`), `ruff`/`pyflakes` for unused imports and locals, then an explicit reference search.
- JavaScript/TypeScript: `knip` for unused files, exports, and dependencies (ensure framework plugins/entry points are configured), plus `eslint`/`tsc` and `ts-prune` where useful.
- Other stacks: enable the compiler/linter unused-symbol warnings.
- For each candidate symbol or file, run an explicit whole-repository reference search across source, tests, configs, templates, and data (catch string and dynamic references).

### Phase 3 — Verify and Classify

For every candidate, record the definition site, every reference (static and dynamic), and reachability from any entry point. Assign one class:

| Class | Meaning | Default action |
|---|---|---|
| DEAD | No references and not reachable; high confidence | Delete |
| LEGACY-WIRED | Off the current main path but still imported by CLI/tests/old wrappers | Confirm with user, then decouple or remove |
| DYNAMIC-RISK | Referenced implicitly (reflection, registry, decorators, entry points, string names, serialization) | Keep or add a tool whitelist |
| KEEP | Public API, entry point, tested, or business-critical | Keep; state why |

Start with the smallest leaf candidates — unused imports, locals, private helpers — before removing whole files or modules.

### Phase 4 — Safety Gate

- Confirm a clean VCS state; create a branch if the change set is non-trivial.
- Run existing tests, lint, typecheck, and an import/startup smoke to capture a green baseline.
- If the target lacks coverage, add characterization tests (output/golden snapshots) that pin current behavior before editing.
- Ask the user before deleting exported/public or LEGACY-WIRED symbols. Private/local DEAD candidates may proceed directly.

### Phase 5 — Delete in Dependency Order

- Remove call sites, imports, and re-exports first; delete the definition only after no references remain. For a file, remove all importers, then the file.
- Keep each step small. After every removal, rerun the fast checks (import/startup, lint, targeted tests).
- Remove now-unused dependencies from manifests only after confirming no remaining references.
- Re-run the discovery tool: the symbol must be gone and the change must not expose a new chain of unused code that was previously masked.

### Phase 6 — Verify and Report

- Run the full test suite, lint/typecheck, build, and an entry-point/application smoke test.
- Confirm behavior is unchanged (only intended removals occurred).
- Deliver a concise report plus a deletion-order checklist.

## Behavior-Preserving Refactoring (when change beyond deletion is needed)

- Refactor in small steps with tests green after each step; make one catalog move at a time (extract/inline function, rename, move, change signature, introduce parameter object).
- Let a code smell trigger the refactor; never bundle refactoring with a feature or bug fix.
- For large legacy replacement, use the Strangler Fig approach: identify seams, build the new component alongside the old, route traffic gradually with transitional architecture, and delete the old path only after parity is verified. Avoid a big-bang rewrite.

## Tooling Notes

- **Vulture**: code called implicitly may be falsely reported; run it over both the library and tests and use whitelist directives for confirmed false positives.
- **Knip**: accuracy depends on correctly configured entry points and framework plugins; misconfiguration produces false positives.
- Always cross-check any tool result with an explicit reference search and entry-point reachability before deleting.

## Output Format

A compact table with one row per candidate: `item | class | location | evidence (references found / none) | action | risk`. Follow it with an ordered deletion checklist and a short list of kept items with reasons and open follow-ups.
