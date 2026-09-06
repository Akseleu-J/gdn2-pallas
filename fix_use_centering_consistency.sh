#!/usr/bin/env bash
# fix_use_centering_consistency.sh
#
# The `atomic_ops` package (configs.py / gdn2_fwd.py / gdn2_bwd.py) ships
# with NO use_centering / unsafe_allow_centering / dgn_acc code at all --
# there is no such KernelConfig field, no kernel branch, no
# NotImplementedError gate. That part of the codebase is already clean.
#
# What is NOT clean is the documentation and one test file, which talk
# about use_centering as if it were an existing, gated, isolated-tested
# code path (README.md, CHANGELOG.md, KNOWN_LIMITATIONS.md, Roadmap.md,
# tests/test_gdn2_full_math_correctness.py). That is stale/aspirational
# text describing a future investigation, not the shipped code, and it
# currently breaks CI (the test calls
# `KernelConfig(..., use_centering=True)`, a keyword argument that does
# not exist).
#
# This script:
#   1. Removes the use_centering-based test function and its two call
#      sites from tests/test_gdn2_full_math_correctness.py (it exercises
#      code that does not exist -- CI is currently red because of it).
#   2. Rewrites Roadmap.md's top section so `use_centering` is documented
#      as a HYPOTHESIS to evaluate only after the beta hybrid path
#      (beta/gdn2_hybrid.py) is fully validated end-to-end -- and only if
#      the hybrid path does not fully close the fwd/bwd gap on its own.
#      No code, no KernelConfig field, no gate is implied to exist.
#   3. Fixes KNOWN_LIMITATIONS.md (section 6 + the v0.2.0 roadmap block)
#      so it stops implying use_centering is already implemented/gated.
#   4. Fixes README.md and CHANGELOG.md limitation bullets to match.
#
# Nothing in atomic_ops/*.py is touched -- there is nothing to strip
# there. beta/gdn2_hybrid.py and its KNOWN_LIMITATIONS.md section 5 are
# left untouched (separate, already-correctly-labeled experimental path).
#
# Usage:
#   ./fix_use_centering_consistency.sh            # dry-run, reports only
#   ./fix_use_centering_consistency.sh --fix      # applies changes, .bak backups made
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

FIX_MODE=false
for arg in "$@"; do
  case "$arg" in
    --fix) FIX_MODE=true ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

note()  { printf '  \033[2m%s\033[0m\n' "$1"; }
warn()  { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
fixed() { printf '  \033[32m[FIXED]\033[0m %s\n' "$1"; }
ok()    { printf '  \033[32m[OK]\033[0m %s\n' "$1"; }
section(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

TEST_FILE="tests/test_gdn2_full_math_correctness.py"
ROADMAP="Roadmap.md"
KNOWN_LIM="KNOWN_LIMITATIONS.md"
README="README.md"
CHANGELOG="CHANGELOG.md"

if ! $FIX_MODE; then
  section "DRY RUN -- no files will be changed"
  note "run with --fix to apply"
fi

PY_MODE="$([ "$FIX_MODE" = true ] && echo write || echo dryrun)"

python3 - "$PY_MODE" "$TEST_FILE" "$ROADMAP" "$KNOWN_LIM" "$README" "$CHANGELOG" <<'PYEOF'
import sys, re, shutil, pathlib

mode, test_file, roadmap, known_lim, readme, changelog = sys.argv[1:7]
write = (mode == "write")

def backup(path):
    p = pathlib.Path(path)
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy(p, bak)
        print(f"  backed up {path} -> {bak}")

def exact_replace_once(path, old, new, label):
    """Refuses if old appears 0 or >1 times (ambiguous)."""
    text = pathlib.Path(path).read_text()
    count = text.count(old)
    if count == 0:
        print(f"  [absent] {path}: {label}")
        return False
    if count > 1:
        print(f"  [SKIP-AMBIGUOUS] {path}: '{label}' matched {count} times -- review manually.")
        return False
    print(f"  [FOUND] {path}: {label}")
    if write:
        new_text = text.replace(old, new, 1)
        backup(path)
        pathlib.Path(path).write_text(new_text)
        print(f"  [WRITTEN] {path}")
    return True

def exact_replace_all(path, old, new, label, expected_count=None):
    """Replaces every occurrence. If expected_count is given, warns (but
    still proceeds) when the actual count differs."""
    text = pathlib.Path(path).read_text()
    count = text.count(old)
    if count == 0:
        print(f"  [absent] {path}: {label}")
        return False
    if expected_count is not None and count != expected_count:
        print(f"  [WARN] {path}: '{label}' expected {expected_count} occurrence(s), found {count} -- proceeding anyway.")
    print(f"  [FOUND] {path}: {label} (x{count})")
    if write:
        new_text = text.replace(old, new)
        backup(path)
        pathlib.Path(path).write_text(new_text)
        print(f"  [WRITTEN] {path}")
    return True

def regex_replace_once(path, pattern, new, label, flags=re.DOTALL):
    text = pathlib.Path(path).read_text()
    matches = list(re.finditer(pattern, text, flags))
    if not matches:
        print(f"  [absent] {path}: {label}")
        return False
    if len(matches) > 1:
        print(f"  [SKIP-AMBIGUOUS] {path}: '{label}' matched {len(matches)} times -- review manually.")
        return False
    print(f"  [FOUND] {path}: {label}")
    if write:
        new_text = re.sub(pattern, lambda m: new, text, count=1, flags=flags)
        backup(path)
        pathlib.Path(path).write_text(new_text)
        print(f"  [WRITTEN] {path}")
    return True


# ==========================================================================
# 1. tests/test_gdn2_full_math_correctness.py
#    Remove the use_centering test function (exercises a KernelConfig
#    kwarg that does not exist -- this is why CI is currently red) and
#    its two call sites.
# ==========================================================================
print("\n-- tests/test_gdn2_full_math_correctness.py --")

_TEST_FN = '''def test_kernel_a_use_centering_matches_default(cfg):
    print("\\n--- Section 4b: Kernel A use_centering=True vs False (CPU) ---")
    bt = 64  # small chunk keeps decay well inside the +-20 clip window
    config_off = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), use_centering=False)
    config_on = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), use_centering=True)
    key = jax.random.PRNGKey(654)
    q, k, v, w, b, g, _ = _make_inputs(key, cfg["bsz"], 1, bt, cfg["H"], cfg["D"], decay_scale=0.02)

    Aqk_off, Akk_off = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config_off, interpret=True)
    Aqk_on, Akk_on = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config_on, interpret=True)

    _check("kernel_a.centering_equiv.Aqk", _rel_err(Aqk_on, Aqk_off), 5e-3)
    _check("kernel_a.centering_equiv.Akk", _rel_err(Akk_on, Akk_off), 5e-3)


'''
exact_replace_once(
    test_file, _TEST_FN, "",
    "remove test_kernel_a_use_centering_matches_default (exercises nonexistent KernelConfig field)"
)

exact_replace_all(
    test_file,
    '    test_kernel_a_use_centering_matches_default(cfg)\n',
    '',
    "remove both call sites (test_all_sections + main)",
    expected_count=2,
)

# ==========================================================================
# 2. Roadmap.md -- rewrite the top section: use_centering becomes an
#    explicitly-unimplemented HYPOTHESIS, gated behind the beta hybrid
#    path being validated first.
# ==========================================================================
print("\n-- Roadmap.md --")

_ROADMAP_HYPOTHESIS = '''## Hypothesis (post-beta, not implemented): MXU-factorized pairwise decay (`use_centering`)

**Status:** HYPOTHESIS ONLY. No code for this exists anywhere in the
package -- no `KernelConfig` field, no kernel branch in `gdn2_fwd.py` /
`gdn2_bwd.py`, no `NotImplementedError` gate. Nothing below has been
measured in this repository; it is written down here so the idea isn't
lost or re-invented from scratch, and so it isn't attempted before its
listed prerequisite.

**Why this is being tracked at all:** `KNOWN_LIMITATIONS.md` section 1/6
identifies the pairwise decay computation in Kernel A
(`build_chunk_scores_pallas`) and its backward counterpart B4
(`intra_backward_pallas`) as VPU-bound (`_weighted_pair_sum` /
`_dL_pair_sum` / `_dR_pair_sum` / `_dgc_pair_sum`: broadcast + elementwise
multiply + manual reduction) rather than MXU-bound. In principle,
centering the pairwise decay term `exp(gc_i - gc_j)` around a shared
per-chunk reference point `gn` (e.g. `gn = gc[bt // 2]`) factors it into
two real matmuls (`q_scaled @ k_scaled.T`) instead of a VPU reduction,
which is the kind of change that could meaningfully close the forward gap
described in `KNOWN_LIMITATIONS.md` section 2.

**Explicit precondition -- do not start this before it is met:** the
`beta/gdn2_hybrid.py` path (JAX-forward + fused Pallas-backward, see
`KNOWN_LIMITATIONS.md` section 5) must be fully validated end-to-end
first (residual-parity test, BF16 numbers, memory numbers, full
deep-correctness suite -- see that section's open-items list). The
hybrid path is a smaller, already-working change; if it turns out to
close the forward/backward gap on its own, an MXU-factorized rewrite of
Kernel A/B4 may not be worth its implementation and validation cost. This
hypothesis is the fallback plan **if and only if** the hybrid path is
validated and still leaves a meaningful gap versus JAX_REF/PALLAS.

**What "validating this hypothesis" would require, if pursued (none of
this exists yet):**
1. A from-scratch implementation of the centered factorization in
   `_kernel_a_body`, gated behind a new, explicitly-named opt-in
   `KernelConfig` field (with its own `NotImplementedError` safety gate,
   matching how every other experimental knob in this codebase is
   introduced) -- not assumed to already exist.
2. Isolated correctness test (vs. the default/non-centered path) and
   isolated speed benchmark for Kernel A, then the same for the B4
   backward counterpart, including the backward gradient contribution
   through the shared reference point `gn` (chain rule through
   `eq_i = exp(clip(gc_i - gn))`, `ek_j = exp(clip(gn - gc_j))`) --
   this is exactly the kind of shared-variable backward term that is
   easy to compute but easy to forget to write back; any implementation
   must have an explicit isolated test for it, independently re-derived
   (not copy-pasted from the forward kernel), before it is trusted.
3. Full `custom_vjp` pipeline correctness (multi-seed vs.
   `gdn2_token_serial_reference`, finite-difference, `wy_eps` damping
   interaction, bf16 coverage, `KAGGLE_SMALL` blocking) -- per the
   layered strategy in `docs/TESTING_STRATEGY.md`.
4. End-to-end fwd/bwd/fwdbwd wall-clock through
   `gdn2_pallas_forward_trainable`, not just isolated kernel calls.
5. Peak HBM (`run_memory_benchmark.py`) for the new path.
6. A repeat of the kernel-gap diagnostic (sum of isolated per-kernel
   timings vs. full pipeline) to rule out a new dispatch/scheduling gap
   from the changed intermediate shapes.

**Do not** add a `use_centering` (or similarly named) field to
`KernelConfig`, add branches to `_kernel_a_body`/`_kernel_b4_body`, or
reference this hypothesis as an existing/gated/tested code path in
`README.md`, `CHANGELOG.md`, or `KNOWN_LIMITATIONS.md` until steps 1-6
above have actually been done. Until then this section is the only place
in the repo where this idea should be mentioned.

---

'''

_roadmap_old_pattern = r"## Now: `use_centering=True` — ISOLATED-ONLY, pending e2e validation.*?(?=## Open, no fix scheduled: Kernel B \(WY-solve\))"
regex_replace_once(roadmap, _roadmap_old_pattern, _ROADMAP_HYPOTHESIS,
                    "replace 'Now: use_centering=True ISOLATED-ONLY' section with the hypothesis note")

# Fix the "Why this now matters more" paragraph in the Kernel B section,
# which referenced the (now-removed) gating-criteria section as if
# use_centering were an active, in-progress code path.
_old_kb_para = '''**Why this now matters more:** once `use_centering=True` clears the
gating criteria above, Kernel B becomes the dominant forward cost by a
wide margin (an estimated ~51ms of ~59.6ms forward, i.e. ~86%, based on
isolated kernel-level extrapolation — itself subject to the same
"not yet e2e-validated" caveat as everything else in this document).'''
_new_kb_para = '''**Why this could matter later:** *if* the post-beta `use_centering`
hypothesis above is ever implemented and validated, Kernel B would likely
become the dominant forward cost by a wide margin (an estimated ~51ms of
~59.6ms forward, i.e. ~86%, extrapolated from today's isolated Kernel
A/B4 VPU-vs-MXU numbers) -- itself unconfirmed and entirely contingent on
that hypothesis being pursued at all (see the section above).'''
exact_replace_once(roadmap, _old_kb_para, _new_kb_para,
                    "fix 'Why this now matters more' paragraph in Kernel B section")

_old_completed_note = '''(Nothing from the `use_centering=True` work appears in this section
until the gating criteria above are met.)'''
_new_completed_note = '''(Nothing from the `use_centering` hypothesis appears in this section --
see the hypothesis note above; no code for it exists yet.)'''
exact_replace_once(roadmap, _old_completed_note, _new_completed_note,
                    "fix parenthetical note in 'Completed (VALIDATED, shipped)' section")

# ==========================================================================
# 3. KNOWN_LIMITATIONS.md -- stop implying use_centering is implemented.
# ==========================================================================
print("\n-- KNOWN_LIMITATIONS.md --")

_old_sec6_tail = '''Kernel A and its backward counterpart B4 use the same
broadcast-multiply-reduce pattern (`_weighted_pair_sum` /
`_dL_pair_sum`/`_dR_pair_sum`/`_dgc_pair_sum`) instead of a true MXU
matmul; this is the same computation the `use_centering=True` path
already factors into real `jnp.dot` calls (section 1). This is currently
the leading, evidence-backed hypothesis for the majority of the
fwd/bwd slowdown vs JAX_REF -- not kernel-count/dispatch overhead.'''
_new_sec6_tail = '''Kernel A and its backward counterpart B4 use the same
broadcast-multiply-reduce pattern (`_weighted_pair_sum` /
`_dL_pair_sum`/`_dR_pair_sum`/`_dgc_pair_sum`) instead of a true MXU
matmul. An MXU-factorized alternative (tentatively `use_centering`) is a
documented hypothesis in `ROADMAP.md`, not yet implemented anywhere in
this codebase. This VPU/MXU gap is currently the leading, evidence-backed
hypothesis for the majority of the fwd/bwd slowdown vs JAX_REF -- not
kernel-count/dispatch overhead.'''
exact_replace_once(known_lim, _old_sec6_tail, _new_sec6_tail,
                    "section 6: stop implying use_centering path already exists in code")

_old_v020 = '''- Complete `dgn` gradient propagation in B4 (`_kernel_b4_body`), lift the
  `use_centering=True` restriction in `KernelConfig`.
- Re-run the full correctness suite (`tests/extended/test_gdn2_deep_correctness.py`)
  and the kernel-gap diagnostic with `use_centering=True` on Kernel A/B4.
- If confirmed to close most of the forward gap: this becomes the primary
  fix, and the section 5 hybrid path is downgraded to a documented
  alternative rather than the default recommendation (its main advantage
  -- fast forward -- would be subsumed by a genuinely fast Pallas forward).'''
_new_v020 = '''- Validate the section 5 hybrid path fully end-to-end (see its open
  items list); this is a precondition, not optional.
- Only if the hybrid path does not fully close the fwd/bwd gap: prototype
  and validate the `use_centering` MXU-factorization hypothesis from
  scratch (see `ROADMAP.md` -- no code for it exists yet), including its
  own isolated correctness/speed tests and the full deep-correctness
  suite (`tests/extended/test_gdn2_deep_correctness.py`).
- If validated and shown to close most of the forward gap: this becomes
  the primary fix, and the section 5 hybrid path is downgraded to a
  documented alternative rather than the default recommendation.'''
exact_replace_once(known_lim, _old_v020, _new_v020,
                    "v0.2.0 (planned): make use_centering conditional on hybrid validation")

# ==========================================================================
# 4. README.md -- fix the Limitations bullet.
# ==========================================================================
print("\n-- README.md --")

_old_readme = '''- `use_centering=True` is intentionally disabled: the B4 backward kernel does not propagate
gradients through the shared centering reference point, so the public config refuses to
construct it (see [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md)).'''
_new_readme = '''- The pairwise decay computation (Kernel A / B4) currently uses a VPU-bound broadcast-reduce
pattern rather than an MXU matmul. An MXU-factorized alternative has been sketched as a
post-beta hypothesis but is not implemented in this release; see
[`ROADMAP.md`](ROADMAP.md) and [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).'''
exact_replace_once(readme, _old_readme, _new_readme,
                    "Limitations bullet: use_centering -> VPU/MXU hypothesis pointer")

# ==========================================================================
# 5. CHANGELOG.md -- fix the [0.1.0] Known limitations bullet.
# ==========================================================================
print("\n-- CHANGELOG.md --")

_old_changelog = '''- `use_centering=True` is disabled by `KernelConfig` (B4 backward does not
propagate gradients through the centering reference point). See
`KNOWN_LIMITATIONS.md`.'''
_new_changelog = '''- The pairwise decay computation (Kernel A / B4) is VPU-bound rather than MXU-bound; an
MXU-factorized `use_centering` alternative is a documented post-beta hypothesis, not
implemented in this release. See `KNOWN_LIMITATIONS.md` and `ROADMAP.md`.'''
exact_replace_once(changelog, _old_changelog, _new_changelog,
                    "[0.1.0] Known limitations: use_centering -> hypothesis pointer")

print("\nDone. Review any [SKIP-AMBIGUOUS]/[absent]/[WARN] lines above.")
PYEOF

section "Summary"
if $FIX_MODE; then
  ok "Text surgery applied where patterns matched. .bak backups made next to each edited file."
  ok "Review any [SKIP-AMBIGUOUS]/[absent]/[WARN] messages above -- those need a manual look."
  ok "Now run: pytest tests/test_gdn2_full_math_correctness.py -v"
  ok "If green, git add -A && git commit && git push."
else
  note "Dry-run complete -- nothing written. Re-run with --fix to apply, then commit/push."
fi