#!/usr/bin/env bash
# One-time setup for the LIBERO-plus robustness benchmark (https://github.com/sylvestf/LIBERO-plus).
#
# LIBERO-plus is a drop-in fork of the `libero` package with 10,030 perturbed tasks across seven
# perturbation dimensions. It is deliberately NOT pip-installed here: it would collide with the
# installed LIBERO (both ship a top-level `libero` package). Instead the checkout lives at
# third_party/LIBERO-plus and is activated per-run via `env.libero_plus=true`, which makes
# robometer_policy_learning.envs.libero_plus.activate() shadow the installed LIBERO on sys.path.
#
# This script:
#   1. initialises the third_party/LIBERO-plus submodule
#   2. downloads assets.zip (~6.4 GB) from the HF dataset Sylvest/LIBERO-plus
#   3. unzips it to third_party/LIBERO-plus/libero/libero/assets (ignored by LIBERO-plus's .gitignore)
#   4. checks the extra runtime deps needed by the Sensor Noise perturbation
#
# Usage:
#   bash scripts/setup_libero_plus.sh              # full setup
#   SKIP_ASSETS=1 bash scripts/setup_libero_plus.sh  # submodule + dependency check only
#
# Re-running is safe: an already-extracted assets dir is left alone unless FORCE_ASSETS=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUS_ROOT="${LIBERO_PLUS_ROOT:-$REPO_ROOT/third_party/LIBERO-plus}"
ASSETS_DIR="$PLUS_ROOT/libero/libero/assets"
# The zip is kept out of the submodule so it can be deleted independently of the checkout.
ZIP_DIR="${LIBERO_PLUS_ZIP_DIR:-$REPO_ROOT/data/libero_plus}"
ZIP_PATH="$ZIP_DIR/assets.zip"
HF_REPO="Sylvest/LIBERO-plus"

log() { printf '\033[1m[libero-plus]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[libero-plus]\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------- 1. submodule
log "repo root:     $REPO_ROOT"
log "LIBERO-plus:   $PLUS_ROOT"
if [ ! -f "$PLUS_ROOT/setup.py" ]; then
    log "initialising submodule third_party/LIBERO-plus"
    git -C "$REPO_ROOT" submodule update --init --recursive third_party/LIBERO-plus
fi
[ -f "$PLUS_ROOT/setup.py" ] || { warn "LIBERO-plus checkout missing at $PLUS_ROOT"; exit 1; }

# The 6.4 GB assets tree is already in LIBERO-plus's .gitignore, so the submodule stays clean.
# Belt-and-braces for the local clone in case upstream ever drops that entry.
EXCLUDE_FILE="$(git -C "$PLUS_ROOT" rev-parse --git-path info/exclude)"
if [ -f "$EXCLUDE_FILE" ] && ! grep -qx "libero/libero/assets/" "$EXCLUDE_FILE" 2>/dev/null; then
    echo "libero/libero/assets/" >>"$EXCLUDE_FILE"
fi

# ------------------------------------------------------------------ 2. assets
if [ "${SKIP_ASSETS:-0}" = "1" ]; then
    log "SKIP_ASSETS=1 -> skipping the assets download"
elif [ -d "$ASSETS_DIR" ] && [ "${FORCE_ASSETS:-0}" != "1" ]; then
    log "assets already present ($(du -sh "$ASSETS_DIR" | cut -f1)) -> skipping download (FORCE_ASSETS=1 to redo)"
else
    mkdir -p "$ZIP_DIR"
    # `hf download` resumes a partial file and no-ops when the blob is already complete.
    log "downloading assets.zip (~6.4 GB) from HF dataset $HF_REPO -> $ZIP_PATH"
    if command -v hf >/dev/null 2>&1; then HF_BIN=hf
    elif [ -x "$REPO_ROOT/.venv/bin/hf" ]; then HF_BIN="$REPO_ROOT/.venv/bin/hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then HF_BIN=huggingface-cli
    else warn "no hf / huggingface-cli found (pip install huggingface-hub)"; exit 1
    fi
    "$HF_BIN" download "$HF_REPO" assets.zip --repo-type dataset --local-dir "$ZIP_DIR"

    # The archive was packed from an absolute path, so `assets/` sits ~9 levels deep under
    # inspire/hdd/project/.../LIBERO-plus-0/. unzip has no --strip-components, so extract into a
    # staging dir, then move the assets tree into place (same filesystem -> instant).
    STAGE="$ZIP_DIR/_extract"
    log "unzipping to $STAGE (~9.5 GB extracted, takes a few minutes)"
    rm -rf "$STAGE" && mkdir -p "$STAGE"
    unzip -q -o "$ZIP_PATH" -d "$STAGE"
    SRC="$(find "$STAGE" -type d -name assets -print -quit)"
    [ -n "$SRC" ] || { warn "no assets/ directory inside $ZIP_PATH -- upstream archive layout changed"; exit 1; }
    log "moving $(basename "$SRC") into $ASSETS_DIR"
    rm -rf "$ASSETS_DIR"
    mv "$SRC" "$ASSETS_DIR"
    rm -rf "$STAGE"
    [ -d "$ASSETS_DIR" ] || { warn "move finished but $ASSETS_DIR is missing"; exit 1; }
    log "assets extracted: $(du -sh "$ASSETS_DIR" | cut -f1)"
    log "assets.zip kept at $ZIP_PATH (safe to delete: rm $ZIP_PATH)"
fi

# ----------------------------------------------------- 3. Sensor Noise extras
# LIBERO-plus's env_wrapper.py imports wand + skimage at module import time, so both are needed for
# ANY LIBERO-plus env. `wand` additionally needs the system ImageMagick library. When ImageMagick is
# absent, libero_plus.activate() installs a stub `wand` module so every non-motion-blur task still
# runs, and raises only if a motion-blur task (noise severity 1-10) is actually stepped.
log "checking Sensor Noise extras (wand, scikit-image)"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" - <<'PY' || true
import importlib.util as u
missing = [m for m, p in (("wand", "wand"), ("scikit-image", "skimage")) if u.find_spec(p) is None]
if missing:
    print(f"  missing python packages: {', '.join(missing)}")
    print("  install with:  uv sync --group libero-plus     (or: uv pip install wand scikit-image)")
else:
    print("  wand + scikit-image importable")
PY
if ! ldconfig -p 2>/dev/null | grep -qi "MagickWand"; then
    warn "system ImageMagick (libMagickWand) not found."
    warn "  Motion-blur Sensor Noise tasks (severity 1-10) need it; everything else works without it."
    warn "  To enable them:  sudo apt-get install -y libmagickwand-dev"
fi

log "done. Smoke-test with:"
log "  uv run python -c \"from robometer_policy_learning.envs.libero_plus import activate, list_tasks; activate(); print(list_tasks('libero_spatial', categories='camera')[:3])\""
