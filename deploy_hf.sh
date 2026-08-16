#!/usr/bin/env bash
# Deploy the current commit to the private Hugging Face Space.
#
#   ./deploy_hf.sh            # deploy HEAD
#   ./deploy_hf.sh --commit   # stage+commit any changed viewer artifacts first
#
# GitHub is NOT in this path. The Space is an independent git remote; pushing to
# GitHub is optional and does not trigger a rebuild. (The GitHub Action described
# in ../DL-research-demo's constitution does not actually exist there either.)
#
# Why an orphan branch instead of `git push hf HEAD:main`:
# this repo's history contains PNGs committed as raw git blobs, and the HF Hub's
# pre-receive hook rejects any push containing raw binaries. Rather than rewrite
# shared history with `git lfs migrate` (which would force-push GitHub too), each
# deploy rebuilds a single-commit orphan branch in a throwaway worktree, where
# .gitattributes puts every binary through LFS. The Hub gets a clean snapshot and
# no upstream history; your working tree is never touched — which matters when a
# backtest is mid-run writing into data/processed.
set -euo pipefail
cd "$(dirname "$0")"

SPACE_URL="https://huggingface.co/spaces/JJ-JIN12345/ml-short-reversion"
HF_USER="JJ-JIN12345"
WT="$(mktemp -d)/hfdeploy"

DO_COMMIT=0
SKIP_SYNTAX=0
for arg in "$@"; do
    case "$arg" in
        --commit)             DO_COMMIT=1 ;;
        --skip-syntax-check)  SKIP_SYNTAX=1 ;;
        *) echo "unknown flag: $arg (want --commit / --skip-syntax-check)" >&2; exit 2 ;;
    esac
done

if [ "$DO_COMMIT" = "1" ]; then
    # Only the whitelisted viewer artifacts in .gitignore can land here.
    git add data/processed notebooks/figures 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "chore(data): refresh viewer artifacts for Space deploy"
    else
        echo "nothing new to commit"
    fi
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty. Commit first, or run with --commit." >&2
    exit 1
fi

# Parse every file the Space executes under the Python the Space actually runs.
# The dev venv may be newer, in which case `python -m py_compile` and even a
# streamlit AppTest run locally prove nothing: PEP 701 f-strings parse on 3.12
# and raise "unterminated string literal" on 3.11. Streamlit compiles a page
# lazily on first view, so such a break is invisible until a user clicks it.
# The tree is clean by the check above, so the working copy == what gets pushed.
if [ "$SKIP_SYNTAX" = "0" ]; then
    PYVER=$(grep -oP '^FROM python:\K[0-9]+\.[0-9]+' Dockerfile)
    LOCALVER=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    CHECK='
import ast, pathlib, sys
bad = []
for p in sorted(list(pathlib.Path("app").rglob("*.py")) + list(pathlib.Path("src").rglob("*.py"))):
    try:
        ast.parse(p.read_text(), str(p))
    except SyntaxError as e:
        bad.append("  %s:%s  %s" % (p, e.lineno, e.msg))
if bad:
    sys.stderr.write("ERROR: these files do not parse under Python %d.%d:\n%s\n"
                     % (sys.version_info[0], sys.version_info[1], "\n".join(bad)))
    raise SystemExit(1)
'
    if [ "$LOCALVER" = "$PYVER" ]; then
        python -c "$CHECK"
    elif command -v docker >/dev/null 2>&1; then
        docker run --rm -v "$PWD":/w -w /w "python:$PYVER-slim" python -c "$CHECK"
    else
        echo "ERROR: cannot check syntax for Python $PYVER (venv is $LOCALVER, no docker)." >&2
        echo "       Match the venv to $PYVER, install docker, or --skip-syntax-check." >&2
        exit 1
    fi
    echo "syntax OK for Python $PYVER"
fi

: "${HF_API_KEY:=$(grep -oP '^HF_API_KEY=\K.*' .env | tr -d '"'"'"'')}"
if [ -z "${HF_API_KEY:-}" ]; then
    echo "ERROR: HF_API_KEY not found in environment or .env" >&2
    exit 1
fi
export HF_API_KEY

ASKPASS="$(mktemp)"
cat > "$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  Username*) echo "$HF_USER_ENV" ;;
  Password*) echo "$HF_API_KEY" ;;
esac
EOF
chmod +x "$ASKPASS"
export HF_USER_ENV="$HF_USER"

HEADSHA=$(git rev-parse HEAD)
echo "deploying $HEADSHA -> $SPACE_URL"

# Clear any leftover deploy worktree from an interrupted run FIRST. A branch
# that is checked out in a worktree cannot be deleted, so `git branch -D` alone
# fails here and the whole deploy aborts.
git worktree list --porcelain \
    | awk '/^worktree /{p=$2} /^branch refs\/heads\/hf-main$/{print p}' \
    | while read -r old; do git worktree remove --force "$old" 2>/dev/null || true; done
git worktree prune
git branch -D hf-main 2>/dev/null || true
git worktree add --detach "$WT" "$HEADSHA" >/dev/null
trap 'git worktree remove --force "$WT" 2>/dev/null || true; rm -f "$ASKPASS"' EXIT

(
    cd "$WT"
    git checkout -q --orphan hf-main

    # Publish only what the Space executes. This is a throwaway worktree, so
    # these deletions never touch your checkout.
    #
    # tests/ is the reason this exists: HF's secret scanner reads a pytest
    # function named `test_` + ~35 chars as a Lob API key and flags the Space
    # on every deploy. Renaming the offenders does not hold -- the file already
    # carries a comment from the last time, and six names still match. Not
    # shipping the directory ends the false positives permanently, and the
    # Space never ran the suite anyway.
    #
    # scripts/ holds the IBKR order-placement path; .specify/ is spec prose.
    # Neither is imported by app/ (which pulls only src.config,
    # src.backtest.{portfolio,live_trackrecord,return_calibration} and
    # src.research.{deepdive,technicals}), so both are dead weight on the Hub.
    #
    # notebooks/figures MUST stay: app/pages/03_shap.py renders the persisted
    # SHAP PNGs from there. Only the .ipynb files go.
    rm -rf tests scripts .specify
    find notebooks -maxdepth 1 -name '*.ipynb' -delete 2>/dev/null || true

    git add -A
    # Unchanged files skip the LFS clean filter, so force it across the tree —
    # without this, previously-committed PNGs stay raw blobs and HF rejects them.
    git add --renormalize .
    git commit -q -m "Deploy: ML Short Reversion viewer ($(echo "$HEADSHA" | cut -c1-8))"

    # Fail loudly rather than let the Hub reject the push after a long upload.
    raw=$(git ls-files | grep -iE '\.(png|jpg|jpeg|parquet|db|joblib|h5|npy)$' \
          | while read -r f; do
                git show ":$f" | head -c 40 | grep -q git-lfs || echo "$f"
            done)
    if [ -n "$raw" ]; then
        echo "ERROR: these binaries are not LFS pointers:" >&2
        echo "$raw" >&2
        exit 1
    fi

    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
        git push "$SPACE_URL" hf-main:main --force
)

echo
echo "pushed. the Space rebuilds automatically (~2-4 min):"
echo "  $SPACE_URL"
