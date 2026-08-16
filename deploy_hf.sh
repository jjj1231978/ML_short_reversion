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

if [ "${1:-}" = "--commit" ]; then
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
