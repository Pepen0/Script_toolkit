#!/usr/bin/env bash
# Concatenate project files into context.txt with a filtered directory tree,
# STRICTLY excluding .git, venv/.venv (and all descendants), __pycache__,
# .next, .open-next, .wrangler, node_modules, the output file itself, concat.sh,
# .gitignore, ./package.json, ./package-lock.json, .tmp-seed.sql,
# and the specific path venv/bin/flask.

set -euo pipefail

OUT="${1:-context.txt}"
OUT_BASENAME="$(basename "$OUT")"

# Start fresh
: > "$OUT"

# --- 1) Write a filtered directory tree at the top ---
{
  printf "Project tree (excluding: .git, .venv, venv, __pycache__, .next, .open-next, .wrangler, node_modules, %s, concat.sh, .gitignore, .tmp-seed.sql, ./package.json, ./package-lock.json, venv/bin/flask)\n" "$OUT_BASENAME"
  python3 - <<'PY' "$OUT_BASENAME"
import os, sys

out_name = sys.argv[1]
skip_dirs = {
    '.git',
    '.venv',
    'venv',
    '__pycache__',
    '.next',
    '.open-next',
    '.wrangler',
    'node_modules',
}
# basename-level skips
skip_files = {out_name, 'concat.sh', '.gitignore', '.tmp-seed.sql'}
skip_exact_paths = {
    os.path.normpath('venv/bin/flask'),
    os.path.normpath('./package.json'),
    os.path.normpath('./package-lock.json'),
    os.path.normpath('package.json'),
    os.path.normpath('package-lock.json'),
}

def is_skipped_path(path):
    npath = os.path.normpath(path)
    # Skip any exact bad path
    if npath in skip_exact_paths:
        return True
    parts = npath.split(os.sep)
    # Skip if any component is a skipped directory
    if any(p in skip_dirs for p in parts):
        return True
    # Skip files by basename
    base = os.path.basename(npath)
    if base in skip_files:
        return True
    return False

class Node:
    def __init__(self, path, is_dir):
        self.path = path
        self.name = os.path.basename(path) if path != '.' else '.'
        self.is_dir = is_dir
        self.children = []

root = Node('.', True)
path_to_node = {'.': root}

for cur, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if not is_skipped_path(os.path.join(cur, d))]
    files = [f for f in files if not is_skipped_path(os.path.join(cur, f))]
    dirs.sort(); files.sort()
    parent = path_to_node[cur]
    for d in dirs:
        p = os.path.join(cur, d)
        node = Node(p, True)
        parent.children.append(node)
        path_to_node[p] = node
    for f in files:
        parent.children.append(Node(os.path.join(cur, f), False))

dirs_count = files_count = 0

def rec(n, prefix='', is_last=True):
    global dirs_count, files_count
    if n.path == '.':
        print('.')
    else:
        connector = '└── ' if is_last else '├── '
        print(prefix + connector + n.name)
    if n.is_dir:
        if n.path != '.':
            dirs_count += 1
        new_prefix = prefix + ('    ' if is_last else '│   ')
        for i, child in enumerate(n.children):
            rec(child, new_prefix, i == len(n.children) - 1)
    else:
        files_count += 1

rec(root)
print(f"\n{dirs_count} directories, {files_count} files")
PY
  printf '\n'
} >> "$OUT"

# --- 2) Append concatenated file contents with strict filtering ---
find . \
  -type d \( \
    -name '.git' -o \
    -name '.venv' -o \
    -name 'venv' -o \
    -name '__pycache__' -o \
    -name '.next' -o \
    -name '.open-next' -o \
    -name '.wrangler' -o \
    -name 'node_modules' \
  \) -prune -o \
  -type f \
  ! -name "$OUT_BASENAME" \
  ! -name 'concat.sh' \
  ! -name '.gitignore' \
  ! -name '.open-next' \
  ! -name '.wrangler' \
  ! -name '.tmp-seed.sql' \
  ! -path './.git/*' \
  ! -path './.venv/*' \
  ! -path './venv/*' \
  ! -path './__pycache__/*' \
  ! -path './.next/*' \
  ! -path './.open-next/*' \
  ! -path './.wrangler/*' \
  ! -path './node_modules/*' \
  ! -path './venv/bin/flask' \
  ! -path './package.json' \
  ! -path './package-lock.json' \
  -print0 |
while IFS= read -r -d '' file; do
  if grep -Iq . "$file" || [ ! -s "$file" ]; then
    printf '===== BEGIN %s =====\n' "$file" >> "$OUT"
    cat "$file" >> "$OUT"
    printf '\n===== END %s =====\n\n' "$file" >> "$OUT"
  fi
done

# --- 3) Report how many lines were written ---
LINE_COUNT=$(wc -l < "$OUT" | tr -d '[:space:]')
printf 'Wrote %s lines of concatenated context to %s\n' "$LINE_COUNT" "$OUT"
