#!/usr/bin/env bash
# Concatenate project files into context.txt with a filtered directory tree,
# STRICTLY excluding:
#   .git, .venv, venv, __pycache__, .next, .open-next, .wrangler,
#   node_modules, vendor, dist,
#   build outputs and vendor trees (build, .pio, .idea, .vscode, third_party),
#   the output file itself, concat.sh,
#   .gitignore, ./package.json, ANY package-lock.json, ANY *.lock,
#   .tmp-seed.sql, venv/bin/flask, .DS_Store, *.excalidraw,
#   ANY *.tsbuildinfo files, and binary artifacts (*.elf, *.bin, *.hex, *.map, *.o, *.a, *.d, *.su).

set -euo pipefail

OUT="${1:-context.txt}"
OUT_BASENAME="$(basename "$OUT")"

# Start fresh
: > "$OUT"

# --- 1) Write a filtered directory tree at the top ---
{
  printf "Project tree (excluding: .git, .venv, venv, __pycache__, .next, .open-next, .wrangler, node_modules, vendor, dist, build, .pio, .idea, .vscode, third_party, %s, concat.sh, .gitignore, .tmp-seed.sql, ./package.json, all package-lock.json, all *.lock, all *.tsbuildinfo, venv/bin/flask, .DS_Store, *.excalidraw, and binary artifacts)\n" "$OUT_BASENAME"
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
    'vendor',
    'dist',
    'build',
    '.pio',
    '.idea',
    '.vscode',
    'third_party',
}
# basename-level skips (applies anywhere)
skip_files = {
    out_name,
    'concat.sh',
    '.gitignore',
    '.tmp-seed.sql',
    '.DS_Store',
    'package-lock.json',  # skip all package-lock.json
}
# exact-path skips (for specific paths only)
skip_exact_paths = {
    os.path.normpath('venv/bin/flask'),
    os.path.normpath('./package.json'),
    os.path.normpath('package.json'),
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
    # Skip files by basename (applies everywhere)
    base = os.path.basename(npath)
    if base in skip_files:
        return True
    # Skip any file ending with .excalidraw
    if base.endswith('.excalidraw'):
        return True
    # Skip any lock file (bun.lock, yarn.lock, etc.)
    if base.endswith('.lock'):
        return True
    # Skip TS build-info files
    if base.endswith('.tsbuildinfo'):
        return True
    # Skip compiled/binary outputs that don't add source context
    if base.endswith(('.elf', '.bin', '.hex', '.map', '.o', '.a', '.d', '.su')):
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
# Skip all package-lock.json, *.lock, *.tsbuildinfo, .DS_Store, *.excalidraw,
# compiled artifacts, and vendor/build trees anywhere.
find . \
  -type d \( \
    -name '.git' -o \
    -name '.venv' -o \
    -name 'venv' -o \
    -name '__pycache__' -o \
    -name '.next' -o \
    -name '.open-next' -o \
    -name '.wrangler' -o \
    -name 'node_modules' -o \
    -name 'vendor' -o \
    -name 'dist' -o \
    -name 'build' -o \
    -name '.pio' -o \
    -name '.idea' -o \
    -name '.vscode' -o \
    -name 'third_party' \
  \) -prune -o \
  -type f \
  ! -name "$OUT_BASENAME" \
  ! -name 'concat.sh' \
  ! -name '.gitignore' \
  ! -name '.open-next' \
  ! -name '.wrangler' \
  ! -name '.tmp-seed.sql' \
  ! -name '.DS-Store' \
  ! -name '.DS_Store' \
  ! -name 'package-lock.json' \
  ! -name '*.lock' \
  ! -name '*.tsbuildinfo' \
  ! -name '*.excalidraw' \
  ! -name '*.elf' \
  ! -name '*.bin' \
  ! -name '*.hex' \
  ! -name '*.map' \
  ! -name '*.o' \
  ! -name '*.a' \
  ! -name '*.d' \
  ! -name '*.su' \
  ! -name '*.svg' \
  ! -path './.git/*' \
  ! -path './.venv/*' \
  ! -path './venv/*' \
  ! -path './__pycache__/*' \
  ! -path './.next/*' \
  ! -path './.open-next/*' \
  ! -path './.wrangler/*' \
  ! -path './node_modules/*' \
  ! -path './vendor/*' \
  ! -path './dist/*' \
  ! -path './build/*' \
  ! -path './.pio/*' \
  ! -path './.idea/*' \
  ! -path './.vscode/*' \
  ! -path './third_party/*' \
  ! -path './venv/bin/flask' \
  ! -path './package.json' \
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
