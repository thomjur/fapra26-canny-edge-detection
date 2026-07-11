#!/bin/bash

ROOT="${1:-.}"

IGNORE=("brot_assets" "logs" "target" ".git")

should_ignore() {
    local name="$1"
    for ignored in "${IGNORE[@]}"; do
        [ "$name" = "$ignored" ] && return 0
    done
    return 1
}

print_tree() {
    local dir="$1"
    local indent="$2"

    for entry in "$dir"/*; do
        [ -e "$entry" ] || continue
        local name
        name=$(basename "$entry")

        should_ignore "$name" && continue

        if [ -d "$entry" ]; then
            echo "${indent}${name}/"
            print_tree "$entry" "${indent}    "
        else
            echo "${indent}${name}"
        fi
    done
}

OUTPUT="$(basename "$ROOT")/
$(print_tree "$ROOT" "    ")"

echo "$OUTPUT"

# Copy to clipboard
if command -v pbcopy &>/dev/null; then
    echo "$OUTPUT" | pbcopy
    echo "Copied to clipboard (pbcopy)"
elif command -v xclip &>/dev/null; then
    echo "$OUTPUT" | xclip -selection clipboard
    echo "Copied to clipboard (xclip)"
elif command -v xsel &>/dev/null; then
    echo "$OUTPUT" | xsel --clipboard --input
    echo "Copied to clipboard (xsel)"
else
    echo "No clipboard tool found (pbcopy / xclip / xsel)"
fi