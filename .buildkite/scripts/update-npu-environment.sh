#!/bin/bash
# Purpose: Updates an NPU test container to match the requested VIME commit.
#          - Reads the image's persisted OLD patch series and exact patch bytes
#          - Updates VIME, then reconciles OLD -> NEW in declared series order
#          - Installs the current VIME checkout and normalizes visible devices
# Usage: Called by Buildkite pipeline during NPU test runs
set -e -o pipefail

VIME_DIR="${VIME_DIR:-/root/vime}"
VIME_NPU_PATCH_STATE_DIR="${VIME_NPU_PATCH_STATE_DIR:-/opt/npu_patch}"
VIME_NPU_PATCH_SOURCE_ROOT="${VIME_NPU_PATCH_SOURCE_ROOT:-${VIME_DIR}}"
PATCH_SERIES_RELATIVE_PATH="docker/npu_patch/series.conf"

sha256_stdin() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

SERIES_ENTRIES=()

load_series() {
    local series_file="$1"
    local line target image_patch source_patch extra

    SERIES_ENTRIES=()
    if [ ! -f "$series_file" ]; then
        echo "ERROR: Patch series not found: $series_file" >&2
        return 1
    fi

    while IFS= read -r line || [ -n "$line" ]; do
        if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi

        IFS='|' read -r target image_patch source_patch extra <<< "$line"
        if [ -z "$target" ] || [ -z "$image_patch" ] || [ -z "$source_patch" ] || [ -n "$extra" ]; then
            echo "ERROR: Invalid patch series entry: $line" >&2
            return 1
        fi
        if [[ "$image_patch" = */* ]]; then
            echo "ERROR: Image patch must be a flat file name: $image_patch" >&2
            return 1
        fi
        if [[ "$source_patch" = /* || "/$source_patch/" = *"/../"* ]]; then
            echo "ERROR: Source patch must stay under the VIME root: $source_patch" >&2
            return 1
        fi
        SERIES_ENTRIES+=("${target}|${image_patch}|${source_patch}")
    done < "$series_file"
}

validate_series_entry() {
    local target="$1"
    local patch_path="$2"

    if ! git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "ERROR: Patch target is not a Git worktree: $target" >&2
        return 1
    fi
    if [ ! -f "$patch_path" ]; then
        echo "ERROR: Patch file not found: $patch_path" >&2
        return 1
    fi
}

series_digest() {
    local series_file="$1"
    local patch_root="$2"
    local path_kind="$3"
    local entry target image_patch source_patch patch_ref patch_path patch_sha

    load_series "$series_file"
    for entry in "${SERIES_ENTRIES[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        case "$path_kind" in
            image) patch_ref="$image_patch" ;;
            source) patch_ref="$source_patch" ;;
            *) echo "ERROR: Unknown patch path kind: $path_kind" >&2; return 1 ;;
        esac
        patch_path="${patch_root}/${patch_ref}"
        if [ ! -f "$patch_path" ]; then
            echo "ERROR: Patch file not found: $patch_path" >&2
            return 1
        fi
    done

    {
        printf 'vime-npu-patch-series-v1\0'
        for entry in "${SERIES_ENTRIES[@]}"; do
            IFS='|' read -r target image_patch source_patch <<< "$entry"
            if [ "$path_kind" = "image" ]; then
                patch_ref="$image_patch"
            else
                patch_ref="$source_patch"
            fi
            patch_path="${patch_root}/${patch_ref}"
            patch_sha=$(sha256_file "$patch_path")
            printf '%s\0%s\0%s\0%s\0' "$target" "$image_patch" "$source_patch" "$patch_sha"
        done
    } | sha256_stdin
}

apply_series() {
    local series_file="$1"
    local source_root="$2"
    local entry target image_patch source_patch patch_path

    load_series "$series_file"
    for entry in "${SERIES_ENTRIES[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        patch_path="${source_root}/${source_patch}"
        validate_series_entry "$target" "$patch_path"
        echo "INFO: Applying $source_patch to $target"
        git -C "$target" apply --whitespace=nowarn "$patch_path"
    done
}

revert_series() {
    local series_file="$1"
    local image_root="$2"
    local i entry target image_patch source_patch patch_path

    load_series "$series_file"
    for ((i=${#SERIES_ENTRIES[@]}-1; i>=0; i--)); do
        entry="${SERIES_ENTRIES[$i]}"
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        patch_path="${image_root}/${image_patch}"
        validate_series_entry "$target" "$patch_path"
        echo "INFO: Reverting $image_patch from $target"
        git -C "$target" apply --reverse --whitespace=nowarn "$patch_path"
    done
}

# Swap one installed patch. If the checkout's patch does not apply on this
# image, put the image patch back so the rest of the series can still run.
swap_patch() {
    local target="$1"
    local old_patch="$2"
    local new_patch="$3"
    local label

    label="$(basename "$new_patch")"
    echo "INFO: Reverting $(basename "$old_patch") from $target"
    git -C "$target" apply --reverse --whitespace=nowarn "$old_patch"
    if git -C "$target" apply --check --whitespace=nowarn "$new_patch"; then
        echo "INFO: Applying $label to $target"
        git -C "$target" apply --whitespace=nowarn "$new_patch"
        return
    fi
    echo "WARNING: $label does not apply on this image; keeping the image patch" >&2
    git -C "$target" apply --whitespace=nowarn "$old_patch"
}

reconcile_series() {
    local old_series="$1"
    local old_root="$2"
    local new_series="$3"
    local new_root="$4"
    local entry target image_patch source_patch key old_path new_path
    local -a old_entries new_entries changed_keys
    local -A old_source_by_key new_source_by_key

    if [ "$(series_digest "$old_series" "$old_root" image)" = "$(series_digest "$new_series" "$new_root" source)" ]; then
        echo "INFO: Patch series is unchanged"
        return
    fi

    # The default image can lag the branch. Replaying every patch then fails on
    # files the image vllm does not have. Update only the same installed patch
    # whose bytes changed, and keep the image copy when the new one does not apply.
    load_series "$old_series"
    old_entries=("${SERIES_ENTRIES[@]}")
    load_series "$new_series"
    new_entries=("${SERIES_ENTRIES[@]}")

    for entry in "${old_entries[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        key="${target}|${image_patch}"
        old_source_by_key["$key"]="${old_root}/${image_patch}"
    done
    for entry in "${new_entries[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        key="${target}|${image_patch}"
        new_source_by_key["$key"]="${new_root}/${source_patch}"
    done

    changed_keys=()
    for entry in "${old_entries[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        key="${target}|${image_patch}"
        new_path="${new_source_by_key[$key]:-}"
        if [ -z "$new_path" ]; then
            echo "WARNING: Image patch $image_patch is not in the checkout series; leaving it applied" >&2
            continue
        fi
        if [ "$(sha256_file "${old_source_by_key[$key]}")" = "$(sha256_file "$new_path")" ]; then
            continue
        fi
        changed_keys+=("$key")
    done

    # Reverse series order matches revert_series, so later patches come off first.
    local i
    for ((i=${#changed_keys[@]}-1; i>=0; i--)); do
        key="${changed_keys[$i]}"
        target="${key%%|*}"
        old_path="${old_source_by_key[$key]}"
        new_path="${new_source_by_key[$key]}"
        validate_series_entry "$target" "$old_path"
        validate_series_entry "$target" "$new_path"
        swap_patch "$target" "$old_path" "$new_path"
    done

    for entry in "${new_entries[@]}"; do
        IFS='|' read -r target image_patch source_patch <<< "$entry"
        key="${target}|${image_patch}"
        if [ -n "${old_source_by_key[$key]:-}" ]; then
            continue
        fi
        echo "WARNING: Checkout patch $source_patch is not installed in this image; skipping it" >&2
    done
}

update_vime_code() {
    echo "INFO: Updating VIME code..."

    if [ -n "${BUILDKITE_COMMIT:-}" ]; then
        echo "INFO: Fetching and checking out commit ${BUILDKITE_COMMIT}"
        git -C "$VIME_DIR" fetch origin "${BUILDKITE_COMMIT}"
        git -C "$VIME_DIR" checkout "${BUILDKITE_COMMIT}"
    else
        echo "INFO: BUILDKITE_COMMIT not set, skipping code update"
    fi
}

install_vime_code() {
    pip install -e "$VIME_DIR" --no-deps --break-system-packages || pip install -e "$VIME_DIR" --no-deps
}

sort_ascend_visible_devices() {
    export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-}}"
    echo "Value: ${ASCEND_VISIBLE_DEVICES}"
    if [ -n "${ASCEND_VISIBLE_DEVICES}" ]; then
        SORTED_DEVICES=$(echo "${ASCEND_VISIBLE_DEVICES}" | tr ',' '\n' | sort -n | tr '\n' ',')
        SORTED_DEVICES=${SORTED_DEVICES%,}
        export ASCEND_VISIBLE_DEVICES=$SORTED_DEVICES
        echo "Sorted ASCEND_VISIBLE_DEVICES: $ASCEND_VISIBLE_DEVICES"
    fi
}

main() {
    local old_series="${VIME_NPU_PATCH_STATE_DIR}/series.conf"
    local new_series="${VIME_NPU_PATCH_SOURCE_ROOT}/${PATCH_SERIES_RELATIVE_PATH}"

    echo "=== Step 1: Sort ASCEND_VISIBLE_DEVICES ==="
    sort_ascend_visible_devices

    echo "=== Step 2: Update VIME code ==="
    update_vime_code

    if [ ! -f "$old_series" ]; then
        echo "ERROR: The selected image does not contain NPU patch state: $old_series" >&2
        echo "ERROR: Build a patch-state-enabled NPU image before running this commit." >&2
        return 1
    fi

    echo "=== Step 3: Reconcile image patches with current VIME patches ==="
    reconcile_series "$old_series" "$VIME_NPU_PATCH_STATE_DIR" "$new_series" "$VIME_NPU_PATCH_SOURCE_ROOT"

    echo "=== Step 4: Install current VIME code ==="
    install_vime_code

    echo "INFO: NPU environment update completed successfully"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        series-digest)
            if [ "$#" -ne 4 ]; then
                echo "Usage: $0 series-digest SERIES_FILE PATCH_ROOT image|source" >&2
                exit 2
            fi
            series_digest "$2" "$3" "$4"
            exit
            ;;
        reconcile)
            if [ "$#" -ne 5 ]; then
                echo "Usage: $0 reconcile OLD_SERIES OLD_ROOT NEW_SERIES NEW_ROOT" >&2
                exit 2
            fi
            reconcile_series "$2" "$3" "$4" "$5"
            exit
            ;;
    esac
fi

main
