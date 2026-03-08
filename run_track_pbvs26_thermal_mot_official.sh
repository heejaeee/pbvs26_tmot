#!/bin/bash

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .

# Add python path
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/src         # .
export CUDA_LAUNCH_BLOCKING=1

START_TIME="$(date -u +%s.%N)"
###########################################################################################################
# shellcheck disable=SC2128

create_submission_folder() {
    local method="$1"
    shift
    local output_dir="$DIR_TSS/data/tmot_dataset/output_pbvs25/tracking/$method"
    local date_tag
    date_tag=$(date +%Y_%m_%d)
    local index=1
    local submission_dir
    local seqs=""
    local config_file
    local line

    while [ -d "$output_dir/submission_${method}_${date_tag}_${index}" ]; do
        index=$((index + 1))
    done

    submission_dir="$output_dir/submission_${method}_${date_tag}_${index}"
    mkdir -p "$submission_dir"

    for config_file in "$@"; do
        if [ ! -f "$config_file" ]; then
            continue
        fi
        line=$(awk -F':' '/^[[:space:]]*data_dir_seq[[:space:]]*:/ {print $2; exit}' "$config_file")
        if [ -n "$line" ]; then
            line=${line%%#*}
            line=${line//[/}
            line=${line//]/}
            line=${line//\"/}
            line=${line//\'/}
            line=${line//,/ }
            line=$(echo "$line")
            seqs="$seqs $line"
        fi
    done

    if [ -z "$seqs" ]; then
        for seq_dir in "$output_dir"/seq*; do
            [ -d "$seq_dir" ] || continue
            seqs="$seqs $(basename "$seq_dir")"
        done
    fi

    for seq_name in $seqs; do
        local bbox_file="$output_dir/$seq_name/thermal/${seq_name}_thermal.txt"
        if [ -f "$bbox_file" ]; then
            cp "$bbox_file" "$submission_dir/"
        else
            echo "Warning: missing bbox file: $bbox_file"
        fi
    done

    if ls "$submission_dir"/*.txt >/dev/null 2>&1; then
        (cd "$submission_dir" && zip -q "submission_${method}_${date_tag}_${index}.zip" ./*.txt)
    else
        echo "Warning: no txt files found to zip in $submission_dir"
    fi

    echo "Created submission folder: $submission_dir"
}

skip_detection=0
skip_postprocess=0
skip_draw=0
for arg in "$@"; do
    case "$arg" in
        --skip-detection)
            skip_detection=1
            ;;
        --skip-postprocess)
            skip_postprocess=1
            ;;
        --skip-draw)
            skip_draw=1
            ;;
    esac
done

echo "###########################"
echo "STARTING"
echo "###########################"

# NOTE: DETECTION PROCESS
if [ "$skip_detection" -eq 1 ]; then
    echo "*****************"
    echo "DETECTION PROCESS SKIPPED"
    echo "*****************"
else
    echo "*****************"
    echo "DETECTION PROCESS"
    echo "*****************"
    python main.py  \
        --detection  \
        --run_image  \
        --drawing  \
        --config $DIR_TSS"/configs/pbvs25_thermal_mot_sort.yaml"
fi

# NOTE: TRACKING PROCESS
echo "****************"
echo "TRACKING PROCESS"
echo "****************"
yaml_files=(pbvs25_thermal_mot.yaml)
# Extract tracker method from the first config, default to "sort".
tracker_method="sort"
first_config="$DIR_TSS/configs/${yaml_files[0]}"
if [ -f "$first_config" ]; then
    tracker_method=$(awk '
        /^[[:space:]]*tracker:[[:space:]]*$/ {in_tracker=1; next}
        in_tracker && /^[^[:space:]]/ {in_tracker=0}
        in_tracker && /^[[:space:]]*folder_out[[:space:]]*:/ {
            gsub(/#.*/, "", $0)
            sub(/.*:[[:space:]]*/, "", $0)
            gsub(/"/, "", $0)
            print $0
            exit
        }
    ' "$first_config")
    if [ -z "$tracker_method" ]; then
        tracker_method="sort"
    fi
fi
# shellcheck disable=SC2068
for yaml_file in ${yaml_files[@]};
do
    python main.py  \
        --tracking  \
        --run_image  \
        --drawing  \
        --config $DIR_TSS"/configs/"$yaml_file
done

# NOTE: POST-PROCESS TRACKING OUTPUTS
if [ "$skip_postprocess" -eq 1 ]; then
    echo "*****************"
    echo "POST-PROCESS SKIPPED"
    echo "*****************"
else
    echo "*****************"
    echo "POST-PROCESS TRACKS"
    echo "*****************"
    draw_flag=()
    if [ "$skip_draw" -eq 0 ]; then
        draw_flag=(--draw)
    fi
    python scripts/postprocess_track_stitch.py \
        --input-root "$DIR_TSS/data/tmot_dataset/output_pbvs25/tracking/$tracker_method" \
        --config "$DIR_TSS/configs/${yaml_files[0]}" \
        "${draw_flag[@]}"

    python scripts/postprocess_track_relink.py \
        --input-root "$DIR_TSS/data/tmot_dataset/output_pbvs25/tracking/$tracker_method" \
        --config "$DIR_TSS/configs/${yaml_files[0]}" \
        --in-place
fi

config_paths=()
for yaml_file in "${yaml_files[@]}"; do
    config_paths+=("$DIR_TSS/configs/$yaml_file")
done
create_submission_folder "$tracker_method" "${config_paths[@]}"

echo "###########################"
echo "ENDING"
echo "###########################"

###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
