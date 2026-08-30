#!/bin/bash
set -e

LAUNCHER="/workspace/isaaclab/isaaclab.sh"
PLAY="scripts/rsl_rl/play.py"
TASK="Ankybot-Play-v2"
BASE="logs/rsl_rl/imu_sensor/baseline"
LOG="/workspace/Ankybot_v2/scripts/record_videos.log"

cd /workspace/Ankybot_v2

run_recording() {
    local model=$1
    local steps=$2
    echo "=== $(date) | START model_${model}.pt | ${steps} steps ===" | tee -a "$LOG"
    "$LAUNCHER" -p "$PLAY" \
        --task "$TASK" \
        --num_envs 128 \
        --viz kit \
        --video \
        --video_length "$steps" \
        --video_tag "model_${model}" \
        --checkpoint "${BASE}/model_${model}.pt" \
        2>&1 | tee -a "$LOG"
    echo "=== $(date) | DONE  model_${model}.pt ===" | tee -a "$LOG"
}

run_recording 100 1500
run_recording 200 1500
run_recording 500 1500

echo "=== $(date) | ALL RECORDINGS COMPLETE ===" | tee -a "$LOG"
