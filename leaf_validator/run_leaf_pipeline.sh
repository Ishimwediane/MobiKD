#!/usr/bin/env bash
# =============================================================================
#  run_leaf_pipeline.sh
#  Full automation script for the Leaf Validator pipeline (leaf vs not_leaf).
#
#  Stages:
#    1. Environment check  (Python, pip, GPU)
#    2. Dependency install (tensorflow, opencv, sklearn, etc.)
#    3. leaf_stage1_train.py   — transfer learning
#    4. leaf_stage2_export.py  — TFLite export & validation
#    5. leaf_stage3_evaluate.py— robustness evaluation + plots
#
#  Usage:
#    chmod +x run_leaf_pipeline.sh
#    ./run_leaf_pipeline.sh              # interactive (default)
#    ./run_leaf_pipeline.sh --skip-install   # skip pip install if already done
#    ./run_leaf_pipeline.sh --stage 2        # start from a specific stage (1/2/3)
#    ./run_leaf_pipeline.sh --log            # tee all output to pipeline.log
# =============================================================================

set -euo pipefail   # exit on error, unbound var, or pipe failure

# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
banner()  {
    echo ""
    echo -e "${BOLD}${CYAN}=================================================================${RESET}"
    echo -e "${BOLD}${CYAN}  $*${RESET}"
    echo -e "${BOLD}${CYAN}=================================================================${RESET}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# DEFAULT FLAGS
# ─────────────────────────────────────────────────────────────
SKIP_INSTALL=false
START_STAGE=1
LOG_MODE=false
LOG_FILE="leaf_pipeline_run.log"

# ─────────────────────────────────────────────────────────────
# PARSE ARGUMENTS
# ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=true; shift ;;
        --stage)
            START_STAGE="$2"
            [[ "$START_STAGE" =~ ^[123]$ ]] || error "--stage must be 1, 2, or 3."
            shift 2 ;;
        --log) LOG_MODE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--skip-install] [--stage 1|2|3] [--log]"
            exit 0 ;;
        *) error "Unknown option: $1. Use --help." ;;
    esac
done

# Tee output to log file if requested
if $LOG_MODE; then
    exec > >(tee -a "$LOG_FILE") 2>&1
    info "Logging to $LOG_FILE"
fi

# ─────────────────────────────────────────────────────────────
# 0. LOCATE PROJECT ROOT (same dir as this script)
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

banner "LEAF VALIDATOR — Automated Pipeline"
info "Project root : $PROJECT_DIR"
info "Start stage  : $START_STAGE"
info "Skip install : $SKIP_INSTALL"
echo ""

# Switch to project directory
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────
# 1. PYTHON CHECK
# ─────────────────────────────────────────────────────────────
banner "STEP 1 — Python & Environment Check"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1)
        info "Found: $cmd → $VER"
        PYTHON_CMD="$cmd"
        break
    fi
done
[[ -z "$PYTHON_CMD" ]] && error "Python3 not found. Install it: sudo apt install python3 python3-pip python3-venv"

# ─────────────────────────────────────────────────────────────
# 2. VIRTUAL ENVIRONMENT
# ─────────────────────────────────────────────────────────────
VENV_DIR="$PROJECT_DIR/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR …"
    $PYTHON_CMD -m venv "$VENV_DIR" || error "Failed to create venv. Install: sudo apt install python3-venv"
    success "Virtual environment created."
else
    info "Virtual environment already exists at $VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtual environment activated."

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ─────────────────────────────────────────────────────────────
# 3. INSTALL DEPENDENCIES
# ─────────────────────────────────────────────────────────────
if ! $SKIP_INSTALL; then
    banner "STEP 2 — Installing Dependencies"
    info "Upgrading pip …"
    $PIP install --upgrade pip --quiet

    info "Installing TensorFlow with CUDA support …"
    $PIP install "tensorflow[and-cuda]" --quiet || {
        warn "tensorflow[and-cuda] failed — falling back to tensorflow-gpu"
        $PIP install tensorflow-gpu --quiet || {
            warn "tensorflow-gpu also failed — falling back to tensorflow (CPU only)"
            $PIP install tensorflow --quiet
        }
    }

    info "Installing other dependencies …"
    $PIP install \
        numpy \
        opencv-python-headless \
        Pillow \
        scikit-learn \
        matplotlib \
        seaborn \
        pandas \
        --quiet

    success "All dependencies installed."
else
    info "Skipping dependency installation (--skip-install)."
fi

# ─────────────────────────────────────────────────────────────
# 4. GPU VERIFICATION
# ─────────────────────────────────────────────────────────────
banner "STEP 3 — GPU Detection"

GPU_CHECK=$($PYTHON - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow : {tf.__version__}")
print(f"GPUs found : {len(gpus)}")
for g in gpus:
    print(f"  → {g.name}")
if not gpus:
    print("WARNING: No GPU detected — training will run on CPU (slower).")
EOF
)
echo "$GPU_CHECK"

if echo "$GPU_CHECK" | grep -q "WARNING: No GPU"; then
    warn "No GPU detected. Training will proceed on CPU."
    warn "To fix: ensure NVIDIA drivers are installed on Windows and nvidia-smi works in WSL."
    echo ""
    read -rp "Continue without GPU? (y/n): " CONT
    [[ "$CONT" =~ ^[Yy]$ ]] || { info "Aborted by user."; exit 0; }
else
    success "GPU is ready."
fi

# ─────────────────────────────────────────────────────────────
# 5. DATASET & BACKBONE CHECKS
# ─────────────────────────────────────────────────────────────
banner "STEP 4 — Dataset & Backbone Verification"

DATASET_DIR="$PROJECT_DIR/lead and not leaf"
LEAF_DIR="$DATASET_DIR/leaf"
NON_LEAF_DIR="$DATASET_DIR/non_leaf"
BACKBONE="$PROJECT_DIR/models/kd_mobilenetv2.keras"

check_dir_nonempty() {
    local dir="$1" label="$2"
    if [[ ! -d "$dir" ]]; then
        error "$label directory not found: $dir"
    fi
    local count
    count=$(find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" \) | wc -l)
    if [[ "$count" -eq 0 ]]; then
        error "$label directory is empty (no images): $dir"
    fi
    success "$label : $count images found in $dir"
}

check_dir_nonempty "$LEAF_DIR"     "leaf"
check_dir_nonempty "$NON_LEAF_DIR" "non_leaf"

if [[ ! -f "$BACKBONE" ]]; then
    error "Backbone model not found: $BACKBONE
    Make sure kd_mobilenetv2.keras is in the models/ folder.
    Copy from Windows if needed:
      cp /mnt/c/Users/HP\ VICTUS/Desktop/MobiKD/models/kd_mobilenetv2.keras $PROJECT_DIR/models/"
fi
success "Backbone found: $BACKBONE"
echo ""

# ─────────────────────────────────────────────────────────────
# HELPER: run a Python stage with timing + error handling
# ─────────────────────────────────────────────────────────────
run_stage() {
    local stage_num="$1"
    local script="$2"
    local label="$3"

    if [[ "$stage_num" -lt "$START_STAGE" ]]; then
        info "Skipping Stage $stage_num ($label) — --stage=$START_STAGE"
        return 0
    fi

    banner "STAGE $stage_num — $label"
    info "Running: $PYTHON $script"
    echo "──────────────────────────────────────────────────────────────"

    local t_start
    t_start=$(date +%s)

    if $PYTHON "$script"; then
        local t_end elapsed
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        echo "──────────────────────────────────────────────────────────────"
        success "Stage $stage_num complete in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
    else
        echo "──────────────────────────────────────────────────────────────"
        error "Stage $stage_num FAILED: $script exited with error code $?"
    fi
    echo ""
}

# ─────────────────────────────────────────────────────────────
# 6. RUN THE THREE PIPELINE STAGES
# ─────────────────────────────────────────────────────────────
run_stage 1 "leaf_stage1_train.py"   "Transfer Learning Training"
run_stage 2 "leaf_stage2_export.py"  "TFLite Export & Validation"
run_stage 3 "leaf_stage3_evaluate.py" "Robustness Evaluation & Plots"

# ─────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────
banner "PIPELINE COMPLETE"

echo -e "${BOLD}Outputs:${RESET}"
echo ""

# Models
echo -e "  ${BOLD}Models (models/):${RESET}"
for f in \
    "models/kd_mobilenetv2_leaf_validator.keras" \
    "models/kd_mobilenetv2_leaf_validator_fp32.tflite" \
    "models/kd_mobilenetv2_leaf_validator_float16.tflite"; do
    if [[ -f "$PROJECT_DIR/$f" ]]; then
        SIZE=$(du -sh "$PROJECT_DIR/$f" | cut -f1)
        echo -e "    ${GREEN}✓${RESET} $f  ($SIZE)"
    else
        echo -e "    ${RED}✗${RESET} $f  (not found)"
    fi
done

# Results
echo ""
echo -e "  ${BOLD}Results (results/leaf/):${RESET}"
RESULTS_DIR="$PROJECT_DIR/results/leaf"
if [[ -d "$RESULTS_DIR" ]]; then
    for f in "$RESULTS_DIR"/*; do
        echo -e "    ${GREEN}✓${RESET} $(basename "$f")"
    done
else
    echo -e "    ${RED}✗${RESET} results/leaf/ not found"
fi

# Plots
echo ""
echo -e "  ${BOLD}Plots (plots/leaf/):${RESET}"
PLOTS_DIR="$PROJECT_DIR/plots/leaf"
if [[ -d "$PLOTS_DIR" ]]; then
    for f in "$PLOTS_DIR"/*.png; do
        echo -e "    ${GREEN}✓${RESET} $(basename "$f")"
    done
else
    echo -e "    ${RED}✗${RESET} plots/leaf/ not found"
fi

echo ""
echo -e "${BOLD}${GREEN}All done! To copy results back to Windows:${RESET}"
echo ""
echo "  cp models/kd_mobilenetv2_leaf_validator*.keras \\"
echo "     models/*.tflite \\"
echo "     /mnt/c/Users/HP\ VICTUS/Desktop/MobiKD/models/"
echo ""
echo "  cp -r results/leaf/ /mnt/c/Users/HP\ VICTUS/Desktop/MobiKD/results/"
echo "  cp -r plots/leaf/   /mnt/c/Users/HP\ VICTUS/Desktop/MobiKD/plots/"
echo ""
echo -e "${BOLD}Next: test inference with${RESET}"
echo "  python deployment/inference_leaf_validator.py <image_path>"
echo ""
