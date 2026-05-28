# reorganize.ps1
# Cleans up the MobiKD project folder structure.
# Run from the MobiKD root:  powershell -ExecutionPolicy Bypass -File reorganize.ps1

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ''
Write-Host '=== MobiKD Project Reorganizer ===' -ForegroundColor Cyan
Write-Host "Root: $Root"
Write-Host ''

function Move-Safe {
    param($Src, $Dst)
    if (Test-Path $Src) {
        $DstDir = Split-Path $Dst -Parent
        if (-not (Test-Path $DstDir)) { New-Item -ItemType Directory -Path $DstDir -Force | Out-Null }
        Move-Item -Path $Src -Destination $Dst -Force
        Write-Host "  Moved: $Src" -ForegroundColor Green
    } else {
        Write-Host "  Skip : $Src" -ForegroundColor DarkGray
    }
}

function New-Dir {
    param($Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "  Created: $Path" -ForegroundColor Yellow
    }
}

# --- 1. CREATE FOLDERS ---
Write-Host '[ 1 ] Creating folder structure...' -ForegroundColor Cyan
New-Dir "$Root\literob"
New-Dir "$Root\leaf_validator"
New-Dir "$Root\models\literob"
New-Dir "$Root\models\leaf_validator"
New-Dir "$Root\plots\literob"
New-Dir "$Root\plots\leaf_validator"
New-Dir "$Root\results\literob"
New-Dir "$Root\results\leaf_validator"
New-Dir "$Root\deployment\literob"
New-Dir "$Root\deployment\leaf_validator"
New-Dir "$Root\data"
New-Dir "$Root\setup"

# --- 2. LITEROB SCRIPTS ---
Write-Host ''
Write-Host '[ 2 ] Moving LiteRob pipeline scripts...' -ForegroundColor Cyan
Move-Safe "$Root\config.py"            "$Root\literob\config.py"
Move-Safe "$Root\stage1_data.py"       "$Root\literob\stage1_data.py"
Move-Safe "$Root\stage2_teacher.py"    "$Root\literob\stage2_teacher.py"
Move-Safe "$Root\stage3_baseline.py"   "$Root\literob\stage3_baseline.py"
Move-Safe "$Root\stage5_literob.py"    "$Root\literob\stage5_literob.py"
Move-Safe "$Root\stage5_tflite_fix.py" "$Root\literob\stage5_tflite_fix.py"
Move-Safe "$Root\stage6_evaluate.py"   "$Root\literob\stage6_evaluate.py"

# --- 3. LEAF VALIDATOR SCRIPTS ---
Write-Host ''
Write-Host '[ 3 ] Moving Leaf Validator scripts...' -ForegroundColor Cyan
Move-Safe "$Root\leaf_validator_config.py" "$Root\leaf_validator\leaf_validator_config.py"
Move-Safe "$Root\leaf_stage1_train.py"     "$Root\leaf_validator\leaf_stage1_train.py"
Move-Safe "$Root\leaf_stage2_export.py"    "$Root\leaf_validator\leaf_stage2_export.py"
Move-Safe "$Root\leaf_stage3_evaluate.py"  "$Root\leaf_validator\leaf_stage3_evaluate.py"
Move-Safe "$Root\run_leaf_pipeline.sh"     "$Root\leaf_validator\run_leaf_pipeline.sh"

# --- 4. MODELS ---
Write-Host ''
Write-Host '[ 4 ] Moving models...' -ForegroundColor Cyan
Move-Safe "$Root\models\kd_mobilenetv2.keras"                         "$Root\models\literob\kd_mobilenetv2.keras"
Move-Safe "$Root\models\kd_mobilenetv2_fp32.tflite"                   "$Root\models\literob\kd_mobilenetv2_fp32.tflite"
Move-Safe "$Root\models\kd_mobilenetv2_float16.tflite"                "$Root\models\literob\kd_mobilenetv2_float16.tflite"
Move-Safe "$Root\models\teacher_resnet50.keras"                       "$Root\models\literob\teacher_resnet50.keras"
Move-Safe "$Root\models\teacher_ckpt.keras"                           "$Root\models\literob\teacher_ckpt.keras"
Move-Safe "$Root\models\kd_mobilenetv2_leaf_validator.keras"          "$Root\models\leaf_validator\kd_mobilenetv2_leaf_validator.keras"
Move-Safe "$Root\models\kd_mobilenetv2_leaf_validator_fp32.tflite"    "$Root\models\leaf_validator\kd_mobilenetv2_leaf_validator_fp32.tflite"
Move-Safe "$Root\models\kd_mobilenetv2_leaf_validator_float16.tflite" "$Root\models\leaf_validator\kd_mobilenetv2_leaf_validator_float16.tflite"
Move-Safe "$Root\models\leaf_validator_ckpt.keras"                    "$Root\models\leaf_validator\leaf_validator_ckpt.keras"

# --- 5. PLOTS ---
Write-Host ''
Write-Host '[ 5 ] Moving plots...' -ForegroundColor Cyan
$literobPlots = @(
    'accuracy_comparison.png', 'accuracy_drop.png', 'class_performance_heatmap.png',
    'confusion_matrix_clean.png', 'confusion_matrix_combined.png',
    'error_analysis_heatmap.png', 'kd_training_accuracy_curve.png',
    'kd_training_loss_curve.png', 'kd_vs_baseline.png',
    'robustness_drop_curve.png', 'size_vs_accuracy_tradeoff.png'
)
foreach ($f in $literobPlots) {
    Move-Safe "$Root\plots\$f" "$Root\plots\literob\$f"
}
if (Test-Path "$Root\plots\leaf") {
    Get-ChildItem "$Root\plots\leaf\*" | ForEach-Object {
        Move-Safe $_.FullName "$Root\plots\leaf_validator\$($_.Name)"
    }
    Remove-Item "$Root\plots\leaf" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host '  Removed old plots\leaf folder' -ForegroundColor DarkGray
}

# --- 6. RESULTS ---
Write-Host ''
Write-Host '[ 6 ] Moving results...' -ForegroundColor Cyan
Move-Safe "$Root\results\accuracy_table.csv"           "$Root\results\literob\accuracy_table.csv"
Move-Safe "$Root\results\classification_metrics.csv"   "$Root\results\literob\classification_metrics.csv"
Move-Safe "$Root\results\tflite_evaluation_report.csv" "$Root\results\literob\tflite_evaluation_report.csv"
if (Test-Path "$Root\results\leaf") {
    Get-ChildItem "$Root\results\leaf\*" | ForEach-Object {
        Move-Safe $_.FullName "$Root\results\leaf_validator\$($_.Name)"
    }
    Remove-Item "$Root\results\leaf" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host '  Removed old results\leaf folder' -ForegroundColor DarkGray
}

# --- 7. DEPLOYMENT ---
Write-Host ''
Write-Host '[ 7 ] Moving deployment files...' -ForegroundColor Cyan
Move-Safe "$Root\deployment\inference.py"                "$Root\deployment\literob\inference.py"
Move-Safe "$Root\deployment\README.md"                   "$Root\deployment\literob\README.md"
Move-Safe "$Root\deployment\label_map.json"              "$Root\deployment\literob\label_map.json"
Move-Safe "$Root\deployment\inference_leaf_validator.py" "$Root\deployment\leaf_validator\inference_leaf_validator.py"
Move-Safe "$Root\deployment\README_leaf_validator.md"    "$Root\deployment\leaf_validator\README_leaf_validator.md"

# --- 8. DATASET ---
Write-Host ''
Write-Host '[ 8 ] Moving dataset...' -ForegroundColor Cyan
if (Test-Path "$Root\lead and not leaf") {
    Move-Safe "$Root\lead and not leaf" "$Root\data\lead and not leaf"
}

# --- 9. SETUP FILES ---
Write-Host ''
Write-Host '[ 9 ] Moving setup files...' -ForegroundColor Cyan
Move-Safe "$Root\cuda-keyring_1.1-1_all.deb" "$Root\setup\cuda-keyring_1.1-1_all.deb"

# --- 10. FINAL STRUCTURE ---
Write-Host ''
Write-Host '=== DONE - Final Structure ===' -ForegroundColor Green
Get-ChildItem $Root -Exclude '.git' | ForEach-Object {
    if ($_.PSIsContainer) {
        Write-Host "  [$($_.Name)/]" -ForegroundColor Yellow
        Get-ChildItem $_.FullName | ForEach-Object {
            if ($_.PSIsContainer) {
                Write-Host "    [$($_.Name)/]" -ForegroundColor DarkYellow
            } else {
                Write-Host "    $($_.Name)"
            }
        }
    } else {
        Write-Host "  $($_.Name)"
    }
}

Write-Host ''
Write-Host '  Done! Run leaf pipeline from: leaf_validator/' -ForegroundColor Cyan
Write-Host ''
