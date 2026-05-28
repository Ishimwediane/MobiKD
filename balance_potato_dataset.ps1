# balance_potato_dataset.ps1
# Trims not_potato_leaf down to TARGET images (randomly keeps a balanced sample)
# Run from MobiKD root: powershell -ExecutionPolicy Bypass -File balance_potato_dataset.ps1

$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dir    = Join-Path $Root 'data\potato\not_potato_leaf'
$TARGET = 2400

$files = Get-ChildItem $Dir -File
Write-Host ('Current not_potato_leaf count: ' + $files.Count) -ForegroundColor Cyan

if ($files.Count -le $TARGET) {
    Write-Host "Already at or below $TARGET images. Nothing to do." -ForegroundColor Green
    exit 0
}

$toDelete = $files | Get-Random -Count ($files.Count - $TARGET)
Write-Host ('Deleting ' + $toDelete.Count + ' excess images...') -ForegroundColor Yellow
$toDelete | Remove-Item -Force

$remaining = (Get-ChildItem $Dir -File).Count
Write-Host ('Done! not_potato_leaf now has: ' + $remaining + ' images') -ForegroundColor Green
Write-Host ''
Write-Host '=== Final Dataset Balance ===' -ForegroundColor Cyan
$classes = @('early_blight','healthy','late_blight','not_potato_leaf')
foreach ($cls in $classes) {
    $count = (Get-ChildItem (Join-Path $Root "data\potato\$cls") -File).Count
    Write-Host ("  $cls : $count") -ForegroundColor White
}
