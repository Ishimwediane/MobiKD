# extract_potato.ps1
# Extracts potato images + balanced sample of other plant leaves from archive.zip
# Output structure:
#   data/potato/early_blight/     <- Potato early blight
#   data/potato/healthy/          <- Potato healthy
#   data/potato/late_blight/      <- Potato late blight
#   data/potato/not_potato_leaf/  <- Other plant leaves (balanced sample)
#
# Run from MobiKD root: powershell -ExecutionPolicy Bypass -File extract_potato.ps1

Add-Type -AssemblyName System.IO.Compression.FileSystem

$Root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$ZipPath = Join-Path $Root 'archive.zip'
$OutBase = Join-Path $Root 'data\potato'

# Target: extract ALL non-potato images (no cap)
# Training uses class weights to handle the imbalance automatically.
$TARGET_NOT_POTATO = 999999999

$PotatoMap = @{
    'Potato___Early_blight' = 'early_blight'
    'Potato___Late_blight'  = 'late_blight'
    'Potato___healthy'      = 'healthy'
}

# All non-potato classes will contribute to not_potato_leaf
$NonPotatoKeywords = @('Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape',
                       'Orange', 'Peach', 'Pepper', 'Raspberry', 'Soybean',
                       'Squash', 'Strawberry', 'Tomato')

# Create output folders
$AllClasses = @('early_blight', 'healthy', 'late_blight', 'not_potato_leaf')
foreach ($cls in $AllClasses) {
    $dir = Join-Path $OutBase $cls
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host 'Opening ZIP...' -ForegroundColor Cyan
$zip     = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
$entries = $zip.Entries
Write-Host ('Total ZIP entries: ' + $entries.Count)
Write-Host ''

# Separate entries into potato and non-potato
Write-Host 'Scanning entries...' -ForegroundColor Cyan
$potatoEntries    = [System.Collections.Generic.List[object]]::new()
$nonPotatoEntries = [System.Collections.Generic.List[object]]::new()

foreach ($entry in $entries) {
    $name = $entry.FullName
    if ($name.EndsWith('/'))          { continue }
    if ($name -notmatch '/train/')    { continue }
    $ext = [System.IO.Path]::GetExtension($name).ToLower()
    if ($ext -notmatch '\.(jpg|jpeg|png|bmp)') { continue }

    if ($name -match 'Potato___') {
        $potatoEntries.Add($entry) | Out-Null
    } else {
        foreach ($kw in $NonPotatoKeywords) {
            if ($name -match $kw) {
                $nonPotatoEntries.Add($entry) | Out-Null
                break
            }
        }
    }
}

Write-Host ('  Potato entries  : ' + $potatoEntries.Count)
Write-Host ('  Non-potato entries: ' + $nonPotatoEntries.Count)
Write-Host ''

# Helper: extract a single entry to destination path
function Extract-Entry {
    param($Entry, $DestPath)
    if (Test-Path $DestPath) { return $false }
    $stream  = $Entry.Open()
    $outFile = [System.IO.File]::Create($DestPath)
    $stream.CopyTo($outFile)
    $outFile.Close()
    $stream.Close()
    return $true
}

# ── Extract Potato classes ────────────────────────────────────────────────────
Write-Host '[ 1 ] Extracting Potato class images...' -ForegroundColor Cyan
$potatoCounts = @{ 'early_blight' = 0; 'healthy' = 0; 'late_blight' = 0 }

foreach ($entry in $potatoEntries) {
    $name = $entry.FullName
    $matchedClass = $null
    foreach ($key in $PotatoMap.Keys) {
        if ($name -match [regex]::Escape($key)) {
            $matchedClass = $PotatoMap[$key]
            break
        }
    }
    if ($null -eq $matchedClass) { continue }

    $fileName = [System.IO.Path]::GetFileName($name)
    $destPath = Join-Path $OutBase "$matchedClass\$fileName"
    $extracted = Extract-Entry -Entry $entry -DestPath $destPath
    if ($extracted) { $potatoCounts[$matchedClass]++ }
}

foreach ($cls in $potatoCounts.Keys | Sort-Object) {
    Write-Host ('  ' + $cls + ' : ' + $potatoCounts[$cls] + ' extracted') -ForegroundColor White
}

# ── Extract Not-Potato leaves (balanced sample) ───────────────────────────────
Write-Host ''
Write-Host "[ 2 ] Extracting not_potato_leaf sample (target: $TARGET_NOT_POTATO images)..." -ForegroundColor Cyan

# Shuffle non-potato entries randomly for balanced sampling
$shuffled = $nonPotatoEntries | Get-Random -Count ([Math]::Min($TARGET_NOT_POTATO * 3, $nonPotatoEntries.Count))
$notPotatoCount = 0

foreach ($entry in $shuffled) {
    if ($notPotatoCount -ge $TARGET_NOT_POTATO) { break }
    $fileName = [System.IO.Path]::GetFileName($entry.FullName)
    $destPath = Join-Path $OutBase "not_potato_leaf\$fileName"
    $extracted = Extract-Entry -Entry $entry -DestPath $destPath
    if ($extracted) { $notPotatoCount++ }
}

Write-Host ('  not_potato_leaf : ' + $notPotatoCount + ' extracted') -ForegroundColor White

$zip.Dispose()

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ''
Write-Host '=== Extraction Complete ===' -ForegroundColor Green
Write-Host ''
foreach ($cls in $AllClasses | Sort-Object) {
    $dir   = Join-Path $OutBase $cls
    $count = (Get-ChildItem $dir -File).Count
    Write-Host ("  $cls : $count images total") -ForegroundColor White
}
Write-Host ''
Write-Host 'Output: data\potato\' -ForegroundColor Cyan
Write-Host 'Next  : copy to WSL and run potato_stage1_train.py' -ForegroundColor Cyan
