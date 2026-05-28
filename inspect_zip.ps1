Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = 'C:\Users\HP VICTUS\Desktop\MobiKD\archive.zip'
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$entries = $zip.Entries

Write-Host '--- All unique class folders in train/ ---'
$trainEntries = $entries | Where-Object { $_.FullName -match '/train/' -and -not $_.FullName.EndsWith('/') }
$classes = $trainEntries | ForEach-Object {
    $parts = $_.FullName -split '/'
    # class folder is the part after 'train/'
    $trainIdx = [Array]::IndexOf($parts, 'train')
    if ($trainIdx -ge 0 -and $parts.Count -gt $trainIdx + 1) {
        $parts[$trainIdx + 1]
    }
} | Sort-Object -Unique

$classes | ForEach-Object { Write-Host "  $_" }

Write-Host ''
Write-Host '--- Potato classes with image counts ---'
$potatoEntries = $trainEntries | Where-Object { $_.FullName -match 'Potato' }
$potatoClasses = $potatoEntries | ForEach-Object {
    $parts = $_.FullName -split '/'
    $trainIdx = [Array]::IndexOf($parts, 'train')
    if ($trainIdx -ge 0 -and $parts.Count -gt $trainIdx + 1) {
        $parts[$trainIdx + 1]
    }
} | Group-Object | Sort-Object Name

$potatoClasses | ForEach-Object {
    Write-Host ('  ' + $_.Name + ' : ' + $_.Count + ' images')
}

$zip.Dispose()
