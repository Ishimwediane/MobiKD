$msiUrl = 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11+9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.msi'
$dest = 'C:\Users\HP VICTUS\AppData\Local\jdk17.msi'

Write-Host "Downloading JDK 17 MSI installer..."
$wc = New-Object System.Net.WebClient
$wc.DownloadFile($msiUrl, $dest)
Write-Host "Download complete: $dest"
Write-Host "File size: $((Get-Item $dest).Length) bytes"
