[CmdletBinding()]
param(
    [string]$DistroName = "DeformTransport-Ubuntu-22.04",
    [string]$InstallRoot = "F:\WSL\DeformTransport-Ubuntu-22.04",
    [string]$DownloadRoot = "F:\WSL\downloads",
    [string]$SourceRepository = "E:\DeformTransport",
    [string]$LinuxUser = "a"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-SafePath {
    param([string]$Path, [string]$RequiredPrefix)

    $absolutePath = [IO.Path]::GetFullPath($Path)
    $absolutePrefix = [IO.Path]::GetFullPath($RequiredPrefix)
    if (-not $absolutePath.StartsWith($absolutePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside $absolutePrefix`: $absolutePath"
    }
    return $absolutePath
}

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Description)

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-IsAdministrator)) {
    throw "Run this script once from an Administrator PowerShell terminal."
}
if (-not (Test-Path -LiteralPath "F:\")) {
    throw "F: is not available. No system changes were made."
}
if ($LinuxUser -notmatch "^[a-z_][a-z0-9_-]*$") {
    throw "LinuxUser is not a valid Linux account name: $LinuxUser"
}
if (-not (Test-Path -LiteralPath $SourceRepository -PathType Container)) {
    throw "Source repository not found: $SourceRepository"
}

$InstallRoot = Assert-SafePath -Path $InstallRoot -RequiredPrefix "F:\WSL\"
$DownloadRoot = Assert-SafePath -Path $DownloadRoot -RequiredPrefix "F:\WSL\"

$restartRequired = $false
foreach ($featureName in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
    if ($feature.State -ne "Enabled") {
        Write-Host "Enabling Windows feature: $featureName"
        $result = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
        $restartRequired = $restartRequired -or $result.RestartNeeded
    }
}

if ($restartRequired) {
    Write-Host "Windows features are enabled. Restart Windows manually, then run this script again."
    exit 3010
}

Invoke-Checked -Description "WSL update" -Command {
    wsl.exe --update --web-download
}
Invoke-Checked -Description "WSL2 default selection" -Command {
    wsl.exe --set-default-version 2
}

$installedDistros = @(wsl.exe --list --quiet) | ForEach-Object {
    $_.Trim([char]0).Trim()
} | Where-Object { $_ }

if ($installedDistros -contains $DistroName) {
    Write-Host "Distribution already exists; leaving it unchanged: $DistroName"
    exit 0
}

if (Test-Path -LiteralPath $InstallRoot) {
    $existingItems = @(Get-ChildItem -LiteralPath $InstallRoot -Force)
    if ($existingItems.Count -gt 0) {
        throw "Install directory is not empty; refusing to overwrite it: $InstallRoot"
    }
} else {
    New-Item -ItemType Directory -Path $InstallRoot | Out-Null
}
New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null

$imageName = "ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz"
$imageUrl = "https://cloud-images.ubuntu.com/wsl/jammy/current/$imageName"
$sumsUrl = "https://cloud-images.ubuntu.com/wsl/jammy/current/SHA256SUMS"
$imagePath = Join-Path $DownloadRoot $imageName
$sumsPath = Join-Path $DownloadRoot "SHA256SUMS-jammy-wsl"

if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
    Write-Host "Downloading official Ubuntu 22.04 WSL rootfs (about 325 MB) to $imagePath"
    Invoke-WebRequest -Uri $imageUrl -OutFile $imagePath
}
Invoke-WebRequest -Uri $sumsUrl -OutFile $sumsPath

$sumLine = Get-Content -LiteralPath $sumsPath | Where-Object { $_ -match [regex]::Escape($imageName) }
if (-not $sumLine) {
    throw "Could not find $imageName in Ubuntu SHA256SUMS"
}
$expectedHash = ($sumLine -split "\s+")[0].ToLowerInvariant()
$actualHash = (Get-FileHash -LiteralPath $imagePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "Ubuntu rootfs SHA-256 mismatch. Refusing to import the image."
}

Invoke-Checked -Description "Ubuntu import" -Command {
    wsl.exe --import $DistroName $InstallRoot $imagePath --version 2
}

$sourceWslPath = "/mnt/" + $SourceRepository.Substring(0, 1).ToLowerInvariant() + "/" + (
    $SourceRepository.Substring(3) -replace "\\", "/"
)
$linuxSetup = @"
set -e
if ! id -u '$LinuxUser' >/dev/null 2>&1; then
  useradd -m -s /bin/bash '$LinuxUser'
fi
usermod -aG sudo '$LinuxUser'
printf '%s ALL=(ALL) NOPASSWD:ALL\n' '$LinuxUser' > /etc/sudoers.d/90-deformtransport
chmod 0440 /etc/sudoers.d/90-deformtransport
printf '[user]\ndefault=$LinuxUser\n' > /etc/wsl.conf
install -d -o '$LinuxUser' -g '$LinuxUser' '/home/$LinuxUser/DeformTransport'
if [ ! -e '/home/$LinuxUser/DeformTransport/.git' ]; then
  tar -C '$sourceWslPath' --exclude='./.venv' --exclude='./.venv-wsl' -cf - . | \
    tar -C '/home/$LinuxUser/DeformTransport' -xf -
  chown -R '${LinuxUser}:$LinuxUser' '/home/$LinuxUser/DeformTransport'
fi
install -d -o '$LinuxUser' -g '$LinuxUser' '/home/$LinuxUser/.cache/pip'
install -d -o '$LinuxUser' -g '$LinuxUser' '/home/$LinuxUser/.conda/pkgs'
install -d -o '$LinuxUser' -g '$LinuxUser' '/home/$LinuxUser/tmp'
"@

Invoke-Checked -Description "Ubuntu user and workspace setup" -Command {
    wsl.exe -d $DistroName -u root -- bash -lc $linuxSetup
}
Invoke-Checked -Description "Ubuntu shutdown" -Command {
    wsl.exe --terminate $DistroName
}

Write-Host "WSL2 Ubuntu 22.04 is installed at $InstallRoot"
Write-Host "Linux workspace: /home/$LinuxUser/DeformTransport"
Write-Host "Open it with: wsl.exe -d $DistroName --cd /home/$LinuxUser/DeformTransport"
