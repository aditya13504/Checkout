# Publishes Checkout to a Hugging Face Space. Run it from anywhere in the project:
#
#   powershell -ExecutionPolicy Bypass -File deploy\huggingface\push_space.ps1 -Space <user>/Checkout
#
# Before the first run:
#   1. Create the Space at https://huggingface.co/new-space: name Checkout, SDK Docker, hardware CPU basic.
#   2. Create a token with write access at https://huggingface.co/settings/tokens.
# When git asks for a password, give that token (and your Hugging Face user name as the user name).
#
# It pushes what is committed (git HEAD), with this folder's README.md and Dockerfile at the Space's root,
# and leaves out the files the site does not need (the report and the demonstration archive).
param([Parameter(Mandatory = $true)][string]$Space)
$ErrorActionPreference = 'Stop'
$root = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
$copy = Join-Path $env:TEMP 'checkout-space'
if (Test-Path $copy) { Remove-Item $copy -Recurse -Force }
New-Item -ItemType Directory $copy | Out-Null

git -C $root -c core.autocrlf=false archive --format=tar -o "$copy.tar" HEAD  # Linux line endings, as stored
tar -xf "$copy.tar" -C $copy
Remove-Item "$copy.tar"
Copy-Item (Join-Path $PSScriptRoot 'README.md') (Join-Path $copy 'README.md') -Force
Copy-Item (Join-Path $PSScriptRoot 'Dockerfile') (Join-Path $copy 'Dockerfile') -Force
foreach ($skip in 'submission', 'baseline_models\data\il_trajs_finalized_images.zip') {
    $path = Join-Path $copy $skip
    if (Test-Path $path) { Remove-Item $path -Recurse -Force }
}

Push-Location $copy
try {
    git init -q -b main
    git lfs install --local | Out-Null
    git lfs track '*.png' | Out-Null  # Hugging Face takes binary files only through LFS
    git add -A
    git commit -q -m 'Checkout on a Hugging Face Space'
    git push --force "https://huggingface.co/spaces/$Space" main
    if ($LASTEXITCODE -ne 0) { throw 'The push failed. Check that the Space exists and that the token can write.' }
    Write-Host "`nPushed. The Space builds the image, about 30 minutes the first time, then it is at:" -ForegroundColor Green
    Write-Host "  https://huggingface.co/spaces/$Space"
} finally {
    Pop-Location
}
