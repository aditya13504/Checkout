# Checkout: gets, sets up and runs the whole project on Windows, end to end.
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/aditya13504/Checkout/main/run.ps1 | iex"
#
# Inside a clone, `powershell -ExecutionPolicy Bypass -File run.ps1` does the same. Finished steps
# are skipped, so later runs start in a few minutes. Set CHECKOUT_SKIP_TESTS=1 to skip the tests.
# The first run downloads about 7 GB and needs about 20 GB of free disk; 8 GB stays in data\.
& {  # a block, so that `irm | iex` leaves the caller's session as it was
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Repo = 'https://github.com/aditya13504/Checkout'
$ModelsUrl = "$Repo/releases/download/models-v1"
$CatalogueUrl = 'https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main'

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Yellow }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)." } }
function Fetch($url, $file, $sha256) {
    if (Test-Path $file) { return }
    New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
    curl.exe -fL --retry 5 --retry-delay 5 -C - -o "$file.part" $url; Check "Downloading $url"
    if ((Get-FileHash "$file.part" -Algorithm SHA256).Hash.ToLower() -ne $sha256) {
        Remove-Item "$file.part"; throw "$url arrived damaged; run the script again."
    }
    Move-Item "$file.part" $file
}

# ---------------------------------------------------------------- the code
Step 'Getting the code'
$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if (-not (Test-Path (Join-Path $root 'kaggle\pipeline.py'))) { $root = Join-Path (Get-Location).Path 'Checkout' }
$git = Get-Command git -ErrorAction SilentlyContinue
if (Test-Path (Join-Path $root 'kaggle\pipeline.py')) {
    if ($git -and (Test-Path (Join-Path $root '.git'))) {
        git -C $root pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Host 'Could not update the code (local changes?); using it as it is.' }
    }
} elseif ($git) {
    git clone "$Repo.git" $root; Check 'git clone'
} else {
    $zip = Join-Path $env:TEMP 'checkout-main.zip'
    curl.exe -fL -o $zip "$Repo/archive/refs/heads/main.zip"; Check 'Downloading the code'
    Expand-Archive $zip -DestinationPath (Split-Path $root) -Force
    Rename-Item (Join-Path (Split-Path $root) 'Checkout-main') $root
}
Set-Location $root
$data = Join-Path $root 'data'
$py = Join-Path $data 'venv\Scripts\python.exe'
New-Item -ItemType Directory -Force $data | Out-Null
$freeGB = [math]::Round((Get-PSDrive ((Get-Item $root).PSDrive.Name)).Free / 1GB)
if (-not (Test-Path (Join-Path $data 'products.sqlite')) -and $freeGB -lt 20) {
    throw "The first run needs about 20 GB of free disk; this drive has $freeGB GB."
}

# ---------------------------------------------------------------- Python
Step 'Python 3.10 environment'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}
$ready = Join-Path $data 'venv\.installed'
if (-not (Test-Path $ready)) {
    if (-not (Test-Path $py)) { uv venv --python 3.10 (Join-Path $data 'venv'); Check 'Creating the environment' }
    $req = Join-Path $data 'requirements_cpu.txt'
    Get-Content requirements.txt | Where-Object { $_ -notmatch '^torch' } | Set-Content -Encoding ascii $req
    uv pip install --python $py -r $req pypandoc_binary; Check 'Installing the requirements'
    uv pip install --python $py torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu; Check 'Installing PyTorch'
    uv pip install --python $py 'en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl'
    Check 'Installing the spaCy model'
    New-Item -ItemType File $ready | Out-Null
}

$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HOME = Join-Path $data 'hf'
$prepared = Join-Path $data 'prepared_test'
$index = Join-Path $prepared 'webshop.sqlite'
$store = Join-Path $data 'products.sqlite'
$env:WEBSHOP_ITEMS_PATH = Join-Path $prepared 'items.json'
$env:WEBSHOP_ATTR_PATH = Join-Path $prepared 'attrs.json'
$env:WEBSHOP_SQLITE_INDEX = $index
$env:WEBSHOP_HUMAN_PATH = Join-Path $data 'items_human_ins.json'
$env:WEBSHOP_PRODUCT_STORE = $store

# ---------------------------------------------------------------- the store
Step 'Product catalogue, search index and product store'
Copy-Item baseline_models\data\items_human_ins.json $env:WEBSHOP_HUMAN_PATH -Force
if (-not ((Test-Path $index) -and (Test-Path $store))) {
    Write-Host 'Downloading 5.7 GB of products, then indexing them (about 20 minutes).'
    Fetch "$CatalogueUrl/items_shuffle.json" (Join-Path $data 'items_shuffle.json') '2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8'
    Fetch "$CatalogueUrl/items_ins_v2.json" (Join-Path $data 'items_ins_v2.json') '1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e'
    Remove-Item $index, $store -ErrorAction SilentlyContinue
    & $py search_engine\prepare_catalog.py --raw_items (Join-Path $data 'items_shuffle.json') --raw_attrs (Join-Path $data 'items_ins_v2.json') --human $env:WEBSHOP_HUMAN_PATH --out_dir $prepared
    Check 'Building the search index'
    & $py -m web_agent_site.engine.product_store --items $env:WEBSHOP_ITEMS_PATH --attrs $env:WEBSHOP_ATTR_PATH --human $env:WEBSHOP_HUMAN_PATH --out $store
    Check 'Building the product store'
    # Only the index and the store are needed from here on.
    Remove-Item (Join-Path $data 'items_shuffle.json'), (Join-Path $data 'items_ins_v2.json'), $env:WEBSHOP_ITEMS_PATH, $env:WEBSHOP_ATTR_PATH -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- models
Step 'BERT and the trained agents'
& $py -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
Check 'Downloading BERT'
$runs = Join-Path $root 'kaggle_outputs\outputs_full\runs'
Fetch "$ModelsUrl/il_rl.pt" (Join-Path $runs 'il_rl\model_best.pt') 'abc18e43721eb30fa3cc8de4326894f873ed1ff037cde84487cf52d4b3a0471a'
Fetch "$ModelsUrl/rl_scratch.pt" (Join-Path $runs 'rl_scratch\model_best.pt') '5e23149bc671f991be376a429bae6de976e72152115a60be381017907f594705'
Fetch "$ModelsUrl/il_seed0.pth" (Join-Path $runs 'il_seed0\model_best.pth') '3d26ab737a5f77b8ec46fc99f7d980a40beed9e6e34e4382ea2e54c5c45f1648'

# ---------------------------------------------------------------- checks and rebuilds
if ($env:CHECKOUT_SKIP_TESTS -ne '1') {
    Step 'Tests'
    & $py -m pytest -q -p no:cacheprovider --disable-warnings tests; Check 'The tests'
}
Step 'Report, rebuilt from the run outputs'
& $py report\build_submission.py --out (Join-Path $data 'report'); Check 'Building the report'

# ---------------------------------------------------------------- run
Step 'Starting the store and the live demo'
# CHECKOUT_SHARE=1 (or SHARE_DEMO.bat) also gives it a free public link through a Cloudflare Quick Tunnel.
# The tunnel carries one address, so the store is then served through the demo's address too.
$share = $env:CHECKOUT_SHARE -eq '1'
$logs = Join-Path $data 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
function Open-Tunnel {
    $cf = Join-Path $data 'bin\cloudflared.exe'
    if (-not (Test-Path $cf)) {
        New-Item -ItemType Directory -Force (Split-Path $cf) | Out-Null
        curl.exe -fL -o $cf https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
        Check 'Downloading cloudflared'
    }
    $log = Join-Path $logs 'tunnel.log'
    Remove-Item $log -ErrorAction SilentlyContinue
    $tunnel = Start-Process $cf -ArgumentList 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:3100' -PassThru `
        -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput (Join-Path $logs 'tunnel.out.log')
    $deadline = (Get-Date).AddSeconds(90)
    do {
        Start-Sleep 2
        $found = if (Test-Path $log) { Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1 } else { $null }
    } until ($found -or $tunnel.HasExited -or (Get-Date) -gt $deadline)
    if (-not $found) { throw "The tunnel did not start; see $log." }
    $url = $found.Matches[0].Value
    Write-Host "`nPublic link, open while this window runs:  $url" -ForegroundColor Green
    Set-Content (Join-Path $logs 'public_url.txt') $url
    Start-Process $url
    return $tunnel
}
$running = $null
try { $running = Invoke-RestMethod http://127.0.0.1:3100/api/setup -TimeoutSec 3 } catch { }
if ($running) {
    if (-not $share) {
        Write-Host 'Checkout is already running; opening it.' -ForegroundColor Green
        Start-Process 'http://127.0.0.1:3100/'; Start-Process 'http://127.0.0.1:3000/new'
        return
    }
    if ($running.store -ne '') { throw 'Checkout is already running without sharing. Close it, then run this again.' }
    $tunnel = Open-Tunnel
    Write-Host 'Press Ctrl+C to stop sharing.'
    try { Wait-Process -Id $tunnel.Id } finally { if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force } }
    return
}
if ($share) {
    $env:WEBSHOP_DEMO_URL = '/'                             # the store links back to the viewer on the same address
    $env:CHECKOUT_STORE_INTERNAL = 'http://127.0.0.1:3000'  # and the demo server passes the store's pages through
}
$tunnel = $null
$storeProc = Start-Process $py -ArgumentList '-m', 'web_agent_site.app', '--attrs' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs 'store.log') -RedirectStandardError (Join-Path $logs 'store.err.log')
$demoProc = Start-Process $py -ArgumentList 'demo\live_demo.py', '--no_browser' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs 'demo.log') -RedirectStandardError (Join-Path $logs 'demo.err.log')
try {
    Write-Host 'Loading 1,181,430 products into both servers (1 to 3 minutes)...'
    $deadline = (Get-Date).AddMinutes(15)
    $warm = $null
    do {
        Start-Sleep 3
        if ($storeProc.HasExited -or $demoProc.HasExited) { throw "A server stopped; see the logs in $logs." }
        if (-not $warm) {
            # The store loads its products on the first page request, so it gets exactly one.
            try { Invoke-WebRequest http://127.0.0.1:3000/static/store.css -UseBasicParsing -TimeoutSec 3 | Out-Null
                  $warm = Start-Job { Invoke-WebRequest http://127.0.0.1:3000/new -UseBasicParsing -TimeoutSec 900 | Out-Null } } catch { }
        }
        try { $demo = (Invoke-RestMethod http://127.0.0.1:3100/api/setup -TimeoutSec 5).status } catch { $demo = '' }
    } until (($demo -eq 'ready' -and $warm) -or (Get-Date) -gt $deadline)
    if ($demo -ne 'ready') { throw "The demo did not start; see the logs in $logs." }
    if ($warm) { Wait-Job $warm -Timeout 900 | Out-Null; Remove-Job $warm -Force }
    $storeUrl = if ($share) { 'http://127.0.0.1:3100/new' } else { 'http://127.0.0.1:3000/new' }
    Start-Process 'http://127.0.0.1:3100/'
    Write-Host "`nCheckout is running:" -ForegroundColor Green
    Write-Host '  Live agent demo    http://127.0.0.1:3100/'
    Write-Host "  Store to try       $storeUrl"
    Write-Host '  Report             data\report\REPORT.md and REPORT.docx, rebuilt just now'
    if ($share) { $tunnel = Open-Tunnel } else { Start-Process $storeUrl }
    Write-Host "`nPress Ctrl+C to stop."
    Wait-Process -Id $demoProc.Id
} finally {
    foreach ($p in $storeProc, $demoProc, $tunnel) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
}
}
