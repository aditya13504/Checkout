# One-time setup for the live demo on Windows. Safe to run again: finished steps are skipped.
#   powershell -ExecutionPolicy Bypass -File demo\setup_demo.ps1
# Needs Python 3.10 (py launcher) and about 20 GB of free disk. Everything goes into data\.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$data = Join-Path $root 'data'
$py = Join-Path $data 'venv\Scripts\python.exe'
New-Item -ItemType Directory -Force $data | Out-Null

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Yellow }
function Check() { if ($LASTEXITCODE -ne 0) { throw "The last step failed (exit code $LASTEXITCODE)." } }

Step 'Python environment'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { py -3.10 -m pip install uv; Check }
if (-not (Test-Path $py)) {
    uv venv --python 3.10 -q (Join-Path $data 'venv'); Check
    Get-Content requirements.txt | Where-Object { $_ -notmatch '^torch' } | Set-Content -Encoding ascii (Join-Path $data 'requirements_cpu.txt')
    uv pip install -q --python $py -r (Join-Path $data 'requirements_cpu.txt'); Check
    uv pip install -q --python $py torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu; Check
    uv pip install -q --python $py 'en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl'; Check
}

Copy-Item baseline_models\data\items_human_ins.json (Join-Path $data 'items_human_ins.json') -Force
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $data 'hf'
$prepared = Join-Path $data 'prepared_test'
$store = Join-Path $data 'products.sqlite'
$index = Join-Path $prepared 'webshop.sqlite'
$env:WEBSHOP_ITEMS_PATH = Join-Path $prepared 'items.json'
$env:WEBSHOP_ATTR_PATH = Join-Path $prepared 'attrs.json'
$env:WEBSHOP_SQLITE_INDEX = $index
$env:WEBSHOP_HUMAN_PATH = Join-Path $data 'items_human_ins.json'

# The demo only needs the search index and the product store. The raw download and the prepared
# JSON are just the way to build them, so they are fetched when one is missing and deleted after.
if (-not ((Test-Path $index) -and (Test-Path $store))) {
    Step 'Product catalogue (5.7 GB download)'
    $hashes = @{
        'items_shuffle.json' = '2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8'
        'items_ins_v2.json'  = '1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e'
    }
    foreach ($name in $hashes.Keys) {
        $file = Join-Path $data $name
        if (Test-Path $file) { continue }
        curl.exe -fL --retry 5 --retry-delay 5 -C - -o "$file.part" "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/$name"; Check
        if ((Get-FileHash "$file.part" -Algorithm SHA256).Hash.ToLower() -ne $hashes[$name]) { throw "$name has the wrong hash; delete $file.part and run again." }
        Move-Item "$file.part" $file
    }

    Step 'Full catalogue and BM25 search index (about 15 minutes)'
    Remove-Item $index -ErrorAction SilentlyContinue
    & $py search_engine\prepare_catalog.py --raw_items (Join-Path $data 'items_shuffle.json') --raw_attrs (Join-Path $data 'items_ins_v2.json') --human (Join-Path $data 'items_human_ins.json') --out_dir $prepared; Check

    Step 'Product store on disk'
    Remove-Item $store -ErrorAction SilentlyContinue
    & $py -m web_agent_site.engine.product_store --items (Join-Path $prepared 'items.json') --attrs (Join-Path $prepared 'attrs.json') --human (Join-Path $data 'items_human_ins.json') --out $store; Check

    Step 'Removing the build files (about 8 GB)'
    foreach ($file in (Join-Path $data 'items_shuffle.json'), (Join-Path $data 'items_ins_v2.json'),
                      (Join-Path $prepared 'items.json'), (Join-Path $prepared 'attrs.json')) {
        Remove-Item $file -ErrorAction SilentlyContinue
    }
}

Step 'BERT tokenizer and weights'
& $py -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"; Check

Write-Host "`nSetup finished. Start the demo with demo\START_DEMO.bat" -ForegroundColor Green
