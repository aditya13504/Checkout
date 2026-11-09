@echo off
rem Starts the Checkout live demo: the store for people (port 3000) and the agent viewer (port 3100).
rem Run demo\setup_demo.ps1 once before the first use.
title Checkout live demo
cd /d "%~dp0.."
if not exist data\venv\Scripts\python.exe (
  echo The demo is not set up yet. Run:  powershell -ExecutionPolicy Bypass -File demo\setup_demo.ps1
  pause
  exit /b 1
)
set PYTHONUTF8=1
set TOKENIZERS_PARALLELISM=false
set HF_HOME=%CD%\data\hf
set WEBSHOP_ITEMS_PATH=%CD%\data\prepared_test\items.json
set WEBSHOP_ATTR_PATH=%CD%\data\prepared_test\attrs.json
set WEBSHOP_SQLITE_INDEX=%CD%\data\prepared_test\webshop.sqlite
set WEBSHOP_HUMAN_PATH=%CD%\data\items_human_ins.json
set WEBSHOP_PRODUCT_STORE=%CD%\data\products.sqlite

start "Checkout - WebShop store on http://127.0.0.1:3000" data\venv\Scripts\python.exe -m web_agent_site.app --attrs
data\venv\Scripts\python.exe demo\live_demo.py
pause
