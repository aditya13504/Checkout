@echo off
rem Double-click to run Checkout and share it through a free public link (a Cloudflare Quick Tunnel).
rem The link works while this window stays open and the computer is on. See run.ps1.
title Checkout (shared)
set CHECKOUT_SHARE=1
set CHECKOUT_SKIP_TESTS=1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\run.ps1"
pause
