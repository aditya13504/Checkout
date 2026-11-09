@echo off
rem Double-click to set up Checkout (the first time) and run it. The steps are in run.ps1.
title Checkout
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\run.ps1"
pause
