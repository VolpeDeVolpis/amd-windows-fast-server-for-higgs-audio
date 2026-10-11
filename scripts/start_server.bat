@echo off
cd /d %~dp0..
set MIOPEN_LOG_LEVEL=1
rem HIGGS_PYTHON = python.exe of the environment with ROCm torch (default: python on PATH)
if "%HIGGS_PYTHON%"=="" set HIGGS_PYTHON=python
"%HIGGS_PYTHON%" -u src\server.py %*

