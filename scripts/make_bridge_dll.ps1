# Creates src\llama_engine\bridge_dll\hipblas.dll as a copy of LM Studio's libhipblas.dll (your own LM Studio install).
# The official llama.cpp HIP release looks for "hipblas.dll" but LM Studio ships it as "libhipblas.dll". We never redistribute it.
$source = Get-ChildItem "$env:USERPROFILE\.lmstudio\extensions\backends\vendor\win-llama-rocm-vendor-*\bin\libhipblas.dll" | Select-Object -Last 1
if (-not $source) { "libhipblas.dll not found: install LM Studio with its ROCm runtime first."; exit 1 }
$target = Join-Path $PSScriptRoot "..\src\llama_engine\bridge_dll"
New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item $source.FullName (Join-Path $target "hipblas.dll") -Force
"Created $target\hipblas.dll from $($source.FullName)"
