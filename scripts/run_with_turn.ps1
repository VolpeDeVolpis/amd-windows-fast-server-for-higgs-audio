# Esegue UN lavoro pesante sulla scheda rispettando le simple rules: one heavy job at a time (lock file), memoria video libera,
# ComfyUI fermo, sorveglianza della memoria condivisa (se sale oltre la soglia il lavoro viene fermato). Rilascia sempre il turno.
# Uso: run_with_turn.ps1 -MinFreeGB 20 -Script <file.py> [-ScriptArgs "..."]
param([double]$MinFreeGB = 20, [string]$Script, [string]$ScriptArgs = "", [double]$MaxSharedRiseGB = 0.5)
$lock = if ($env:HIGGS_LOCK_FILE) { $env:HIGGS_LOCK_FILE } else { Join-Path $env:TEMP "higgs_heavy_job.lock" }
$python = if ($env:HIGGS_PYTHON) { $env:HIGGS_PYTHON } else { "python" }
$totalGB = if ($env:HIGGS_VRAM_GB) { [double]$env:HIGGS_VRAM_GB } else { 24 }  # total video memory of your card
function Dedicated { ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum / 1GB }
function Shared { ((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum / 1GB }
if (Test-Path $lock) { "TURNO OCCUPATO: $(Get-Content $lock)"; exit 2 }
try { $queue = (Invoke-WebRequest http://127.0.0.1:8188/queue -UseBasicParsing -TimeoutSec 3).Content } catch { $queue = "" }
if ($queue -match '"queue_running": \[\[') { "ComfyUI sta generando: non parto."; exit 2 }
$free = $totalGB - (Dedicated)
if ($free -lt $MinFreeGB) { "Memoria video libera {0:N1} GB, ne servono {1}: non parto." -f $free, $MinFreeGB; exit 2 }
Set-Content -Encoding utf8 $lock "higgs $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $Script"
$sharedStart = Shared; $peakDed = 0; $peakShared = $sharedStart
try {
    $env:MIOPEN_LOG_LEVEL = "1"
    $out = Join-Path $env:TEMP "higgs_job_out.txt"
    $proc = Start-Process -FilePath $python -ArgumentList "-u `"$Script`" $ScriptArgs" -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    while (-not $proc.HasExited) {
        Start-Sleep 2
        $d = Dedicated; $s = Shared
        if ($d -gt $peakDed) { $peakDed = $d }; if ($s -gt $peakShared) { $peakShared = $s }
        if ($s - $sharedStart -gt $MaxSharedRiseGB) { Stop-Process -Id $proc.Id -Force; "FERMATO: memoria condivisa salita di {0:N2} GB" -f ($s - $sharedStart); break }
    }
    Get-Content $out -ErrorAction SilentlyContinue
    Get-Content "$out.err" -ErrorAction SilentlyContinue | Select-String 'Error|Traceback|error:' | Select-Object -First 8
    "picco dedicata {0:N1} GB, condivisa {1:N2} -> {2:N2} GB" -f $peakDed, $sharedStart, $peakShared
} finally {
    Remove-Item -LiteralPath $lock -Confirm:$false -ErrorAction SilentlyContinue
}
