# Prints dedicated and shared video memory in use (GB). Works for any adapter, no nvidia-smi needed.
function Sum($counter) { ((Get-Counter $counter).CounterSamples | Measure-Object CookedValue -Sum).Sum / 1GB }
"Dedicated used: {0:N2} GB" -f (Sum '\GPU Adapter Memory(*)\Dedicated Usage')
"Shared used:    {0:N2} GB (should stay near its idle value; a rise means spill into system RAM)" -f (Sum '\GPU Adapter Memory(*)\Shared Usage')
