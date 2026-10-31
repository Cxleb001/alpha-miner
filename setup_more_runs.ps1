# Longer runs, Nemotron as the backup idea writer, and a third daily run.
# Run from your alpha-miner folder:  powershell -ExecutionPolicy Bypass -File .\setup_more_runs.ps1
Set-Location $PSScriptRoot

function Set-EnvValue([string]$Name, [string]$Value) {
    $lines = @()
    if (Test-Path .env) { $lines = @(Get-Content .env) }
    if ($lines | Where-Object { $_ -match "^$Name=" }) {
        $lines = @($lines | ForEach-Object { if ($_ -match "^$Name=") { "$Name=$Value" } else { $_ } })
    } else {
        $lines += "$Name=$Value"
    }
    Set-Content -Path .env -Value $lines
}

# 1. Longer runs: 15 rounds of ideas instead of 5
Set-EnvValue "MAX_ROUNDS" "15"
"1. MAX_ROUNDS is now 15 in .env - each run tests about three times as many ideas."

# 2. Nemotron, asked only when Gemini can't write a round's ideas
$secure = Read-Host "2. Paste your NVIDIA API key from build.nvidia.com (it starts with nvapi-; typing is hidden), or press Enter to skip" -AsSecureString
$key = (New-Object System.Net.NetworkCredential("", $secure)).Password.Trim()
if ($key) {
    Set-EnvValue "NVIDIA_API_KEY" $key
    "   Saved in .env. Nemotron only takes over when Gemini fails."
} else {
    "   Skipped. Run this script again once you have a key."
}

# 3. A third daily run at 14:30, copying your main Alpha Miner task
if (-not (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue)) {
    "3. Scheduled tasks need Windows - skipped."
} else {
    $tasks = @(Get-ScheduledTask -TaskName "Alpha Miner*" -ErrorAction SilentlyContinue)
    $main = $tasks | Where-Object { $_.TaskName -ne "Alpha Miner Afternoon" -and $_.Actions[0].Arguments -notmatch "--backup|--catch-up|--try" } | Select-Object -First 1
    if ($tasks.TaskName -contains "Alpha Miner Afternoon") {
        "3. 'Alpha Miner Afternoon' already exists - left as it is."
    } elseif (-not $main) {
        "3. Found no Alpha Miner task to copy, so no afternoon run was added. Send Claude the output of: Get-ScheduledTask -TaskName 'Alpha Miner*'"
    } else {
        $trigger = New-ScheduledTaskTrigger -Daily -At "14:30"
        try {
            Register-ScheduledTask -TaskName "Alpha Miner Afternoon" -Action $main.Actions -Trigger $trigger -Settings $main.Settings -Principal $main.Principal -ErrorAction Stop | Out-Null
        } catch {
            Register-ScheduledTask -TaskName "Alpha Miner Afternoon" -Action $main.Actions -Trigger $trigger -Settings $main.Settings -ErrorAction Stop | Out-Null
        }
        "3. Added 'Alpha Miner Afternoon': daily at 14:30, the same command as '$($main.TaskName)'."
    }
    ""
    "Your Alpha Miner runs:"
    Get-ScheduledTask -TaskName "Alpha Miner*" | ForEach-Object {
        "   {0,-28} next run {1}" -f $_.TaskName, ($_ | Get-ScheduledTaskInfo).NextRunTime
    }
}
