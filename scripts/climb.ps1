# Run the native BuildABot runner unattended to climb ladder with the
# starter deck, detached from any terminal, logging to tmp\live.
#
#   .\scripts\climb.ps1                start (10 hours, starter deck, Giant push)
#   .\scripts\climb.ps1 -Hours 4       shorter session
#   .\scripts\climb.ps1 -Matches 1     stop after one finished match
#   .\scripts\climb.ps1 -Strategy buildabot   upstream's per-card rules instead
#   .\scripts\climb.ps1 -Strategy ia          the policy trained in the simulator
#                                             (-Checkpoint tmp\rl\starter_long_best.pt)
#   .\scripts\climb.ps1 -Stop          stop it (the display is reset too)
#
# Watch it:  vision.cmd (window beside MuMu), or
#            Get-Content tmp\live\buildabot-native.log -Wait -Tail 20
# The control panel (panel.cmd) also shows it, as a bot started outside it.

param(
    [double]$Hours = 10,
    [int]$Matches = 0,
    [ValidateSet('beatdown', 'buildabot', 'pekka', 'pekka2', 'pekka2c', 'pekka_hier', 'ia', 'hybride', 'hybride_sorts')][string]$Strategy = 'beatdown',
    [string]$Checkpoint = 'tmp\rl\starter_long_best.pt',
    [ValidateSet('starter', 'default', 'pekka')][string]$Deck = 'starter',
    # Our YOLO detector trained on our own frames (tmp/yolo/runs/chosen.txt)
    # with the side classifier trained on our recordings: on 48 checked frames
    # of our arena (scripts/eval_truth.py, 2026-11-06) 75% of units exactly
    # right and 3% ghosts, against 57% and 20% for BuildABot's model, which
    # stays available with -Vision buildabot.
    [ValidateSet('buildabot', 'yolo', 'auto')][string]$Vision = 'yolo',
    [string]$Serial = $(if ($env:HASTYCR_SERIAL) { $env:HASTYCR_SERIAL } else { '127.0.0.1:16384' }),
    # Keep a battle frame a second with what the bot saw and played, for
    # growing the detector's dataset (scriptsutolabel.py).
    [switch]$Record,
    [switch]$Stop
)

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
$adb = 'C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe'

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like '*mumu_buildabot_native.py*' }
if ($Stop) {
    $running | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # A killed runner never reaches its `finally`, so the size is reset here.
    & $adb -s $Serial shell wm size reset | Out-Null
    Remove-Item (Join-Path $root 'tmp\live\stop_after_match.flag') -ErrorAction SilentlyContinue
    Write-Host "stopped $(@($running).Count) process(es); display reset"
    exit 0
}
if ($running) {
    Write-Host 'a native runner is already running; use -Stop first' -ForegroundColor Yellow
    exit 1
}

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:HASTYCR_SERIAL = $Serial
if (-not $env:HASTYCR_INSTANCE) { $env:HASTYCR_INSTANCE = 'Android Device' }
# Start-Process in PowerShell 5.1 does not quote list arguments, so the
# command line is built by hand: the adb path contains a space.
$argline = "-u `"$root\scripts\mumu_buildabot_native.py`" --adb `"$adb`" --serial $Serial " +
           "--deck $Deck --strategy $Strategy --vision $Vision --fast-capture --hours $Hours"
if ($Matches -gt 0) { $argline += " --max-matches $Matches" }
if ($Record) { $argline += " --record-dataset `"$(Join-Path $root 'tmp\dataset\live')`"" }
if ($Strategy -in @('ia', 'hybride', 'hybride_sorts')) { $argline += " --rl-checkpoint `"$(Join-Path $root $Checkpoint)`"" }
$logs = Join-Path $root 'tmp\logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$process = Start-Process -FilePath $python -ArgumentList $argline -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput "$logs\climb_stdout.log" `
    -RedirectStandardError "$logs\climb_stderr.log" -PassThru
Write-Host "climbing: pid $($process.Id), deck $Deck, strategy $Strategy, $Hours h - log tmp\live\buildabot-native.log"
