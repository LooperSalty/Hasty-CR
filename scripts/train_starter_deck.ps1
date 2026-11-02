# Train a policy for the starter deck in the simulator, unattended:
# behaviour clone from the scripted teacher, then PPO from that clone.
#
#   .\scripts\train_starter_deck.ps1                 1000 clone episodes, 2M PPO steps
#   .\scripts\train_starter_deck.ps1 -Steps 4000000  longer PPO
#
# Runs at BelowNormal priority (inherited by every worker) so a live bot and
# the emulator keep the CPU first. Progress: tmp\live\rl_train.log
# Results: tmp\rl\starter_clone.pt, tmp\rl\starter_ppo_best.pt
#
# This policy is for the simulator: the live bridge (scripts\brain\rl_policy.py)
# still reads a Hog 2.6 hand, so it cannot drive the real game yet.

param(
    [int]$Episodes = 1000,
    [int]$Steps = 2000000,
    [int]$Envs = 8,
    # Continue from a trained policy instead of cloning first, under a new name
    # so the policy it starts from is kept:
    #   -Init tmp\rl\starter_ppo_last.pt -Name starter_long -Steps 30000000
    [string]$Init = '',
    [string]$Name = 'starter_ppo',
    [int]$EvalEvery = 250000,
    # Past selves kept as opponents (self-play league); 0 = scripted opponents only.
    # Against a fixed opponent the evals saturate (24/24) while live stays ~50%.
    [int]$League = 0,
    [ValidateSet('beatdown', 'control', 'cycle', 'mix')][string]$Style = 'beatdown',
    # Weak-point training: the decks met live (data\live_decks.json, built by
    # scripts\build_live_decks.py) with stronger opponent units, as their
    # higher card levels make them live:
    #   -Opponent live -Boost 1.0-1.45 -BrainShare 0.25 -EvalOpponent 'live@1.33,meta@1.21'
    [ValidateSet('meta', 'live')][string]$Opponent = 'meta',
    [string]$Boost = '',
    [double]$BrainShare = 0.5,
    [string]$EvalOpponent = 'brain,meta',
    # Simulator card names, in the order that fixes each hand slot's meaning
    # (clone and PPO must share it). The P.E.K.K.A deck since 2026-10-31:
    #   -Deck 'knight,pekka,minions,arrows,inferno_tower,archer,musketeer,minipekka'
    [string]$Deck = 'knight,archer,minions,arrows,fireball,giant,musketeer,minipekka'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.venvs\buildabot\Scripts\python.exe'
# The order fixes the meaning of each hand slot; clone and PPO must share it.
$deck = $Deck
$env:PYTHONIOENCODING = 'utf-8'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
Set-Location $root

if (-not $Init) {
    & $python -m sim.clone --deck $deck --teacher scripted --teacher-style $Style `
        --episodes $Episodes --epochs 8 --name "$($Name)_clone"
    if ($LASTEXITCODE -ne 0) { throw "clone failed ($LASTEXITCODE)" }
    $Init = "tmp\rl\$($Name)_clone.pt"
}

$leagueArgs = @()
if ($League -gt 0) { $leagueArgs = @('--league', $League, '--scripted-share', 0.5) }
if ($Boost) { $leagueArgs += @('--opponent-boost', $Boost) }
& $python -m sim.train_ppo --name $Name --deck $deck `
    --init $Init --value-warmup 200000 --lr 1e-4 --target-kl 0.02 `
    --opponent $Opponent --scripted-alt $Opponent --brain-share $BrainShare `
    --eval-opponent $EvalOpponent `
    --envs $Envs --rollout 128 --steps $Steps --eval-every $EvalEvery --eval-episodes 24 `
    @leagueArgs
if ($LASTEXITCODE -ne 0) { throw "ppo failed ($LASTEXITCODE)" }
