@echo off
REM ============================================================================
REM bus-monitor.cmd  --  SecuredChat bus monitor with af_nicole speech (PCLA)
REM
REM Watches the prometheus-relay room and SPEAKS inbound messages aloud via
REM Kokoro (af_nicole). Also prints [bus] text notifications to the console.
REM Ctrl-C to stop. Long-running; run it in its own terminal window.
REM
REM Pieces (see memory/native-voice-and-bus-monitor.md):
REM   - SecuredChat CLI watch --json   ->  emits inbound messages as JSON
REM   - bus_speak.py (voice venv)       ->  speaks "<sender> says: <summary>"
REM
REM Defaults below are overridable by setting the env vars before launch.
REM To hear ALL room traffic (not just messages addressed to you), remove
REM   --addressed-to-me --exclude-self  from the chat.py line.
REM ============================================================================

setlocal

if "%SECUREDCHAT_BUS%"==""      set "SECUREDCHAT_BUS=D:\FromGitHubEtc\securedchat-bus"
if "%SECUREDCHAT_ROOM%"==""     set "SECUREDCHAT_ROOM=prometheus-relay"
if "%SECUREDCHAT_IDENTITY%"=="" set "SECUREDCHAT_IDENTITY=windows-claude"
if "%BUS_VOICE%"==""            set "BUS_VOICE=af_nicole"

set "SC_CLI=D:\FromGitHubEtc\SecuredChat\cli\chat.py"
set "VENV_PY=D:\FromGitHubEtc\pc-native-voice-models\.venv\Scripts\python.exe"
set "BRIDGE=D:\FromGitHubEtc\pc-native-voice-models\integrations\bus_speak.py"

echo [bus-monitor] room=%SECUREDCHAT_ROOM% identity=%SECUREDCHAT_IDENTITY% voice=%BUS_VOICE%
echo [bus-monitor] watching... (Ctrl-C to stop)

python "%SC_CLI%" --bus "%SECUREDCHAT_BUS%" --room "%SECUREDCHAT_ROOM%" --identity "%SECUREDCHAT_IDENTITY%" ^
    watch --addressed-to-me --exclude-self --poll 30 --json 2>&1 ^
    | "%VENV_PY%" "%BRIDGE%"

endlocal
