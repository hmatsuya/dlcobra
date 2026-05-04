@echo off
:: ============================================================
:: usi_ssh_proxy.bat
::
:: Connects ShogiHome (or any USI GUI) on Windows to a dlshogi
:: USI engine running on a Vast.ai server via SSH.
::
:: The script is registered as a USI engine in ShogiHome.
:: ShogiHome talks to this script via stdin/stdout; this script
:: forwards everything over SSH to /workspace/run_usi.sh on the
:: remote server.
::
:: Prerequisites:
::   - OpenSSH client installed (built-in on Windows 10/11)
::     or Git for Windows / PuTTY's plink.exe
::   - SSH key added to the Vast.ai instance (or password auth)
::
:: Configuration — edit the three variables below:
::   VAST_HOST   Public IP of the Vast.ai instance
::   VAST_PORT   SSH port shown in the Vast.ai dashboard
::   VAST_KEY    Path to your private key file (e.g. id_ed25519)
::
:: How to register in ShogiHome:
::   Engine > Add engine > select this .bat file
::   (ShogiHome will call it directly; no extra arguments needed)
:: ============================================================

:: ── User configuration ────────────────────────────────────────
set VAST_HOST=YOUR_VAST_IP
set VAST_PORT=YOUR_VAST_PORT
set VAST_KEY=%USERPROFILE%\.ssh\id_ed25519
:: ─────────────────────────────────────────────────────────────

:: Optional: override the remote command (default: run_usi.sh)
set REMOTE_CMD=/workspace/run_usi.sh

:: Validate that the user has filled in the settings
if "%VAST_HOST%"=="YOUR_VAST_IP" (
    echo ERROR: Edit usi_ssh_proxy.bat and set VAST_HOST, VAST_PORT, VAST_KEY >&2
    exit /b 1
)

:: Launch SSH in stdio-forwarding mode.
::   -T          disable pseudo-TTY allocation (pure pipe mode)
::   -o ...      suppress host-key prompts that would break the USI stream
::   -i          identity file (private key)
::   -p          port
ssh -T ^
    -o StrictHostKeyChecking=no ^
    -o UserKnownHostsFile=NUL ^
    -o ServerAliveInterval=30 ^
    -o ServerAliveCountMax=3 ^
    -i "%VAST_KEY%" ^
    -p %VAST_PORT% ^
    root@%VAST_HOST% ^
    "%REMOTE_CMD%"
