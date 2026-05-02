#Requires -Version 5.1
<#
.SYNOPSIS
    USI engine proxy — forwards stdin/stdout to a dlshogi engine on Vast.ai via SSH.

.DESCRIPTION
    Register this script as a USI engine in ShogiHome (or any USI GUI).
    The GUI talks to this script via stdin/stdout; the script tunnels
    everything over SSH to /workspace/run_usi.sh on the remote server.

    Prerequisites:
      - OpenSSH client (built-in on Windows 10/11: Settings > Optional Features)
        OR Git for Windows (includes ssh.exe)
      - SSH key pair. Add the public key to the Vast.ai instance:
          vastai create instance ... --ssh
        then copy your public key via the Vast.ai web console or:
          ssh-copy-id -p <PORT> root@<HOST>

    How to register in ShogiHome:
      Engine > Add engine > select this .ps1 file
      (ShogiHome must be configured to allow PowerShell engines, or use the
       .bat wrapper below which calls this script.)

.NOTES
    Edit the "User configuration" section below before use.
#>

# ── User configuration ─────────────────────────────────────────────────────────
$VastHost   = "YOUR_VAST_IP"          # e.g. "123.45.67.89"
$VastPort   = "YOUR_VAST_PORT"        # e.g. "12345"  (shown in Vast.ai dashboard)
$KeyFile    = "$env:USERPROFILE\.ssh\id_rsa"   # path to your private key
$RemoteCmd  = "/workspace/run_usi.sh"          # command to run on the server
# ──────────────────────────────────────────────────────────────────────────────

# Validate configuration
if ($VastHost -eq "YOUR_VAST_IP" -or $VastPort -eq "YOUR_VAST_PORT") {
    [Console]::Error.WriteLine("ERROR: Edit usi_ssh_proxy.ps1 and set VastHost, VastPort, KeyFile")
    exit 1
}

if (-not (Test-Path $KeyFile)) {
    [Console]::Error.WriteLine("ERROR: SSH key not found: $KeyFile")
    exit 1
}

# Find ssh.exe — prefer the Windows built-in, fall back to Git for Windows
$SshExe = Get-Command "ssh.exe" -ErrorAction SilentlyContinue |
          Select-Object -ExpandProperty Source -First 1

if (-not $SshExe) {
    $GitSsh = "C:\Program Files\Git\usr\bin\ssh.exe"
    if (Test-Path $GitSsh) { $SshExe = $GitSsh }
}

if (-not $SshExe) {
    [Console]::Error.WriteLine("ERROR: ssh.exe not found. Install OpenSSH (Windows Optional Features) or Git for Windows.")
    exit 1
}

# Build SSH argument list
$SshArgs = @(
    "-T"                              # no pseudo-TTY — pure pipe mode
    "-o", "StrictHostKeyChecking=no"  # skip host-key prompt (would break USI stream)
    "-o", "UserKnownHostsFile=NUL"    # don't write to known_hosts
    "-o", "ServerAliveInterval=30"    # keep-alive every 30 s
    "-o", "ServerAliveCountMax=3"     # drop after 3 missed keep-alives
    "-i", $KeyFile
    "-p", $VastPort
    "root@$VastHost"
    $RemoteCmd
)

# Start SSH and inherit stdin/stdout/stderr from this process.
# The USI GUI writes to our stdin → SSH reads it → remote engine receives it.
# Remote engine writes to stdout → SSH forwards it → USI GUI reads it.
$proc = Start-Process -FilePath $SshExe `
                      -ArgumentList $SshArgs `
                      -NoNewWindow `
                      -PassThru

$proc.WaitForExit()
exit $proc.ExitCode
