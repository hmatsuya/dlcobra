@echo off
:: Thin wrapper so ShogiHome can launch the PowerShell proxy script.
:: Register THIS .bat file as the engine in ShogiHome.
:: The actual configuration (IP, port, key) lives in usi_ssh_proxy.ps1
:: in the same folder as this file.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0usi_ssh_proxy.ps1"
