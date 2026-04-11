@echo off
REM Run Claude Code from the latent-space-cartography project root.
REM Usage: run_claude.bat [args passed through to claude]
cd /d "%~dp0"
claude %*
