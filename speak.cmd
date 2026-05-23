@echo off
REM speak.cmd — launcher so you can run `speak "..."` from anywhere.
REM Put this dir on PATH (or copy this file to a dir already on PATH), then:
REM   speak "Hello from anywhere."
REM   speak --file notes.txt
REM   speak --list-voices
"%~dp0.venv\Scripts\python.exe" "%~dp0speak.py" %*
