@echo off
cd /d "%~dp0"
start /min python -m http.server 8080
timeout /t 2 /nobreak >nul
start msedge --app=http://localhost:8080/daily-todo.html