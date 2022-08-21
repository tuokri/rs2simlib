@echo off

set /a result=0

flake8 rs2simlib --count --select=E9,F63,F7,F82 --show-source --statistics
set /a result = result + %ERRORLEVEL%

flake8 rs2simlib --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
set /a result = result + %ERRORLEVEL%

if %ERRORLEVEL% NEQ 0 (exit 1) else (exit 0)
