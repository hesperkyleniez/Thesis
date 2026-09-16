@echo off
cd /d D:\Thesis\code

echo ========================================
echo Waiting for train.py to finish...
echo ========================================
echo.

:WAIT
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -match 'train\.py' }; if ($p) { exit 0 } else { exit 1 }"

if not errorlevel 1 (
    echo train.py is still running...
    timeout /t 30 /nobreak >nul
    goto WAIT
)

echo.
echo ========================================
echo train.py has finished.
echo Starting retrain.py...
echo ========================================
echo.

python "D:\Thesis\code\retrain.py"

if errorlevel 1 (
    echo.
    echo retrain.py FAILED. Stopping.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting run_degraded_test.py...
echo ========================================
echo.

python "D:\Thesis\code\run_degraded_test.py"

if errorlevel 1 (
    echo.
    echo run_degraded_test.py FAILED. Stopping.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting evaluate.py...
echo ========================================
echo.

python "D:\Thesis\code\evaluate.py"

if errorlevel 1 (
    echo.
    echo evaluate.py FAILED. Stopping.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting latency_test.py...
echo ========================================
echo.

python "D:\Thesis\code\latency_test.py"

if errorlevel 1 (
    echo.
    echo latency_test.py FAILED. Stopping.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ALL STEPS COMPLETED SUCCESSFULLY
echo ========================================
pause