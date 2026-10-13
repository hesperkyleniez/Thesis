@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 exit /b 1

if not exist "train.py" (
    echo ERROR: Put this BAT file inside D:\Thesis\code beside the Python files.
    pause
    exit /b 1
)
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not on PATH.
    pause
    exit /b 1
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg is not on PATH. Install it before running this pipeline.
    pause
    exit /b 1
)

rem Reuse existing held-out features; create them when no cache is present.
set "HAVE_CLEAN="
set "HAVE_DEGRADED="
for %%F in ("D:\Thesis\features\test\*.npz") do if exist "%%~fF" set "HAVE_CLEAN=1"
for %%F in ("D:\Thesis\features\test_degraded\*.npz") do if exist "%%~fF" set "HAVE_DEGRADED=1"
if defined HAVE_CLEAN (
    call :run python prepare_dataset.py --train-only
) else (
    call :run python prepare_dataset.py
)
if errorlevel 1 goto :failed
if not defined HAVE_DEGRADED (
    call :run python run_degraded_test.py
    if errorlevel 1 goto :failed
)

call :run python train.py --model all
if errorlevel 1 goto :failed
call :run python retrain.py --model all
if errorlevel 1 goto :failed
call :run python evaluate.py --model all
if errorlevel 1 goto :failed
call :run python cqcc_train.py
if errorlevel 1 goto :failed
call :run python cqcc_retrain.py
if errorlevel 1 goto :failed
call :run python f0_train.py
if errorlevel 1 goto :failed
call :run python f0_retrain.py
if errorlevel 1 goto :failed
call :run python cqcc_f0_fusion.py --rebuild-cache
if errorlevel 1 goto :failed
call :run python latency_test.py --model CNN-only
if errorlevel 1 goto :failed
call :run python latency_test.py --model CNN-GRU
if errorlevel 1 goto :failed
call :run python latency_test.py --model CNN-GRU-F
if errorlevel 1 goto :failed
call :run python latency_test.py --model CNN-GRU-CQCC-F0-F
if errorlevel 1 goto :failed

echo.
echo COMPLETE: All four models finished. Outputs are under D:\Thesis\models and D:\Thesis\results.
pause
exit /b 0

:run
set "CURRENT_STEP=%*"
echo.
echo Running: %CURRENT_STEP%
%*
exit /b %errorlevel%

:failed
echo.
echo ERROR: Pipeline stopped at: %CURRENT_STEP%
echo Fix the error shown above before rerunning.
pause
exit /b 1
