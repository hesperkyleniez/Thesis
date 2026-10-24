@echo off
REM ImprovedCQCC: start-from-audio development workflow; no held-out model selection.
REM Expects the original WAV sets, acoustic resources, and speaker split.
cd /d "%~dp0"
python improvedCQCC_prepare_dataset.py --stage all --workers 2 || exit /b 1
python improvedCQCC_run_degraded_test.py --stage all --workers 2 || exit /b 1
python improvedCQCC_prepare_dataset.py --stage verify || exit /b 1
python improvedCQCC_run_degraded_test.py --stage verify || exit /b 1
python improvedCQCC_train.py --model all || exit /b 1
python improvedCQCC_retrain.py --model all || exit /b 1
python improvedCQCC_cqcc_train.py --stage check || exit /b 1
python improvedCQCC_cqcc_train.py --stage cv || exit /b 1
python improvedCQCC_f0_train.py --stage cv || exit /b 1
python improvedCQCC_cqcc_f0_fusion.py --stage self-test || exit /b 1
python improvedCQCC_cqcc_f0_fusion.py --stage oof || exit /b 1
python improvedCQCC_cqcc_f0_fusion.py --stage fit || exit /b 1
python improvedCQCC_cqcc_retrain.py || exit /b 1
python improvedCQCC_f0_retrain.py || exit /b 1
echo.
echo DEVELOPMENT COMPLETE. Review all validation results before any held-out evaluation.
echo If approved, run: python improvedCQCC_evaluate.py --model all
echo Then: python improvedCQCC_cqcc_f0_fusion.py --stage evaluate
echo For latency: python improvedCQCC_latency_test.py --model CNN-GRU-CQCC-F0-F --stage benchmark
