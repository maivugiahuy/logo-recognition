@echo off
REM Copy inference files to a destination folder
REM Usage: export_inference_files.bat [destination]
REM Example: export_inference_files.bat D:\backup\logo-inference

set DEST=%1
if "%DEST%"=="" set DEST=inference_export

echo Exporting inference files to: %DEST%
mkdir "%DEST%" 2>nul
mkdir "%DEST%\checkpoints" 2>nul
mkdir "%DEST%\galleries" 2>nul
mkdir "%DEST%\detector" 2>nul

copy "checkpoints\vit_hn.pt"                                                    "%DEST%\checkpoints\vit_hn.pt"
copy "runs\detect\checkpoints\yolo26m_logo\weights\best.pt"                     "%DEST%\detector\best.pt"
copy "data\galleries\openlogodet3k.faiss"                                        "%DEST%\galleries\openlogodet3k.faiss"
copy "data\galleries\openlogodet3k_labels.json"                                  "%DEST%\galleries\openlogodet3k_labels.json"

echo.
echo Done. Files saved to: %DEST%
dir "%DEST%" /s /b
