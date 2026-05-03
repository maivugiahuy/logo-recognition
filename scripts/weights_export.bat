@echo off
REM Copy inference files to a destination folder
REM Usage: weights_export.bat [destination]
REM Example: weights_export.bat D:\backup\logo-inference

set DEST=%1
if "%DEST%"=="" set DEST=weights_export

echo Exporting inference files to: %DEST%

mkdir "%DEST%\checkpoints" 2>nul
mkdir "%DEST%\data\galleries" 2>nul
mkdir "%DEST%\runs\detect\checkpoints\yolo26m_logo\weights" 2>nul

copy "checkpoints\vit_hn.pt"                                                "%DEST%\checkpoints\vit_hn.pt"
copy "runs\detect\checkpoints\yolo26m_logo\weights\best.pt"                 "%DEST%\runs\detect\checkpoints\yolo26m_logo\weights\best.pt"
copy "data\galleries\openlogodet3k.faiss"                                   "%DEST%\data\galleries\openlogodet3k.faiss"
copy "data\galleries\openlogodet3k_labels.json"                             "%DEST%\data\galleries\openlogodet3k_labels.json"

echo.
echo Done. Files saved to: %DEST%
dir "%DEST%" /s /b
