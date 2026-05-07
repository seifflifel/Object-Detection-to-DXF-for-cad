@echo off
SETLOCAL

REM === Step 1: Launch Python installer with GUI ===
echo Launching Python installer. Complete the installation, then press any key to continue...
start "" "python-3.13.0-amd64.exe"

REM === Step 2: Wait for user to finish installing ===
pause

REM === Step 3: Check if Python is installed and available ===
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Python was not found in PATH. Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM === Step 4: Create virtual environment ===
python -m venv venv

REM === Step 5: Activate virtual environment ===
call venv\Scripts\activate.bat

REM === Step 6: Install requirements ===
pip install -r requirements.txt

REM === Step 7: Install MobileSam in editable mode ===
cd MobileSam
pip install -e .
cd ..

REM === Step 8: Copy TCL folder into venv ===
xcopy /E /I /Y "C:\CP files\tcl" "venv\tcl"

REM === Step 9: Set TCL_LIBRARY env var ===
set TCL_LIBRARY=C:\CP files\venv\tcl\tcl8.6

echo.
echo ✅ All steps completed successfully.
echo (TCL_LIBRARY set to: %TCL_LIBRARY%)
pause
