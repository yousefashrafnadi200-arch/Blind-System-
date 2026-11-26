@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Try interpreters in order; use the first one that has the dependencies.
for %%P in (
  "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
  "%LOCALAPPDATA%\Python\bin\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%USERPROFILE%\anaconda3\python.exe"
  "python"
) do (
  "%%~P" -c "import cv2, onnxruntime, insightface" >nul 2>&1
  if !errorlevel! equ 0 (
    echo Using interpreter: %%~P
    "%%~P" main.py
    goto :done
  )
)

echo.
echo No Python interpreter with opencv + onnxruntime + insightface was found.
echo Install the dependencies with:
echo     python -m pip install -r requirements.txt
echo.

:done
pause
