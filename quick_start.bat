@echo off
REM Kronos Quick Start
REM 适合已手动安装好依赖的用户（使用系统 Python，无需虚拟环境）
REM 首次使用请先运行 deploy.bat 完成完整部署

echo ================================================
echo Kronos Web UI - Quick Start
echo ================================================
echo.
echo 前提：已通过 deploy.bat 完成过一次完整部署
echo       或已在系统 Python 中手动安装全部依赖：
echo       pip install flask flask-cors pandas numpy plotly torch
echo            tushare tqsdk huggingface_hub einops safetensors
echo.

cd /d "%~dp0\webui"

echo 启动服务：http://localhost:7070
echo Hugging Face 镜像：hf-mirror.com
echo.
echo 按 Ctrl+C 停止服务
echo.

set HF_ENDPOINT=https://hf-mirror.com
python app.py

pause

