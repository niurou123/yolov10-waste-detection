@echo off
chcp 65001 >nul
echo ============================================
echo  YOLOv10 垃圾分类检测系统
echo ============================================
echo.
echo [1/2] 启动 FastAPI 后端 (端口 8000)...
start "YOLOv10-Backend" cmd /c "cd /d %~dp0 && d:\app\Anaconda3\envs\opencv\python.exe -m uvicorn backend.server:app --host 0.0.0.0 --port 8080"
echo [2/2] 启动前端 (端口 3000)...
start "YOLOv10-Frontend" cmd /c "cd /d %~dp0 && python -m http.server 3000 --directory frontend"
echo.
echo ============================================
echo  后端 API:  http://localhost:8080/docs
echo  前端页面:  http://localhost:3000
echo ============================================
echo  按任意键关闭全部...
pause >nul
taskkill /FI "WINDOWTITLE eq YOLOv10*" /T 2>nul
