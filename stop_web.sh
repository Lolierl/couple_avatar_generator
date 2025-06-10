#!/bin/bash

echo "🛑 停止 CUPID AI情侣头像生成器"
echo "=========================="

echo "🧹 停止 Gradio 应用..."
pkill -f "python demo.py" && echo "✅ Gradio 应用已停止" || echo "ℹ️  未找到运行的 Gradio 应用"

echo "🧹 停止 ngrok 隧道..."
pkill -f "ngrok" && echo "✅ ngrok 隧道已停止" || echo "ℹ️  未找到运行的 ngrok 进程"

echo "🔍 检查端口 7860 使用情况..."
lsof -i :7860 2>/dev/null && echo "⚠️  端口 7860 仍被占用" || echo "✅ 端口 7860 已释放"

echo "🏁 停止完成！" 