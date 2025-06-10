#!/bin/bash

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate control

echo "🚀 启动 CUPID AI情侣头像生成器网络版"
echo "=================================="
echo "🐍 使用Python环境: $(which python)"
echo "🐍 Python版本: $(python --version)"

# 检查ngrok是否存在
if [ ! -f "./ngrok" ]; then
    echo "❌ ngrok 文件不存在！"
    exit 1
fi

# 确保ngrok有执行权限
chmod +x ./ngrok

# 杀死可能存在的旧进程
echo "🧹 清理旧进程..."
pkill -f "python demo.py" 2>/dev/null || true
pkill -f "ngrok" 2>/dev/null || true

# 等待端口释放
sleep 2

echo "🔧 启动 Gradio 应用..."
# 后台启动gradio应用
nohup python demo.py > gradio.log 2>&1 &
GRADIO_PID=$!

echo "⏳ 等待应用启动（15秒）..."
sleep 15

# 检查应用是否启动成功
if ! ps -p $GRADIO_PID > /dev/null; then
    echo "❌ Gradio 应用启动失败！请检查 gradio.log"
    cat gradio.log
    exit 1
fi

echo "✅ Gradio 应用启动成功 (PID: $GRADIO_PID)"

echo "🌐 启动 ngrok 隧道..."
# 启动ngrok（前台运行，显示URL）
./ngrok http 7860

echo "🛑 应用已停止"
echo "清理进程..."
kill $GRADIO_PID 2>/dev/null || true 