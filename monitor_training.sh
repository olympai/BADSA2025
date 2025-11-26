#!/bin/bash
# Training Monitor Script

echo "====================================="
echo "Skin Cancer Training Monitor"
echo "====================================="
echo ""

# Check if training process is running
PROCESS=$(ps aux | grep "python train_model.py" | grep -v grep)

if [ -z "$PROCESS" ]; then
    echo "❌ Training process is NOT running"
    echo ""
    echo "Start training with:"
    echo "  python train_model.py > training.log 2>&1 &"
    exit 1
else
    echo "✅ Training process is running:"
    echo "$PROCESS" | awk '{print "   PID: " $2 " | CPU: " $3"% | Memory: " $4"% | Time: " $10}'
    echo ""
fi

# Check if training log exists
if [ -f "training.log" ]; then
    echo "📊 Latest training output:"
    echo "-------------------------------------"
    tail -n 30 training.log
else
    echo "⚠️  No training.log file found"
    echo ""
    echo "The training is running but not logging to a file."
    echo "To see live output, restart training with:"
    echo "  pkill -f 'python train_model.py'"
    echo "  python train_model.py > training.log 2>&1 &"
fi

echo ""
echo "-------------------------------------"
echo "Commands:"
echo "  Monitor live: tail -f training.log"
echo "  Stop training: pkill -f 'python train_model.py'"
echo "  Check models: ls -lh models/"
echo "====================================="
