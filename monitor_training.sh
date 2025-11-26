#!/bin/bash
# Training Monitor Script

echo "====================================="
echo "Skin Cancer Training Monitor"
echo "====================================="
echo ""

# Check if training process is running (recall-optimized or standard)
PROCESS=$(ps aux | grep "python train_model" | grep -v grep)

if [ -z "$PROCESS" ]; then
    echo "❌ Training process is NOT running"
    echo ""
    echo "Start training with:"
    echo "  Standard:         python train_model.py > training.log 2>&1 &"
    echo "  Recall-optimized: python train_model_recall_optimized.py > training_recall.log 2>&1 &"
    exit 1
else
    echo "✅ Training process is running:"
    echo "$PROCESS" | awk '{print "   PID: " $2 " | CPU: " $3"% | Memory: " $4"% | Time: " $10}'

    # Detect which training is running
    if echo "$PROCESS" | grep -q "recall_optimized"; then
        echo "   Type: RECALL-OPTIMIZED Training"
    else
        echo "   Type: Standard Training"
    fi
    echo ""
fi

# Check for log files
if [ -f "training_recall.log" ]; then
    echo "📊 Latest training output (Recall-Optimized):"
    echo "-------------------------------------"
    tail -n 30 training_recall.log
elif [ -f "training.log" ]; then
    echo "📊 Latest training output (Standard):"
    echo "-------------------------------------"
    tail -n 30 training.log
else
    echo "⚠️  No training log file found"
    echo ""
    echo "The training is running but not logging to a file."
    echo "To see live output, restart training with:"
    echo "  pkill -f 'python train_model'"
    echo "  python train_model_recall_optimized.py > training_recall.log 2>&1 &"
fi

echo ""
echo "-------------------------------------"
echo "Commands:"
if [ -f "training_recall.log" ]; then
    echo "  Monitor live: tail -f training_recall.log"
else
    echo "  Monitor live: tail -f training.log"
fi
echo "  Stop training: pkill -f 'python train_model'"
echo "  Check models: ls -lh models/"
echo "  View recall log: cat models/training_recall_log.csv"
echo "====================================="
