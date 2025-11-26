# Training Commands - Quick Reference

## Status Checks

```bash
# Quick status check
./monitor_training.sh

# Is training running?
ps aux | grep "python train_model.py" | grep -v grep

# Check saved models
ls -lh models/

# Model file sizes
du -sh models/*.h5

# Process uptime
ps -o etime= -p $(pgrep -f "python train_model.py")
```

## Live Monitoring

```bash
# Follow training log (if using log file)
tail -f training_recall.log

# Watch with auto-refresh every 2 seconds
watch -n 2 './monitor_training.sh'

# System resource usage
top -pid $(pgrep -f "python train_model.py")
```

## Control Training

```bash
# Stop training
pkill -f "python train_model.py"

# Start training (no logging)
python train_model.py &

# Start training with logging (recommended)
python train_model.py > training.log 2>&1 &

# Start and detach (keeps running after logout)
nohup python train_model.py > training.log 2>&1 &
```

## After Training

```bash
# View training history plot
open models/training_history.png

# Check model files
ls -lh models/

# Test the Streamlit app
streamlit run app.py
```

## Troubleshooting

```bash
# Kill all Python training processes
pkill -9 -f "python train_model.py"

# Check for errors in log
tail -n 50 training.log | grep -i error

# Check available disk space
df -h .

# Check memory usage
free -h  # Linux
vm_stat  # macOS
```

## Current Training Info

- **Dataset**: HAM10000 (~10,000 images)
- **Batch size**: 16
- **Initial epochs**: 15
- **Fine-tuning epochs**: 5
- **Total training time**: ~19 hours on CPU
- **Model saves to**: `models/best_model.h5`

## Tips

1. Use `nohup` if you want training to continue after closing terminal
2. Always check `models/` directory for saved checkpoints
3. The best model is automatically saved based on validation accuracy
4. Use `tail -f training.log` for live monitoring
5. Training can be stopped and resumed (model checkpoints are saved)
