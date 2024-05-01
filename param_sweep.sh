for idx in {0..9}; do
    echo "Running with delta_idx = $idx"
    nohup python param_sweep.py --delta_idx $idx >> log/log_$idx.txt 2>&1 & 
done