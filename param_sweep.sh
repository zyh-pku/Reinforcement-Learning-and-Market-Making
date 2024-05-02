# for idx in {3..9}; do
#     echo "Running with delta_idx = $idx"
#     nohup python param_sweep.py --delta_idx $idx >> log/log_$idx.txt 2>&1 & 
# done

idx=0
echo "Running with delta_idx = $idx"
nohup python param_sweep.py --delta_idx $idx --iter 100000 --v_error 0.1>> log/53_log_$idx.txt 2>&1 & 
