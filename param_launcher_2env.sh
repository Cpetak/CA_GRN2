for residual in 0 1
do
  for rule_combo in 0 1 2
  do
    for season_len in 50 300
    do
    sbatch launch_2env $residual $rule_combo $season_len 
    sleep 0.1
  done
  done
done
