for rule in 0 1 2
do
  for seed in 4147842 1238860 2481300 69904 149796 1024
  do
  sbatch launcher one $rule $seed
  sleep 0.1
done
done
