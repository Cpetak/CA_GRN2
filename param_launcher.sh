for rule in 22
do
  for seed in 1024
  do
  sbatch launcher one $rule $seed
  sleep 0.1
done
done
