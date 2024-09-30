for rule in 122 70 118 #22 126 146
do
  for seed in 69904 149796
  do
  sbatch launcher one $rule $seed
  sleep 0.1
done
done
