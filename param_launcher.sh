for rule in {0..2} #{0..255}
do
  for seed in 69904 149796
  do
  sbatch launcher one $rule $seed
  sleep 0.1
done
done
