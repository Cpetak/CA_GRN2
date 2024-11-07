for rule in 22 126 154 18 122 70 54 94 30 50 62 110 90 150 102
do
  #for seed in 4147842 1238860
  #do
  sbatch launcher one $rule 4147842 #$seed
  sleep 0.1
#done
done
