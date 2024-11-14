for rule in 90 150 102 #30 50 62 110 #22 126 154 18 122 70 54 94 # 
do
  for ((i=0; i<10; i+=1)); #i<10 results in 5 pair of initial conditions, for static just change i+=2 to i+=1
  do
  sbatch launcher one $rule $i $((i+1))
  sleep 0.1
done
done
