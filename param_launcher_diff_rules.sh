#!/bin/bash

rules=(22 18 154 126 122 70 54 94 30 62 110 90 150 102)
init_conds=(69904 149796 4147842 1238860 1677032 1410667 2607162 1754028 286634 159689 2961036 153075 3430997 2634107)

counter=0
for ((i=0; i<${#rules[@]}; i+=2)); do
    first=${rules[i]}
    second=${rules[i+1]}
    for ((j=0; j<${#init_conds[@]}; j+=1)); do
        ((counter++))
        echo "Pair: $first and $second with init cond ${init_conds[j]}"
        sbatch launcher_diff_rules $first $second $init_conds[j] $init_conds[j]
        sleep 0.1
    done
    
done
echo $counter