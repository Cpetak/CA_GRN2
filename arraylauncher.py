from pathlib import Path
from time import sleep
import datetime
import subprocess

USER = "cpetak"
SLEEP_TIME = 30

def get_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")  # 2023-04-15 14:30:45

def run_check_output_safely(command):
    try:
        output = subprocess.check_output(command, stderr=subprocess.PIPE, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error message: {e.stderr}")
        return None
    except Exception as e:
        print(f"Failed to execute command: {e}")
        return None

if __name__ == "__main__":

    rules=[22,18,154,126,122,70,54,94,30,62,110,90,150,102]
    init_conds=[69904,149796,4147842,1238860,1677032,1410667,2607162,1754028,286634,159689,2961036,153075,3430997,2634107]

    all_combos = len(rules)/2 * len(init_conds)
    print(all_combos)
    counter1=0
    counter2=0
    while all_combos > 0:
        first = rules[counter1]
        second = rules[counter1+1]

        #CMD = f"sbatch --array=1-{len(lines)} run_geva.sh guide_files/guide_file_{mychr}.txt 0.01 geva_results {input_vcf}"
        CMD = f"sbatch launcher_diff_rules {first} {second} {init_conds[counter2]} {init_conds[counter2]}"
        
        parts = CMD.split(" ")

        print(f"[{get_time()}] Processing {CMD}")
        print(f"[CMD] {CMD}")

        success = False
        wait = SLEEP_TIME
        while not success:
            output = run_check_output_safely(parts)
            if not output:
                print(
                    f"    [{get_time()}] Submission FAILED, sleeping for {wait} seconds"
                )
                sleep(wait)
                wait *= 2
            else:
                success = True
        print(f"    [{get_time()}] Submitted {CMD}")
        print()

        all_combos=all_combos-1
        if counter2 == len(init_conds)-1:
            counter1=counter1+2
            counter2=0
        else:
            counter2=counter2+1

    print(f"[{get_time()}] DONE")
