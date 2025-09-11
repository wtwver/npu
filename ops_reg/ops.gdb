run 
break submitTask
run
shell python3 dump.py 1 
shell python3 dump.py 2
shell python3 dump.py 3 
shell python3 dump.py 4
shell python3 dump.py 5
shell python3 dump.py 1 | grep REG_DPU_EW_CFG
shell echo "============"
c
q