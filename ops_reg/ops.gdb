set pagination off
set confirm off
break submitTask
run
shell python3 dump.py 1 | head -n 40
shell python3 dump.py 2 | head -n 160
shell echo "============"
c
q
