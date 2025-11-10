set pagination off
set confirm off

break submitTask
break mem_destroy
run
shell python3 dump.py 1 
shell python3 dump.py 2 
# shell python3 dump.py 3 
c

echo ==================================================\n
echo before mem_destroy\n
echo ==================================================\n
shell python3 dump.py 4
c
q
