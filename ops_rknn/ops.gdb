break rknn_run
break rknn_destroy
run

echo rknn_run============
# shell python3 dump.py 1
shell python3 dump.py 2 
# shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN"
# shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN" | wc -l
# shell python3 dump.py 2 | grep "PC_BASE_ADDRESS_PC_SOURCE_ADDR" 
# shell python3 dump.py 2 | grep "PC_BASE_ADDRESS_PC_SOURCE_ADDR" | wc -l
echo rknn_run============
c

echo rknn_destroy============
# shell python3 dump.py 3
# shell python3 dump.py 4
# shell python3 dump.py 5
# shell python3 dump.py 6
echo rknn_destroy============
c

q