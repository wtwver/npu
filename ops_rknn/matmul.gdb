set pagination off
set breakpoint pending on

break rknn_matmul_run
commands 
    printf "begin rknn_matmul_run============\n"
    printf "\n[matmul gem1 emit]\n"
    shell python3 dump.py 1 
    shell python3 dump.py 1 | grep EMIT | grep -v "0x00000000" | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_rknn_matmul_emit
    
    printf "\n[matmul gem2 task]\n"
    shell python3 dump.py 2 

    printf "\n[matmul gem3 weight]\n"
    shell python3 dump.py 3 | head 
    #shell python3 dump.py 3  > /tmp/ops_rknn_matmul_weight 
    
    printf "\n[matmul gem4 input]\n"
    shell python3 dump.py 4 | head 
    #shell python3 dump.py 4 > /tmp/ops_rknn_matmul_input  
    printf "rknn_matmul_run============\n"
    continue
end

break rknn_destroy_mem
commands
    printf "begin rknn_destroy_mem============\n"
    #shell python3 dump.py 3
    #shell python3 dump.py 4
    #shell python3 dump.py 5
    shell python3 dump.py 6
    printf "rknn_destroy_mem============\n"
    continue
end

run
q