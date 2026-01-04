break rknn_matmul_run
#break rknn_destroy_mem

commands 1
    printf "begin rknn_matmul_run============\n"
    shell python3 dump.py 1 | grep EMIT | grep -v "0x00000000"
    shell python3 dump.py 1 | grep EMIT | grep -v "0x00000000" | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_rknn_matmul 
    shell python3 dump.py 2 

    printf "\n[matmul gem3 weight]\n"
    # shell python3 dump.py 3 | grep -E "\[00.*]" | tee /tmp/ops_rknn_weight | head
    
    printf "\n[matmul gem4 input]\n"
    # shell python3 dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_rknn_input | head
    printf "rknn_matmul_run============\n"
    continue
end

# commands 2
#     printf "begin rknn_destroy_mem============\n"
#     shell python3 dump.py 3
#     shell python3 dump.py 4
#     shell python3 dump.py 5
#     shell python3 dump.py 6
#     printf "rknn_destroy_mem============\n"
#     continue
# end

run
q