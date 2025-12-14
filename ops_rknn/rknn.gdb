break rknn_run
commands 
    printf "rknn_run============\n"
    shell python3 dump.py 1
    shell python3 dump.py 2
    shell python3 dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' | grep -v "0x00000000" > /tmp/ops_rknn_gem2 
    printf "rknn_run============\n"
    continue
end

break rknn_destroy
commands 2
    printf "rknn_destroy============\n"
    shell python3 dump.py 3
    shell python3 dump.py 4
    shell python3 dump.py 5
    shell python3 dump.py 6
    printf "rknn_destroy============\n"
    continue
end

run
q