# ------------------------------------------------------------------
# 1. Breakpoints
# ------------------------------------------------------------------
break rknn_run
break rknn_destroy

# ------------------------------------------------------------------
# 2. Commands that run automatically when the breakpoint is hit
# ------------------------------------------------------------------
commands 1
    printf "rknn_run============\n"
    shell python3 dump.py 1
    shell python3 dump.py 2 
    shell python3 dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_rknn_gem2 
    printf "rknn_run============\n"
    continue
end

commands 2
    printf "rknn_destroy============\n"
    shell python3 dump.py 3
    shell python3 dump.py 4
    shell python3 dump.py 5
    shell python3 dump.py 6
    printf "rknn_destroy============\n"
    continue
end

# ------------------------------------------------------------------
# 3. Start the program
# ------------------------------------------------------------------
run
q