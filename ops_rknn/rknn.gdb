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
    # (the two commented-out greps are kept as examples)
    # shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN"
    # shell python3 dump.py 2 | grep "PC_OPERATION_ENABLE_OP_EN" | wc -l
    printf "rknn_run============\n"
    continue
end

commands 2
    #silent
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