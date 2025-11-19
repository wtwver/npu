# ------------------------------------------------------------------
# 1. Breakpoints
# ------------------------------------------------------------------
break rknn_run
# break rknn_destroy

# ------------------------------------------------------------------
# 2. Commands that run automatically when the breakpoint is hit
# ------------------------------------------------------------------
commands 1
    printf "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    printf "before task modify\n"
    printf "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    shell python3 dump.py 1
    printf "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    printf "after task modify\n"
    printf "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    shell python patch_task_buffer.py 1    
    shell python3 dump.py 1
    continue
end

commands 2
    printf "rknn_destroy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    shell python3 dump.py 3
    shell python3 dump.py 4
    shell python3 dump.py 5
    shell python3 dump.py 6
    printf "rknn_destroy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    continue
end

# ------------------------------------------------------------------
# 3. Start the program
# ------------------------------------------------------------------
run
