set pagination off
set confirm off

# --------------------------------------------------------------
# 1. Set breakpoints
# --------------------------------------------------------------
break submitTask
break pack_nc1hwc2_fp16

# --------------------------------------------------------------
# 2. Attach automatic commands to each breakpoint
# --------------------------------------------------------------

# Breakpoint 1: submitTask
commands 1
    shell python3 dump.py 1
    shell python3 dump.py 2
    shell python3 dump.py 3
    printf "==================================================\n"
    continue
end

# Breakpoint 2: pack_nc1hwc2_fp16
commands 2
    silent
    printf "==================================================\n"
    printf "before pack_nc1hwc2_fp16\n"
    printf "==================================================\n"
    shell python3 dump.py 4
    continue
end

# --------------------------------------------------------------
# 3. Run the program (and let it finish)
# --------------------------------------------------------------
run
q