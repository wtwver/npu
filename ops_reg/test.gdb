set pagination off
set breakpoint pending on

break submitTask
commands
  shell python dump.py 1
  shell python dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_reg_emit

  printf "\n[gem2 (matmul weight)]\n"
  # Dump the full packed weight buffer (no truncation).
  shell python dump.py 2 | tee /tmp/ops_reg_weight

  printf "\n[gem3 (matmul input)]\n"
  shell python dump.py 3 | grep -E "\[00.*]" | tee /tmp/ops_reg_input

  printf "\n[gem4 (output before submit)]\n"
  shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output
  continue
end

break unpack_nc1hwc2_fp16
commands
  shell python dump.py 4
  continue
end

break should_print_matmul
commands
  printf "\n[gem4 (matmul output)]\n"
  shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output 
  continue
end

break breakpoint 
commands
  printf "\n[gem3 (input)]\n"
  shell python dump.py 3 | grep -E "\[00.*]" | tee /tmp/ops_reg_input

  printf "\n[gem4 (output)]\n"
  shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output
  continue
end

r
q
