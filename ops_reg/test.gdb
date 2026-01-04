set pagination off
set breakpoint pending on

break submitTask
commands
#   shell python dump.py 1
#   shell python dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_reg_emit

  #  printf "\n[gem2 (matmul weight)]\n"
  #  shell python dump.py 2 | tee /tmp/ops_reg_weight

   printf "\n[gem2 (matmul EMIT)]\n"
   shell python dump.py 2 | grep EMIT | grep -v "0x00000000" | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' | tee /tmp/ops_reg_emit

#   printf "\n[gem3 (matmul input)]\n"
#   shell python dump.py 3 | grep -E "\[00.*]" | tee /tmp/ops_reg_input

#   printf "\n[gem4 (output before submit)]\n"
#   shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output
  
#   printf "\n[gem6 (cmplt op2)]\n"
#   shell python dump.py 6 | tee /tmp/ops_reg_weight2
  continue
end

# break unpack_nc1hwc2_fp16
# commands
#   shell python dump.py 4
#   continue
# end

# break should_print_matmul
# commands
#   printf "\n[gem4 (matmul output)]\n"
#   shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output 
#   continue
# end

# break breakpoint 
# commands
#   printf "\n[gem3 (input)]\n"
#   shell python dump.py 3 | grep -E "\[00.*]" | tee /tmp/ops_reg_input

#   printf "\n[gem4 (output)]\n"
#   shell python dump.py 4 | grep -E "\[00.*]" | tee /tmp/ops_reg_output

#   printf "\n[gem8 (output)]\n"
#   shell python dump.py 8 | grep -E "\[00.*]" | tee /tmp/ops_reg_output3
#   continue
# end

r
q
