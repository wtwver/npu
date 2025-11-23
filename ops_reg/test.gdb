set pagination off
set breakpoint pending on

break submitTask
commands
  shell python dump.py 1
  shell python dump.py 2 
  shell python dump.py 2 | grep EMIT | sed 's/\x1B\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/ops_reg_gem2 
  shell python dump.py 3
  continue
end

break unpack_nc1hwc2_fp16
commands
  shell python dump.py 4
  shell python dump.py 5
  shell python dump.py 6
  continue
end

r
q