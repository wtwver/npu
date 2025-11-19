set pagination off
set breakpoint pending on

break submitTask
commands
  shell python dump.py 2
  continue
end

r
q