#!/bin/bash
# Script to run ops.gdb silently first time to read symbols, then with output

# Check if the quiet init script exists, if not create it
if [ ! -f "/home/orangepi/npu/cpp/ops_init.gdb" ]; then
    cat > /home/orangepi/npu/cpp/ops_init.gdb << 'EOF'
# First run to read symbols (quiet mode)
set confirm off
run > /dev/null 2>&1
break ioctl
run > /dev/null 2>&1
c 70
quit
EOF
fi

# Run gdb with the init script to read symbols quietly
echo "Initializing symbols (no output expected)..."
gdb -x /home/orangepi/npu/cpp/ops_init.gdb --args ./ops_int32 mul 1x10 > /dev/null 2>&1

# Now run the main ops.gdb script with normal output
echo "Running main script with output..."
gdb -x /home/orangepi/npu/cpp/ops.gdb --args ./ops_int32 mul 1x10