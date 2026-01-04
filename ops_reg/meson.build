project('ops_reg', 'c')

incdir = include_directories('../include')
ops_reg_exe = executable('ops_reg', 'main.c',
  include_directories: incdir,
  c_args: ['-Wno-unused-variable', '-Wno-unused-function'],
  link_args: ['-ldrm', '-lm'],
)

test('matmul 8x8x8', ops_reg_exe, is_parallel: false, args: ['matmul', '8', '8', '8'])
test('matmul 9x9x9', ops_reg_exe, is_parallel: false, args: ['matmul', '9', '9', '9'])
test('matmul 32x32x32', ops_reg_exe, is_parallel: false, args: ['matmul', '32', '32', '32'])
test('matmul 64x64x64', ops_reg_exe, is_parallel: false, args: ['matmul', '64', '64', '64'])
test('matmul 256x256x256', ops_reg_exe, is_parallel: false, args: ['matmul', '256', '256', '256'])

test('matmul 1x32x16', ops_reg_exe, is_parallel: false, args: ['matmul', '1', '32', '16'])
test('matmul 1x768x768', ops_reg_exe, is_parallel: false, args: ['matmul', '1', '768', '768'])
test('matmul 1x768x2048', ops_reg_exe, is_parallel: false, args: ['matmul', '1', '768', '2048'])
# test('matmul 1x8192x7744', ops_reg_exe, is_parallel: false, args: ['matmul', '1', '8192', '7744'])