project('ops_reg', 'c')

incdir = include_directories('../include')
ops_reg_exe = executable('ops_reg', 'main.c',
  include_directories: incdir,
  c_args: ['-Wno-unused-variable', '-Wno-unused-function'],
  link_args: ['-ldrm', '-lm'],
)



matmul_cases = [
  ['32', '32', '32'],
  ['64', '64', '64'],
  ['256', '256', '256'],
  ['1', '32', '16'],
  ['1', '768', '768'],
  ['1', '768', '2048'],
  ['1', '2048', '2048'],
  ['33', '33', '33'],
  # ['1', '8192', '7744'],
]

foreach dims : matmul_cases
  test('matmul @0@x@1@x@2@'.format(dims[0], dims[1], dims[2]),
    ops_reg_exe,
    is_parallel: false,
    args: ['matmul', dims[0], dims[1], dims[2]])
endforeach

foreach n : range(1, 64)
  test('matmul 1x@0@x@0@'.format(n),
    ops_reg_exe,
    is_parallel: false,
    args: ['matmul', n.to_string(), n.to_string(), n.to_string()])
endforeach

# Run with: ninja -C build test_failfast
run_target('test_fail',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root()],
)
