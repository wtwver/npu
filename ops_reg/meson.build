project('ops_reg', 'c')

incdir = include_directories('../include')
ops_reg_exe = executable('ops_reg', 'main.c',
  include_directories: incdir,
  c_args: ['-Wno-unused-variable', '-Wno-unused-function'],
  link_args: ['-ldrm', '-lm'],
)

matmul_cases = [
  ['1', '8192', '8192'], 
  ['1', '8192', '8193'], 
  ['1', '8193', '8192'], 
  ['1', '8193', '8193'], # failed
  ['32', '32', '32'],
  ['64', '64', '64'],
  ['128', '128', '128'],
  ['256', '256', '256'],
  # ['394', '394', '394'], # failed
  ['1', '32', '16'],
  ['1', '394', '394'],
  ['1', '768', '768'],
  ['1', '768', '2048'],
  ['1', '2048', '2048'],
  ['1', '4096', '4096'],
]

# foreach dims : matmul_cases
#  test('matmul @0@x@1@x@2@'.format(dims[0], dims[1], dims[2]),
#    ops_reg_exe,
#    is_parallel: false,
#    suite: 'matmul',
#    args: ['matmul', dims[0], dims[1], dims[2]])
# endforeach

foreach n : range(10302, 20000, 1)
  test('matmul 1x@0@x8192'.format(n),
    ops_reg_exe,
    is_parallel: false,
    suite: 'matmul',
    timeout: 180,
    args: ['matmul', '1', n.to_string(), '8192'])
endforeach

# llm_linear_cases = [
#   ['1', '1024', '2048', 'attn_q_proj'],
#   ['1', '1024', '1024', 'attn_k_proj'],
#   ['1', '1024', '1024', 'attn_v_proj'],
#   ['1', '2048', '1024', 'attn_o_proj'],
#   ['1', '1024', '3072', 'ffn_gate_proj'],
#   ['1', '1024', '3072', 'ffn_up_proj'],
#   ['1', '3072', '1024', 'ffn_down_proj'],
#   ['1', '1024', '151936', 'lm_head'], # >30s
# ]

# foreach dims : llm_linear_cases
#   test('matmul @0@'.format(dims[3]),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'matmul',
#     timeout: 180,
#     args: ['matmul', dims[0], dims[1], dims[2]])
# endforeach


##################

# conv2d_cases = [
#   ['1', '3', '5', '7', '6', '3', '2', '1', '1', 'conv2d_i1357_w6321'],
#   ['1', '3', '5', '7', '6', '3', '2', '3', '1', 'conv2d_i1357_w6323'],
#   ['1', '3', '5', '7', '6', '3', '2', '5', '1', 'conv2d_i1357_w6325'],
#   ['1', '3', '5', '7', '6', '3', '3', '1', '1', 'conv2d_i1357_w6331'],
#   ['1', '3', '5', '7', '6', '3', '3', '3', '1', 'conv2d_i1357_w6333'],
#   ['1', '3', '5', '7', '6', '1', '3', '3', '3', 'conv2d_i1357_w6133_g3'],
#   ['1', '3', '5', '7', '6', '3', '3', '5', '1', 'conv2d_i1357_w6335'],
# ]

# foreach dims : conv2d_cases
#   test(dims[9],
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'conv2d',
#     args: ['conv2d', dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8]])
# endforeach

# foreach n : range(1, 513)
#   test('conv2d_1x3_@0@x@0@_k1'.format(n),
#     ops_reg_exe,
#     is_parallel: false,
#     suite: 'conv2d',
#     args: ['conv2d', '1', '3', n.to_string(), n.to_string(), '6', '3', '1', '1', '1'])
# endforeach

#################
# Run with: ninja -C build test_failfast
run_target('test_matmul',
  command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'matmul'],
)

# run_target('test_conv2d',
#   command: ['meson', 'test', '--maxfail=1', '-C', meson.project_build_root(), '--suite', 'conv2d'],
# )
