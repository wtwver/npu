int gen_matmul_int8(matmul_params_t *params) {
    npu_cna_desc cna_desc;
    npu_core_desc core_desc;
    npu_dpu_desc dpu_desc;
  
    unsigned int fd_bytes;
    unsigned int fd_banks;
    unsigned int weight_banks;
    int surf_stride;
  
    cna_desc.conv_mode = direct_convolution;
    # here
    cna_desc.in_precision = precision_int8;
    cna_desc.proc_precision = precision_int8;
  
    cna_desc.kernel_groups = 0;
    cna_desc.feature_grains = params->m+1;
    cna_desc.conv_x_stride = 1;
    cna_desc.conv_y_stride = 1;
  
    cna_desc.datain_width = 1;
    cna_desc.datain_height = params->m;
    cna_desc.datain_channel = params->k;
    cna_desc.dataout_width = 1;
    cna_desc.dataout_height = params->m;
    cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;
  
    cna_desc.weight_width = 1;
    cna_desc.weight_height = 1;
    cna_desc.weight_kernels = params->n;
    # here
    cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
      cna_desc.datain_channel * sizeof(int8_t);
    cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;
  
    # here
    fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(int8_t);
    fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
    fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks +1;
    weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
    weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE)==0) ? weight_banks : weight_banks + 1;
    
    printf("DEBUG: int8 CBUF calculations:\n");
    printf("  fd_bytes=%u, weight_bytes=%u\n", fd_bytes, cna_desc.weight_bytes);
    printf("  fd_banks=%u, weight_banks=%u (available: %u)\n", 
           fd_banks, weight_banks, NPU_CBUF_BANKS);
    
    if ((fd_banks) > NPU_CBUF_BANKS-1) {
      printf("DEBUG: ERROR: fd_banks (%u) > NPU_CBUF_BANKS-1 (%u), returning -1\n", fd_banks, NPU_CBUF_BANKS-1);
      return -1;
    } else {
        if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
         weight_banks = NPU_CBUF_BANKS - fd_banks;
         printf("DEBUG: weight_banks recalculated to %u\n", weight_banks);
        } else {
          printf("DEBUG: ERROR: weight_bytes_per_kernel (%u) > NPU_CBUF_BANK_SIZE (%u), returning -2\n", 
                 cna_desc.weight_bytes_per_kernel, NPU_CBUF_BANK_SIZE);
          return -2;
        }
    }
  
    cna_desc.weight_bank = weight_banks;
    cna_desc.data_bank = fd_banks;
    # here
    cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 64;
    cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % 64) == 0) ?
      cna_desc.data_entries : cna_desc.data_entries +1;
    cna_desc.data_sign = 0x1;
    cna_desc.cvt_type  = 0x1;
    cna_desc.cvt_bypass = 0x1;
    cna_desc.cvt_scale0 = 0x1;
    cna_desc.cvt_scale1 = 0x1;
    cna_desc.cvt_scale2 = 0x1;
    cna_desc.cvt_scale3 = 0x1;
    cna_desc.fc_skip_en = 0;
    cna_desc.data_offset = 0x0;
    cna_desc.pad_left = 0;
    cna_desc.pad_top = 0;
    cna_desc.feature_base_addr = params->input_dma;
    cna_desc.weight_offset = 0;
    cna_desc.weight_burst_len = 0xf;
    cna_desc.data_burst_len = 0xf;
    cna_desc.line_stride = cna_desc.datain_width * 4;
    surf_stride = cna_desc.line_stride * ((cna_desc.datain_height / 4)-1);
    surf_stride = surf_stride < 0 ? surf_stride + 1 : surf_stride;
    cna_desc.surf_stride = surf_stride;
    cna_desc.dma_width = cna_desc.datain_width;
    cna_desc.dma_height = cna_desc.datain_height;
    cna_desc.dma_channel = cna_desc.datain_channel;
    cna_desc.decompress_addr0 = params->weights_dma;

    # here
    core_desc.proc_precision = precision_int8;
    core_desc.qd_en = 0;
    core_desc.dataout_height = cna_desc.dataout_height - 1;
    core_desc.dataout_width = cna_desc.dataout_width - 1;
    core_desc.dataout_channel = cna_desc.weight_kernels -1;
  
    dpu_desc.burst_len = 0xf;
    dpu_desc.conv_mode = direct_convolution;
    dpu_desc.output_mode = 0x2;
    dpu_desc.flying_mode = 0x0;

    # here
    dpu_desc.out_precision = precision_int32;
    dpu_desc.in_precision = precision_int8;
    dpu_desc.proc_precision = precision_int8;
    dpu_desc.dst_base_addr = params->output_dma;
    dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
    dpu_desc.width = core_desc.dataout_width ;
    dpu_desc.height = core_desc.dataout_height;
    dpu_desc.channel = core_desc.dataout_channel;
    dpu_desc.bs_bypass = 1;
    dpu_desc.bs_alu_bypass = 1;
    dpu_desc.bs_mul_bypass = 1;
    dpu_desc.bs_relu_bypass = 1;
    dpu_desc.bn_bypass =1;
    dpu_desc.bn_alu_bypass = 1;
    dpu_desc.bn_mul_bypass = 1;
    dpu_desc.bn_relu_bypass = 1;
    dpu_desc.ew_bypass =1;
    dpu_desc.ew_op_bypass =1;
    dpu_desc.ew_lut_bypass =1;
    dpu_desc.ew_op_cvt_bypass =1;
    dpu_desc.ew_relu_bypass=1;

    # here
    dpu_desc.fp32tofp16_en =0;
    dpu_desc.out_cvt_scale =1;

    # here
    dpu_desc.size_e_2 = 7;
    dpu_desc.size_e_1 = 7;
    dpu_desc.size_e_0 = 7;
    dpu_desc.od_bypass = 1;
    dpu_desc.width_wdma = core_desc.dataout_width;
    dpu_desc.height_wdma = core_desc.dataout_height;
    dpu_desc.channel_wdma = core_desc.dataout_channel;

    # here
    dpu_desc.surf_add = dpu_desc.dst_surf_stride * 8;
  
    gen_matmul_task(params->tasks,&cna_desc,&core_desc,&dpu_desc);
  
    return 0;
  }
  