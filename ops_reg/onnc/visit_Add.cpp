#include <onnc/IR/Compute/Add.h>


void CodeEmitVisitor::visit(const Add& pOp)
{
  printf("visit(Add) is called\n");
  
  // Get tensor attributes.
  const Tensor& first = *(pOp.getInput(0));
  const Tensor& second = *(pOp.getInput(1));
  const Tensor& output = *(pOp.getOutput(0));

  // For this example, we only support a special case where the first tensor is activation data
  // stored in memory and the 2nd tensor is a constant
  assert( (!isConstant(first) && isConstant(second)) &&
          "support only the case that the first tensor is activation data and the second constant");

  NvDlaDlaOperation* operation = new NvDlaDlaOperation();
  // Set hardware block type.
  operation->op_dep.op_type = DLA_OP_SDP;

  struct dla_sdp_op_desc& desc = (struct dla_sdp_op_desc&)(operation->op_desc);
  desc.src_precision     = PRECISION_FP16;
  desc.dst_precision     = PRECISION_FP16;
  desc.lut_index         = -1;



  desc.batch_num         = 1;
  desc.batch_stride      = 0;
  desc.x1_op.enable      = 1;
  desc.x1_op.alu_type    = SDP_ALU_OP_SUM;    //SUM/MIN/MAX
  desc.x1_op.type        = SDP_OP_ADD;        // (SDP_OP_NONE) / (SDP_OP_ADD) / (SDP_OP_MUL) / ALU+MUL (SDP_OP_BOTH)
  desc.x1_op.mode        = SDP_OP_PER_POINT;  // per_layer/per_channel/per_point , getSdpOpMode(category)
  desc.x1_op.act         = ACTIVATION_NONE;   // Disable ReLU


  desc.x1_op.precision   = PRECISION_FP16;

  struct dla_sdp_surface_desc& surface = (struct dla_sdp_surface_desc&)(operation->op_surf);

  const NvDlaCubeInfo firstCubeInfo   = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, first);
  // The 1st input tensor can be read from:
  //   external DRAM via the interface of MCIF: DLA_MEM_MC
  //   SRAM via the interface of CVIF: DLA_MEM_CV
  //   the output of CONV hardware block: DLA_MEM_HW
  // In this example, we only support the 1st input tensor is stored at external DRAM.
  surface.src_data.type               = DLA_MEM_MC;
  // Setup memory allocation and DMA configuration for 1st input tensor.
  surface.src_data.address            = issueDlaAddr(first, firstCubeInfo);
  surface.src_data.size               = m_pMeta.getMemoryListEntrySize(first);
  surface.src_data.width              = firstCubeInfo.dim_w;
  surface.src_data.height             = firstCubeInfo.dim_h;
  surface.src_data.channel            = firstCubeInfo.dim_c;
  surface.src_data.line_stride        = firstCubeInfo.stride_line;
  surface.src_data.surf_stride        = firstCubeInfo.stride_surface;

  MemoryListEntryId   memoryId;
  const NvDlaCubeInfo secondCubeInfo = makeCubeInfo(*this, getSdpXSingleCubeType(second, DLA_PRECISION), second);
  // The 2nd input tensor is stored at DRAM and accessed through the interface of MCIF.
  surface.x1_data.type               = DLA_MEM_MC;
  // Setup memory allocation and DMA configuration for 2nd input tensor.
  // In addition, the 2nd tensor is constant so need be packed into a blob and becomes a part of loadable.
  surface.x1_data.address            = issueSDPOperand(second, secondCubeInfo, memoryId);
  surface.x1_data.size               = m_pMeta.getMemoryListEntrySize(memoryId);
  surface.x1_data.width              = secondCubeInfo.dim_w;
  surface.x1_data.height             = secondCubeInfo.dim_h;
  surface.x1_data.channel            = secondCubeInfo.dim_c;
  surface.x1_data.line_stride        = secondCubeInfo.stride_line;
  surface.x1_data.surf_stride        = secondCubeInfo.stride_surface;

  const NvDlaCubeInfo outputCubeInfo = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, output);
  // The output tensor is stored at DRAM.
  surface.dst_data.type         = DLA_MEM_MC;
  surface.dst_data.address      = issueDlaAddr(output, outputCubeInfo);
  surface.dst_data.size         = m_pMeta.getMemoryListEntrySize(output);
  surface.dst_data.width        = outputCubeInfo.dim_w;
  surface.dst_data.height       = outputCubeInfo.dim_h;
  surface.dst_data.channel      = outputCubeInfo.dim_c;
  surface.dst_data.line_stride  = outputCubeInfo.stride_line;
  surface.dst_data.surf_stride  = outputCubeInfo.stride_surface;

  issueDlaOp(operation, NULL, m_pMeta.m_pPrevOp);
}

void CodeEmitVisitor::emitSdp(std::uint8_t opType, const Tensor& first, const Tensor& second, const Tensor& output)
{
  assert(!(isConstant(first) && isConstant(second)) && "cannot support 2 constant tensors");
  assert(opType == SDP_OP_ADD || opType == SDP_OP_MUL);

  // make sure the 'first' tensor is always non-constant
  if (isConstant(first)) {
    emitSdp(opType, second, first, output);
    return;
  }

  assert(!isConstant(first));

  const BroadcastCategory category =
    (isConstant(second) ? getBroadcastCategory(second, first) : getBroadcastCategory(first, second));
  auto operation = makeNvDlaOp(NvDlaOpType::sdp);

  auto& desc             = getDesc<NvDlaOpType::sdp>(*operation);
  desc.src_precision     = DLA_PRECISION;
  desc.dst_precision     = DLA_PRECISION;
  desc.lut_index         = -1;
  desc.out_cvt.scale     = 1;
  desc.out_cvt.truncate  = 0;
  desc.out_cvt.enable    = 1;
  desc.out_cvt.offset    = 0;
  desc.conv_mode         = CONV_MODE_DIRECT;
  desc.batch_num         = 1;
  desc.batch_stride      = 0;
  desc.x1_op.enable      = 1;
  desc.x1_op.alu_type    = SDP_ALU_OP_SUM;
  desc.x1_op.type        = opType;
  desc.x1_op.mode        = getSdpOpMode(category);
  desc.x1_op.act         = ACTIVATION_NONE;
  desc.x1_op.shift_value = 0;
  desc.x1_op.truncate    = 0;
  desc.x1_op.precision   = DLA_PRECISION;
  if (category == BroadcastCategory::LAYER) {
    const auto operand = to_<std::vector<float>>(second);
    assert(size(operand) == 1);

    switch (opType) {
    case SDP_OP_ADD:
      desc.x1_op.mul_operand = 0;
      desc.x1_op.alu_operand = f2float16_ieee(*begin(operand));
      break;
    case SDP_OP_MUL:
      desc.x1_op.alu_operand = 0;
      desc.x1_op.mul_operand = f2float16_ieee(*begin(operand));
      break;
    default:
      assert(false && "should not reach here");
    }
  }

  auto& surface = getSurface<NvDlaOpType::sdp>(*operation);

  const NvDlaCubeInfo firstCubeInfo = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, first);
  NvDlaDataCubeModifier(surface.src_data, NvDlaMemType::mc)
    .setSize(m_pMeta.getMemoryListEntrySize(first))
    .setAddress(issueDlaAddr(first, firstCubeInfo))
    .setInfo(firstCubeInfo);

  if (category == BroadcastCategory::LAYER) {
    NvDlaDataCubeModifier(surface.x1_data, NvDlaMemType::hw).setAddress(-1);
  } else {
    MemoryListEntryId   memoryId;
    const NvDlaCubeInfo secondCubeInfo = makeCubeInfo(*this, getSdpXSingleCubeType(second, DLA_PRECISION), second);
    NvDlaDataCubeModifier(surface.x1_data, NvDlaMemType::mc)
      .setAddress(issueSDPOperand(second, secondCubeInfo, memoryId))
      .setSize(m_pMeta.getMemoryListEntrySize(memoryId))
      .setInfo(secondCubeInfo);
  }

  const NvDlaCubeInfo outputCubeInfo = makeCubeInfo(*this, NVDLA_CUBE_FEATURE, output);
  NvDlaDataCubeModifier(surface.dst_data, NvDlaMemType::mc)
    .setSize(m_pMeta.getMemoryListEntrySize(output))
    .setAddress(issueDlaAddr(output, outputCubeInfo))
    .setInfo(outputCubeInfo);

  issueDlaOp(std::move(operation));
}

NvDlaError engine_ast::SDPElementWiseOpNode::emitOp(Graph *g,
  DLAInterface *target_dla,
  NvU32 op_slot, NvU32 batch_id,
  DLACommonOpDescAccessor       dep,
  DLAOperationContainerAccessor op,
  DLASurfaceContainerAccessor   surf)
{
NvDlaError e = NvDlaSuccess;
DLASDPOpDescAccessor  sdp_op = op.sdpOpDescAccessor(0);
DLACVTParamAccessor out_cvt_acc = sdp_op.outCVTAccessor();
DLASDPOpAccessor x1_op_acc = sdp_op.x1OpAccessor();
DLASDPOpAccessor x2_op_acc = sdp_op.x2OpAccessor();
DLASDPOpAccessor y_op_acc  = sdp_op.yOpAccessor();
DLASDPSurfaceDescAccessor surf_acc  = surf.sdpSurfaceDescAccessor(0);
DLADataCubeAccessor src_data_acc    = surf_acc.srcDataAccessor();
DLADataCubeAccessor dst_data_acc    = surf_acc.dstDataAccessor();
DLADataCubeAccessor x1_data_acc     = surf_acc.x1DataAccessor();
DLADataCubeAccessor x2_data_acc     = surf_acc.x2DataAccessor();
DLADataCubeAccessor y_data_acc     = surf_acc.yDataAccessor();
DLAConsumerAccessor fused_acc = dep.fusedParentAccessor();
NVDLA_UNUSED(x2_data_acc);
NVDLA_UNUSED(y_data_acc);
NVDLA_UNUSED(fused_acc);

surface::TensorSurfaceDesc *src_one_tsd  = g->nodeInputTensorSurface(this, 0, supportedInSurfCategories());
surface::TensorSurfaceDesc *src_two_tsd  = g->nodeInputTensorSurface(this, 1, supportedInSurfCategories());
surface::TensorSurfaceDesc *dst_tsd  = g->nodeOutputTensorSurface(this, 0, supportedOutSurfCategories());

*sdp_op.srcPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, src_one_tsd->surfaceFormat().precision());
*sdp_op.dstPrecision()   = ASTToDLAInterface::getSDPPrecision(target_dla, dst_tsd->surfaceFormat().precision());
*sdp_op.LUTIndex()       = -1;
*sdp_op.batchNum()       = 1;
*sdp_op.batchStride()    = 0;

*out_cvt_acc.scale()     = params().outCVT().scale();
*out_cvt_acc.truncate()  = params().outCVT().truncate();
*out_cvt_acc.offset()    = params().outCVT().offset();
*out_cvt_acc.enable()    = static_cast<NvU8>(params().outCVT().isEnable());

*x1_op_acc.enable()      = ASTToDLAInterface::getSDPEnable(target_dla, params(batch_id).x1Params().enabled());
*x1_op_acc.ALUType()     = ASTToDLAInterface::getSDPALUType(target_dla, params(batch_id).x1Params().aluType());
*x1_op_acc.type()        = ASTToDLAInterface::getSDPOpType(target_dla, params(batch_id).x1Params().opType());
*x1_op_acc.mode()        = ASTToDLAInterface::getSDPMode(target_dla, params(batch_id).x1Params().mode());
*x1_op_acc.act()         = ASTToDLAInterface::getSDPActType(target_dla, params(batch_id).x1Params().actType());
*x1_op_acc.shiftValue()  = 0;
*x1_op_acc.ALUOperand()  = 0;
*x1_op_acc.MulOperand()  = 1;
*x1_op_acc.truncate()    = 0;
*x1_op_acc.precision()   = *sdp_op.srcPrecision(); // precision of engine = precision of its input tensor

*x2_op_acc.enable() = 0;
*y_op_acc.enable()  = 0;

emitDependencyParams(target_dla, dep, batch_id);
setDataCubeAccessor(src_data_acc, src_one_tsd, IODirectionEnum::INPUT, batch_id);
setDataCubeAccessor(x1_data_acc, src_two_tsd, IODirectionEnum::UNKNOWN, batch_id);
setDataCubeAccessor(dst_data_acc, dst_tsd, IODirectionEnum::OUTPUT, batch_id);

if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_WINOGRAD )
{
ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported WINOGRAD Conv mode for %s", name().c_str());
}
else if ( params(batch_id).convMode().v() == ConvolutionModeEnum::CONV_DIRECT )
{
*sdp_op.convMode() = sdp_op.convMode_Direct();
}
else
{
ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Conv mode for %s", name().c_str());
}

if ( g->debugOps() )
{
gLogInfo << "SDP EW node @ op_slot = " << op_slot << " batch_id = " << batch_id << endl;
gLogInfo << "\tsrc precision " << (int)*sdp_op.srcPrecision() << endl;
gLogInfo << "\tdst precision " << (int)*sdp_op.dstPrecision() << endl;
gLogInfo << "\tx1 enable " << (int)*x1_op_acc.enable() << endl;
if (*x1_op_acc.enable())
{
gLogInfo << "\tx1 precision " << (int)*x1_op_acc.precision() << endl;
gLogInfo << "\tx1 aluType " << (int)*x1_op_acc.ALUType() << endl;
gLogInfo << "\tx1 type " << (int)*x1_op_acc.type() << endl;
gLogInfo << "\tx1 mode " << (int)*x1_op_acc.mode() << endl;
gLogInfo << "\tx1 act " << (int)*x1_op_acc.act() << endl;
gLogInfo << "\tx1 shiftValue " << (int)*x1_op_acc.shiftValue() << endl;
gLogInfo << "\tx1 aluOperand " << (int)*x1_op_acc.ALUOperand() << endl;
gLogInfo << "\tx1 mulOperand " << (int)*x1_op_acc.MulOperand() << endl;
gLogInfo << "\tx1 truncate " << (int)*x1_op_acc.truncate() << endl;
}
gLogInfo << "\tx2 enable " << (int)*x2_op_acc.enable() << endl;
if (*x2_op_acc.enable())
{
gLogInfo << "\tx2 precision " << (int)*x2_op_acc.precision() << endl;
gLogInfo << "\tx2 aluType " << (int)*x2_op_acc.ALUType() << endl;
gLogInfo << "\tx2 type " << (int)*x2_op_acc.type() << endl;
gLogInfo << "\tx2 mode " << (int)*x2_op_acc.mode() << endl;
gLogInfo << "\tx2 act " << (int)*x2_op_acc.act() << endl;
gLogInfo << "\tx2 shiftValue " << (int)*x2_op_acc.shiftValue() << endl;
gLogInfo << "\tx2 aluOperand " << (int)*x2_op_acc.ALUOperand() << endl;
gLogInfo << "\tx2 mulOperand " << (int)*x2_op_acc.MulOperand() << endl;
gLogInfo << "\tx2 truncate " << (int)*x2_op_acc.truncate() << endl;
}
gLogInfo << "\ty enable " << (int)*y_op_acc.enable() << endl;
if (*y_op_acc.enable())
{
gLogInfo << "\ty precision " << (int)*y_op_acc.precision() << endl;
gLogInfo << "\ty aluType " << (int)*y_op_acc.ALUType() << endl;
gLogInfo << "\ty type " << (int)*y_op_acc.type() << endl;
gLogInfo << "\ty mode " << (int)*y_op_acc.mode() << endl;
gLogInfo << "\ty act " << (int)*y_op_acc.act() << endl;
gLogInfo << "\ty shiftValue " << (int)*y_op_acc.shiftValue() << endl;
gLogInfo << "\ty aluOperand " << (int)*y_op_acc.ALUOperand() << endl;
gLogInfo << "\ty mulOperand " << (int)*y_op_acc.MulOperand() << endl;
gLogInfo << "\ty truncate " << (int)*y_op_acc.truncate() << endl;
}
gLogInfo << "\tsrc1 tsd:" << src_one_tsd->id() << endl;
gLogInfo << "\tsrc2 tsd:" << src_two_tsd->id() << endl;
gLogInfo << "\tdst tsd:" << dst_tsd->id() << endl;
gLogInfo << "\tdependencyCount" << (int)*dep.dependencyCount() << endl;
gLogInfo << "\tsrc1 addr=" << *src_data_acc.address() << endl;
gLogInfo << "\tsrc1 type=" << (int)*src_data_acc.type() << endl;
gLogInfo << "\tsrc1 size " << *src_data_acc.size()    << endl;
gLogInfo << "\tsrc1 width " << *src_data_acc.width()   << endl;
gLogInfo << "\tsrc1 height " << *src_data_acc.height()   << endl;
gLogInfo << "\tsrc1 channel " << *src_data_acc.channel()  << endl;
gLogInfo << "\tsrc1 linestride " << *src_data_acc.lineStride() << endl;
gLogInfo << "\tsrc1 surfstride " << *src_data_acc.surfStride()  << endl;
gLogInfo << "\tsrc2 addr=" << *x1_data_acc.address() << endl;
gLogInfo << "\tsrc2 type=" << (int)*x1_data_acc.type() << endl;
gLogInfo << "\tsrc2 size " << *x1_data_acc.size()    << endl;
gLogInfo << "\tsrc2 width " << *x1_data_acc.width()   << endl;
gLogInfo << "\tsrc2 height " << *x1_data_acc.height()   << endl;
gLogInfo << "\tsrc2 channel " << *x1_data_acc.channel()  << endl;
gLogInfo << "\tsrc2 linestride " << *x1_data_acc.lineStride() << endl;
gLogInfo << "\tsrc2 surfstride " << *x1_data_acc.surfStride()  << endl;
gLogInfo << "\tdst addr=" << *dst_data_acc.address() << endl;
gLogInfo << "\tdst type=" << (int)*dst_data_acc.type() << endl;
gLogInfo << "\tdst size " << *dst_data_acc.size()    << endl;
gLogInfo << "\tdst width " << *dst_data_acc.width()   << endl;
gLogInfo << "\tdst height " << *dst_data_acc.height()   << endl;
gLogInfo << "\tdst channel " << *dst_data_acc.channel()  << endl;
gLogInfo << "\tdst linestride " << *dst_data_acc.lineStride() << endl;
gLogInfo << "\tdst surfstride " << *dst_data_acc.surfStride()  << endl;
gLogInfo << "\tout_cvt enable " << (int)*out_cvt_acc.enable() << endl;
gLogInfo << "\tout_cvt scale " << (int)*out_cvt_acc.scale() << endl;
gLogInfo << "\tout_cvt offset " << (int)*out_cvt_acc.offset() << endl;
gLogInfo << "\tout_cvt truncate " << (int)*out_cvt_acc.truncate() << endl;
}

fail:
return e;
}