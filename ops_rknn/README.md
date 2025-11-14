g++ -o ops_int32 ops_int32.cpp -I../include -lrknnrt -std=c++11
g++ -o ops_int8 ops_int8.cpp -I../include -lrknnrt -std=c++11
g++ -o ops_float16 ops_float16.cpp -I../include -lrknnrt -std=c++11

gdb -x ops.gdb --args ./ops_int32 and 1x1
./ops_int32 add 1x3
./ops_int32 sub 1x4
./ops_int32 div 1x5

The implementation demonstrates the core RKNN workflow:
1. Model loading and initialization (using local model files)
2. Input tensor setup with proper metadata
3. Inference execution
4. Output retrieval and display
5. Resource cleanup

## deterministic conv1d inputs

`conv1d_simple.cpp` now prefers the deterministic tensors created by `generate_conv1d_inputs.py`.
Run the helper before exercising the RKNN path (`python3 generate_conv1d_inputs.py`) to seed NumPy exactly once per case
and export the fp16 `input.bin`/`kernel.bin` pairs plus a JSON descriptor under `conv1d_simple_data/<case>`.
Each descriptor records shapes, kernel height, stride, dilation, rtol/atol, and references a CPU `expected.bin`
produced with `torch.nn.functional.conv1d`. The RKNN driver falls back to the prior ramp data if the binaries
are missing, so you just need to rerun the generator to refresh the vectors after changing the shapes.

## plan to align RKNN vs CPU conv1d outputs

1. **Capture a reference failure**: run `sh run.sh` after regenerating deterministic tensors to collect `run_output.txt` plus the dumps under `dump/`. Note the exact `Native output dims`, format, and any GEM padding that shows up in the log; those details are required to understand how NC1HWC2 buffers are laid out for this model.
2. **Mirror the on-device layout**: inspect `conv2d_multi.cpp`’s `nc1hwc2_fp16_to_nchw` helper and port the logic into `conv1d_simple.cpp`. Feed it with the driver-reported `native_output_attr` (batch, c1, w_stride, c2). This will keep the host-side index in lock‑step with the padded NC1HWC2 buffer instead of assuming a compact `[N, C, W]` blob.
3. **Honor metadata tolerances**: read `conv1d_simple_data/<case>/metadata.json` (or hard-code conservative tolerances) so the comparison uses the same `rtol/atol` thresholds that were used when exporting `expected.bin`. This avoids reporting mismatches that are purely due to fp16 accumulation noise.
4. **Wire up a structured diff**: after converting the RKNN output to `NCHW`, compare it against the CPU reference (or the stored `expected.bin`) and emit a summarized mismatch report: worst index, expected/actual pair, max absolute and relative error, plus a “Status: match/not matched” footer after the tensor printouts.
5. **Re-run with instrumentation**: execute `sh run.sh` again. If the comparison still fails, dump the offending channel slice together with its NC1HWC2 offsets (C1, C2, padded width) so you can rule out stride or channel packing bugs without re-running gdb.
6. **Close the loop**: once the converted tensor matches the CPU baseline, archive the `run_output.txt` snippet that shows the “Status: match” line so regression hunting can start from a known-good reference point. Keep the helper around for future conv1d shapes so new kernels benefit from the fixed conversion path automatically.

## inspecting native NC1HWC2 dumps

Set `RKNN_DUMP_NATIVE=native_dump.bin` before running `conv1d_simple` to capture the raw fp16 buffer. The new
`inspect_nc1hwc2.py` helper shows how the driver multiplexes the logical batches/channels onto the C1/C2 lanes and
verifies the conversion against the CPU reference:

```
python inspect_nc1hwc2.py \
  --dump native_dump.bin \
  --native-shape 8,2,1,12,8 \
  --stride-w 12 \
  --logical-shape 8,6,11 \
  --expected conv1d_simple_data/conv1d_simple_bs8/expected.bin
```

The script prints one row per lane along with the logical `(batch, channel)` slice that each `n` slot carries. That
mapping is what the C++ converter mirrors, so mismatches between RKNN and the CPU reference can now be traced back to
the exact hardware lane with a single command.
