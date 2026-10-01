# firmware/ — custom OpenMV N6 firmware for the flight policy

The reported flights ran on a lightly customized OpenMV N6 firmware. The policy itself runs through
the stock Neural-ART runtime (`ml.Model`), so **stock OpenMV N6 firmware also works** — the mission
scripts in `openmv_scripts/` auto-detect the customization and fall back. The two changes below exist
to remove an in-flight **garbage-collection freeze** that otherwise stalls control for hundreds of ms.

The changes are kept as patches against the OpenMV tree rather than a fork; apply them to
`github.com/openmv/openmv` (board `OPENMV_N6`) and build with ARM GCC 14.3.

## The two changes (`n6_patches/`)

1. **`py_framestack.c`** — a zero-allocation C module that builds the 3-frame observation stack.
   `push`/`fill` do the channel shift + `uint8→float` in C; `push_ds`/`fill_ds` additionally do a
   fused 5×5 box-downsample (320→64) off a cached copy of the (uncacheable) camera framebuffer. This
   removes the ~260 B/frame ulab churn that, on the stock path, makes the GC fire mid-flight.
   Drop it into `openmv/modules/` (it is auto-compiled by the `modules/*.c` wildcard; it registers
   `MP_QSTR_framestack`).

2. **Shrink the GC heap** in `boards/OPENMV_N6/board_config.h`:

   ```c
   #define OMV_GC_BLOCK1_SIZE   (4M)   // was 24M: gc.collect() sweeps the whole GC heap,
                                        // so 24M -> ~610 ms per collect, 4M -> ~63 ms.
   ```

   `gc.collect()` cost is proportional to heap size (a full heap walk), independent of how much is
   garbage. 24 MB made a single collect ~610 ms — a guaranteed control dropout. 4 MB is ~63 ms and,
   combined with `framestack` (near-zero churn → collects almost never fire), keeps the loop steady.

## Build & flash

```bash
# in an OpenMV checkout with the two patches applied:
make -C src TARGET=OPENMV_N6 -j$(nproc)
```

Flash the resulting firmware **with the OpenMV IDE** (Tools ▸ Run Bootloader). Do **not** flash the N6
with the generic STM32CubeProgrammer N6 loaders — those are for ST dev boards and risk bricking the
OpenMV N6.

## Not included

The GenX320 DIFF3D driver and the Neural-ART runtime are part of the upstream OpenMV N6 port and are
not duplicated here. `deploy/contracts.md` specifies the observation layout, action semantics and
control rate needed to reimplement the onboard interface against a different airframe or sensor.
