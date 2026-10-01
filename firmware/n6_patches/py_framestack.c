/*
 * This file is part of the OpenMV project.
 *
 * framestack -- zero-allocation rolling frame-stack builder for the N6 deploy loop.
 *
 * The event-obs deploy policies take a [1, G, G, K] NHWC float32 observation: a K-deep stack of
 * downsampled GenX320 frames, newest in the last channel. Building that in Python/ulab costs ~50 KB
 * of temporaries PER FRAME -- each `img[0,:,:,i] = img[0,:,:,i+1]` channel shift slice-copies a
 * G*G float block, plus the uint8->float conversion of the new frame. That churn forces MicroPython
 * GC, and on the N6 a full GC sweep walks the whole DRAM GC heap (~0.1-0.6 s = a control freeze).
 *
 * framestack.push(dst, src) does the entire shift + uint8->float convert in ONE C pass over G*G*K
 * with NO heap allocation, so the deploy loop never churns and GC essentially never runs in flight.
 * framestack.fill(dst, src) writes the frame into ALL K channels (== the env-reset / first frame).
 */
#include "py/runtime.h"
#include "py/obj.h"

#include "imlib.h"
#include "py_helper.h"

#if MICROPY_PY_ULAB
#include "ulab/code/ndarray.h"

// Validate dst is a C-contiguous float32 [1, H, W, K] ndarray matching a grayscale HxW image,
// and return the flat float buffer plus H, W, K. Raises on any mismatch.
static mp_float_t *framestack_check(mp_obj_t dst_obj, image_t *img, size_t *H, size_t *W, size_t *K) {
    if (!mp_obj_is_type(dst_obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("dst must be an ndarray"));
    }
    ndarray_obj_t *nd = MP_OBJ_TO_PTR(dst_obj);
    if (nd->dtype != NDARRAY_FLOAT || nd->itemsize != sizeof(mp_float_t)) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be float"));
    }
    if (nd->ndim != 4 || nd->shape[ULAB_MAX_DIMS - 4] != 1) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be [1,H,W,K]"));
    }
    size_t h = nd->shape[ULAB_MAX_DIMS - 3];
    size_t w = nd->shape[ULAB_MAX_DIMS - 2];
    size_t k = nd->shape[ULAB_MAX_DIMS - 1];
    // Require full C-contiguity (byte strides) so the flat p*K + c index is valid.
    int32_t is = (int32_t) nd->itemsize;
    if (nd->strides[ULAB_MAX_DIMS - 1] != is ||
        nd->strides[ULAB_MAX_DIMS - 2] != is * (int32_t) k ||
        nd->strides[ULAB_MAX_DIMS - 3] != is * (int32_t) (k * w)) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be C-contiguous"));
    }
    if ((size_t) img->w != w || (size_t) img->h != h) {
        mp_raise_ValueError(MP_ERROR_TEXT("image size != [H,W]"));
    }
    *H = h; *W = w; *K = k;
    return (mp_float_t *) nd->array;
}

// framestack.push(dst[1,H,W,K] float32, src grayscale HxW image):
//   shift channels toward 0 (drop oldest), write src into the last channel as float. Zero alloc.
static mp_obj_t py_framestack_push(mp_obj_t dst_obj, mp_obj_t img_obj) {
    image_t *img = py_helper_arg_to_image(img_obj, ARG_IMAGE_GRAYSCALE | ARG_IMAGE_UNCOMPRESSED);
    size_t H, W, K;
    mp_float_t *buf = framestack_check(dst_obj, img, &H, &W, &K);
    const uint8_t *px = (const uint8_t *) img->data;
    size_t npix = H * W;
    for (size_t p = 0; p < npix; p++) {
        mp_float_t *base = buf + p * K;
        for (size_t c = 0; c + 1 < K; c++) {
            base[c] = base[c + 1];
        }
        base[K - 1] = (mp_float_t) px[p];
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_framestack_push_obj, py_framestack_push);

// framestack.fill(dst[1,H,W,K] float32, src grayscale HxW image):
//   write src into ALL K channels (first frame == env reset). Zero alloc.
static mp_obj_t py_framestack_fill(mp_obj_t dst_obj, mp_obj_t img_obj) {
    image_t *img = py_helper_arg_to_image(img_obj, ARG_IMAGE_GRAYSCALE | ARG_IMAGE_UNCOMPRESSED);
    size_t H, W, K;
    mp_float_t *buf = framestack_check(dst_obj, img, &H, &W, &K);
    const uint8_t *px = (const uint8_t *) img->data;
    size_t npix = H * W;
    for (size_t p = 0; p < npix; p++) {
        mp_float_t v = (mp_float_t) px[p];
        mp_float_t *base = buf + p * K;
        for (size_t c = 0; c < K; c++) {
            base[c] = v;
        }
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_framestack_fill_obj, py_framestack_fill);

// --- Fused box-downsample + stack, so we can skip the ~16ms draw_image(AREA) M55 pass. ---
// Validate dst is C-contiguous float32 [1,H,W,K] and src grayscale (fy*H)x(fx*W); return buf + dims.
static mp_float_t *framestack_check_ds(mp_obj_t dst_obj, image_t *img,
                                       size_t *H, size_t *W, size_t *K, size_t *fx, size_t *fy) {
    if (!mp_obj_is_type(dst_obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("dst must be an ndarray"));
    }
    ndarray_obj_t *nd = MP_OBJ_TO_PTR(dst_obj);
    if (nd->dtype != NDARRAY_FLOAT || nd->itemsize != sizeof(mp_float_t)) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be float"));
    }
    if (nd->ndim != 4 || nd->shape[ULAB_MAX_DIMS - 4] != 1) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be [1,H,W,K]"));
    }
    size_t h = nd->shape[ULAB_MAX_DIMS - 3], w = nd->shape[ULAB_MAX_DIMS - 2], k = nd->shape[ULAB_MAX_DIMS - 1];
    int32_t is = (int32_t) nd->itemsize;
    if (nd->strides[ULAB_MAX_DIMS - 1] != is ||
        nd->strides[ULAB_MAX_DIMS - 2] != is * (int32_t) k ||
        nd->strides[ULAB_MAX_DIMS - 3] != is * (int32_t) (k * w)) {
        mp_raise_ValueError(MP_ERROR_TEXT("dst must be C-contiguous"));
    }
    if (w == 0 || h == 0 || (size_t) img->w % w || (size_t) img->h % h) {
        mp_raise_ValueError(MP_ERROR_TEXT("src not an integer multiple of dst"));
    }
    *H = h; *W = w; *K = k; *fx = (size_t) img->w / w; *fy = (size_t) img->h / h;
    return (mp_float_t *) nd->array;
}

// Box-average the (fy x fx) block for output pixel (oy,ox) of a grayscale src.
static inline mp_float_t framestack_box(const uint8_t *sp, size_t Ws, size_t oy, size_t ox,
                                        size_t fx, size_t fy, mp_float_t inv) {
    uint32_t acc = 0;
    const uint8_t *blk = sp + (oy * fy) * Ws + ox * fx;
    for (size_t dy = 0; dy < fy; dy++, blk += Ws) {
        for (size_t dx = 0; dx < fx; dx++) {
            acc += blk[dx];
        }
    }
    return acc * inv;
}

// framestack.push_ds(dst[1,H,W,K], src grayscale (fy*H)x(fx*W)):
//   box-downsample src -> shift channels toward 0 -> write into last channel. Replaces draw_image(AREA)+push.
static mp_obj_t py_framestack_push_ds(mp_obj_t dst_obj, mp_obj_t img_obj) {
    image_t *img = py_helper_arg_to_image(img_obj, ARG_IMAGE_GRAYSCALE | ARG_IMAGE_UNCOMPRESSED);
    size_t H, W, K, fx, fy;
    mp_float_t *buf = framestack_check_ds(dst_obj, img, &H, &W, &K, &fx, &fy);
    const uint8_t *sp = (const uint8_t *) img->data;
    size_t Ws = (size_t) img->w;
    mp_float_t inv = (mp_float_t) 1 / (mp_float_t) (fx * fy);
    for (size_t oy = 0; oy < H; oy++) {
        for (size_t ox = 0; ox < W; ox++) {
            mp_float_t v = framestack_box(sp, Ws, oy, ox, fx, fy, inv);
            mp_float_t *base = buf + (oy * W + ox) * K;
            for (size_t c = 0; c + 1 < K; c++) {
                base[c] = base[c + 1];
            }
            base[K - 1] = v;
        }
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_framestack_push_ds_obj, py_framestack_push_ds);

// framestack.fill_ds(dst[1,H,W,K], src): box-downsample src into ALL K channels (first frame == reset).
static mp_obj_t py_framestack_fill_ds(mp_obj_t dst_obj, mp_obj_t img_obj) {
    image_t *img = py_helper_arg_to_image(img_obj, ARG_IMAGE_GRAYSCALE | ARG_IMAGE_UNCOMPRESSED);
    size_t H, W, K, fx, fy;
    mp_float_t *buf = framestack_check_ds(dst_obj, img, &H, &W, &K, &fx, &fy);
    const uint8_t *sp = (const uint8_t *) img->data;
    size_t Ws = (size_t) img->w;
    mp_float_t inv = (mp_float_t) 1 / (mp_float_t) (fx * fy);
    for (size_t oy = 0; oy < H; oy++) {
        for (size_t ox = 0; ox < W; ox++) {
            mp_float_t v = framestack_box(sp, Ws, oy, ox, fx, fy, inv);
            mp_float_t *base = buf + (oy * W + ox) * K;
            for (size_t c = 0; c < K; c++) {
                base[c] = v;
            }
        }
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_framestack_fill_ds_obj, py_framestack_fill_ds);

static const mp_rom_map_elem_t globals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_framestack) },
    { MP_ROM_QSTR(MP_QSTR_push),     MP_ROM_PTR(&py_framestack_push_obj) },
    { MP_ROM_QSTR(MP_QSTR_fill),     MP_ROM_PTR(&py_framestack_fill_obj) },
    { MP_ROM_QSTR(MP_QSTR_push_ds),  MP_ROM_PTR(&py_framestack_push_ds_obj) },  // fused box-downsample + push
    { MP_ROM_QSTR(MP_QSTR_fill_ds),  MP_ROM_PTR(&py_framestack_fill_ds_obj) },  // fused box-downsample + fill
};
static MP_DEFINE_CONST_DICT(globals_dict, globals_dict_table);

const mp_obj_module_t framestack_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_t) &globals_dict,
};

MP_REGISTER_MODULE(MP_QSTR_framestack, framestack_module);
#endif // MICROPY_PY_ULAB
