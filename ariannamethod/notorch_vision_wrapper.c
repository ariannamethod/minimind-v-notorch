// notorch_vision_wrapper.c — export notorch_vision.h functions for ctypes
//
// notorch_vision.h uses static functions; this wrapper re-exports them
// as non-static symbols for the shared library.

#include "notorch_vision.h"

// Image loading
nt_image* ntv_image_load(const char* path, int channels) {
    return nt_image_load(path, channels);
}
nt_image* ntv_image_load_mem(const unsigned char* buf, int len, int channels) {
    return nt_image_load_mem(buf, len, channels);
}
void ntv_image_free(nt_image* img) {
    nt_image_free(img);
}

// Transforms
nt_image* ntv_image_resize(const nt_image* src, int tw, int th) {
    return nt_image_resize(src, tw, th);
}
nt_image* ntv_image_center_crop(const nt_image* src, int tw, int th) {
    return nt_image_center_crop(src, tw, th);
}
void ntv_image_normalize(nt_image* img, const float* mean, const float* std) {
    nt_image_normalize(img, mean, std);
}

// Patch extraction
nt_tensor* ntv_image_to_patches(const nt_image* img, int ph, int pw) {
    return nt_image_to_patches(img, ph, pw);
}
nt_tensor* ntv_image_to_tensor(const nt_image* img) {
    return nt_image_to_tensor(img);
}

// Convenience pipelines
nt_tensor* ntv_vit_preprocess(const char* path, int img_size, int patch_size) {
    return nt_vit_preprocess(path, img_size, patch_size);
}

// Memory buffer → patches pipeline (for training from parquet/bytes)
nt_tensor* ntv_vit_preprocess_mem(const unsigned char* buf, int buf_len,
                                   int img_size, int patch_size) {
    nt_image* img = nt_image_load_mem(buf, buf_len, 3);
    if (!img) return NULL;

    float scale = (float)img_size / (img->width < img->height ? img->width : img->height);
    int rw = (int)(img->width * scale + 0.5f);
    int rh = (int)(img->height * scale + 0.5f);
    nt_image* resized = nt_image_resize(img, rw, rh);
    nt_image_free(img);

    nt_image* cropped = nt_image_center_crop(resized, img_size, img_size);
    nt_image_free(resized);

    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[]  = {0.229f, 0.224f, 0.225f};
    nt_image_normalize(cropped, mean, std);

    nt_tensor* patches = nt_image_to_patches(cropped, patch_size, patch_size);
    nt_image_free(cropped);

    return patches;
}
