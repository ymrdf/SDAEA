#include <cuda.h>

#include <cstdio>
#include <cstdint>
#include <unistd.h>

static void report(const char *stage, CUresult result) {
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    std::fprintf(stderr, "%s: %s (%d)\n", stage,
                 message ? message : "unknown CUDA error", result);
}

struct ExternalImage {
    CUexternalMemory memory = nullptr;
    CUmipmappedArray mipmap = nullptr;
    CUarray array = nullptr;
    unsigned width = 0;
    unsigned height = 0;
};

// PyTorch must have created the current CUDA context before import. CUDA
// consumes an OpaqueFd on successful import.
extern "C" void *external_image_create(int fd, unsigned long long allocation_size,
                                         unsigned width, unsigned height) {
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
    handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    handle.handle.fd = fd;
    handle.size = allocation_size;
    handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    auto *image_handle = new ExternalImage;
    image_handle->width = width;
    image_handle->height = height;
    CUresult result = cuImportExternalMemory(&image_handle->memory, &handle);
    if (result != CUDA_SUCCESS) {
        close(fd); // Failed import did not consume the received FD.
        report("cuImportExternalMemory", result);
        delete image_handle;
        return nullptr;
    }

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC image{};
    image.offset = 0;
    image.arrayDesc.Width = width;
    image.arrayDesc.Height = height;
    image.arrayDesc.Depth = 0;
    image.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    image.arrayDesc.NumChannels = 4;
    image.numLevels = 1;

    result = cuExternalMemoryGetMappedMipmappedArray(
        &image_handle->mipmap, image_handle->memory, &image);
    if (result != CUDA_SUCCESS) {
        report("cuExternalMemoryGetMappedMipmappedArray", result);
        cuDestroyExternalMemory(image_handle->memory);
        delete image_handle;
        return nullptr;
    }

    result = cuMipmappedArrayGetLevel(&image_handle->array, image_handle->mipmap, 0);
    if (result != CUDA_SUCCESS) {
        report("cuMipmappedArrayGetLevel", result);
        cuMipmappedArrayDestroy(image_handle->mipmap);
        cuDestroyExternalMemory(image_handle->memory);
        delete image_handle;
        return nullptr;
    }
    return image_handle;
}

// dst_device_ptr must point to a contiguous H x W x 4 CUDA tensor owned by
// PyTorch. The copy stays entirely on the GPU.
extern "C" int external_image_copy(void *opaque, void *dst_device_ptr) {
    auto *image_handle = static_cast<ExternalImage *>(opaque);
    if (!image_handle || !dst_device_ptr) return 1;
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = image_handle->array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst_device_ptr);
    copy.dstPitch = image_handle->width * 4;
    copy.WidthInBytes = image_handle->width * 4;
    copy.Height = image_handle->height;
    CUresult result = cuMemcpy2D(&copy);
    if (result == CUDA_SUCCESS) result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) report("CUDA array copy", result);
    return result == CUDA_SUCCESS ? 0 : 2;
}

extern "C" void external_image_destroy(void *opaque) {
    auto *image_handle = static_cast<ExternalImage *>(opaque);
    if (!image_handle) return;
    cuCtxSynchronize();
    cuMipmappedArrayDestroy(image_handle->mipmap);
    cuDestroyExternalMemory(image_handle->memory);
    delete image_handle;
}

extern "C" int copy_external_rgba(int fd, unsigned long long allocation_size,
                                    unsigned width, unsigned height,
                                    void *dst_device_ptr) {
    void *image_handle = external_image_create(fd, allocation_size, width, height);
    if (!image_handle) return 1;
    int result = external_image_copy(image_handle, dst_device_ptr);
    external_image_destroy(image_handle);
    return result;
}
