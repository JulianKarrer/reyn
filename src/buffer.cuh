#ifndef BUFFER_H_
#define BUFFER_H_

#include <thrust/device_vector.h>
#include "gui.h"

template <typename T>
struct ExternallyManagedBuffer
{
    bool active{false};
    T *raw{nullptr};
    GUI *gui{nullptr};
    size_t size{0};
};

/// @brief A convenience wrapper for device buffers with RAII handling of alloc and free, wrapping a thrust device vector for use of thrust algorithms and with convencience methods for acquiring either the thrust vector or a raw device pointer for use in CUDA kernels, template-parametrized over the type of data stored.
/// @tparam T
template <typename T>
class DeviceBuffer
{
private:
    thrust::device_vector<T> buf;
    ExternallyManagedBuffer<T> ext;

public:
    /// @brief Create a device-side buffer using a `thrust::device_vector` of the datatype specified by the template parameter `T` able to hold `N` elements
    /// @param N number of elements to store in the buffer
    DeviceBuffer(size_t N) : buf(N), ext({.active = false, .raw = nullptr, .gui = nullptr, .size = 0}) {};

    /// @brief Create a device-side buffer using a `thrust::device_vector` of the datatype specified by the template parameter `T` able to hold `N` elements, all initialized to `init`
    /// @param N number of elements to store in the buffer
    /// @param init value to initialize all entries to
    DeviceBuffer(size_t N, T init) : buf(N, init), ext({.active = false, .raw = nullptr, .gui = nullptr, .size = 0}) {};

    /// @brief Extend some functionality of the `DeviceBuffer` to CUDA arrays managed externally, e.g. through OpenGL interop with VBOs - in particular, `get` and `resize` should work.
    DeviceBuffer(GUI *_gui) : ext({.active = true, .raw = nullptr, .gui = _gui, .size = _gui->N}) {};

    size_t size() const
    {
        if (ext.active)
        {
            return ext.size;
        }
        else
        {
            return buf.size();
        }
    }

    /// @brief Resize the device buffer, providing the data to intialize new entries with.
    /// @param new_size the new size of the device buffer
    /// @param data the data to initialize new entries in the buffer with
    void resize(size_t new_size, const T &data)
    {
        if (ext.active)
            throw std::runtime_error("Resizing an externally managed `DeviceBuffer` using data to initialize additionally allocated elements with is unimplemented.");
        buf.resize(new_size, data);
    };

    /// @brief Resize the device buffer, initializing any new elements to the default initial value
    /// @param new_size the new size of the device buffer
    void resize(size_t new_size)
    {
        if constexpr (std::is_same_v<T, float3>)
        {
            // if the resized buffer is a float3 buffer, we might have to call the gui
            // to resize it for us
            if (ext.active)
            {
                ext.size = new_size;
                ext.raw = ext.gui->resize_mapped_buffer(new_size);
            }
            else
            {
                buf.resize(new_size);
            }
        }
        else
        {
            // otherwise, externally managed buffers are unsupported
            if (ext.active)
                throw std::runtime_error("Externally managed buffers that are not `float3` are currently not supported, since they are not resized using `gui->resize_mapped_buffer(new_size)`.");
            buf.resize(new_size);
        }
    };

    /// @brief Obtain `thrust::device_vector` underlying the buffer
    /// @return reference to the underlying `thrust::device_vector`
    thrust::device_vector<T> &get()
    {
        if (ext.active)
            throw std::runtime_error("Trying to obtain the `thrust::device_vector` of an externally managed `DeviceBuffer` cannot succeed.");
        else
            return buf;
    };

    /// @brief Obtain `thrust::device_vector` underlying the buffer
    /// @return reference to the underlying `thrust::device_vector`
    const thrust::device_vector<T> &get() const
    {
        if (ext.active)
            throw std::runtime_error("Trying to obtain the `thrust::device_vector` of an externally managed `DeviceBuffer` cannot succeed.");
        else
            return buf;
    };

    /// @brief Obtain the raw CUDA pointer to the array underlying the device buffer, for use in CUDA kernels.
    /// Uses `thrust::raw_pointer_cast`.
    /// @return Pointer to the raw data
    T *ptr()
    {
        if (ext.active)
            return ext.raw;
        else
            return thrust::raw_pointer_cast(buf.data());
    };

    /// @brief Obtain the raw CUDA pointer to the array underlying the device buffer, for use in CUDA kernels.
    /// Uses `thrust::raw_pointer_cast`.
    /// @return Pointer to the raw data
    const T *ptr() const
    {
        if (ext.active)
            return ext.raw;
        else
            return thrust::raw_pointer_cast(buf.data());
    };

    /// @brief If the `DeviceBuffer` is externally managed and the pointer to the underlying resource has changed (i.e. due to unmapping and remapping in OpenGL interop), use this function to update the raw device pointer to the underlying data.
    /// @param new_ptr
    void update_raw_ptr(T *new_ptr)
    {
        if (!ext.active)
            throw std::runtime_error("Trying to update the raw pointer to a `DeviceBuffer` that was not constructed to be externally managed.");
        else
            ext.raw = new_ptr;
    }

    // prevent copying
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

#endif // BUFFER_H_