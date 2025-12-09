#ifndef BUFFER_H_
#define BUFFER_H_

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include "gui.cuh"

/// @brief A convenience wrapper for device buffers with RAII handling of alloc
/// and free, wrapping a thrust device vector for use of thrust algorithms and
/// with convencience methods for acquiring either the thrust vector or a raw
/// device pointer for use in CUDA kernels, template-parametrized over the type
/// of data stored.
/// @tparam T
template <typename T> class DeviceBuffer {

private:
    struct ExternallyManagedBuffer {
        bool active { false };
        T* raw { nullptr };
        size_t size { 0 };
    };
    thrust::device_vector<T> buf;
    ExternallyManagedBuffer ext;

public:
    /// @brief Create a device-side buffer using a `thrust::device_vector` of
    /// the datatype specified by the template parameter `T` able to hold `N`
    /// elements
    /// @param N number of elements to store in the buffer
    DeviceBuffer(size_t N)
        : buf(N)
        , ext({ false, nullptr, 0 }) {};

    /// @brief Create a device-side buffer using a `thrust::device_vector` of
    /// the datatype specified by the template parameter `T` able to hold `N`
    /// elements, all initialized to `init`
    /// @param N number of elements to store in the buffer
    /// @param init value to initialize all entries to
    DeviceBuffer(size_t N, T init)
        : buf(N, init)
        , ext({ false, nullptr, 0 }) {};

    /// @brief Create a device-side buffer using a `thrust::device_vector` of
    /// the datatype specified by the template parameter `T` from the beginning
    /// and end iterators of a standard vector, copying the contents of the
    /// vector to the device.
    /// @param begin `std::vector<T>::begin()` of the vector containing data to
    /// construct the buffer with
    /// @param end `std::vector<T>::end()` of the vector containing data to
    /// construct the buffer with
    DeviceBuffer(const std::vector<T>::const_iterator begin,
        const std::vector<T>::const_iterator end)
        : buf(begin, end)
        , ext({ false, nullptr, 0 }) {};

    /// @brief Extend some functionality of the `DeviceBuffer` to CUDA
    /// arrays managed externally, e.g. through OpenGL interop with VBOs -
    /// in particular, `get` and `resize` should work.
    DeviceBuffer(GUI* _gui)
        : ext({ true, nullptr, _gui->N }) {};

    size_t size() const
    {
        if (ext.active) {
            return ext.size;
        } else {
            return buf.size();
        }
    }

    /// @brief Resize the device buffer, providing the data to intialize new
    /// entries with.
    /// @param new_size the new size of the device buffer
    /// @param data the data to initialize new entries in the buffer with
    void resize(size_t new_size, const T& data)
    {
        if (ext.active) {
            throw std::runtime_error(
                "Resizing an externally managed `DeviceBuffer` using data to "
                "initialize additionally allocated elements with is "
                "unimplemented.");
        } else {
            buf.resize(new_size, data);
        }
    };

    /// @brief Resize the device buffer, initializing any new elements to the
    /// default initial value
    /// @param new_size the new size of the device buffer
    void resize(size_t new_size)
    {
        if (ext.active) {
            throw std::runtime_error(
                "Cannot call resize on externmally managed buffer. Try "
                "GUI::resize_mapped_buffers if applicable");
        }
        if (buf.size() != new_size) {
            buf.resize(new_size);
        }
    };

    /// @brief Obtain `thrust::device_vector` underlying the buffer
    /// @return reference to the underlying `thrust::device_vector`
    thrust::device_vector<T>& get()
    {
        if (ext.active) {
            throw std::runtime_error(
                "Trying to obtain the `thrust::device_vector` of an externally "
                "managed `DeviceBuffer` cannot succeed.");
        } else {
            return buf;
        }
    };

    /// @brief Obtain `thrust::device_vector` underlying the buffer
    /// @return reference to the underlying `thrust::device_vector`
    const thrust::device_vector<T>& get() const
    {
        if (ext.active) {
            throw std::runtime_error(
                "Trying to obtain the `thrust::device_vector` of an externally "
                "managed `DeviceBuffer` cannot succeed.");
        } else {
            return buf;
        }
    };

    /// @brief Obtain the raw CUDA pointer to the array underlying the device
    /// buffer, for use in CUDA kernels. Uses `thrust::raw_pointer_cast`.
    /// @return Pointer to the raw data
    T* ptr()
    {
        if (ext.active) {
            return ext.raw;
        } else {
            return thrust::raw_pointer_cast(buf.data());
        }
    };

    /// @brief Obtain the raw CUDA pointer to the array underlying the device
    /// buffer, for use in CUDA kernels. Might use `thrust::raw_pointer_cast`.
    /// @return Pointer to the raw data
    const T* ptr() const
    {
        if (ext.active)
            return ext.raw;
        else
            return thrust::raw_pointer_cast(buf.data());
    };

    /// @brief If the `DeviceBuffer` is externally managed and the pointer to
    /// the underlying resource has changed (i.e. due to unmapping and remapping
    /// in OpenGL interop), use this function to update the raw device pointer
    /// to the underlying data.
    /// @param new_ptr
    void update_raw_ptr(T* new_ptr, size_t N)
    {
        if (!ext.active) {
            throw std::runtime_error(
                "Trying to update the raw pointer to a `DeviceBuffer` that was "
                "not constructed to be externally managed.");
        } else {
            ext.raw = new_ptr;
            ext.size = N;
        }
    }

    /// @brief Whether this `DeviceBuffer` has externally managed memory
    /// underlying it (e.g. mapped for use by CUDA from OpenGL) or not (meaning
    /// it is a `thrust::device_vector` and corresponding functionality may be
    /// used)
    /// @return `true` if externally managed, `false` if `thrust::device_vector`
    bool is_externally_managed() const { return this->ext.active; }

    /// @brief Reorder the buffer in the order provided by the map `sorted`,
    /// which must be a permutation of the numbers \f$[0; N-1]\f$. This is
    /// essentially a gather operation.
    /// @param sorted permutation of \f$[0; N-1]\f$ to resort
    /// the buffer with. Must have the same length as the buffer itself.
    /// @param tmpa temporary buffer used to resort efficiently (and not
    /// in-place). Must not be an externally managed `DeviceBuffer` since
    /// `thrust` functionality is used, so this would throw an error.
    void reorder(const DeviceBuffer<uint>& sorted, DeviceBuffer<float>& tmp)
    {
        if (tmp.is_externally_managed() || sorted.is_externally_managed()) {
            throw std::runtime_error(
                "map and tmp in reorder operation is must be a "
                "thrust::device_vector internally");
        }

        if (this->is_externally_managed()) {
            // manually reorder and copy
            uint N { static_cast<uint>(this->size()) };
            auto data_d { this->ptr() };
            thrust::transform(sorted.get().begin(), sorted.get().end(),
                tmp.get().begin(),
                [data_d] __device__(uint map_i) { return data_d[map_i]; });

            cudaMemcpy(this->ptr(), tmp.ptr(), N * sizeof(float),
                cudaMemcpyDeviceToDevice);
        } else {
            // this, sorting and tmp are all thrust-arrays:
            // use thrust gather
            thrust::gather(sorted.get().begin(), sorted.get().end(),
                this->get().begin(), tmp.get().begin());
            tmp.get().swap(this->get());
        }
    }

    /// @brief Reduce and return the minimum value of the buffer
    /// @return minimum element
    T min() const
    {
        if (ext.active) {
            throw std::runtime_error("min operation is not supported on "
                                     "externally managed DeviceBuffer");
        } else {
            return thrust::min_element(get().begin(), get().end())[0];
        }
    }

    ///@brief Reduce and return the maximum value of the buffer
    ///@return maximum element
    T max() const
    {
        if (ext.active) {
            throw std::runtime_error("max operation is not supported on "
                                     "externally managed DeviceBuffer");
        } else {
            return thrust::max_element(get().begin(), get().end())[0];
        }
    }

    ///@brief Reduce the sum of the elements in the buffer
    ///@return sum of elements
    T sum() const
    {
        if (ext.active) {
            throw std::runtime_error("sum operation is not supported on "
                                     "externally managed DeviceBuffer");
        } else {
            return thrust::reduce(get().begin(), get().end());
        }
    }

    ///@brief Average value of the elements in the buffer - cast to float if not
    /// already
    ///@return average value of elements
    float avg() const
    {
        if (ext.active) {
            throw std::runtime_error("avg operation is not supported on "
                                     "externally managed DeviceBuffer");
        } else {
            return static_cast<float>(sum()) / static_cast<float>(size());
        }
    }

    /// prevent copying
    DeviceBuffer(const DeviceBuffer&) = delete;
    /// prevent copying
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// allow moving
    DeviceBuffer(DeviceBuffer&& other)
        : ext({ other.ext.active, other.ext.raw, other.ext.size })
        , buf(std::move(other.buf)) {};
};

#endif // BUFFER_H_