#ifndef ONEMKL_TEST_COMMON_HPP
#define ONEMKL_TEST_COMMON_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

template <typename T, int D>
void copy_to_host(sycl::queue sycl_queue, sycl::buffer<T, D> &buffer_in, std::vector<T> &host_out) {
    sycl_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor accessor = buffer_in.get_access(cgh);
        cgh.copy(accessor, host_out.data());
    });
    sycl_queue.wait();
}

template <typename T, int D>
void copy_to_device(sycl::queue sycl_queue, std::vector<T> &host_in,
                    sycl::buffer<T, D> &buffer_out) {
    sycl_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor accessor = buffer_out.get_access(cgh);
        cgh.copy(host_in.data(), accessor);
    });
    sycl_queue.wait();
}

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
            print_error_code(e);
        }
    }
};
#endif //ONEMKL_TEST_COMMON_HPP