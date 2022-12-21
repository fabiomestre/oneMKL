#ifndef ONEMKL_COMPUTE_TESTER_HPP
#define ONEMKL_COMPUTE_TESTER_HPP

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"
#include "test_helper.hpp"
#include "../../blas/include/test_common.hpp"
#include "test_common.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "reference_dft.hpp"

extern std::vector<sycl::device *> devices;

namespace {
using std::vector;
using namespace oneapi::mkl;
using namespace sycl;

//constexpr size_t default_1d_lengths = 6;
constexpr size_t default_1d_lengths = 137;

/* FIXME Delete this function before submitting PR */
template <typename VectorType>
void print_dft_output(VectorType output) {
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
}

template <dft::precision precision, dft::domain domain>
struct DFT_Test {
    using PrecisionType =
        typename std::conditional<precision == dft::precision::SINGLE, float, double>::type;

    using InputType = typename std::conditional<domain == dft::domain::REAL, PrecisionType,
                                                std::complex<PrecisionType>>::type;
    using OutputType = std::complex<PrecisionType>;

    const int conjugate_even_size = 2 * (default_1d_lengths / 2 + 1);
    static constexpr int error_margin = 200; //FIXME this is way too large

    sycl::device *dev;
    sycl::queue sycl_queue;
    context cxt;

    std::vector<InputType> input;
    std::vector<PrecisionType> input_re;
    std::vector<PrecisionType> input_im;
    std::vector<OutputType> out_host_ref;

    DFT_Test(sycl::device *dev);

    bool skip_test();
    bool run_reference();
    bool init();
    int test_in_place_buffer();
    int test_in_place_real_real_buffer();
    int test_out_of_place_buffer();
    int test_out_of_place_real_real_buffer();
    int test_in_place_USM();
    int test_in_place_real_real_USM();
    int test_out_of_place_USM();
    int test_out_of_place_real_real_USM();
};

template <dft::precision precision, dft::domain domain>
DFT_Test<precision, domain>::DFT_Test(sycl::device *dev)
        : dev{ dev },
          sycl_queue{ *dev, exception_handler }, cxt{sycl_queue.get_context()} {
    input = std::vector<InputType>(default_1d_lengths);
    input_re = std::vector<PrecisionType>(default_1d_lengths);
    input_im = std::vector<PrecisionType>(default_1d_lengths);

    out_host_ref = std::vector<OutputType>(default_1d_lengths);
    rand_vector(input, default_1d_lengths, 1);

    if constexpr (domain == dft::domain::REAL) {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i] };
            input_im[i] = 0;
        }
    }
    else {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i].real() };
            input_re[i] = { input[i].real() };
        }
    }
}

template <dft::precision precision, dft::domain domain>
bool DFT_Test<precision, domain>::skip_test() {
    if constexpr (precision == dft::detail::precision::DOUBLE) {
        if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
            std::cout << "Device does not support double precision." << std::endl;
            return true;
        }
    }

#ifndef CALL_RT_API
    std::cout << "CT tests are not implemented yet." << std::endl;
    return true;
#endif

    return false;
}

template <dft::precision precision, dft::domain domain>
bool DFT_Test<precision, domain>::run_reference() {
    sycl::buffer<InputType, 1> in_dev_ref{ sycl::range<1>(default_1d_lengths) };
    sycl::buffer<OutputType, 1> out_dev_ref{ sycl::range<1>(default_1d_lengths) };
    copy_to_device(sycl_queue, input, in_dev_ref);
    bool success =
        reference_forward_dft<InputType, OutputType>(sycl_queue, in_dev_ref, out_dev_ref);
    copy_to_host(sycl_queue, out_dev_ref, out_host_ref);

    return success;
}

template <dft::precision precision, dft::domain domain>
bool DFT_Test<precision, domain>::init() {
    return static_cast<int>(!skip_test() && run_reference());
}

#include "compute_in.hpp"
#include "compute_in_real_real.hpp"
#include "compute_out.hpp"
#include "compute_out_real_real.hpp"

} // namespace
#endif //ONEMKL_COMPUTE_TESTER_HPP
