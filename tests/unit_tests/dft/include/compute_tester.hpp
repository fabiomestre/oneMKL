/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef ONEMKL_COMPUTE_TESTER_HPP
#define ONEMKL_COMPUTE_TESTER_HPP

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"
#include "test_helper.hpp"
#include "test_common.hpp"
#include "reference_dft.hpp"

using namespace oneapi::mkl;

template <dft::precision precision, dft::domain domain>
struct DFT_Test {
    using descriptor_t = dft::descriptor<precision, domain>;
    using PrecisionType =
        typename std::conditional_t<precision == dft::precision::SINGLE, float, double>;

    using InputType = typename std::conditional_t<domain == dft::domain::REAL, PrecisionType,
                                                std::complex<PrecisionType>>;
    using OutputType = std::complex<PrecisionType>;

    const std::int64_t size;
    const std::int64_t conjugate_even_size;
    static constexpr int error_margin = 200;

    sycl::device *dev;
    sycl::queue sycl_queue;
    context cxt;

    std::vector<InputType> input;
    std::vector<PrecisionType> input_re;
    std::vector<PrecisionType> input_im;
    std::vector<OutputType> out_host_ref;

    DFT_Test(sycl::device *dev, std::int64_t size);

    bool skip_test();
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
DFT_Test<precision, domain>::DFT_Test(sycl::device *dev, std::int64_t size)
        : dev{ dev },
          size{ static_cast<std::int64_t>(size) },
          conjugate_even_size{ 2 * (size / 2 + 1) },
          sycl_queue{ *dev, exception_handler },
          cxt{ sycl_queue.get_context() } {
    input = std::vector<InputType>(size);
    input_re = std::vector<PrecisionType>(size);
    input_im = std::vector<PrecisionType>(size);

    out_host_ref = std::vector<OutputType>(size);
    rand_vector(input, size, 1);

    if constexpr (domain == dft::domain::REAL) {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i] };
            input_im[i] = 0;
        }
    }
    else {
        for (int i = 0; i < input.size(); ++i) {
            input_re[i] = { input[i].real() };
            input_im[i] = { input[i].imag() };
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

    return false;
}

template <dft::precision precision, dft::domain domain>
bool DFT_Test<precision, domain>::init() {
    reference_forward_dft<InputType, OutputType>(sycl_queue, input, out_host_ref);
    return static_cast<int>(!skip_test());
}

#include "compute_inplace.hpp"
#include "compute_inplace_real_real.hpp"
#include "compute_out_of_place.hpp"
#include "compute_out_of_place_real_real.hpp"

#endif //ONEMKL_COMPUTE_TESTER_HPP
