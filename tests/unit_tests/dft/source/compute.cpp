/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

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
#include "oneapi/mkl.hpp"
#include "test_helper.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "reference_dft.hpp"

#include <gtest/gtest.h>

using std::vector;
using namespace oneapi::mkl;
using namespace sycl;

extern std::vector<sycl::device *> devices;

namespace {

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception e) {
            std::cout << "Caught asynchronous SYCL exception during AXPY:\n"
                      << e.what() << std::endl;
            print_error_code(e);
        }
    }
};

constexpr size_t default_1d_lengths = 4;

template <dft::precision precision, dft::domain domain>
int test(sycl::device *dev, dft::descriptor<precision, domain>& descriptor) {

    sycl::queue main_queue(*dev, exception_handler);

//    dft::descriptor<precision, domain> descriptor{ length };
    size_t length = default_1d_lengths;
    descriptor.get_value(dft::config_param::LENGTHS, &length);
    std::cout << length << std::endl;

    dft::precision precision2;
    descriptor.get_value(dft::config_param::PRECISION, &precision2);
    std::cout << static_cast<size_t>(precision2) << std::endl;

    std::vector<double> in_host{1,2,3,4};
    std::vector<std::complex<double>> out_host{1,2,3,4};

    sycl::buffer<double, 1> in_dev_ref{sycl::range<1>(length)};
    sycl::buffer<std::complex<double>, 1> out_dev_ref{sycl::range<1>(length)};

//    std::vector<std::complex<double>> in_host{1,2,3,4};
//    std::vector<std::complex<double>> out_host{1,2,3,4};
//
//    sycl::buffer<std::complex<double>, 1> in_dev_ref{sycl::range<1>(length)};
//    sycl::buffer<std::complex<double>, 1> out_dev_ref{sycl::range<1>(length)};



    main_queue.submit([&](handler &cgh) {
        accessor a_in_dev_ref = in_dev_ref.get_access<access_mode::write>(cgh);
        cgh.copy(in_host.data(), a_in_dev_ref);
    });

//    host_accessor h_d{in_dev_ref};
//    for (size_t i = 0; i < h_d.size(); ++i) {
//        std::cout << h_d[i] << " ";
//    }
//    std::cout << std::endl;
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor.commit(main_queue);

    reference_forward_dft(main_queue, descriptor, in_dev_ref, out_dev_ref);

    host_accessor h_a{out_dev_ref};
    for (size_t i = 0; i < h_a.size(); ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

#ifdef CALL_RT_API
    std::cout << "HELLO, IM RT" << std::endl;
#else
    std::cout << "HELLO, IM CT" << std::endl;
#endif

    return 1;
}



class ComputeTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

/* Default Tests */
TEST_P(ComputeTests, RealSinglePrecisionDefaults) {
    dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> descriptor{ default_1d_lengths };
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), descriptor));
}

TEST_P(ComputeTests, RealDoublePrecisionDefaults) {
    dft::descriptor<dft::precision::DOUBLE, dft::domain::REAL> descriptor{ default_1d_lengths };
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), descriptor));
}

TEST_P(ComputeTests, ComplexSinglePrecisionDefaults) {
    dft::descriptor<dft::precision::SINGLE, dft::domain::COMPLEX> descriptor{ default_1d_lengths };
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), descriptor));
}

TEST_P(ComputeTests, ComplexDoublePrecisionDefaults) {
    dft::descriptor<dft::precision::DOUBLE, dft::domain::COMPLEX> descriptor{ default_1d_lengths };
    EXPECT_TRUEORSKIP(test(std::get<0>(GetParam()), descriptor));
}

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
