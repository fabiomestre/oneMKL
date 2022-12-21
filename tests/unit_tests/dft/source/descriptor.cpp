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

extern std::vector<sycl::device*> devices;

namespace {

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
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

constexpr std::int64_t default_1d_lengths = 4;
const std::vector<std::int64_t> default_3d_lengths{ 124, 5, 3 };

template <dft::precision precision, dft::domain domain>
inline void set_and_get_lengths(sycl::queue& sycl_queue) {
    /* Negative Testing */
    {
        dft::descriptor<precision, domain> descriptor{ default_3d_lengths };
        EXPECT_THROW(descriptor.set_value(dft::config_param::LENGTHS, nullptr),
                     oneapi::mkl::invalid_argument);
    }

    /* 1D */
    {
        dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

        std::int64_t lengths_value{ 0 };
        std::int64_t new_lengths{ 2345 };
        std::int64_t dimensions_before_set{ 0 };
        std::int64_t dimensions_after_set{ 0 };

        descriptor.get_value(dft::config_param::LENGTHS, &lengths_value);
        descriptor.get_value(dft::config_param::DIMENSION, &dimensions_before_set);
        EXPECT_EQ(default_1d_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, 1);

        descriptor.set_value(dft::config_param::LENGTHS, new_lengths);
        descriptor.get_value(dft::config_param::LENGTHS, &lengths_value);
        descriptor.get_value(dft::config_param::DIMENSION, &dimensions_after_set);
        EXPECT_EQ(new_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, dimensions_after_set);

        descriptor.commit(sycl_queue);
    }

    /* >= 2D */
    {
        const std::int64_t dimensions = 3;

        dft::descriptor<precision, domain> descriptor{ default_3d_lengths };

        std::vector<std::int64_t> lengths_value(3);
        std::vector<std::int64_t> new_lengths{ 1, 2, 7 };
        std::int64_t dimensions_before_set{ 0 };
        std::int64_t dimensions_after_set{ 0 };

        descriptor.get_value(dft::config_param::LENGTHS, lengths_value.data());
        descriptor.get_value(dft::config_param::DIMENSION, &dimensions_before_set);

        EXPECT_EQ(default_3d_lengths, lengths_value);
        EXPECT_EQ(dimensions, dimensions_before_set);

        descriptor.set_value(dft::config_param::LENGTHS, new_lengths.data());
        descriptor.get_value(dft::config_param::LENGTHS, lengths_value.data());
        descriptor.get_value(dft::config_param::DIMENSION, &dimensions_after_set);

        EXPECT_EQ(new_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, dimensions_after_set);

        //        descriptor.commit(sycl_queue); FIXME Commiting multiple dimensions does not work.
    }
}

template <dft::precision precision, dft::domain domain>
inline void set_and_get_strides(sycl::queue& sycl_queue) {
    dft::descriptor<precision, domain> descriptor{ default_3d_lengths };

    EXPECT_THROW(descriptor.set_value(dft::config_param::INPUT_STRIDES, nullptr),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(dft::config_param::OUTPUT_STRIDES, nullptr),
                 oneapi::mkl::invalid_argument);

    constexpr std::int64_t strides_size = 4;
    const std::int64_t default_stride_d1 = default_3d_lengths[2] * default_3d_lengths[1];
    const std::int64_t default_stride_d2 = default_3d_lengths[2];
    const std::int64_t default_stride_d3 = 1;

    std::vector<std::int64_t> default_strides_value{ 0, default_stride_d1, default_stride_d2,
                                                     default_stride_d3 };
    std::vector<std::int64_t> strides_value{ 50, default_stride_d1 * 2, default_stride_d2 * 2,
                                             default_stride_d3 * 2 };

    std::vector<std::int64_t> input_strides_before_set(strides_size);
    std::vector<std::int64_t> input_strides_after_set(strides_size);

    descriptor.get_value(dft::config_param::INPUT_STRIDES, input_strides_before_set.data());
    EXPECT_EQ(default_strides_value, input_strides_before_set);
    descriptor.set_value(dft::config_param::INPUT_STRIDES, strides_value.data());
    descriptor.get_value(dft::config_param::INPUT_STRIDES, input_strides_after_set.data());
    EXPECT_EQ(strides_value, input_strides_after_set);

    std::vector<std::int64_t> output_strides_before_set(strides_size);
    std::vector<std::int64_t> output_strides_after_set(strides_size);
    descriptor.get_value(dft::config_param::OUTPUT_STRIDES, output_strides_before_set.data());
    EXPECT_EQ(default_strides_value, output_strides_before_set);
    descriptor.set_value(dft::config_param::OUTPUT_STRIDES, strides_value.data());
    descriptor.get_value(dft::config_param::OUTPUT_STRIDES, output_strides_after_set.data());
    EXPECT_EQ(strides_value, output_strides_after_set);

    //    descriptor.commit(sycl_queue); //FIXME Commiting multiple dimensions does not work.
}

/* TODO More negative scenario tests (e.g. setting the wrong value and commiting) */
template <dft::precision precision, dft::domain domain>
inline void set_and_get_values(sycl::queue& sycl_queue) {
    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    //    using Precision_Type =
    //        typename std::conditional<precision == dft::precision::SINGLE, float, double>::type;
    using Precision_Type = typename std::conditional<precision == dft::precision::SINGLE, double,
                                                     double>::type; //FIXME Waiting for fix

    {
        Precision_Type forward_scale_set_value{ 143.5 };
        Precision_Type forward_scale_before_set;
        Precision_Type forward_scale_after_set;

        descriptor.get_value(dft::config_param::FORWARD_SCALE, &forward_scale_before_set);
        EXPECT_EQ(1.0, forward_scale_before_set);
        descriptor.set_value(dft::config_param::FORWARD_SCALE, forward_scale_set_value);
        descriptor.get_value(dft::config_param::FORWARD_SCALE, &forward_scale_after_set);
        EXPECT_EQ(forward_scale_set_value, forward_scale_after_set);
    }

    {
        Precision_Type backward_scale_set_value{ 143.5 };
        Precision_Type backward_scale_before_set;
        Precision_Type backward_scale_after_set;

        descriptor.get_value(dft::config_param::BACKWARD_SCALE, &backward_scale_before_set);
        EXPECT_EQ(1.0, backward_scale_before_set);
        descriptor.set_value(dft::config_param::BACKWARD_SCALE, backward_scale_set_value);
        descriptor.get_value(dft::config_param::BACKWARD_SCALE, &backward_scale_after_set);
        EXPECT_EQ(backward_scale_set_value, backward_scale_after_set);
    }

    {
        std::int64_t n_transforms_set_value{ 12 };
        std::int64_t n_transforms_before_set;
        std::int64_t n_transforms_after_set;

        descriptor.get_value(dft::config_param::NUMBER_OF_TRANSFORMS, &n_transforms_before_set);
        EXPECT_EQ(1, n_transforms_before_set);
        descriptor.set_value(dft::config_param::NUMBER_OF_TRANSFORMS, n_transforms_set_value);
        descriptor.get_value(dft::config_param::NUMBER_OF_TRANSFORMS, &n_transforms_after_set);
        EXPECT_EQ(n_transforms_set_value, n_transforms_after_set);
    }

    {
        std::int64_t fwd_distance_set_value{ 12 };
        std::int64_t fwd_distance_before_set;
        std::int64_t fwd_distance_after_set;

        descriptor.get_value(dft::config_param::FWD_DISTANCE, &fwd_distance_before_set);
        EXPECT_EQ(1, fwd_distance_before_set);
        descriptor.set_value(dft::config_param::FWD_DISTANCE, fwd_distance_set_value);
        descriptor.get_value(dft::config_param::FWD_DISTANCE, &fwd_distance_after_set);
        EXPECT_EQ(fwd_distance_set_value, fwd_distance_after_set);
    }

    {
        std::int64_t bwd_distance_set_value{ 12 };
        std::int64_t bwd_distance_before_set;
        std::int64_t bwd_distance_after_set;

        descriptor.get_value(dft::config_param::BWD_DISTANCE, &bwd_distance_before_set);
        EXPECT_EQ(1, bwd_distance_before_set);
        descriptor.set_value(dft::config_param::BWD_DISTANCE, bwd_distance_set_value);
        descriptor.get_value(dft::config_param::BWD_DISTANCE, &bwd_distance_after_set);
        EXPECT_EQ(bwd_distance_set_value, bwd_distance_after_set);
    }

    {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(dft::config_value::INPLACE, value);

        descriptor.set_value(dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE);
        descriptor.get_value(dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(dft::config_value::NOT_INPLACE, value);

        descriptor.set_value(dft::config_param::PLACEMENT, dft::config_value::INPLACE);
        descriptor.get_value(dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(dft::config_value::INPLACE, value);
    }

    {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(dft::config_value::COMPLEX_COMPLEX, value);

        descriptor.set_value(dft::config_param::COMPLEX_STORAGE, dft::config_value::REAL_REAL);
        descriptor.get_value(dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(dft::config_value::REAL_REAL, value);

        descriptor.set_value(dft::config_param::COMPLEX_STORAGE,
                             dft::config_value::COMPLEX_COMPLEX);
        descriptor.get_value(dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(dft::config_value::COMPLEX_COMPLEX, value);
    }

    {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
        EXPECT_EQ(dft::config_value::COMPLEX_COMPLEX, value);

        descriptor.set_value(dft::config_param::CONJUGATE_EVEN_STORAGE,
                             dft::config_value::COMPLEX_COMPLEX);

        value = dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
        EXPECT_EQ(dft::config_value::COMPLEX_COMPLEX, value);
    }

    /* Unimplemented Values */
    auto real_storage = [&]() {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::REAL_STORAGE, &value);
        EXPECT_EQ(dft::config_value::REAL_REAL, value);

        descriptor.set_value(dft::config_param::REAL_STORAGE, dft::config_value::REAL_REAL);

        value = dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(dft::config_param::REAL_STORAGE, &value);
        EXPECT_EQ(dft::config_value::REAL_REAL, value);
    };
    EXPECT_THROW(real_storage(), oneapi::mkl::unimplemented);

    auto ordering = [&]() {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::ORDERING, &value);
        EXPECT_EQ(dft::config_value::ORDERED, value);

        descriptor.set_value(dft::config_param::ORDERING, dft::config_value::BACKWARD_SCRAMBLED);
        descriptor.get_value(dft::config_param::ORDERING, &value);
        EXPECT_EQ(dft::config_value::BACKWARD_SCRAMBLED, value);

        descriptor.set_value(dft::config_param::ORDERING, dft::config_value::ORDERED);
        descriptor.get_value(dft::config_param::ORDERING, &value);
        EXPECT_EQ(dft::config_value::ORDERED, value);
    };
    EXPECT_THROW(ordering(), oneapi::mkl::unimplemented);

    auto transpose = [&]() {
        bool value = true;
        descriptor.get_value(dft::config_param::TRANSPOSE, &value);
        EXPECT_EQ(false, value);

        descriptor.set_value(dft::config_param::TRANSPOSE, true);
        descriptor.get_value(dft::config_param::TRANSPOSE, &value);
        EXPECT_EQ(true, value);
    };
    EXPECT_THROW(transpose(), oneapi::mkl::unimplemented);

    auto packed_format = [&]() {
        dft::config_value value{ dft::config_value::COMMITTED }; // Initialize with invalid value
        descriptor.get_value(dft::config_param::PACKED_FORMAT, &value);
        EXPECT_EQ(dft::config_value::CCE_FORMAT, value);

        descriptor.set_value(dft::config_param::PACKED_FORMAT, dft::config_value::CCE_FORMAT);

        value = dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(dft::config_param::PACKED_FORMAT, &value);
        EXPECT_EQ(dft::config_value::CCE_FORMAT, value);
    };
    EXPECT_THROW(packed_format(), oneapi::mkl::unimplemented);
    descriptor.commit(sycl_queue);
}

template <dft::precision precision, dft::domain domain>
inline void get_readonly_values(sycl::queue& sycl_queue) {
    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    dft::domain domain_value;
    descriptor.get_value(dft::config_param::FORWARD_DOMAIN, &domain_value);
    EXPECT_EQ(domain_value, domain);

    dft::precision precision_value;
    descriptor.get_value(dft::config_param::PRECISION, &precision_value);
    EXPECT_EQ(precision_value, precision);

    std::int64_t dimension_value;
    descriptor.get_value(dft::config_param::DIMENSION, &dimension_value);
    EXPECT_EQ(dimension_value, 1);

    dft::descriptor<precision, domain> descriptor3D{ std::vector<std::int64_t>(3) };
    descriptor3D.get_value(dft::config_param::DIMENSION, &dimension_value);
    EXPECT_EQ(dimension_value, 3);

    bool commit_status;
    descriptor.get_value(dft::config_param::COMMIT_STATUS, &commit_status);
    EXPECT_EQ(commit_status, false);

    descriptor.commit(sycl_queue);
    descriptor.get_value(dft::config_param::COMMIT_STATUS, &commit_status);
    EXPECT_EQ(commit_status, true);
}

template <dft::precision precision, dft::domain domain>
inline void set_readonly_values(sycl::queue& sycl_queue) {
    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    EXPECT_THROW(descriptor.set_value(dft::config_param::FORWARD_DOMAIN, dft::domain::REAL),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(dft::config_param::FORWARD_DOMAIN, dft::domain::COMPLEX),
                 oneapi::mkl::invalid_argument);

    EXPECT_THROW(descriptor.set_value(dft::config_param::PRECISION, dft::precision::SINGLE),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(dft::config_param::PRECISION, dft::precision::DOUBLE),
                 oneapi::mkl::invalid_argument);

    std::int64_t set_dimension{ 3 };
    EXPECT_THROW(descriptor.set_value(dft::config_param::DIMENSION, set_dimension),
                 oneapi::mkl::invalid_argument);

    EXPECT_THROW(
        descriptor.set_value(dft::config_param::COMMIT_STATUS, dft::config_value::COMMITTED),
        oneapi::mkl::invalid_argument);
    EXPECT_THROW(
        descriptor.set_value(dft::config_param::COMMIT_STATUS, dft::config_value::UNCOMMITTED),
        oneapi::mkl::invalid_argument);

    descriptor.commit(sycl_queue);
}

template <dft::precision precision, dft::domain domain>
int test(sycl::device* dev) {
    sycl::queue sycl_queue(*dev, exception_handler);

    set_and_get_lengths<precision, domain>(sycl_queue);
    set_and_get_strides<precision, domain>(sycl_queue);
    set_and_get_values<precision, domain>(sycl_queue);
    get_readonly_values<precision, domain>(sycl_queue);
    set_readonly_values<precision, domain>(sycl_queue);

    return !::testing::Test::HasFailure();
}

class DescriptorTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(DescriptorTests, DescriptorTests) {
    EXPECT_TRUEORSKIP((test<dft::precision::SINGLE, dft::domain::REAL>(std::get<0>(GetParam()))));
    EXPECT_TRUEORSKIP((test<dft::precision::DOUBLE, dft::domain::REAL>(std::get<0>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<dft::precision::SINGLE, dft::domain::COMPLEX>(std::get<0>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<dft::precision::DOUBLE, dft::domain::COMPLEX>(std::get<0>(GetParam()))));
}

INSTANTIATE_TEST_SUITE_P(DescriptorTestSuite, DescriptorTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
