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

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "../../blas/include/test_common.hpp"
#include "test_common.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"
#include "oneapi/mkl/dft/types.hpp"
#include "reference_dft.hpp"
#include "compute_tester.hpp"
#include <gtest/gtest.h>

namespace {

class ComputeTests : public ::testing::TestWithParam<sycl::device *> {};

/* test_in_place_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

/* test_in_place_real_real_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

/* test_out_of_place_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

/* test_out_of_place_real_real_buffer */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

/* test_in_place_USM */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

/* test_in_place_real_real_USM */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

/* test_out_of_place_USM */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

/* test_out_of_place_real_real_USM */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::REAL>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::SINGLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<dft::precision::DOUBLE, dft::domain::COMPLEX>{ GetParam() };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
