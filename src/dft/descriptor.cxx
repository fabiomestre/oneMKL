/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/mkl/detail/exceptions.hpp"
#include "oneapi/mkl/dft/descriptor.hpp"

namespace oneapi {
namespace mkl {
namespace dft {
namespace detail {

template <precision prec, domain dom>
void descriptor<prec, dom>::set_value(config_param param, ...) {
    va_list vl;
    va_start(vl, param);
    switch (param) {
        case config_param::FORWARD_DOMAIN:
            throw mkl::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::DIMENSION:
            throw mkl::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::LENGTHS: {
            if (values_.rank == 1) {
                values_.dimensions = std::vector<std::int64_t>{ va_arg(vl, std::int64_t) };
            }
            else {
                auto ptr = va_arg(vl, std::int64_t*);
                std::copy(ptr, ptr + values_.rank + 1, values_.dimensions.begin());
            }
            break;
        }
        case config_param::PRECISION:
            throw mkl::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::INPUT_STRIDES:
        case config_param::OUTPUT_STRIDES: {
            auto strides = va_arg(vl, std::int64_t*);
            if (strides == nullptr) {
                throw mkl::invalid_argument("DFT", "set_value", "Invalid config_param argument.");
            } else if (param == config_param::INPUT_STRIDES) {
                std::copy(strides, strides + values_.rank + 1, values_.input_strides.begin());
            } else if (param == config_param::OUTPUT_STRIDES){ 
                std::copy(strides, strides + values_.rank + 1, values_.output_strides.begin());
            }
            break;
        }
        case config_param::FORWARD_SCALE: values_.fwd_scale = va_arg(vl, double); break;
        case config_param::BACKWARD_SCALE: values_.bwd_scale = va_arg(vl, double); break;
        case config_param::NUMBER_OF_TRANSFORMS:
            values_.number_of_transforms = va_arg(vl, int64_t);
            break;
        case config_param::FWD_DISTANCE: values_.fwd_dist = va_arg(vl, std::int64_t); break;
        case config_param::BWD_DISTANCE: values_.bwd_dist = va_arg(vl, std::int64_t); break;
        case config_param::PLACEMENT: values_.placement = va_arg(vl, config_value); break;
        case config_param::COMPLEX_STORAGE:
            values_.complex_storage = va_arg(vl, config_value);
            break;
        case config_param::REAL_STORAGE:
            throw mkl::unimplemented("DFT", "set_value", "Real storage not implemented.");
            break;
        case config_param::CONJUGATE_EVEN_STORAGE:
            values_.conj_even_storage = va_arg(vl, config_value);
            break;
        case config_param::ORDERING:
            throw mkl::unimplemented("DFT", "set_value", "Ordering not implemented.");
            break;
        case config_param::TRANSPOSE:
            throw mkl::unimplemented("DFT", "set_value", "Transpose not implemented.");
            break;
        case config_param::PACKED_FORMAT:
            throw mkl::unimplemented("DFT", "set_value", "Packed format not implemented.");
            break;
        case config_param::COMMIT_STATUS:
            throw mkl::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        default: throw mkl::invalid_argument("DFT", "set_value", "Invalid config_param argument.");
    }
    va_end(vl);
}

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimensions)
        : dimensions_(std::move(dimensions)),
          rank_(dimensions.size()) {
    // Compute default strides.
    std::vector<std::int64_t> defaultStrides(rank_, 1);
    for (int i = rank_ - 1; i < 0; --i) {
        defaultStrides[i] = defaultStrides[i - 1] * dimensions_[i];
    }
    defaultStrides[0] = 0;
    values_.input_strides = defaultStrides;
    values_.output_strides = std::move(defaultStrides);
    values_.bwd_scale = 1.0;
    values_.fwd_scale = 1.0;
    values_.number_of_transforms = 1;
    values_.fwd_dist = 1;
    values_.bwd_dist = 1;
    values_.placement = config_value::INPLACE;
    values_.complex_storage = config_value::COMPLEX_COMPLEX;
    values_.conj_even_storage = config_value::COMPLEX_COMPLEX;
    values_.dimensions = dimensions_;
    values_.rank = rank_;
    values_.domain = dom;
    values_.precision = prec;
}

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::int64_t length)
        : descriptor<prec, dom>(std::vector<std::int64_t>{ length }) {}

template <precision prec, domain dom>
descriptor<prec, dom>::~descriptor() {}

template <precision prec, domain dom>
void descriptor<prec, dom>::get_value(config_param param, ...) {
    int err = 0;
    va_list vl;
    va_start(vl, param);
    switch (param) {
        case config_param::FORWARD_DOMAIN: *va_arg(vl, dft::domain*) = dom; break;
        case config_param::DIMENSION: *va_arg(vl, std::int64_t*) = values_.rank; break;
        case config_param::LENGTHS:
            std::copy(values_.dimensions.begin(), values_.dimensions.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::PRECISION: *va_arg(vl, dft::precision*) = prec; break;
        case config_param::FORWARD_SCALE: *va_arg(vl, double*) = values_.fwd_scale; break;
        case config_param::BACKWARD_SCALE: *va_arg(vl, double*) = values_.bwd_scale; break;
        case config_param::NUMBER_OF_TRANSFORMS:
            *va_arg(vl, std::int64_t*) = values_.number_of_transforms;
            break;
        case config_param::COMPLEX_STORAGE:
            *va_arg(vl, config_value*) = values_.complex_storage;
            break;
        case config_param::REAL_STORAGE:
            throw mkl::unimplemented("DFT", "get_value", "Real storage not implemented.");
            break;
        case config_param::CONJUGATE_EVEN_STORAGE:
            *va_arg(vl, config_value*) = values_.conj_even_storage;
            break;
        case config_param::PLACEMENT: *va_arg(vl, config_value*) = values_.placement; break;
        case config_param::INPUT_STRIDES:
            std::copy(values_.input_strides.begin(), values_.input_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::OUTPUT_STRIDES:
            std::copy(values_.output_strides.begin(), values_.output_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::FWD_DISTANCE: *va_arg(vl, std::int64_t*) = values_.fwd_dist; break;
        case config_param::BWD_DISTANCE: *va_arg(vl, std::int64_t*) = values_.bwd_dist; break;
        case config_param::WORKSPACE: *va_arg(vl, config_value*) = values_.workspace; break;
        case config_param::ORDERING:
            throw mkl::unimplemented("DFT", "get_value", "Ordering not implemented.");
            break;
        case config_param::TRANSPOSE:
            throw mkl::unimplemented("DFT", "get_value", "Real storage not implemented.");
            break;
        case config_param::PACKED_FORMAT:
            throw mkl::unimplemented("DFT", "get_value", "Real storage not implemented.");
            break;
        case config_param::COMMIT_STATUS: *va_arg(vl, bool*) = static_cast<bool>(pimpl_); break;
        default: throw mkl::invalid_argument("DFT", "get_value", "Invalid config_param argument.");
    }
    va_end(vl);
}

template class descriptor<precision::SINGLE, domain::COMPLEX>;
template class descriptor<precision::SINGLE, domain::REAL>;
template class descriptor<precision::DOUBLE, domain::COMPLEX>;
template class descriptor<precision::DOUBLE, domain::REAL>;

} //namespace detail
} //namespace dft
} //namespace mkl
} //namespace oneapi
