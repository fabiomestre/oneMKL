#ifndef ONEMKL_COMPUTE_OUT_HPP
#define ONEMKL_COMPUTE_OUT_HPP

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

/* Note: Domain Real is not implemented */
template <dft::precision precision, dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_buffer() {

    if (!init()) {
        return test_skipped;
    }

    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor.commit(sycl_queue);

    std::vector<OutputType> out_host(default_1d_lengths);
    std::vector<InputType> out_host_back(default_1d_lengths);
    sycl::buffer<InputType, 1> in_dev{ sycl::range<1>(default_1d_lengths) };
    sycl::buffer<OutputType, 1> out_dev{ sycl::range<1>(default_1d_lengths) };
    sycl::buffer<InputType, 1> out_back_dev{ sycl::range<1>(default_1d_lengths) };

    copy_to_device(sycl_queue, input, in_dev);

    try {
        dft::compute_forward<std::remove_reference_t<decltype(descriptor)>, InputType, OutputType>(
            descriptor, in_dev, out_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, out_dev, out_host);

    EXPECT_TRUE(check_equal_vector(out_host.data(), out_host_ref.data(), out_host.size(), 1,
                                   error_margin, std::cout));


    dft::descriptor<precision, domain> descriptor_back{ default_1d_lengths };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(dft::config_param::BACKWARD_SCALE, (1.0 / default_1d_lengths));
    descriptor_back.commit(sycl_queue);

    try {
        dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>, OutputType, InputType>(
            descriptor_back, out_dev, out_back_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, out_back_dev, out_host_back);

    EXPECT_TRUE(check_equal_vector(out_host_back.data(), input.data(), input.size(), 1, error_margin,
                                   std::cout));

    return !::testing::Test::HasFailure();
}

template <dft::precision precision, dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_USM() {
    if (!init()) {
        return test_skipped;
    }

    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor.commit(sycl_queue);

    std::vector<OutputType> out_host(default_1d_lengths);
    std::vector<InputType> out_host_back(default_1d_lengths);

    auto ua_input = usm_allocator<InputType, usm::alloc::shared, 64>(cxt, *dev);
    auto ua_output = usm_allocator<OutputType, usm::alloc::shared, 64>(cxt, *dev);

    std::vector<InputType, decltype(ua_input)> in(default_1d_lengths,ua_input);
    std::vector<OutputType, decltype(ua_output)> out(default_1d_lengths,ua_output);

    std::copy(input.begin(), input.end(), in.begin());

    try {
        std::vector<event> dependencies;
        event done = dft::compute_forward<std::remove_reference_t<decltype(descriptor)>, InputType, OutputType>(
            descriptor, in.data(), out.data(), dependencies);
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(check_equal_vector(out.data(), out_host_ref.data(), out.size(), 1,
                                   error_margin, std::cout));


    dft::descriptor<precision, domain> descriptor_back{ default_1d_lengths };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(dft::config_param::BACKWARD_SCALE, (1.0 / default_1d_lengths));
    descriptor_back.commit(sycl_queue);

    std::vector<InputType, decltype(ua_input)> out_back(default_1d_lengths,ua_input);

    try {
        std::vector<event> dependencies;
        event done = dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>, OutputType, InputType>(
            descriptor_back, out.data(), out_back.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(check_equal_vector(out_back.data(), input.data(), input.size(), 1, error_margin,
                                   std::cout));

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_OUT_HPP
