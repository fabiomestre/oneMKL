#ifndef ONEMKL_COMPUTE_IN_REAL_REAL_HPP
#define ONEMKL_COMPUTE_IN_REAL_REAL_HPP

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

/* Test is not implemented because currently there are no available implementation that support it */
template <dft::precision precision, dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_USM() {
    if (!init()) {
        return test_skipped;
    }

    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };
    try {
        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor.commit(sycl_queue);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    return test_skipped;
}

/* Test is not implemented because currently there are no available implementation that support it */
template <dft::precision precision, dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_real_real_buffer() {
    if (!init()) {
        return test_skipped;
    }

    dft::descriptor<precision, domain> descriptor{ default_1d_lengths };
    try {
        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor.commit(sycl_queue);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }
    //
    //    sycl::buffer<PrecisionType, 1> inout_re_dev{ sycl::range<1>(default_1d_lengths) };
    //    sycl::buffer<PrecisionType, 1> inout_im_dev{ sycl::range<1>(default_1d_lengths) };
    //
    //    copy_to_device(sycl_queue, input_re, inout_re_dev);
    //    copy_to_device(sycl_queue, input_im, inout_im_dev);
    //
    //    try {
    //        dft::compute_forward<std::remove_reference_t<decltype(descriptor)>, PrecisionType>(
    //            descriptor, inout_re_dev, inout_im_dev);
    //    }
    //    catch (oneapi::mkl::unimplemented &e) {
    //        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
    //        return test_skipped;
    //    }
    //
    //    std::vector<PrecisionType> out_host_re(default_1d_lengths);
    //    std::vector<PrecisionType> out_host_im(default_1d_lengths);
    //    std::vector<OutputType> out_host(default_1d_lengths);
    //
    //    copy_to_host(sycl_queue, inout_re_dev, out_host_re);
    //    copy_to_host(sycl_queue, inout_re_dev, out_host_im);
    //
    //    for (int i = 0; i < out_host_re.size(); ++i) {
    //        out_host[i] = OutputType{ out_host_re[i], out_host_im[i] };
    //    }
    //
    //    EXPECT_TRUE(check_equal_vector(out_host.data(), out_host_ref.data(), out_host.size(), 1,
    //                                   error_margin, std::cout));
    //
    //    std::cout << "output ref" << std::endl;
    //    print_dft_output(out_host_ref);
    //    std::cout << "output dft" << std::endl;
    //    print_dft_output(out_host);

    return test_skipped;
}

#endif //ONEMKL_COMPUTE_IN_REAL_REAL_HPP
