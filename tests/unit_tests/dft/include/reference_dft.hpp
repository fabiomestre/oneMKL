#ifndef ONEMKL_REFERENCE_DFT_HPP
#define ONEMKL_REFERENCE_DFT_HPP

#include <sycl/sycl.hpp>

using namespace oneapi::mkl;
using namespace sycl;

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

//template <typename TypeIn, typename TypeOut, dft::precision precision, dft::domain domain>
//void reference_backward_dft(sycl::queue &queue, dft::descriptor<precision, domain> &descriptor,
//                           sycl::buffer<TypeIn, 1> &in, sycl::buffer<TypeOut, 1> &out) {
//    static_assert(is_complex_t<TypeIn>());
//
//
//
//}

template <typename TypeIn, typename TypeOut, dft::precision precision, dft::domain domain>
void reference_forward_dft(sycl::queue &queue, dft::descriptor<precision, domain> &descriptor,
                           sycl::buffer<TypeIn, 1> &in, sycl::buffer<TypeOut, 1> &out) {
    static_assert(is_complex_t<TypeOut>());

    using outputPrecision = typename TypeOut::value_type;
    const outputPrecision TWOPI{ 2.0 * std::atan(1.0) * 4.0 };

    queue.submit([&](handler &cgh) {
        auto a_in = in.template get_access<access_mode::read>(cgh);
        auto a_out = out.template get_access<access_mode::write>(cgh);
        cgh.parallel_for(1, [=](id<1> i) {
            size_t N = a_out.size();

            for (int k = 0; k < N; k++) {
                a_out[k] = 0;
                for (int n = 0; n < N; n++) {
                    if constexpr (is_complex_t<TypeIn>()) {
                        sycl::ext::oneapi::experimental::printf("%f %f\n", a_in[n].real(),
                                                                a_in[n].imag());
                        a_out[k] += a_in[n] * TypeOut{ sycl::cos(n * k * TWOPI / N),
                                                       -sycl::sin(n * k * TWOPI / N) };
                    }
                    else {
                        a_out[k] += TypeOut{ a_in[n] * sycl::cos(n * k * TWOPI / N),
                                             -a_in[n] * sycl::sin(n * k * TWOPI / N) };
                    }
                }
            }

            sycl::ext::oneapi::experimental::printf("Hello World %f\n", TWOPI);
    });
});

queue.wait();
}

//template <dft::precision precision, dft::domain domain> //FIXME
//void reference_dft(sycl::device *dev) {
//    if constexpr (precision == dft::precision::SINGLE) {
//        if constexpr (domain == dft::domain::REAL) {
//            reference_dft<float>(dev);
//        }
//        else {
//            reference_dft<std::complex<float>>(dev);
//        }
//    }
//    else {
//        if constexpr (domain == dft::domain::REAL) {
//            reference_dft<double>(dev);
//        }
//        else {
//            reference_dft<std::complex<double>>(dev);
//        }
//    }
//}

#endif //ONEMKL_REFERENCE_DFT_HPP
