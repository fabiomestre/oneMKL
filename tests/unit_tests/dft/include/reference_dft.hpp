#ifndef ONEMKL_REFERENCE_DFT_HPP
#define ONEMKL_REFERENCE_DFT_HPP

#include <sycl/sycl.hpp>

using namespace oneapi::mkl;
using namespace sycl;

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template <typename TypeIn, typename TypeOut>
bool reference_forward_dft(sycl::queue &queue, sycl::buffer<TypeIn, 1> &in,
                           sycl::buffer<TypeOut, 1> &out) {
    static_assert(is_complex_t<TypeOut>());

    double TWOPI = 2.0 * std::atan(1.0) * 4.0;

    std::vector<TypeIn> in_host(in.size());
    std::vector<TypeOut> out_host(out.size());
    copy_to_host(queue, in, in_host);
    copy_to_host(queue, out, out_host);

    std::complex<double> out_temp; /* Do the calculations using double */
    size_t N = out_host.size();
    for (int k = 0; k < N; k++) {
        out_host[k] = 0;
        out_temp = 0;
        for (int n = 0; n < N; n++) {
            if constexpr (is_complex_t<TypeIn>()) {
                out_temp += static_cast<std::complex<double>>(in_host[n]) *
                            std::complex<double>{ std::cos(n * k * TWOPI / N),
                                                  -std::sin(n * k * TWOPI / N) };
            }
            else {
                out_temp += std::complex<double>{
                    static_cast<double>(in_host[n]) * std::cos(n * k * TWOPI / N),
                    static_cast<double>(-in_host[n]) * std::sin(n * k * TWOPI / N)
                };
            }
        }
        out_host[k] = static_cast<TypeOut>(out_temp);
    }

    copy_to_device(queue, out_host, out);

    return true;
}

//template <typename TypeIn, typename TypeOut>
//bool reference_forward_dft(sycl::queue &queue, sycl::buffer<TypeIn, 1> &in,
//                           sycl::buffer<TypeOut, 1> &out) {
//    static_assert(is_complex_t<TypeOut>());
//
//    using PrecisionType = typename TypeOut::value_type;
////    using PrecisionType =
////        typename std::conditional<precision == dft::precision::SINGLE, float, double>::type;
//    PrecisionType TWOPI = static_cast<PrecisionType>(2.0 * std::atan(1.0) * 4.0);
//
//    if constexpr (!std::is_same_v<PrecisionType, double>) {
//        queue.submit([&](handler &cgh) {
//            auto a_in = in.get_access(cgh);
//            auto a_out = out.get_access(cgh);
//            cgh.parallel_for(1, [=](id<1> i) {
//                size_t N = a_out.size();
//                for (int k = 0; k < N; k++) {
//                    a_out[k] = 0;
//                    for (int n = 0; n < N; n++) {
//                        if constexpr (is_complex_t<TypeIn>()) {
//                            a_out[k] +=
//                                a_in[n] * TypeOut{ sycl::cos<PrecisionType>(n * k * TWOPI / N),
//                                                   -sycl::sin<PrecisionType>(n * k * TWOPI / N) };
//                        }
//                        else {
//                            a_out[k] +=
//                                TypeOut{ a_in[n] * sycl::cos<PrecisionType>(n * k * TWOPI / N),
//                                         -a_in[n] * sycl::sin<PrecisionType>(n * k * TWOPI / N) };
//                        }
//                    }
//                }
//
//                //            sycl::ext::oneapi::experimental::printf("Hello World %f\n", TWOPI);
//            });
//        });
//
//        queue.wait();
//    }
//    else {
//        return false;
//    }
//    return true;
//}

#endif //ONEMKL_REFERENCE_DFT_HPP
