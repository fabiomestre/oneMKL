#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

set(LIB_NAME onemkl_blas_mklcpu)
set(LIB_OBJ ${LIB_NAME}_obj)

set(USE_DPCPP_API ON)
find_package(MKL REQUIRED)
set(SOURCES
  mklcpu_level1.cpp mklcpu_level2.cpp mklcpu_level3.cpp mklcpu_batch.cpp mklcpu_extensions.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>: mklcpu_wrappers.cpp>)
add_library(${LIB_NAME})
add_library(${LIB_OBJ} OBJECT ${SOURCES})
if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

target_include_directories(${LIB_OBJ}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
          ${PROJECT_SOURCE_DIR}/src
          ${PROJECT_SOURCE_DIR}/src/include
          ${CMAKE_BINARY_DIR}/bin
          ${MKL_INCLUDE}
)

target_compile_options(${LIB_OBJ} PRIVATE ${ONEMKL_BUILD_COPT} ${MKL_COPT})

target_link_libraries(${LIB_OBJ} PUBLIC ONEMKL::SYCL::SYCL ${MKL_LINK_C} ${MKL_LINK_SYCL})

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${LIB_NAME} PUBLIC ${LIB_OBJ})

#Set libraries as not transitive for dynamic
if(BUILD_SHARED_LIBS)
  set_target_properties(${LIB_NAME} PROPERTIES
    INTERFACE_LINK_LIBRARIES ONEMKL::SYCL::SYCL
  )
endif()

# Add major version to the library
set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# Add the library to install package
install(TARGETS ${LIB_OBJ} EXPORT oneMKLTargets)
install(TARGETS ${LIB_NAME} EXPORT oneMKLTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
