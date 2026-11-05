cmake_minimum_required(VERSION 3.10)
project(ncnn_mobileclip_vulkan)
set(CMAKE_CXX_STANDARD 20)

find_package(ncnn REQUIRED)
find_package(Vulkan REQUIRED)

# 查找 glslangValidator
find_program(GLSLANG_VALIDATOR glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found!")
endif()

# 编译 shader
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv
    COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.comp -o ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.comp
)
add_custom_target(shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/preprocess.spv)

add_executable(ncnn_mobileclip_vulkan
    main_vulkan.cpp
)
add_dependencies(ncnn_mobileclip_vulkan shaders)

target_link_libraries(ncnn_mobileclip_vulkan ncnn Vulkan::Vulkan)
