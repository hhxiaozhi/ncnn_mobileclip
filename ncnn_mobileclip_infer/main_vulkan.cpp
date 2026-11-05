#include <vulkan/vulkan.h>
#include <ncnn/net.h>
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    std::cout << "ncnn_mobileclip_vulkan skeleton" << std::endl;

    // NCNN 初始化（纯 CPU，不开 Vulkan 计算）
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;

    // Vulkan 实例创建（仅验证环境）
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    VkInstance inst;
    if (vkCreateInstance(&instInfo, nullptr, &inst) == VK_SUCCESS) {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(inst, &count, nullptr);
        std::cout << "Vulkan devices: " << count << std::endl;
        if (count > 0) {
            std::vector<VkPhysicalDevice> devs(count);
            vkEnumeratePhysicalDevices(inst, &count, devs.data());
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devs[0], &props);
            std::cout << "Device: " << props.deviceName << std::endl;
        }
        vkDestroyInstance(inst, nullptr);
    } else {
        std::cerr << "Vulkan instance creation failed" << std::endl;
        return 1;
    }

    std::cout << "Compilation Successful!" << std::endl;
    return 0;
}
