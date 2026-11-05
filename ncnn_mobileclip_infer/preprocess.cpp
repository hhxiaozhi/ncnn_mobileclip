#include <vulkan/vulkan.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d @ %s:%d\n", (int)_r, __FILE__, __LINE__); exit(1); } } while (0)

static const uint32_t OUT_W = 224, OUT_H = 224, OUT_C = 3, NUM_CROPS = 5;
static const uint32_t WG_X = 16, WG_Y = 16;
static const size_t OUT_FLOATS = (size_t)NUM_CROPS * OUT_C * OUT_W * OUT_H;
static const size_t OUT_BYTES = OUT_FLOATS * sizeof(float);

struct PushConstants {
    float crop[NUM_CROPS][4];
    float mean[4];
    float invStd[4];
    float reserved[4];
};
static_assert(sizeof(PushConstants) == 128, "push constants must be 128B");

static uint32_t findMemoryType(VkPhysicalDevice phy, uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phy, &mp);
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if (!(typeBits & (1u << i))) continue;
            if ((f & required) != required) continue;
            if (pass == 0 && preferred != 0 && (f & preferred) == 0) continue;
            return i;
        }
    }
    throw std::runtime_error("no suitable memory type");
}

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

static GpuBuffer createBuffer(VkDevice dev, VkPhysicalDevice phy, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) {
    GpuBuffer b;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(phy, req.memoryTypeBits, required, preferred);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(dev, b.memory, 0, size, 0, &b.mapped));
    return b;
}

static void destroyBuffer(VkDevice dev, GpuBuffer& b) {
    if (b.mapped) vkUnmapMemory(dev, b.memory);
    vkDestroyBuffer(dev, b.buffer, nullptr);
    vkFreeMemory(dev, b.memory, nullptr);
}

static std::vector<uint32_t> loadSpv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz == 0 || (sz & 3) != 0) { fprintf(stderr, "invalid SPIR-V\n"); exit(1); }
    std::vector<uint32_t> code(sz / 4);
    f.read(reinterpret_cast<char*>(code.data()), (std::streamsize)sz);
    return code;
}

static uint32_t pickComputeQueueFamily(VkPhysicalDevice phy) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &n, q.data());
    uint32_t fallback = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (!(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) return i;
            if (fallback == 0xFFFFFFFFu) fallback = i;
        }
    }
    if (fallback == 0xFFFFFFFFu) throw std::runtime_error("no compute queue");
    return fallback;
}

static int run(int argc, char** argv) {
    const std::string jpgPath = argc > 1 ? argv[1] : "input.jpg";
    const std::string spvPath = argc > 2 ? argv[2] : "preprocess.spv";
    const std::string outPath = argc > 3 ? argv[3] : "crops.bin";

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

    auto tA = clk::now();
    cv::Mat bgr = cv::imread(jpgPath, cv::IMREAD_COLOR);
    if (bgr.empty()) { fprintf(stderr, "failed to read %s\n", jpgPath.c_str()); return 1; }
    const uint32_t W = (uint32_t)bgr.cols, H = (uint32_t)bgr.rows;

    std::vector<uint8_t> rgba((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; y++) {
        const uint8_t* s = bgr.ptr<uint8_t>(y);
        uint8_t* d = rgba.data() + (size_t)y * W * 4;
        for (uint32_t x = 0; x < W; x++) {
            d[4*x+0] = s[3*x+2];
            d[4*x+1] = s[3*x+1];
            d[4*x+2] = s[3*x+0];
            d[4*x+3] = 255;
        }
    }
    printf("[host] decode + repack %ux%u: %.1f ms\n", W, H, ms(clk::now() - tA));

    struct Rect { uint32_t x, y, w, h; };
    const Rect rects[NUM_CROPS] = {
        {0, 0, W, H},
        {0, 0, W/2, H/2},
        {W/2, 0, W - W/2, H/2},
        {0, H/2, W/2, H - H/2},
        {W/2, H/2, W - W/2, H - H/2},
    };
    PushConstants pc{};
    for (uint32_t i = 0; i < NUM_CROPS; i++) {
        pc.crop[i][0] = (float)rects[i].x / (float)W;
        pc.crop[i][1] = (float)rects[i].y / (float)H;
        pc.crop[i][2] = (float)rects[i].w / ((float)W * (float)OUT_W);
        pc.crop[i][3] = (float)rects[i].h / ((float)H * (float)OUT_H);
    }
    const float imMean[3] = {0.485f, 0.456f, 0.406f};
    const float imStd[3]  = {0.229f, 0.224f, 0.225f};
    for (uint32_t i = 0; i < 3; i++) {
        pc.mean[i] = imMean[i];
        pc.invStd[i] = 1.0f / imStd[i];
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gpu-preproc";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nPhy = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &nPhy, nullptr));
    std::vector<VkPhysicalDevice> phys(nPhy);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &nPhy, phys.data()));

    VkPhysicalDevice phy = VK_NULL_HANDLE;
    for (VkPhysicalDevice p : phys) {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n, nullptr);
        std::vector<VkQueueFamilyProperties> q(n);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n, q.data());
        for (const VkQueueFamilyProperties& e : q)
            if (e.queueFlags & VK_QUEUE_COMPUTE_BIT) { phy = p; break; }
        if (phy) break;
    }
    if (!phy) { fprintf(stderr, "no compute device\n"); return 1; }

    VkPhysicalDeviceProperties pdp{};
    vkGetPhysicalDeviceProperties(phy, &pdp);
    printf("Device: %s\n", pdp.deviceName);
    printf("Max push constants: %u\n", pdp.limits.maxPushConstantsSize);

    const uint32_t qf = pickComputeQueueFamily(phy);

    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(phy, VK_FORMAT_R8G8B8A8_UNORM, &fp);
    if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
        fprintf(stderr, "R8G8B8A8_UNORM linear filter unsupported\n");
        return 1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDevice(phy, &dci, nullptr, &dev));
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qf, 0, &queue);

    VkImage image = VK_NULL_HANDLE;
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.extent = {W, H, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(dev, &ci, nullptr, &image));
    }
    VkDeviceMemory imageMem = VK_NULL_HANDLE;
    {
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(phy, req.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &imageMem));
        VK_CHECK(vkBindImageMemory(dev, image, imageMem, 0));
    }
    VkImageView view = VK_NULL_HANDLE;
    {
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &ci, nullptr, &view));
    }

    GpuBuffer staging = createBuffer(dev, phy, (VkDeviceSize)rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(staging.mapped, rgba.data(), rgba.size());

    GpuBuffer outBuf = createBuffer(dev, phy, (VkDeviceSize)OUT_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkSampler sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.mipLodBias = 0.0f;
        ci.anisotropyEnable = VK_FALSE;
        ci.maxAnisotropy = 1.0f;
        ci.compareEnable = VK_FALSE;
        ci.compareOp = VK_COMPARE_OP_ALWAYS;
        ci.minLod = 0.0f;
        ci.maxLod = 0.0f;
        ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        ci.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(dev, &ci, nullptr, &sampler));
    }

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 2;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &dsl));
    }
    VkDescriptorPool pool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize sz[2] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets = 1;
        ci.poolSizeCount = 2;
        ci.pPoolSizes = sz;
        VK_CHECK(vkCreateDescriptorPool(dev, &ci, nullptr, &pool));
    }
    VkDescriptorSet ds = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dsl;
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &ds));
    }
    {
        VkDescriptorImageInfo dii{};
        dii.sampler = sampler;
        dii.imageView = view;
        dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = outBuf.buffer;
        dbi.offset = 0;
        dbi.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = ds;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &dii;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = ds;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }

    std::vector<uint32_t> spv = loadSpv(spvPath);
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &shaderModule));
    }
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &dsl;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(dev, &ci, nullptr, &pipelineLayout));
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = shaderModule;
        ci.stage.pName = "main";
        ci.layout = pipelineLayout;
        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    }

    VkCommandPool cpool = VK_NULL_HANDLE;
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = qf;
        VK_CHECK(vkCreateCommandPool(dev, &ci, nullptr, &cpool));
    }
    VkFence fence = VK_NULL_HANDLE;
    {
        VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev, &ci, nullptr, &fence));
    }

    {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cpool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));

        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = image;
        bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);

        VkBufferImageCopy cpy{};
        cpy.bufferOffset = 0;
        cpy.bufferRowLength = 0;
        cpy.bufferImageHeight = 0;
        cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cpy.imageOffset = {0, 0, 0};
        cpy.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cb, staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cpy);

        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
        VK_CHECK(vkEndCommandBuffer(cb));

        auto t0 = clk::now();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
        printf("[gpu] upload %.1f MB -> image: %.2f ms\n", rgba.size() / 1048576.0, ms(clk::now() - t0));
        vkFreeCommandBuffers(dev, cpool, 1, &cb);
    }

    VkCommandBuffer ccb = VK_NULL_HANDLE;
    {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cpool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &ccb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VK_CHECK(vkBeginCommandBuffer(ccb, &bi));
        vkCmdBindPipeline(ccb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(ccb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(ccb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDispatch(ccb, OUT_W / WG_X, OUT_H / WG_Y, NUM_CROPS);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(ccb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(ccb));
    }

    auto submitAndWait = [&]() {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &ccb;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    };

    for (int i = 0; i < 10; i++) submitAndWait();

    auto t0 = clk::now();
    submitAndWait();
    printf("[gpu] preprocess single: %.3f ms\n", ms(clk::now() - t0));

    const int N = 100;
    std::vector<VkCommandBuffer> many(N, ccb);
    auto t1 = clk::now();
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = (uint32_t)N;
        si.pCommandBuffers = many.data();
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    }
    printf("[gpu] preprocess %d batch avg: %.4f ms\n", N, ms(clk::now() - t1) / N);

    const float* out = (const float*)outBuf.mapped;
    {
        std::ofstream f(outPath, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot write %s\n", outPath.c_str()); return 1; }
        f.write(reinterpret_cast<const char*>(out), (std::streamsize)OUT_BYTES);
    }
    printf("output %s: %zu bytes / %zu floats = 5x3x224x224 fp32 (CHW)\n", outPath.c_str(), OUT_BYTES, OUT_FLOATS);

    for (uint32_t c = 0; c < NUM_CROPS; c++) {
        for (uint32_t ch = 0; ch < OUT_C; ch++) {
            const float* p = out + ((size_t)c * OUT_C + ch) * OUT_W * OUT_H;
            const size_t n = (size_t)OUT_W * OUT_H;
            double sum = 0.0, sum2 = 0.0;
            for (size_t i = 0; i < n; i++) { sum += p[i]; sum2 += (double)p[i] * p[i]; }
            double m = sum / (double)n;
            double sd = std::sqrt(std::max(0.0, sum2 / (double)n - m * m));
            printf("  crop%u ch%u: mean %+.4f  std %.4f\n", c, ch, m, sd);
        }
    }

    vkDeviceWaitIdle(dev);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
    vkDestroyShaderModule(dev, shaderModule, nullptr);
    vkDestroySampler(dev, sampler, nullptr);
    vkDestroyImageView(dev, view, nullptr);
    vkDestroyImage(dev, image, nullptr);
    vkFreeMemory(dev, imageMem, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    destroyBuffer(dev, staging);
    destroyBuffer(dev, outBuf);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
