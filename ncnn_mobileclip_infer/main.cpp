#include <iostream>
#include <cmath>
#include "SimpleTokenizer.h"
#include "MobileClip.h"
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // 检查参数数量是否足够
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> \"<tag1>\" \"<tag2>\" ..." << std::endl;
        return -1;
    }

    SimpleTokenizer tokenizer("bpe_simple_vocab_16e6.txt");
    MobileClip model("mobileclip_s2");

    // 从命令行参数读取图片路径
    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path);

    // 检查图片是否成功加载
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << image_path << std::endl;
        return -1;
    }

    // 从命令行参数读取标签
    std::vector<std::string> rtext;
    for (int i = 2; i < argc; ++i) {
        rtext.push_back(argv[i]);
    }

    auto text = tokenizer(rtext);
    auto image_features = model.encode_image(image);

    std::vector<cv::Mat> text_features;
    for (const auto &t : text) {
        text_features.push_back(model.encode_text(t));
    }

    std::vector<double> similarities;
    for (const auto &text_feature : text_features) {
        double similarity = image_features.dot(text_feature);
        similarities.push_back(similarity * 100);
    }

    // softmax
    double sum = 0.0;
    for (const auto &sim : similarities) {
        sum += exp(sim);
    }
    for (auto &sim : similarities) {
        sim = exp(sim) / sum;
    }
    for (size_t i = 0; i < text.size(); ++i) {
        std::cout << "Text: " << rtext[i] << ", Prob: " << similarities[i] << std::endl;
    }

    return 0;
}