#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// 装甲板结构体定义
typedef struct Armor
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float score;
    int label;
} armor;

// 定义颜色数组，用于可视化不同类别
const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0),    // 蓝色
    cv::Scalar(0, 0, 255),    // 红色
    cv::Scalar(128, 128, 128), // 灰色
    cv::Scalar(255, 100, 0),   // 亮蓝色
    cv::Scalar(0, 100, 255),   // 亮红色
    cv::Scalar(200, 200, 200), // 浅灰色
    cv::Scalar(180, 105, 255), // 蓝紫色
    cv::Scalar(105, 180, 255), // 红紫色
    cv::Scalar(170, 170, 170), // 中灰色
    cv::Scalar(80, 127, 255),  // 浅蓝色
    cv::Scalar(127, 80, 255),  // 浅红色
    cv::Scalar(225, 225, 225)  // 亮灰色
};

// 类别名称映射
const std::vector<std::string> CLASS_NAMES = {
    "armor_sentry_blue",      // 0
    "armor_sentry_red",       // 1
    "armor_sentry_none",      // 2
    "armor_hero_blue",        // 3
    "armor_hero_red",         // 4
    "armor_hero_none",        // 5
    "armor_engine_blue",      // 6
    "armor_engine_red",       // 7
    "armor_engine_none",      // 8
    "armor_infantry_3_blue",  // 9
    "armor_infantry_3_red",   // 10
    "armor_infantry_3_none",  // 11
    "armor_infantry_4_blue",  // 12
    "armor_infantry_4_red",   // 13
    "armor_infantry_4_none",  // 14
    "armor_infantry_5_blue",  // 15
    "armor_infantry_5_red",   // 16
    "armor_infantry_5_none",  // 17
    "armor_outpost_blue",     // 18
    "armor_outpost_red",      // 19
    "armor_outpost_none",     // 20
    "armor_base_blue",        // 21
    "armor_base_red",         // 22
    "armor_infantry_Big_3_blue", // 23
    "armor_infantry_Big_3_red",  // 24
    "armor_infantry_Big_3_none", // 25
    "armor_infantry_Big_4_blue", // 26
    "armor_infantry_Big_4_red",  // 27
    "armor_infantry_Big_4_none", // 28
    "armor_infantry_Big_5_blue", // 29
    "armor_infantry_Big_5_red",  // 30
    "armor_infantry_Big_5_none", // 31
    "armor_base_purple",      // 32
    "yindaodeng"              // 33
};

// 预处理函数，将图像数据转换为模型输入格式
void preprocess(cv::Mat &image, ov::Tensor &tensor)
{
    // 确保图像是以浮点格式
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3, 1.0/255.0);  // 归一化到[0,1]
    
    int img_w = float_image.cols;
    int img_h = float_image.rows;
    int channels = 3;

    auto data = tensor.data<float>();

    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                // OpenCV默认是BGR格式，将通道顺序从BGR转为RGB
                data[c * img_w * img_h + h * img_w + w] =
                    float_image.at<cv::Vec3f>(h, w)[2 - c];
            }
        }
    }
}

// 计算两个装甲板的IOU
float cal_iou(const Armor a, const Armor b) {
    // 计算两个四边形的外接矩形
    int ax_min = std::min(std::min(std::min(a.x1, a.x2), a.x3), a.x4);
    int ax_max = std::max(std::max(std::max(a.x1, a.x2), a.x3), a.x4);
    int ay_min = std::min(std::min(std::min(a.y1, a.y2), a.y3), a.y4);
    int ay_max = std::max(std::max(std::max(a.y1, a.y2), a.y3), a.y4);

    int bx_min = std::min(std::min(std::min(b.x1, b.x2), b.x3), b.x4);
    int bx_max = std::max(std::max(std::max(b.x1, b.x2), b.x3), b.x4);
    int by_min = std::min(std::min(std::min(b.y1, b.y2), b.y3), b.y4);
    int by_max = std::max(std::max(std::max(b.y1, b.y2), b.y3), b.y4);

    float max_x = std::max(ax_min, bx_min);
    float min_x = std::min(ax_max, bx_max);
    float max_y = std::max(ay_min, by_min);
    float min_y = std::min(ay_max, by_max);

    // 原条件是错误的，正确应该是：
    if(min_x <= max_x || min_y <= max_y)
        return 0;
    
    // 修改为：
    if(max_x >= min_x || max_y >= min_y)
        return 0;
    
    float over_area = (min_x - max_x) * (min_y - max_y);

    float area_a = (ax_max - ax_min) * (ay_max - ay_min);
    float area_b = (bx_max - bx_min) * (by_max - by_min);
    float iou = over_area / (area_a + area_b - over_area);
    return iou;
}

// NMS处理函数
void nms(float* result, float conf_thr, float iou_thr, std::vector<Armor>& armors, int class_nums) {
    // 遍历result，如果conf大于阈值conf_thr，则放入armors
    for(int i = 0; i < 25200; ++i) {
        if(result[8 + i * class_nums] >= conf_thr) {
            Armor temp;
            // 将四个角点放入
            temp.x1 = int(result[0 + i * class_nums]);
            temp.y1 = int(result[1 + i * class_nums]);
            temp.x2 = int(result[2 + i * class_nums]);
            temp.y2 = int(result[3 + i * class_nums]);
            temp.x3 = int(result[4 + i * class_nums]);
            temp.y3 = int(result[5 + i * class_nums]);
            temp.x4 = int(result[6 + i * class_nums]);
            temp.y4 = int(result[7 + i * class_nums]);

            // 找到最大的条件类别概率并乘上conf作为类别概率
            float cls = result[i * class_nums + 9];
            int cnt = 0;
            
            // 修正类别索引计算
            for(int j = i * class_nums + 9; j < i * class_nums + class_nums; ++j) {
                if(cls < result[j]) {
                    cls = result[j];
                    cnt = j - (i * class_nums + 9);  // 计算类别索引
                }
            }
            
            cls *= result[8 + i * class_nums];
            temp.score = cls;
            temp.label = cnt;
            armors.push_back(temp);
        }
    }
    
    // 对得到的armor按score进行降序排序
    std::sort(armors.begin(), armors.end(), [](Armor a, Armor b) { return a.score > b.score; });

    // 只保留置信度最高的一个装甲板
    if (armors.size() > 1) {
        armors.resize(1);
    }

    // 按iou_thr将重合度高的armor进行筛掉
    for(int i = 0; i < int(armors.size()); ++i) {
        for(int j = i + 1; j < int(armors.size()); ++j) {
            // 如果与当前的框iou大于阈值则erase掉
            if(cal_iou(armors[i], armors[j]) > iou_thr) {
                armors.erase(armors.begin() + j);
                --j; // 删除元素后，索引减一继续检查
            }
        }
    }
}
// 重构后的主函数
int main() {
    try {
        // 构建路径
        std::string model_path = std::filesystem::absolute("../model/last.xml").string();
        std::string image_path = std::filesystem::absolute("../img/image.png").string();
        
        // 检查文件是否存在
        if (!std::filesystem::exists(model_path) || !std::filesystem::exists(image_path)) {
            std::cerr << "错误: 找不到必要文件" << std::endl;
            return 1;
        }
        
        // 读取原始图像
        cv::Mat original_image = cv::imread(image_path);
        if (original_image.empty()) {
            std::cerr << "无法读取图像文件" << std::endl;
            return 1;
        }
        
        // 图像预处理
        cv::Mat processed_image;
        float scale = std::min(float(640) / original_image.cols, float(640) / original_image.rows);
        int padding_y = int((640 - original_image.rows * scale) / 2);
        int padding_x = int((640 - original_image.cols * scale) / 2);
        
        cv::resize(original_image, processed_image, cv::Size(original_image.cols * scale, original_image.rows * scale));
        cv::copyMakeBorder(processed_image, processed_image, padding_y, padding_y, padding_x, padding_x, 
                          cv::BORDER_CONSTANT, cv::Scalar(144, 144, 144));
        
        // 初始化推理引擎
        ov::Core core;
        auto model = core.read_model(model_path);
        
        // 优先使用GPU，失败则回退到CPU
        ov::CompiledModel compiled_model;
        try {
            compiled_model = core.compile_model(model, "GPU");
        } catch (const std::exception&) {
            compiled_model = core.compile_model(model, "CPU");
        }
        
        // 创建推理请求并准备输入
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        ov::Tensor input_tensor = infer_request.get_input_tensor();
        std::cout << "输入张量形状: " << input_tensor.get_shape() << std::endl;
        preprocess(processed_image, input_tensor);
        
        // 执行推理
        auto infer_start = std::chrono::steady_clock::now();
        infer_request.infer();
        auto infer_end = std::chrono::steady_clock::now();
        
        std::cout << "推理时间: " << std::chrono::duration<double>(infer_end - infer_start).count() * 1000 << " 毫秒" << std::endl;
        
        // 获取输出并处理
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        std::cout << "输出张量形状: " << output_tensor.get_shape() << std::endl;
        auto result = output_tensor.data<float>();
        
        // 执行NMS获取装甲板检测结果
        std::vector<Armor> armors;
        nms(result, 0.4, 0.45, armors, 43); // 使用43作为类别总数(8个基础坐标 + 1个置信度 + 34个类别)
        
        std::cout << "检测到 " << armors.size() << " 个装甲板" << std::endl;
        
        // 可视化结果
        cv::Mat visualization_image = original_image.clone();
        
        for (const auto& armor : armors) {
            // 映射坐标回原始图像
            int x1 = int((armor.x1 - padding_x) / scale);
            int y1 = int((armor.y1 - padding_y) / scale);
            int x2 = int((armor.x2 - padding_x) / scale);
            int y2 = int((armor.y2 - padding_y) / scale);
            int x3 = int((armor.x3 - padding_x) / scale);
            int y3 = int((armor.y3 - padding_y) / scale);
            int x4 = int((armor.x4 - padding_x) / scale);
            int y4 = int((armor.y4 - padding_y) / scale);
            
            // 确保坐标在图像范围内
            x1 = std::max(0, std::min(x1, original_image.cols - 1));
            y1 = std::max(0, std::min(y1, original_image.rows - 1));
            x2 = std::max(0, std::min(x2, original_image.cols - 1));
            y2 = std::max(0, std::min(y2, original_image.rows - 1));
            x3 = std::max(0, std::min(x3, original_image.cols - 1));
            y3 = std::max(0, std::min(y3, original_image.rows - 1));
            x4 = std::max(0, std::min(x4, original_image.cols - 1));
            y4 = std::max(0, std::min(y4, original_image.rows - 1));
            
            // 绘制装甲板 - 使用装甲板类型的颜色
            cv::Scalar color = COLORS[armor.label % COLORS.size()];
            
            // 根据类型选择不同线宽，蓝色和红色装甲板使用粗线
            int lineWidth = 2;
            if (armor.label % 3 == 0 || armor.label % 3 == 1) { // 蓝色或红色装甲板
                lineWidth = 3;
            }
            
            std::vector<cv::Point> polygon = {
                cv::Point(x1, y1), cv::Point(x2, y2), 
                cv::Point(x3, y3), cv::Point(x4, y4)
            };
            
            cv::polylines(visualization_image, std::vector<std::vector<cv::Point>>{polygon}, true, color, lineWidth);
            
            // 绘制角点
            cv::circle(visualization_image, cv::Point(x1, y1), 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(visualization_image, cv::Point(x2, y2), 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(visualization_image, cv::Point(x3, y3), 5, cv::Scalar(255, 0, 0), -1);
            cv::circle(visualization_image, cv::Point(x4, y4), 5, cv::Scalar(255, 255, 0), -1);
            
            // 获取简短标签名
            std::string shortLabel;
            if (armor.label < CLASS_NAMES.size()) {
                std::string fullName = CLASS_NAMES[armor.label];
                size_t lastUnderscore = fullName.find_last_of('_');
                
                if (lastUnderscore != std::string::npos && lastUnderscore + 1 < fullName.length()) {
                    // 获取最后一个下划线后的内容（颜色信息：blue/red/none）
                    std::string colorInfo = fullName.substr(lastUnderscore + 1);
                    
                    // 获取类型信息（从第一个下划线后到最后一个下划线前）
                    size_t firstUnderscore = fullName.find_first_of('_');
                    if (firstUnderscore != std::string::npos && firstUnderscore < lastUnderscore) {
                        std::string typeInfo = fullName.substr(firstUnderscore + 1, lastUnderscore - firstUnderscore - 1);
                        
                        // 创建简短标签：类型+颜色
                        shortLabel = typeInfo + "_" + colorInfo;
                    } else {
                        shortLabel = fullName;
                    }
                } else {
                    shortLabel = fullName;
                }
            } else {
                shortLabel = "class" + std::to_string(armor.label);
            }
            
            // 标签信息
            std::string label = shortLabel + " " + std::to_string(int(armor.score * 100)) + "%";
            
            cv::putText(visualization_image, label, cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            
            std::cout << "装甲板: 类别=" << CLASS_NAMES[armor.label] << ", 置信度=" << armor.score << std::endl;
        }
        
        // 保存结果
        cv::imwrite(std::filesystem::absolute("../img/output_visualization.jpg").string(), visualization_image);
        std::cout << "推理完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


