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
        std::string video_path = std::filesystem::absolute("../img/1.avi").string();
        int mode = 1; // 0: 图像推理，1: 视频文件推理，2: 摄像头视频流

        // 检查模型文件是否存在
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "错误: 找不到模型文件" << std::endl;
            return 1;
        }
        
        // 初始化推理引擎（提前初始化以避免每一帧都重新加载）
        ov::Core core;
        auto model = core.read_model(model_path);
        
        // 优先使用GPU，失败则回退到CPU
        ov::CompiledModel compiled_model;
        try {
            compiled_model = core.compile_model(model, "GPU");
            std::cout << "使用GPU进行推理" << std::endl;
        } catch (const std::exception&) {
            compiled_model = core.compile_model(model, "CPU");
            std::cout << "使用CPU进行推理" << std::endl;
        }
        
        // 创建推理请求
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        
        // 根据模式选择处理图像或视频
        if (mode == 0) { // 图像模式
            // 检查图像文件是否存在
            if (!std::filesystem::exists(image_path)) {
                std::cerr << "错误: 找不到图像文件" << std::endl;
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
            
            // 创建推理请求并准备输入
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
        }
        else if (mode == 1) { // 视频文件模式
            // 检查视频文件是否存在
            if (!std::filesystem::exists(video_path)) {
                std::cerr << "错误: 找不到视频文件" << std::endl;
                return 1;
            }
            
            // 打开视频文件
            cv::VideoCapture cap(video_path);
            if (!cap.isOpened()) {
                std::cerr << "无法打开视频文件" << std::endl;
                return 1;
            }
            
            // 获取视频属性
            double fps = cap.get(cv::CAP_PROP_FPS);
            int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
            
            std::cout << "视频信息: " << width << "x" << height 
                      << ", FPS: " << fps 
                      << ", 总帧数: " << total_frames << std::endl;
            
            // 创建视频写入器
            std::string output_video_path = std::filesystem::absolute("../img/output_video.mp4").string();
            cv::VideoWriter video_writer(output_video_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
                                         fps, cv::Size(width, height));
            
            if (!video_writer.isOpened()) {
                std::cerr << "无法创建输出视频文件" << std::endl;
                return 1;
            }
            
            // 创建窗口
            cv::namedWindow("装甲板检测", cv::WINDOW_NORMAL);
            cv::resizeWindow("装甲板检测", 800, 600);
            
            cv::Mat frame;
            int frame_count = 0;
            double total_time = 0.0;
            
            while (true) {
                // 读取视频帧
                if (!cap.read(frame)) {
                    break; // 视频结束
                }
                
                // 图像预处理
                cv::Mat processed_image;
                float scale = std::min(float(640) / frame.cols, float(640) / frame.rows);
                int padding_y = int((640 - frame.rows * scale) / 2);
                int padding_x = int((640 - frame.cols * scale) / 2);
                
                cv::resize(frame, processed_image, cv::Size(frame.cols * scale, frame.rows * scale));
                cv::copyMakeBorder(processed_image, processed_image, padding_y, padding_y, padding_x, padding_x, 
                                  cv::BORDER_CONSTANT, cv::Scalar(144, 144, 144));
                
                // 准备输入
                ov::Tensor input_tensor = infer_request.get_input_tensor();
                preprocess(processed_image, input_tensor);
                
                // 执行推理
                auto infer_start = std::chrono::steady_clock::now();
                infer_request.infer();
                auto infer_end = std::chrono::steady_clock::now();
                
                double infer_time = std::chrono::duration<double>(infer_end - infer_start).count() * 1000;
                total_time += infer_time;
                
                // 获取输出并处理
                ov::Tensor output_tensor = infer_request.get_output_tensor();
                auto result = output_tensor.data<float>();
                
                // 执行NMS获取装甲板检测结果
                std::vector<Armor> armors;
                nms(result, 0.4, 0.45, armors, 43); // 使用43作为类别总数
                
                // 可视化结果
                cv::Mat visualization_image = frame.clone();
                
                // 显示当前帧号和处理时间
                cv::putText(visualization_image, 
                           "Frame: " + std::to_string(frame_count) + "/" + std::to_string(total_frames) +
                           ", Time: " + std::to_string(int(infer_time)) + "ms", 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                
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
                    x1 = std::max(0, std::min(x1, frame.cols - 1));
                    y1 = std::max(0, std::min(y1, frame.rows - 1));
                    x2 = std::max(0, std::min(x2, frame.cols - 1));
                    y2 = std::max(0, std::min(y2, frame.rows - 1));
                    x3 = std::max(0, std::min(x3, frame.cols - 1));
                    y3 = std::max(0, std::min(y3, frame.rows - 1));
                    x4 = std::max(0, std::min(x4, frame.cols - 1));
                    y4 = std::max(0, std::min(y4, frame.rows - 1));
                    
                    // 绘制装甲板
                    cv::Scalar color = COLORS[armor.label % COLORS.size()];
                    
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
                            std::string colorInfo = fullName.substr(lastUnderscore + 1);
                            
                            size_t firstUnderscore = fullName.find_first_of('_');
                            if (firstUnderscore != std::string::npos && firstUnderscore < lastUnderscore) {
                                std::string typeInfo = fullName.substr(firstUnderscore + 1, lastUnderscore - firstUnderscore - 1);
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
                }
                
                // 写入视频帧
                video_writer.write(visualization_image);
                
                // 显示结果
                cv::imshow("装甲板检测", visualization_image);
                
                // 按ESC键退出
                int key = cv::waitKey(1);
                if (key == 27) { // ESC键
                    std::cout << "用户中断处理" << std::endl;
                    break;
                }
                
                frame_count++;
                
                // 每秒显示处理进度
                if (frame_count % int(fps) == 0) {
                    std::cout << "已处理 " << frame_count << "/" << total_frames 
                              << " 帧 (" << (frame_count * 100 / total_frames) << "%)" 
                              << ", 平均推理时间: " << (total_time / frame_count) << " 毫秒" << std::endl;
                }
            }
            
            // 关闭资源
            cap.release();
            video_writer.release();
            cv::destroyAllWindows();
            
            std::cout << "视频处理完成！输出文件: " << output_video_path << std::endl;
            std::cout << "总帧数: " << frame_count << ", 平均推理时间: " << (total_time / frame_count) << " 毫秒" << std::endl;
        }
        else if (mode == 2) { // 摄像头视频流模式
            // 打开摄像头
            cv::VideoCapture cap(0);
            if (!cap.isOpened()) {
                std::cerr << "无法打开摄像头" << std::endl;
                return 1;
            }
            
            // 获取摄像头属性
            int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = cap.get(cv::CAP_PROP_FPS);
            
            std::cout << "摄像头信息: " << width << "x" << height << ", FPS: " << fps << std::endl;
            
            // 创建窗口
            cv::namedWindow("装甲板检测 (摄像头)", cv::WINDOW_NORMAL);
            cv::resizeWindow("装甲板检测 (摄像头)", 800, 600);
            
            // 可选：创建视频写入器保存结果
            bool save_video = false;
            cv::VideoWriter video_writer;
            std::string output_video_path;
            
            if (save_video) {
                output_video_path = std::filesystem::absolute("../img/camera_output.mp4").string();
                video_writer.open(output_video_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
                                 30.0, cv::Size(width, height));
                
                if (!video_writer.isOpened()) {
                    std::cerr << "警告: 无法创建输出视频文件，将只显示不保存" << std::endl;
                    save_video = false;
                }
            }
            
            cv::Mat frame;
            int frame_count = 0;
            double total_time = 0.0;
            auto start_time = std::chrono::steady_clock::now();
            
            while (true) {
                // 读取摄像头帧
                if (!cap.read(frame)) {
                    std::cerr << "无法读取摄像头帧" << std::endl;
                    break;
                }
                
                // 图像预处理
                cv::Mat processed_image;
                float scale = std::min(float(640) / frame.cols, float(640) / frame.rows);
                int padding_y = int((640 - frame.rows * scale) / 2);
                int padding_x = int((640 - frame.cols * scale) / 2);
                
                cv::resize(frame, processed_image, cv::Size(frame.cols * scale, frame.rows * scale));
                cv::copyMakeBorder(processed_image, processed_image, padding_y, padding_y, padding_x, padding_x, 
                                  cv::BORDER_CONSTANT, cv::Scalar(144, 144, 144));
                
                // 准备输入
                ov::Tensor input_tensor = infer_request.get_input_tensor();
                preprocess(processed_image, input_tensor);
                
                // 执行推理
                auto infer_start = std::chrono::steady_clock::now();
                infer_request.infer();
                auto infer_end = std::chrono::steady_clock::now();
                
                double infer_time = std::chrono::duration<double>(infer_end - infer_start).count() * 1000;
                total_time += infer_time;
                
                // 获取输出并处理
                ov::Tensor output_tensor = infer_request.get_output_tensor();
                auto result = output_tensor.data<float>();
                
                // 执行NMS获取装甲板检测结果
                std::vector<Armor> armors;
                nms(result, 0.4, 0.45, armors, 43); // 使用43作为类别总数
                
                // 可视化结果
                cv::Mat visualization_image = frame.clone();
                
                // 计算和显示帧率
                auto current_time = std::chrono::steady_clock::now();
                double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();
                double current_fps = frame_count / elapsed_seconds;
                
                // 显示处理信息
                cv::putText(visualization_image, 
                           "FPS: " + std::to_string(int(current_fps)) + 
                           ", 帧数: " + std::to_string(frame_count) +
                           ", 时间: " + std::to_string(int(infer_time)) + "ms", 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                
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
                    x1 = std::max(0, std::min(x1, frame.cols - 1));
                    y1 = std::max(0, std::min(y1, frame.rows - 1));
                    x2 = std::max(0, std::min(x2, frame.cols - 1));
                    y2 = std::max(0, std::min(y2, frame.rows - 1));
                    x3 = std::max(0, std::min(x3, frame.cols - 1));
                    y3 = std::max(0, std::min(y3, frame.rows - 1));
                    x4 = std::max(0, std::min(x4, frame.cols - 1));
                    y4 = std::max(0, std::min(y4, frame.rows - 1));
                    
                    // 绘制装甲板
                    cv::Scalar color = COLORS[armor.label % COLORS.size()];
                    
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
                            std::string colorInfo = fullName.substr(lastUnderscore + 1);
                            
                            size_t firstUnderscore = fullName.find_first_of('_');
                            if (firstUnderscore != std::string::npos && firstUnderscore < lastUnderscore) {
                                std::string typeInfo = fullName.substr(firstUnderscore + 1, lastUnderscore - firstUnderscore - 1);
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
                }
                
                // 保存视频帧（如果启用）
                if (save_video) {
                    video_writer.write(visualization_image);
                }
                
                // 显示结果
                cv::imshow("装甲板检测 (摄像头)", visualization_image);
                
                // 按ESC键退出，按's'键切换保存视频
                int key = cv::waitKey(1);
                if (key == 27) { // ESC键
                    std::cout << "用户中断处理" << std::endl;
                    break;
                }
                else if (key == 's' || key == 'S') {
                    save_video = !save_video;
                    if (save_video && !video_writer.isOpened()) {
                        output_video_path = std::filesystem::absolute("../img/camera_output.mp4").string();
                        video_writer.open(output_video_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
                                         30.0, cv::Size(width, height));
                        
                        if (!video_writer.isOpened()) {
                            std::cerr << "警告: 无法创建输出视频文件，将只显示不保存" << std::endl;
                            save_video = false;
                        } else {
                            std::cout << "开始保存视频到: " << output_video_path << std::endl;
                        }
                    } else if (!save_video && video_writer.isOpened()) {
                        std::cout << "停止保存视频" << std::endl;
                    }
                }
                
                frame_count++;
                
                // 每100帧显示一次状态
                if (frame_count % 100 == 0) {
                    std::cout << "已处理 " << frame_count << " 帧"
                              << ", 当前FPS: " << current_fps
                              << ", 平均推理时间: " << (total_time / frame_count) << " 毫秒" << std::endl;
                }
            }
            
            // 关闭资源
            cap.release();
            if (video_writer.isOpened()) {
                video_writer.release();
                std::cout << "视频已保存到: " << output_video_path << std::endl;
            }
            cv::destroyAllWindows();
            
            auto end_time = std::chrono::steady_clock::now();
            double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
            double avg_fps = frame_count / total_elapsed;
            
            std::cout << "摄像头处理完成！" << std::endl;
            std::cout << "总帧数: " << frame_count 
                      << ", 总运行时间: " << total_elapsed << " 秒"
                      << ", 平均FPS: " << avg_fps 
                      << ", 平均推理时间: " << (total_time / frame_count) << " 毫秒" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


