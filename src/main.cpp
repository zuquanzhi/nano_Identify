#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// 结构体定义 - YOLOv5 xywh格式
typedef struct Detection
{
    float x;      // 中心点x坐标
    float y;      // 中心点y坐标
    float width;  // 宽度
    float height; // 高度
    float score;  // 置信度
    int label;    // 类别
} Detection;

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
    "target_area",           // 0
    "active_target_area",    // 1
    "arrow_lightbar",        // 2
    "active_lightbar",       // 3
    "r_logo",                // 4
    "ten_score"              // 5
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

// 计算两个检测框的IOU (使用YOLOv5 xywh格式)
float cal_iou(const Detection& a, const Detection& b) {
    // 计算两个框的边界
    float a_x1 = a.x - a.width / 2;
    float a_y1 = a.y - a.height / 2;
    float a_x2 = a.x + a.width / 2;
    float a_y2 = a.y + a.height / 2;
    
    float b_x1 = b.x - b.width / 2;
    float b_y1 = b.y - b.height / 2;
    float b_x2 = b.x + b.width / 2;
    float b_y2 = b.y + b.height / 2;
    
    // 计算交集区域
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);
    
    // 检查是否有交集
    if(inter_x1 >= inter_x2 || inter_y1 >= inter_y2)
        return 0;
    
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float a_area = a.width * a.height;
    float b_area = b.width * b.height;
    
    // 计算IOU
    float iou = inter_area / (a_area + b_area - inter_area);
    return iou;
}

// NMS处理函数 - 使用YOLOv5 xywh格式
void nms(float* result, float conf_thr, float iou_thr, std::vector<Detection>& detections, int class_nums) {
    // 遍历result，如果conf大于阈值conf_thr，则放入detections
    for(int i = 0; i < 25200; ++i) {
        if(result[4 + i * class_nums] >= conf_thr) {  // YOLOv5中第5个元素是置信度
            Detection temp;
            // 直接获取中心点和宽高
            temp.x = result[0 + i * class_nums];      // 中心点x
            temp.y = result[1 + i * class_nums];      // 中心点y
            temp.width = result[2 + i * class_nums];  // 宽度
            temp.height = result[3 + i * class_nums]; // 高度

            // 找到最大的条件类别概率并乘上conf作为类别概率
            float max_cls_prob = result[i * class_nums + 5];   // 类别预测从第6个元素开始
            int class_idx = 0;
            
            // 计算类别索引
            for(int j = i * class_nums + 5; j < i * class_nums + class_nums; ++j) {
                if(max_cls_prob < result[j]) {
                    max_cls_prob = result[j];
                    class_idx = j - (i * class_nums + 5);  // 计算类别索引
                }
            }
            
            float conf_score = max_cls_prob * result[4 + i * class_nums];  // 类别概率 = 最大类别概率 * 置信度
            temp.score = conf_score;
            temp.label = class_idx;
            detections.push_back(temp);
        }
    }
    
    // 对得到的detection按score进行降序排序
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });

    // 标准YOLOv5 NMS处理
    for(int i = 0; i < int(detections.size()); ++i) {
        for(int j = i + 1; j < int(detections.size()); ++j) {
            // 如果与当前的框iou大于阈值则删除
            if(cal_iou(detections[i], detections[j]) > iou_thr) {
                detections.erase(detections.begin() + j);
                --j; // 删除元素后，索引减一继续检查
            }
        }
    }
}

// 重构后的主函数
int main() {
    try {
        // 构建路径
        std::string model_path = std::filesystem::absolute("../model/best.xml").string();
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
        
        // 图像预处理 - YOLOv5标准预处理
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
        
        // 执行NMS获取检测结果
        std::vector<Detection> detections;
        nms(result, 0.4, 0.45, detections, 11); // 使用11作为类别总数(4个基础坐标(x,y,w,h) + 1个置信度 + 6个类别)
        
        std::cout << "检测到 " << detections.size() << " 个目标" << std::endl;
        
        // 可视化结果
        cv::Mat visualization_image = original_image.clone();
        
        for (const auto& det : detections) {
            // 映射中心点坐标和宽高回原始图像
            int center_x = int((det.x - padding_x) / scale);
            int center_y = int((det.y - padding_y) / scale);
            int width = int(det.width / scale);
            int height = int(det.height / scale);
            
            // 计算矩形框的四个角点
            int x1 = center_x - width / 2;
            int y1 = center_y - height / 2;
            int x2 = center_x + width / 2;
            int y2 = center_y + height / 2;
            
            // 确保坐标在图像范围内
            x1 = std::max(0, std::min(x1, original_image.cols - 1));
            y1 = std::max(0, std::min(y1, original_image.rows - 1));
            x2 = std::max(0, std::min(x2, original_image.cols - 1));
            y2 = std::max(0, std::min(y2, original_image.rows - 1));
            
            // 使用类别对应的颜色
            cv::Scalar color = COLORS[det.label % COLORS.size()];
            
            // 根据类型选择不同线宽，特定类别使用粗线
            int lineWidth = 2;
            if (det.label == 1 || det.label == 3) { // 激活的区域
                lineWidth = 3;
            }
            
            // 绘制矩形框
            cv::rectangle(visualization_image, cv::Point(x1, y1), cv::Point(x2, y2), color, lineWidth);
            
            // 绘制中心点
            cv::circle(visualization_image, cv::Point(center_x, center_y), 5, cv::Scalar(0, 0, 255), -1);
            
            // 获取标签名
            std::string shortLabel;
            if (det.label < CLASS_NAMES.size()) {
                shortLabel = CLASS_NAMES[det.label];
            } else {
                shortLabel = "class" + std::to_string(det.label);
            }
            
            // 标签信息
            std::string label = shortLabel + " " + std::to_string(int(det.score * 100)) + "%";
            
            cv::putText(visualization_image, label, cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            
            std::cout << "目标: 类别=" << CLASS_NAMES[det.label] << ", 置信度=" << det.score 
                      << ", 中心点=(" << center_x << "," << center_y << ")" 
                      << ", 宽=" << width << ", 高=" << height << std::endl;
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


