#include "yolo11_buff.hpp"

namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path = yaml["model"].as<std::string>();
  if (yaml["ConfidenceThreshold"]) {
    ConfidenceThreshold = yaml["ConfidenceThreshold"].as<double>();
  }
  if (yaml["IouThreshold"]) {
    IouThreshold = yaml["IouThreshold"].as<double>();
  }
  if (yaml["enemy_color"]) {
    set_enemy_color(yaml["enemy_color"].as<std::string>());
  }
  model = core.read_model(model_path);
  compiled_model = core.compile_model(model, "CPU");
  infer_request = compiled_model.create_infer_request();
  input_tensor = infer_request.get_input_tensor();
  input_tensor.set_shape({1, 3, 640, 640});
}

void YOLO11_BUFF::set_enemy_color(const std::string & color)
{
  // 能量机关打我方颜色，与敌方颜色取反
  // enemy_color=blue → 我方red → class_idx=0(red)
  // enemy_color=red  → 我方blue → class_idx=1(blue)
  if (color == "blue")
    enemy_class_idx_ = 0;  // red
  else
    enemy_class_idx_ = 1;  // blue (default enemy=red)
}

std::vector<cv::Point2f> YOLO11_BUFF::remap_keypoints(const std::vector<cv::Point2f> & raw) const
{
  // Raw 9-pt: [0]=R_center, [1,8]=bottom, [2,3]=right, [4,5]=top, [6,7]=left
  // Compute edge-pair midpoints → map to old 6-pt PnP corners
  const cv::Point2f top = (raw[4] + raw[5]) * 0.5f;
  const cv::Point2f left = (raw[6] + raw[7]) * 0.5f;
  const cv::Point2f bottom = (raw[1] + raw[8]) * 0.5f;
  const cv::Point2f right = (raw[2] + raw[3]) * 0.5f;

  std::vector<cv::Point2f> out(REMAPPED_KPTS);
  out[0] = top;     // top edge center
  out[1] = left;    // left edge center
  out[2] = bottom;  // bottom edge center
  out[3] = right;   // right edge center
  out[4] = (top + left + bottom + right) * 0.25f;  // blade center
  out[5] = raw[0];  // R center
  return out;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  const float factor = use_legacy_preprocess_
                         ? fill_tensor_data_image_legacy(input_tensor, image)
                         : fill_tensor_data_image(input_tensor, image);

  infer_request.infer();

  const ov::Tensor output = infer_request.get_output_tensor();
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output_shape[1];   // 33
  const int out_cols = output_shape[2];   // 8400
  const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);

  // 33 = 4 bbox + 2 class + 9*3 kpt
  const int class_row = 4 + enemy_class_idx_;
  const int kpt_rows = NUM_POINTS * KPT_DIM;  // 27

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> labels;
  std::vector<std::vector<float>> objects_keypoints;

  for (int i = 0; i < det_output.cols; ++i) {
    const float score = det_output.at<float>(class_row, i);
    if (score > ConfidenceThreshold) {
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      cv::Rect box;
      box.x = static_cast<int>((cx - 0.5 * ow) * factor);
      box.y = static_cast<int>((cy - 0.5 * oh) * factor);
      box.width = static_cast<int>(ow * factor);
      box.height = static_cast<int>(oh * factor);
      boxes.push_back(box);
      confidences.push_back(score);
      labels.push_back(enemy_class_idx_);

      std::vector<float> keypoints;
      cv::Mat kpts = det_output.col(i).rowRange(KPT_START, KPT_START + kpt_rows);
      std::vector<cv::Point2f> raw_kpts(NUM_POINTS);
      for (int j = 0; j < NUM_POINTS; ++j) {
        const float x = kpts.at<float>(j * KPT_DIM + 0, 0) * factor;
        const float y = kpts.at<float>(j * KPT_DIM + 1, 0) * factor;
        raw_kpts[j] = cv::Point2f(x, y);
      }
      // remap to 6-pt layout expected by solver
      auto remapped = remap_keypoints(raw_kpts);
      for (const auto & p : remapped) {
        keypoints.push_back(p.x);
        keypoints.push_back(p.y);
      }
      objects_keypoints.push_back(keypoints);
    }
  }

  // NMS
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);

  std::vector<Object> object_result;
  for (size_t i = 0; i < indexes.size(); ++i) {
    Object obj;
    const int index = indexes[i];
    obj.rect = boxes[index];
    obj.prob = confidences[index];
    obj.label = labels[index];

    const std::vector<float> & keypoint = objects_keypoints[index];
    for (int j = 0; j < REMAPPED_KPTS; ++j) {
      obj.kpt.push_back(cv::Point2f(keypoint[j * 2], keypoint[j * 2 + 1]));
    }
    object_result.push_back(obj);

    // draw
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = class_names[enemy_class_idx_] + ":" +
                              std::to_string(obj.prob).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40),
              cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  const float factor = use_legacy_preprocess_
                         ? fill_tensor_data_image_legacy(input_tensor, image)
                         : fill_tensor_data_image(input_tensor, image);

  infer_request.infer();

  // 33 = 4 bbox + 2 class + 9*3 kpt
  const ov::Tensor output = infer_request.get_output_tensor();
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output_shape[1];
  const int out_cols = output_shape[2];
  const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);

  const int class_row = 4 + enemy_class_idx_;
  const int kpt_rows = NUM_POINTS * KPT_DIM;

  // find max class score for enemy color
  int best_index = -1;
  float max_confidence = 0.0f;
  for (int i = 0; i < det_output.cols; ++i) {
    const float confidence = det_output.at<float>(class_row, i);
    if (confidence > max_confidence) {
      max_confidence = confidence;
      best_index = i;
    }
  }

  // static int no_detect_count = 0;
  // if (max_confidence <= ConfidenceThreshold) {
  //   no_detect_count++;
  //   if (no_detect_count % 30 == 1)
  //     tools::logger()->debug("[YOLO11_BUFF] max_conf={:.4f} < thr={:.2f} (no_detect x{})",
  //       max_confidence, ConfidenceThreshold, no_detect_count);
  // } else {
  //   if (no_detect_count > 0)
  //     tools::logger()->debug("[YOLO11_BUFF] detected after {} misses, max_conf={:.4f}",
  //       no_detect_count, max_confidence);
  //   no_detect_count = 0;
  // }

  std::vector<Object> object_result;
  if (max_confidence > ConfidenceThreshold) {
    Object obj;
    const float cx = det_output.at<float>(0, best_index);
    const float cy = det_output.at<float>(1, best_index);
    const float ow = det_output.at<float>(2, best_index);
    const float oh = det_output.at<float>(3, best_index);
    obj.rect.x = static_cast<int>((cx - 0.5 * ow) * factor);
    obj.rect.y = static_cast<int>((cy - 0.5 * oh) * factor);
    obj.rect.width = static_cast<int>(ow * factor);
    obj.rect.height = static_cast<int>(oh * factor);
    obj.prob = max_confidence;
    obj.label = enemy_class_idx_;

    cv::Mat kpts = det_output.col(best_index).rowRange(KPT_START, KPT_START + kpt_rows);
    std::vector<cv::Point2f> raw_kpts(NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; ++i) {
      const float x = kpts.at<float>(i * KPT_DIM + 0, 0) * factor;
      const float y = kpts.at<float>(i * KPT_DIM + 1, 0) * factor;
      raw_kpts[i] = cv::Point2f(x, y);
    }

    auto remapped = remap_keypoints(raw_kpts);
    for (const auto & p : remapped) {
      obj.kpt.push_back(p);
    }
    object_result.push_back(obj);

    if (max_confidence < 0.7) save(std::to_string(start), image);

    // draw
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = class_names[enemy_class_idx_] + ":" +
                              std::to_string(max_confidence).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40),
              cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

float YOLO11_BUFF::fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t num_channels = tensor_shape[1];
  const size_t height = tensor_shape[2];
  const size_t width = tensor_shape[3];
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{scale, 0.0, 0.0, 0.0, scale, 0.0};
  cv::Mat blob_image;
  if (scale < 1.0f) {
    cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
    convert(blob_image, blob_image, true, true);
  } else {
    convert(input_image, blob_image, true, true);
    cv::warpAffine(blob_image, blob_image, matrix, cv::Size(width, height));
  }
  float * const input_tensor_data = input_tensor.data<float>();
  for (size_t c = 0; c < num_channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        input_tensor_data[c * width * height + h * width + w] =
          blob_image.at<cv::Vec<float, 3>>(h, w)[c];
      }
    }
  }
  return 1 / scale;
}

float YOLO11_BUFF::fill_tensor_data_image_legacy(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t num_channels = tensor_shape[1];
  const size_t height = tensor_shape[2];
  const size_t width = tensor_shape[3];
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const int h = static_cast<int>(input_image.rows * scale);
  const int w = static_cast<int>(input_image.cols * scale);
  cv::Mat letterbox(static_cast<int>(height), static_cast<int>(width), CV_8UC3, cv::Scalar(0, 0, 0));
  cv::resize(input_image, letterbox(cv::Rect(0, 0, w, h)), {w, h});
  cv::Mat blob_image;
  letterbox.convertTo(blob_image, CV_32F);
  float * const input_tensor_data = input_tensor.data<float>();
  for (size_t c = 0; c < num_channels; c++) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        input_tensor_data[c * width * height + y * width + x] =
          blob_image.at<cv::Vec<float, 3>>(y, x)[c];
      }
    }
  }
  return 1.0f / scale;
}

void YOLO11_BUFF::printInputAndOutputsInfo(const ov::Model & network)
{
  std::cout << "model name: " << network.get_friendly_name() << std::endl;
  const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
  for (const ov::Output<const ov::Node> & input : inputs) {
    std::cout << "    inputs" << std::endl;
    const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
    std::cout << "        input name: " << name << std::endl;
    const ov::element::Type type = input.get_element_type();
    std::cout << "        input type: " << type << std::endl;
    const ov::Shape shape = input.get_shape();
    std::cout << "        input shape: " << shape << std::endl;
  }
  const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
  for (const ov::Output<const ov::Node> & output : outputs) {
    std::cout << "    outputs" << std::endl;
    const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
    std::cout << "        output name: " << name << std::endl;
    const ov::element::Type type = output.get_element_type();
    std::cout << "        output type: " << type << std::endl;
    const ov::Shape shape = output.get_shape();
    std::cout << "        output shape: " << shape << std::endl;
  }
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff
