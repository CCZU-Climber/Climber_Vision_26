#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <atomic>
#include <chrono>
#include <list>
#include <memory>
#include <vector>

#include "decider.hpp"
#include "detection.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"

namespace omniperception
{

class Perceptron
{
public:
  Perceptron(
    const std::vector<io::USBCamera *> & cameras, const std::string & config_path,
    bool debug = false);

  ~Perceptron();

  std::vector<DetectionResult> get_detection_queue();

private:
  void inference_loop();

  std::string config_path_;
  std::thread inference_thread_;
  std::atomic<uint64_t> queue_full_count_;
  tools::ThreadSafeQueue<DetectionResult, true> detection_queue_;

  std::shared_ptr<auto_aim::YOLO> yolo_;
  std::vector<io::USBCamera *> cameras_;
  size_t rr_index_;

  Decider decider_;
  bool stop_flag_;
  bool debug_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace omniperception
#endif
