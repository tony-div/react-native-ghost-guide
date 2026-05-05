#include "HybridGhostGuide.hpp"

namespace margelo::nitro::ghostguide {

  void HybridGhostGuide::loadReference(const std::string& json) {
    if (_ref) {
      ghost_guide_free_reference(_ref);
      _ref = nullptr;
    }
    _ref = ghost_guide_load_reference(json.c_str());
  }

  ProcessResult HybridGhostGuide::processFrame(const Skeleton& userSkeleton) {
    ProcessResult result;
    
    if (!_ref) {
      return result;
    }

    float user_points[33 * 3];
    for (int i = 0; i < 33; i++) {
      user_points[i * 3] = static_cast<float>(userSkeleton.points[i].x);
      user_points[i * 3 + 1] = static_cast<float>(userSkeleton.points[i].y);
      user_points[i * 3 + 2] = static_cast<float>(userSkeleton.points[i].z);
    }

    float result_points[33 * 3];
    uint32_t checkpoint_idx = 0;
    bool is_aligned = false;
    uint32_t rep_count = 0;

    ghost_guide_process_frame(_ref, user_points, result_points, &checkpoint_idx, &is_aligned, &rep_count);

    Skeleton ghost;
    ghost.points.resize(33);
    for (int i = 0; i < 33; i++) {
      ghost.points[i].x = result_points[i * 3];
      ghost.points[i].y = result_points[i * 3 + 1];
      ghost.points[i].z = result_points[i * 3 + 2];
    }

    result.ghostSkeleton = ghost;
    result.currentCheckpointIndex = checkpoint_idx;
    result.isAligned = is_aligned;
    result.repCount = rep_count;

    return result;
  }

} // namespace margelo::nitro::ghostguide
