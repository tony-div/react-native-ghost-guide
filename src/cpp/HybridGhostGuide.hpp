#pragma once

#include <NitroModules/HybridObject.hpp>
#include "nitrogen/generated/shared/c++/HybridGhostGuideSpec.hpp"
#include "nitrogen/generated/shared/c++/Skeleton.hpp"
#include "nitrogen/generated/shared/c++/ProcessResult.hpp"
#include <memory>
#include <cstring>

// Rust FFI declarations
extern "C" {
  void* ghost_guide_load_reference(const char* json_data);
  void ghost_guide_free_reference(void* ref);
  void ghost_guide_process_frame(
    void* ref,
    const float* user_points,
    float* result_points,
    uint32_t* checkpoint_idx,
    bool* is_aligned,
    uint32_t* rep_count
  );
}

namespace margelo::nitro::ghostguide {

  class HybridGhostGuide: public HybridGhostGuideSpec {
    public:
      HybridGhostGuide() : HybridObject(TAG), HybridGhostGuideSpec(), _ref(nullptr) {}
      ~HybridGhostGuide() override {
        if (_ref) ghost_guide_free_reference(_ref);
      }

      void loadReference(const std::string& json) override;
      ProcessResult processFrame(const Skeleton& userSkeleton) override;

    private:
      void* _ref;
  };

} // namespace margelo::nitro::ghostguide
