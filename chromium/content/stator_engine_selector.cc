// chromium/content/stator_engine_selector.cc — Implements the per-context
// engine selection logic described in stator_engine_selector.h.

#include "stator_engine_selector.h"

#include "base/feature_list.h"
#include "third_party/stator/chromium/features/stator_features.h"

namespace content {

bool IsStatorEnabledFor(RendererType type) {
  // The umbrella flag enables Stator for every context type.
  if (base::FeatureList::IsEnabled(features::kStatorEngine)) {
    return true;
  }

  // Fine-grained sub-flags allow selective enablement per context type.
  switch (type) {
    case RendererType::kWorker:
      return base::FeatureList::IsEnabled(features::kStatorEngineWorkers);
    case RendererType::kServiceWorker:
      return base::FeatureList::IsEnabled(
          features::kStatorEngineServiceWorkers);
    case RendererType::kMainFrame:
      return base::FeatureList::IsEnabled(features::kStatorEngineMainFrame);
  }
  return false;
}

}  // namespace content
