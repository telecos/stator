// chromium/features/stator_features.cc — Chrome feature-flag definitions for
// the Stator JavaScript engine.
//
// All flags default to FEATURE_DISABLED_BY_DEFAULT; they must be explicitly
// opt-in via --enable-features=<Name> or a server-side Finch experiment.

#include "stator_features.h"

#include "base/feature_list.h"

namespace features {

// Umbrella flag: enables Stator for all renderer context types simultaneously.
BASE_FEATURE(kStatorEngine,
             "StatorEngine",
             base::FEATURE_DISABLED_BY_DEFAULT);

// Sub-flag 1: dedicated Web Workers (lowest risk — no DOM).
BASE_FEATURE(kStatorEngineWorkers,
             "StatorEngineWorkers",
             base::FEATURE_DISABLED_BY_DEFAULT);

// Sub-flag 2: Service Workers (DOM-free, intercepts network requests).
BASE_FEATURE(kStatorEngineServiceWorkers,
             "StatorEngineServiceWorkers",
             base::FEATURE_DISABLED_BY_DEFAULT);

// Sub-flag 3: main-frame renderer contexts (highest risk — full DOM access).
BASE_FEATURE(kStatorEngineMainFrame,
             "StatorEngineMainFrame",
             base::FEATURE_DISABLED_BY_DEFAULT);

// Differential-testing flag: run both V8 and Stator, compare DOM state.
BASE_FEATURE(kStatorDualEngine,
             "StatorDualEngine",
             base::FEATURE_DISABLED_BY_DEFAULT);

}  // namespace features
