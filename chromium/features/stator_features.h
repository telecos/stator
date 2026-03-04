// chromium/features/stator_features.h — Chrome feature-flag declarations for
// the Stator JavaScript engine.
//
// Each flag is disabled by default and can be enabled at runtime via the
// standard --enable-features=<Name> command-line switch, e.g.:
//
//   chrome --enable-features=StatorEngine
//
// Rollout order
// -------------
// The safe rollout order mirrors the risk profile of each renderer context:
//
//   1. StatorEngineWorkers       — dedicated workers (isolated, no DOM)
//   2. StatorEngineServiceWorkers— service workers   (isolated, controlled)
//   3. StatorEngineMainFrame     — main-frame renderer (full DOM, highest risk)
//
// The umbrella flag StatorEngine enables all three sub-flags simultaneously.
// Fine-grained sub-flags exist so that field experiments can enable only a
// subset of context types (e.g. enable for workers before main frame is ready).
//
// Differential testing
// --------------------
// StatorDualEngine runs both V8 and Stator on the same page and compares the
// resulting DOM/global state after each script evaluation.  This flag is
// intended for canary / dev channel testing and should never be enabled in
// stable due to double execution cost.

#ifndef CHROMIUM_FEATURES_STATOR_FEATURES_H_
#define CHROMIUM_FEATURES_STATOR_FEATURES_H_

#include "base/feature_list.h"

namespace features {

// ---------------------------------------------------------------------------
// Umbrella flag — enable Stator for ALL renderer context types at once.
// Enabling this flag is equivalent to enabling the three sub-flags below.
// ---------------------------------------------------------------------------
BASE_DECLARE_FEATURE(kStatorEngine);

// ---------------------------------------------------------------------------
// Sub-flag 1 — enable Stator for dedicated Web Workers.
// Workers have no access to the DOM and run in isolated V8 contexts, making
// them the lowest-risk context type for a new engine.
// ---------------------------------------------------------------------------
BASE_DECLARE_FEATURE(kStatorEngineWorkers);

// ---------------------------------------------------------------------------
// Sub-flag 2 — enable Stator for Service Workers.
// Service workers are also DOM-free but intercept network requests, so they
// are slightly higher-risk than dedicated workers.
// ---------------------------------------------------------------------------
BASE_DECLARE_FEATURE(kStatorEngineServiceWorkers);

// ---------------------------------------------------------------------------
// Sub-flag 3 — enable Stator for main-frame renderer contexts.
// Main frames have full DOM access and are the highest-risk context type.
// This flag should only be enabled after workers and service workers are
// confirmed stable.
// ---------------------------------------------------------------------------
BASE_DECLARE_FEATURE(kStatorEngineMainFrame);

// ---------------------------------------------------------------------------
// Differential-testing flag.
//
// When enabled, both V8 and Stator execute each script in a renderer context.
// After execution the resulting DOM/global state is serialised and compared.
// Discrepancies are reported via a new DevTools protocol event
// (Stator.dualEngineDiscrepancy).
//
// Cost: roughly 2× script execution time per renderer context.
// NEVER enable in stable builds.
// ---------------------------------------------------------------------------
BASE_DECLARE_FEATURE(kStatorDualEngine);

}  // namespace features

#endif  // CHROMIUM_FEATURES_STATOR_FEATURES_H_
