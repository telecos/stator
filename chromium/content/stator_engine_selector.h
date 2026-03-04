// chromium/content/stator_engine_selector.h — Selects the JS engine (V8 or
// Stator) for a renderer context based on the active Chrome feature flags.
//
// Usage
// -----
// Call IsStatorEnabledFor() at renderer-context creation time to decide which
// engine to instantiate:
//
//   #include "third_party/stator/chromium/content/stator_engine_selector.h"
//
//   auto engine = content::IsStatorEnabledFor(content::RendererType::kWorker)
//                     ? CreateStatorIsolate()
//                     : CreateV8Isolate();
//
// Phased rollout order
// --------------------
// The intended deployment sequence is:
//
//   Phase 1 — enable kStatorEngineWorkers (dedicated workers, no DOM)
//   Phase 2 — enable kStatorEngineServiceWorkers (service workers)
//   Phase 3 — enable kStatorEngineMainFrame (main-frame renderer)
//
// Alternatively, the umbrella flag kStatorEngine enables all three phases
// simultaneously and is the canonical flag for the --enable-features= switch.

#ifndef CHROMIUM_CONTENT_STATOR_ENGINE_SELECTOR_H_
#define CHROMIUM_CONTENT_STATOR_ENGINE_SELECTOR_H_

namespace content {

// The type of renderer context being initialised.
enum class RendererType {
  // A dedicated Web Worker (WorkerGlobalScope, no DOM access).
  kWorker,
  // A Service Worker (ServiceWorkerGlobalScope, intercepts network).
  kServiceWorker,
  // A main-frame or sub-frame renderer (Document, full DOM access).
  kMainFrame,
};

// Returns true if the Stator engine should be used for a renderer context of
// the given type, based on the currently active Chrome feature flags.
//
// The function checks, in priority order:
//   1. kStatorEngine          — umbrella flag (all context types)
//   2. kStatorEngineWorkers   — workers only
//   3. kStatorEngineServiceWorkers — service workers only
//   4. kStatorEngineMainFrame — main frame only
//
// Thread-safety: safe to call from any thread after base::FeatureList has been
// initialised (i.e. after ChromeMainDelegate::BasicStartupComplete).
bool IsStatorEnabledFor(RendererType type);

}  // namespace content

#endif  // CHROMIUM_CONTENT_STATOR_ENGINE_SELECTOR_H_
