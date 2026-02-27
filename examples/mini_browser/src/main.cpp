/**
 * mini_browser — Placeholder C++ application demonstrating Stator engine usage.
 *
 * This sample simulates the lifecycle a real browser embedding would follow:
 *
 *   1. Initialize — create one Stator isolate per browser tab / worker.
 *   2. Load page  — parse HTML, extract inline <script> content (stubbed here).
 *   3. Execute    — run the extracted scripts through the engine (stubbed here).
 *   4. Idle GC    — trigger a minor collection between navigations.
 *   5. Teardown   — destroy the isolate when the tab is closed.
 *
 * NOTE: Script execution is not yet implemented in the engine.  The calls
 *       below show the *intended* API surface; they will be filled in as
 *       stator_core gains an interpreter / bytecode compiler.
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "stator.h"

/* -------------------------------------------------------------------------
 * Minimal HTML parser stub
 *
 * Extracts the content of every <script>…</script> block found in `html`.
 * In a real browser this would be part of a full HTML5 parser.
 * ------------------------------------------------------------------------- */
static std::vector<std::string> extract_scripts(const char *html) {
    std::vector<std::string> scripts;
    const char *open_tag  = "<script>";
    const char *close_tag = "</script>";
    const char *cursor    = html;

    while (true) {
        const char *start = std::strstr(cursor, open_tag);
        if (!start) break;
        start += std::strlen(open_tag);

        const char *end = std::strstr(start, close_tag);
        if (!end) break;

        scripts.emplace_back(start, static_cast<std::size_t>(end - start));
        cursor = end + std::strlen(close_tag);
    }

    return scripts;
}

/* -------------------------------------------------------------------------
 * Browser tab simulation
 * ------------------------------------------------------------------------- */

/** Represents one browser tab, owning its own JS engine isolate. */
class BrowserTab {
public:
    explicit BrowserTab(const char *url) : url_(url), isolate_(nullptr) {
        std::printf("[tab] opening   %s\n", url_.c_str());
        isolate_ = stator_isolate_create();
        if (!isolate_) {
            std::fprintf(stderr, "[tab] ERROR: failed to create Stator isolate\n");
        }
    }

    ~BrowserTab() {
        if (isolate_) {
            std::printf("[tab] closing   %s\n", url_.c_str());
            stator_isolate_destroy(isolate_);
            isolate_ = nullptr;
        }
    }

    /* Disallow copy; allow move (simplified). */
    BrowserTab(const BrowserTab &)            = delete;
    BrowserTab &operator=(const BrowserTab &) = delete;

    /**
     * Simulate loading a page.
     *
     * @param html  Raw HTML source of the page.
     */
    void load(const char *html) {
        if (!isolate_) return;

        std::printf("[tab] loading   %s\n", url_.c_str());

        /* --- (stub) Parse HTML & build DOM -------------------------------- */
        std::printf("[tab] parsed    HTML document\n");

        /* --- Extract and run inline scripts ------------------------------- */
        auto scripts = extract_scripts(html);
        std::printf("[tab] found     %zu inline script(s)\n", scripts.size());

        for (std::size_t i = 0; i < scripts.size(); ++i) {
            std::printf("[tab] executing script[%zu]: %s\n", i,
                        scripts[i].c_str());
            /* TODO: call stator_context_eval(isolate_, scripts[i].c_str())
             *       once the interpreter is implemented. */
            std::printf("[tab] (script execution not yet implemented)\n");
        }

        /* --- Run a GC pass to clean up short-lived allocations ----------- */
        std::printf("[tab] gc        collecting nursery\n");
        stator_isolate_gc(isolate_);

        std::printf("[tab] loaded    %s\n", url_.c_str());
    }

private:
    std::string    url_;
    StatorIsolate *isolate_;
};

/* -------------------------------------------------------------------------
 * Entry point
 * ------------------------------------------------------------------------- */
int main() {
    std::printf("=== Stator mini-browser (placeholder) ===\n\n");

    /* Simulate two tabs with simple HTML pages containing inline scripts. */
    const char *page_a =
        "<html><body>"
        "<h1>Hello, Stator!</h1>"
        "<script>console.log('page A loaded');</script>"
        "</body></html>";

    const char *page_b =
        "<html><body>"
        "<script>const x = 1 + 2;</script>"
        "<script>console.log('x =', x);</script>"
        "</body></html>";

    {
        BrowserTab tab_a("https://example.com/page-a");
        tab_a.load(page_a);
    }

    std::printf("\n");

    {
        BrowserTab tab_b("https://example.com/page-b");
        tab_b.load(page_b);
    }

    std::printf("\nDone.\n");
    return 0;
}
