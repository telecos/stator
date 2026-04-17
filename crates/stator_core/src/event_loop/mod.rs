//! Event loop integration layer.
//!
//! Provides a [`EventLoop`] that coordinates macrotask scheduling, timer
//! management, and microtask queue draining — bridging the Stator engine with
//! embedders (e.g. Chromium's task runner).
//!
//! # Architecture
//!
//! ```text
//!  Embedder (C++)                           Stator
//!  ┌─────────────────┐     FFI            ┌───────────────────┐
//!  │ Chromium         │ ◄──────────────►  │ EventLoop          │
//!  │ MessageLoop      │  post_task /      │  ├─ task_queue     │
//!  │                  │  timer callbacks   │  ├─ timers         │
//!  └─────────────────┘                    │  └─ microtask_queue│
//!                                         └───────────────────┘
//! ```
//!
//! The [`EmbedderCallbacks`] trait lets the embedder inject its own scheduling
//! primitives while the engine manages the JS-visible task lifecycle.

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;

use crate::builtins::promise::MicrotaskQueue;

// ── TimerHandle ────────────────────────────────────────────────────────────────

/// Opaque identifier for a pending timer (returned by `set_timer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimerHandle(u64);

impl TimerHandle {
    /// Returns the raw numeric identifier.
    pub fn id(self) -> u64 {
        self.0
    }

    /// Reconstruct a handle from a raw id (e.g. received over FFI).
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

// ── Task ───────────────────────────────────────────────────────────────────────

/// A macrotask queued for later execution.
type TaskFn = Box<dyn FnOnce()>;

// ── TaskQueue ──────────────────────────────────────────────────────────────────

/// FIFO queue for macrotasks.
///
/// Macrotasks are scheduled by the embedder or by engine internals (e.g.
/// `setTimeout` / `setInterval`).  After each macrotask completes the event
/// loop drains the microtask queue before picking up the next macrotask.
pub struct TaskQueue {
    inner: RefCell<std::collections::VecDeque<TaskFn>>,
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskQueue {
    /// Create an empty task queue.
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(std::collections::VecDeque::new()),
        }
    }

    /// Push a macrotask onto the back of the queue.
    pub fn enqueue(&self, task: TaskFn) {
        self.inner.borrow_mut().push_back(task);
    }

    /// Pop the next macrotask from the front, or `None` if empty.
    pub fn dequeue(&self) -> Option<TaskFn> {
        self.inner.borrow_mut().pop_front()
    }

    /// Returns `true` if there are no pending macrotasks.
    pub fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    /// Returns the number of pending macrotasks.
    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }
}

// ── Timer entry (min-heap by deadline) ─────────────────────────────────────────

/// A scheduled timer with an absolute deadline.
struct TimerEntry {
    handle: TimerHandle,
    /// Absolute deadline in seconds (same epoch as
    /// [`EmbedderCallbacks::monotonic_time`]).
    deadline: f64,
    task: Option<TaskFn>,
}

// BinaryHeap is a max-heap; invert ordering so the *smallest* deadline pops
// first.
impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.handle == other.handle
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so the earliest deadline has highest priority.
        other
            .deadline
            .partial_cmp(&self.deadline)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.handle.0.cmp(&self.handle.0))
    }
}

// ── EmbedderCallbacks ──────────────────────────────────────────────────────────

/// Trait implemented by the embedder to provide platform-level scheduling
/// primitives.
///
/// A default (no-op) implementation is available via [`DefaultCallbacks`] for
/// testing and standalone usage.
pub trait EmbedderCallbacks {
    /// Post a task to the embedder's main-thread task runner.
    ///
    /// The embedder should schedule the provided closure for execution on the
    /// thread that owns the event loop.
    fn post_task(&self, task: TaskFn);

    /// Post a task to be executed after `delay_secs` seconds.
    fn post_delayed_task(&self, task: TaskFn, delay_secs: f64);

    /// Request an idle callback.  The embedder should invoke the provided
    /// closure during an idle period, passing the amount of idle time remaining
    /// (in seconds).
    fn request_idle_callback(&self, callback: Box<dyn FnOnce(f64)>);

    /// Return a monotonically increasing time in seconds.
    fn monotonic_time(&self) -> f64;
}

/// No-op embedder callbacks for standalone / test usage.
///
/// Tasks posted through this implementation are silently dropped (they will
/// never execute).  `monotonic_time` always returns `0.0`.
pub struct DefaultCallbacks;

impl EmbedderCallbacks for DefaultCallbacks {
    fn post_task(&self, _task: TaskFn) {}
    fn post_delayed_task(&self, _task: TaskFn, _delay_secs: f64) {}
    fn request_idle_callback(&self, _cb: Box<dyn FnOnce(f64)>) {}
    fn monotonic_time(&self) -> f64 {
        0.0
    }
}

// ── EventLoop ──────────────────────────────────────────────────────────────────

/// Central event loop that coordinates macrotask scheduling, timer management,
/// and microtask queue draining.
///
/// # Example
///
/// ```
/// use stator_js::builtins::promise::MicrotaskQueue;
/// use stator_js::event_loop::{DefaultCallbacks, EventLoop};
///
/// let mtq = MicrotaskQueue::new();
/// let mut el = EventLoop::new(mtq.clone(), Box::new(DefaultCallbacks));
///
/// // Enqueue a macrotask that itself enqueues a microtask.
/// let mtq2 = mtq.clone();
/// el.post_task(Box::new(move || {
///     mtq2.enqueue(Box::new(|| { /* microtask work */ }));
/// }));
///
/// // Spin the loop — runs the macrotask, then drains microtasks.
/// el.run_until_idle();
/// assert!(el.is_idle());
/// ```
pub struct EventLoop {
    task_queue: TaskQueue,
    microtask_queue: MicrotaskQueue,
    timers: RefCell<BinaryHeap<TimerEntry>>,
    next_timer_id: RefCell<u64>,
    cancelled_timers: RefCell<std::collections::HashSet<u64>>,
    callbacks: Rc<dyn EmbedderCallbacks>,
    running: RefCell<bool>,
}

impl EventLoop {
    /// Create a new event loop.
    ///
    /// `microtask_queue` — the shared microtask queue (also used by the promise
    /// subsystem).
    /// `callbacks` — embedder-provided scheduling hooks.
    pub fn new(microtask_queue: MicrotaskQueue, callbacks: Box<dyn EmbedderCallbacks>) -> Self {
        Self {
            task_queue: TaskQueue::new(),
            microtask_queue,
            timers: RefCell::new(BinaryHeap::new()),
            next_timer_id: RefCell::new(1),
            cancelled_timers: RefCell::new(std::collections::HashSet::new()),
            callbacks: Rc::from(callbacks),
            running: RefCell::new(false),
        }
    }

    // ── Macrotask API ──────────────────────────────────────────────────────

    /// Enqueue a macrotask for execution on the next turn of the loop.
    pub fn post_task(&self, task: TaskFn) {
        self.task_queue.enqueue(task);
    }

    /// Returns the number of pending macrotasks (does **not** count timers).
    pub fn pending_task_count(&self) -> usize {
        self.task_queue.len()
    }

    // ── Timer API ──────────────────────────────────────────────────────────

    /// Schedule a one-shot timer that fires after `delay_secs`.
    ///
    /// Returns a [`TimerHandle`] that can be passed to [`cancel_timer`](Self::cancel_timer).
    pub fn set_timer(&self, delay_secs: f64, task: TaskFn) -> TimerHandle {
        let mut id = self.next_timer_id.borrow_mut();
        let handle = TimerHandle(*id);
        *id = id.wrapping_add(1);

        let deadline = self.callbacks.monotonic_time() + delay_secs;
        self.timers.borrow_mut().push(TimerEntry {
            handle,
            deadline,
            task: Some(task),
        });
        handle
    }

    /// Cancel a previously scheduled timer.
    ///
    /// If the timer has already fired this is a no-op.
    pub fn cancel_timer(&self, handle: TimerHandle) {
        self.cancelled_timers.borrow_mut().insert(handle.0);
    }

    /// Returns the number of live (non-cancelled) timers.
    pub fn pending_timer_count(&self) -> usize {
        let cancelled = self.cancelled_timers.borrow();
        self.timers
            .borrow()
            .iter()
            .filter(|e| !cancelled.contains(&e.handle.0))
            .count()
    }

    // ── Microtask integration ──────────────────────────────────────────────

    /// Drain all pending microtasks (delegates to [`MicrotaskQueue::drain`]).
    pub fn drain_microtasks(&self) {
        self.microtask_queue.drain();
    }

    /// Returns a clone of the shared microtask queue.
    pub fn microtask_queue(&self) -> MicrotaskQueue {
        self.microtask_queue.clone()
    }

    // ── Run loop ───────────────────────────────────────────────────────────

    /// Execute one macrotask (if available) then drain microtasks.
    ///
    /// Returns `true` if a macrotask was executed, `false` if the queue was
    /// empty.
    pub fn tick(&self) -> bool {
        // 1. Fire any ready timers.
        self.fire_ready_timers();

        // 2. Run one macrotask.
        let ran = if let Some(task) = self.task_queue.dequeue() {
            task();
            true
        } else {
            false
        };

        // 3. Drain microtasks after the macrotask.
        self.microtask_queue.drain();

        ran
    }

    /// Spin the event loop until there are no pending macrotasks or ready
    /// timers, draining microtasks after each macrotask.
    pub fn run_until_idle(&self) {
        *self.running.borrow_mut() = true;
        loop {
            self.fire_ready_timers();

            match self.task_queue.dequeue() {
                Some(task) => {
                    task();
                    self.microtask_queue.drain();
                }
                None => break,
            }
        }
        // Final microtask drain in case timers enqueued microtasks.
        self.microtask_queue.drain();
        *self.running.borrow_mut() = false;
    }

    /// Returns `true` when there are no pending macrotasks, no pending
    /// microtasks, and no live timers.
    pub fn is_idle(&self) -> bool {
        self.task_queue.is_empty()
            && self.microtask_queue.is_empty()
            && self.pending_timer_count() == 0
    }

    /// Returns `true` while the event loop is inside [`run_until_idle`](Self::run_until_idle).
    pub fn is_running(&self) -> bool {
        *self.running.borrow()
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Move all timers whose deadline ≤ now into the macrotask queue.
    fn fire_ready_timers(&self) {
        let now = self.callbacks.monotonic_time();
        let cancelled = self.cancelled_timers.borrow();
        let mut timers = self.timers.borrow_mut();

        while let Some(entry) = timers.peek() {
            if entry.deadline > now {
                break;
            }
            let mut entry = timers.pop().expect("peek succeeded");
            if cancelled.contains(&entry.handle.0) {
                continue;
            }
            if let Some(task) = entry.task.take() {
                self.task_queue.enqueue(task);
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc as StdRc;

    /// Test-only callbacks with a controllable clock.
    struct TestCallbacks {
        time: Cell<f64>,
    }

    impl TestCallbacks {
        fn new() -> StdRc<Self> {
            StdRc::new(Self {
                time: Cell::new(0.0),
            })
        }

        fn advance(&self, secs: f64) {
            self.time.set(self.time.get() + secs);
        }
    }

    impl EmbedderCallbacks for TestCallbacks {
        fn post_task(&self, _task: TaskFn) {}
        fn post_delayed_task(&self, _task: TaskFn, _delay_secs: f64) {}
        fn request_idle_callback(&self, _cb: Box<dyn FnOnce(f64)>) {}
        fn monotonic_time(&self) -> f64 {
            self.time.get()
        }
    }

    /// Helper: build an `EventLoop` wired to `TestCallbacks`.
    fn make_loop(clock: StdRc<TestCallbacks>) -> EventLoop {
        // Wrap the Rc'd TestCallbacks in a newtype so we can put it in a Box.
        struct Wrapper(StdRc<TestCallbacks>);
        impl EmbedderCallbacks for Wrapper {
            fn post_task(&self, task: TaskFn) {
                self.0.post_task(task);
            }
            fn post_delayed_task(&self, task: TaskFn, d: f64) {
                self.0.post_delayed_task(task, d);
            }
            fn request_idle_callback(&self, cb: Box<dyn FnOnce(f64)>) {
                self.0.request_idle_callback(cb);
            }
            fn monotonic_time(&self) -> f64 {
                self.0.monotonic_time()
            }
        }

        EventLoop::new(MicrotaskQueue::new(), Box::new(Wrapper(clock)))
    }

    #[test]
    fn test_task_queue_fifo_order() {
        let log = StdRc::new(RefCell::new(Vec::<u32>::new()));
        let q = TaskQueue::new();

        for i in 0..5 {
            let l = StdRc::clone(&log);
            q.enqueue(Box::new(move || l.borrow_mut().push(i)));
        }
        assert_eq!(q.len(), 5);

        while let Some(t) = q.dequeue() {
            t();
        }
        assert_eq!(*log.borrow(), vec![0, 1, 2, 3, 4]);
        assert!(q.is_empty());
    }

    #[test]
    fn test_event_loop_tick_runs_one_task_and_drains_microtasks() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);

        let log = StdRc::new(RefCell::new(Vec::<&str>::new()));
        let mtq = el.microtask_queue();

        let l1 = StdRc::clone(&log);
        let l2 = StdRc::clone(&log);
        let mtq2 = mtq.clone();
        el.post_task(Box::new(move || {
            l1.borrow_mut().push("macro");
            mtq2.enqueue(Box::new(move || l2.borrow_mut().push("micro")));
        }));

        assert!(el.tick());
        assert_eq!(*log.borrow(), vec!["macro", "micro"]);
    }

    #[test]
    fn test_event_loop_tick_returns_false_when_empty() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);
        assert!(!el.tick());
    }

    #[test]
    fn test_event_loop_run_until_idle() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);

        let counter = StdRc::new(Cell::new(0u32));
        for _ in 0..3 {
            let c = StdRc::clone(&counter);
            el.post_task(Box::new(move || {
                c.set(c.get() + 1);
            }));
        }

        el.run_until_idle();
        assert_eq!(counter.get(), 3);
        assert!(el.is_idle());
    }

    #[test]
    fn test_timer_fires_after_deadline() {
        let clock = TestCallbacks::new();
        let el = make_loop(StdRc::clone(&clock));

        let fired = StdRc::new(Cell::new(false));
        let f = StdRc::clone(&fired);
        el.set_timer(1.0, Box::new(move || f.set(true)));

        // Before deadline — should not fire.
        el.tick();
        assert!(!fired.get());

        // Advance past deadline.
        clock.advance(1.5);
        el.tick();
        assert!(fired.get());
    }

    #[test]
    fn test_cancel_timer() {
        let clock = TestCallbacks::new();
        let el = make_loop(StdRc::clone(&clock));

        let fired = StdRc::new(Cell::new(false));
        let f = StdRc::clone(&fired);
        let handle = el.set_timer(1.0, Box::new(move || f.set(true)));

        el.cancel_timer(handle);
        clock.advance(2.0);
        el.run_until_idle();
        assert!(!fired.get());
        assert_eq!(el.pending_timer_count(), 0);
    }

    #[test]
    fn test_multiple_timers_fire_in_deadline_order() {
        let clock = TestCallbacks::new();
        let el = make_loop(StdRc::clone(&clock));

        let log = StdRc::new(RefCell::new(Vec::<u32>::new()));

        for i in (1..=3).rev() {
            let l = StdRc::clone(&log);
            el.set_timer(f64::from(i), Box::new(move || l.borrow_mut().push(i)));
        }

        clock.advance(5.0);
        el.run_until_idle();
        assert_eq!(*log.borrow(), vec![1, 2, 3]);
    }

    #[test]
    fn test_is_idle_reflects_all_queues() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);
        assert!(el.is_idle());

        el.post_task(Box::new(|| {}));
        assert!(!el.is_idle());

        el.run_until_idle();
        assert!(el.is_idle());
    }

    #[test]
    fn test_timer_handle_id() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);

        let h1 = el.set_timer(1.0, Box::new(|| {}));
        let h2 = el.set_timer(2.0, Box::new(|| {}));
        assert_ne!(h1.id(), h2.id());
    }

    #[test]
    fn test_default_callbacks_no_panic() {
        let mtq = MicrotaskQueue::new();
        let el = EventLoop::new(mtq, Box::new(DefaultCallbacks));
        el.post_task(Box::new(|| {}));
        el.run_until_idle();
    }

    #[test]
    fn test_microtask_enqueued_during_macrotask_drains_same_turn() {
        let clock = TestCallbacks::new();
        let el = make_loop(clock);
        let mtq = el.microtask_queue();

        let log = StdRc::new(RefCell::new(Vec::<u32>::new()));
        let l1 = StdRc::clone(&log);
        let l2 = StdRc::clone(&log);
        let l3 = StdRc::clone(&log);
        let mtq2 = mtq.clone();

        el.post_task(Box::new(move || {
            l1.borrow_mut().push(1);
            let l2_inner = l2;
            mtq2.enqueue(Box::new(move || {
                l2_inner.borrow_mut().push(2);
            }));
        }));
        el.post_task(Box::new(move || {
            l3.borrow_mut().push(3);
        }));

        // First tick: macro(1), then micro(2).
        el.tick();
        assert_eq!(*log.borrow(), vec![1, 2]);

        // Second tick: macro(3).
        el.tick();
        assert_eq!(*log.borrow(), vec![1, 2, 3]);
    }
}
