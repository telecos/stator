//! ECMAScript §27 `Promise` built-in and microtask queue.
//!
//! Implements the full Promises/A+ specification:
//! - [`JsPromise`] — a shared-state promise handle.
//! - [`MicrotaskQueue`] — FIFO queue for asynchronous reactions.
//! - Constructor ([`promise_new`]) and prototype methods ([`promise_then`],
//!   [`promise_catch`], [`promise_finally`]).
//! - Static combinators ([`promise_all`], [`promise_all_settled`],
//!   [`promise_any`], [`promise_race`], [`promise_resolve`],
//!   [`promise_reject`], [`promise_with_resolvers`]).
//!
//! # Design notes
//!
//! Promises use `Rc<RefCell<_>>` for shared interior mutability so that multiple
//! handles to the same promise can coexist safely in a single-threaded context.
//! The [`MicrotaskQueue`] is also wrapped in `Rc<RefCell<_>>` so that reactions
//! enqueued *during* a drain can themselves enqueue further reactions.
//!
//! Fulfillment and rejection handlers have the signature
//! `Fn(JsValue) -> Result<JsValue, JsValue>`:
//! - `Ok(v)` → resolve the downstream promise with `v`.
//! - `Err(r)` → reject the downstream promise with `r`.
//!
//! # References
//!
//! * ECMAScript 2025 Language Specification §27 — *Control Abstraction Objects*
//! * Promises/A+ specification — <https://promisesaplus.com/>

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::objects::value::JsValue;

// ── Type aliases ───────────────────────────────────────────────────────────────

/// The inner storage type for [`MicrotaskQueue`].
type MicrotaskQueueInner = Rc<RefCell<VecDeque<Box<dyn FnOnce()>>>>;

/// A handler closure used by [`promise_then`] and related functions.
///
/// Returns `Ok(value)` to resolve the downstream promise or `Err(reason)` to
/// reject it.
pub type PromiseHandler = Box<dyn Fn(JsValue) -> Result<JsValue, JsValue>>;

// ── MicrotaskQueue ─────────────────────────────────────────────────────────────

/// FIFO microtask queue.
///
/// Microtasks are pushed onto the back of the queue by promise reactions and
/// drained from the front by [`MicrotaskQueue::drain`].  The drain loop
/// continues until the queue is empty, picking up any tasks that are enqueued
/// *during* the drain.
///
/// Cloning a [`MicrotaskQueue`] gives a second handle to the **same** underlying
/// queue (shallow copy via [`Rc`]).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::MicrotaskQueue;
///
/// let queue = MicrotaskQueue::new();
/// let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::<i32>::new()));
/// let log2 = std::rc::Rc::clone(&log);
/// queue.enqueue(Box::new(move || log2.borrow_mut().push(1)));
/// assert!(!queue.is_empty());
/// queue.drain();
/// assert!(queue.is_empty());
/// assert_eq!(*log.borrow(), vec![1]);
/// ```
#[derive(Clone)]
pub struct MicrotaskQueue(MicrotaskQueueInner);

impl Default for MicrotaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl MicrotaskQueue {
    /// Create an empty microtask queue.
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(VecDeque::new())))
    }

    /// Push a microtask onto the back of the queue.
    ///
    /// The task will be run in FIFO order by the next call to
    /// [`drain`](Self::drain).
    pub fn enqueue(&self, task: Box<dyn FnOnce()>) {
        self.0.borrow_mut().push_back(task);
    }

    /// Drain the queue: run all pending microtasks in FIFO order, including
    /// any that are enqueued by tasks that run during this drain.
    ///
    /// Returns when the queue is empty.
    pub fn drain(&self) {
        loop {
            let task = self.0.borrow_mut().pop_front();
            match task {
                Some(t) => t(),
                None => break,
            }
        }
    }

    /// Returns `true` if there are no pending microtasks.
    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    /// Returns the number of pending microtasks.
    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }
}

// ── Internal promise state ─────────────────────────────────────────────────────

/// Internal state of a [`JsPromise`].
enum PromiseStateInner {
    Pending {
        /// Reactions fired when the promise is fulfilled.
        fulfill_reactions: Vec<Box<dyn FnOnce(JsValue)>>,
        /// Reactions fired when the promise is rejected.
        reject_reactions: Vec<Box<dyn FnOnce(JsValue)>>,
    },
    Fulfilled(JsValue),
    Rejected(JsValue),
}

struct PromiseInner {
    state: PromiseStateInner,
    /// Whether at least one rejection handler has been attached.
    /// Used by [`UnhandledRejectionTracker`] to detect unhandled rejections.
    is_handled: bool,
}

// ── JsPromise ──────────────────────────────────────────────────────────────────

/// Observable state of a [`JsPromise`].
#[derive(Debug, Clone, PartialEq)]
pub enum PromiseState {
    /// The promise has not yet been settled.
    Pending,
    /// The promise was fulfilled with the given value.
    Fulfilled(JsValue),
    /// The promise was rejected with the given reason.
    Rejected(JsValue),
}

/// A JavaScript `Promise` — a handle to a shared, potentially-deferred
/// computation result.
///
/// Cloning a [`JsPromise`] gives a second handle to the **same** underlying
/// promise (shallow copy via [`Rc`]).
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{
///     MicrotaskQueue, promise_resolve, promise_then,
/// };
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_resolve(JsValue::Smi(42), &queue);
/// let result = std::rc::Rc::new(std::cell::RefCell::new(JsValue::Undefined));
/// let r2 = std::rc::Rc::clone(&result);
/// promise_then(
///     &p,
///     Some(Box::new(move |v| { *r2.borrow_mut() = v.clone(); Ok(v) })),
///     None,
///     &queue,
/// );
/// queue.drain();
/// assert_eq!(*result.borrow(), JsValue::Smi(42));
/// ```
#[derive(Clone)]
pub struct JsPromise(Rc<RefCell<PromiseInner>>);

impl std::fmt::Debug for JsPromise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JsPromise({:?})", self.state())
    }
}

impl PartialEq for JsPromise {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl JsPromise {
    fn new_pending() -> Self {
        Self(Rc::new(RefCell::new(PromiseInner {
            state: PromiseStateInner::Pending {
                fulfill_reactions: Vec::new(),
                reject_reactions: Vec::new(),
            },
            is_handled: false,
        })))
    }

    /// Returns the current observable state of this promise.
    pub fn state(&self) -> PromiseState {
        match &self.0.borrow().state {
            PromiseStateInner::Pending { .. } => PromiseState::Pending,
            PromiseStateInner::Fulfilled(v) => PromiseState::Fulfilled(v.clone()),
            PromiseStateInner::Rejected(r) => PromiseState::Rejected(r.clone()),
        }
    }

    /// Returns `true` if the promise is still pending.
    pub fn is_pending(&self) -> bool {
        matches!(self.0.borrow().state, PromiseStateInner::Pending { .. })
    }

    /// Returns `true` if the promise has been fulfilled.
    pub fn is_fulfilled(&self) -> bool {
        matches!(self.0.borrow().state, PromiseStateInner::Fulfilled(_))
    }

    /// Returns `true` if the promise has been rejected.
    pub fn is_rejected(&self) -> bool {
        matches!(self.0.borrow().state, PromiseStateInner::Rejected(_))
    }

    /// Returns `true` if at least one rejection handler has been attached.
    pub fn is_handled(&self) -> bool {
        self.0.borrow().is_handled
    }

    /// Returns the fulfillment value, or `None` if not yet fulfilled.
    pub fn value(&self) -> Option<JsValue> {
        match &self.0.borrow().state {
            PromiseStateInner::Fulfilled(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Returns the rejection reason, or `None` if not rejected.
    pub fn reason(&self) -> Option<JsValue> {
        match &self.0.borrow().state {
            PromiseStateInner::Rejected(r) => Some(r.clone()),
            _ => None,
        }
    }

    /// Resolve the promise with `value`.
    ///
    /// Transitions state from `Pending` to `Fulfilled(value)` and schedules
    /// all pending fulfillment reactions as microtasks on `queue`.
    /// A no-op if the promise is already settled.
    fn resolve(&self, value: JsValue, queue: &MicrotaskQueue) {
        let reactions = {
            let mut inner = self.0.borrow_mut();
            if !matches!(inner.state, PromiseStateInner::Pending { .. }) {
                return;
            }
            let old = std::mem::replace(
                &mut inner.state,
                PromiseStateInner::Fulfilled(value.clone()),
            );
            match old {
                PromiseStateInner::Pending {
                    fulfill_reactions, ..
                } => fulfill_reactions,
                _ => unreachable!(),
            }
        };
        for reaction in reactions {
            let val = value.clone();
            queue.enqueue(Box::new(move || reaction(val)));
        }
    }

    /// Reject the promise with `reason`.
    ///
    /// Transitions state from `Pending` to `Rejected(reason)` and schedules
    /// all pending rejection reactions as microtasks on `queue`.
    /// A no-op if the promise is already settled.
    fn reject(&self, reason: JsValue, queue: &MicrotaskQueue) {
        let reactions = {
            let mut inner = self.0.borrow_mut();
            if !matches!(inner.state, PromiseStateInner::Pending { .. }) {
                return;
            }
            let old = std::mem::replace(
                &mut inner.state,
                PromiseStateInner::Rejected(reason.clone()),
            );
            match old {
                PromiseStateInner::Pending {
                    reject_reactions, ..
                } => reject_reactions,
                _ => unreachable!(),
            }
        };
        for reaction in reactions {
            let r = reason.clone();
            queue.enqueue(Box::new(move || reaction(r)));
        }
    }

    /// Register a pair of reactions (fulfill, reject) on this promise.
    ///
    /// If the promise is still pending the reactions are stored.  If it is
    /// already settled, the appropriate reaction is scheduled immediately as a
    /// microtask on `queue`.
    fn add_reactions(
        &self,
        fulfill: Box<dyn FnOnce(JsValue)>,
        reject: Box<dyn FnOnce(JsValue)>,
        queue: &MicrotaskQueue,
    ) {
        enum Settled {
            Fulfilled(JsValue),
            Rejected(JsValue),
        }
        let settled = {
            let mut inner = self.0.borrow_mut();
            inner.is_handled = true;
            match &mut inner.state {
                PromiseStateInner::Pending {
                    fulfill_reactions,
                    reject_reactions,
                } => {
                    fulfill_reactions.push(fulfill);
                    reject_reactions.push(reject);
                    return;
                }
                PromiseStateInner::Fulfilled(v) => Settled::Fulfilled(v.clone()),
                PromiseStateInner::Rejected(r) => Settled::Rejected(r.clone()),
            }
        };
        match settled {
            Settled::Fulfilled(v) => queue.enqueue(Box::new(move || fulfill(v))),
            Settled::Rejected(r) => queue.enqueue(Box::new(move || reject(r))),
        }
    }
}

// ── Constructor ────────────────────────────────────────────────────────────────

/// ECMAScript §27.2.3.1 `new Promise(executor)`.
///
/// Calls `executor(resolve, reject)` **synchronously**.  The returned promise is
/// fulfilled when the executor calls `resolve(value)` and rejected when it calls
/// `reject(reason)`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_new};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_new(|resolve, _reject| resolve(JsValue::Smi(1)), &queue);
/// assert!(p.is_fulfilled());
/// assert_eq!(p.value(), Some(JsValue::Smi(1)));
/// ```
pub fn promise_new<F>(executor: F, queue: &MicrotaskQueue) -> JsPromise
where
    F: FnOnce(Box<dyn FnOnce(JsValue)>, Box<dyn FnOnce(JsValue)>),
{
    let p = JsPromise::new_pending();
    let p1 = p.clone();
    let q1 = queue.clone();
    let resolve: Box<dyn FnOnce(JsValue)> = Box::new(move |v| p1.resolve(v, &q1));
    let p2 = p.clone();
    let q2 = queue.clone();
    let reject: Box<dyn FnOnce(JsValue)> = Box::new(move |r| p2.reject(r, &q2));
    executor(resolve, reject);
    p
}

// ── Static: Promise.resolve / Promise.reject ───────────────────────────────────

/// ECMAScript §27.2.4.5 `Promise.resolve(value)`.
///
/// Returns a promise that is already fulfilled with `value`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_resolve};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_resolve(JsValue::Smi(42), &queue);
/// assert!(p.is_fulfilled());
/// assert_eq!(p.value(), Some(JsValue::Smi(42)));
/// ```
pub fn promise_resolve(value: JsValue, queue: &MicrotaskQueue) -> JsPromise {
    promise_new(|resolve, _reject| resolve(value), queue)
}

/// ECMAScript §27.2.4.4 `Promise.reject(reason)`.
///
/// Returns a promise that is already rejected with `reason`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_reject};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_reject(JsValue::String("oops".to_string()), &queue);
/// assert!(p.is_rejected());
/// assert_eq!(p.reason(), Some(JsValue::String("oops".to_string())));
/// ```
pub fn promise_reject(reason: JsValue, queue: &MicrotaskQueue) -> JsPromise {
    promise_new(|_resolve, reject| reject(reason), queue)
}

// ── Prototype: then / catch / finally ─────────────────────────────────────────

/// ECMAScript §27.2.5.4 `Promise.prototype.then(onFulfilled, onRejected)`.
///
/// Returns a new promise that is settled according to the result of whichever
/// handler fires:
///
/// - Handler returns `Ok(v)` → new promise resolves with `v`.
/// - Handler returns `Err(r)` → new promise rejects with `r`.
/// - `None` handler → value/reason passes through unchanged.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_resolve, promise_then};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_resolve(JsValue::Smi(5), &queue);
/// let p2 = promise_then(
///     &p,
///     Some(Box::new(|v| if let JsValue::Smi(n) = v { Ok(JsValue::Smi(n * 2)) } else { Ok(v) })),
///     None,
///     &queue,
/// );
/// queue.drain();
/// assert_eq!(p2.value(), Some(JsValue::Smi(10)));
/// ```
pub fn promise_then(
    promise: &JsPromise,
    on_fulfilled: Option<PromiseHandler>,
    on_rejected: Option<PromiseHandler>,
    queue: &MicrotaskQueue,
) -> JsPromise {
    let p2 = JsPromise::new_pending();

    let p2a = p2.clone();
    let qa = queue.clone();
    let fulfill_reaction: Box<dyn FnOnce(JsValue)> = Box::new(move |v| {
        let result = if let Some(h) = &on_fulfilled {
            h(v)
        } else {
            Ok(v)
        };
        match result {
            Ok(val) => p2a.resolve(val, &qa),
            Err(reason) => p2a.reject(reason, &qa),
        }
    });

    let p2b = p2.clone();
    let qb = queue.clone();
    let reject_reaction: Box<dyn FnOnce(JsValue)> = Box::new(move |r| {
        let result = if let Some(h) = &on_rejected {
            h(r)
        } else {
            Err(r)
        };
        match result {
            Ok(val) => p2b.resolve(val, &qb),
            Err(reason) => p2b.reject(reason, &qb),
        }
    });

    promise.add_reactions(fulfill_reaction, reject_reaction, queue);
    p2
}

/// ECMAScript §27.2.5.1 `Promise.prototype.catch(onRejected)`.
///
/// Equivalent to `promise_then(promise, None, Some(onRejected), queue)`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_reject, promise_catch};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_reject(JsValue::String("err".to_string()), &queue);
/// let result = std::rc::Rc::new(std::cell::RefCell::new(JsValue::Undefined));
/// let r2 = std::rc::Rc::clone(&result);
/// promise_catch(&p, Box::new(move |r| { *r2.borrow_mut() = r.clone(); Ok(JsValue::Undefined) }), &queue);
/// queue.drain();
/// assert_eq!(*result.borrow(), JsValue::String("err".to_string()));
/// ```
pub fn promise_catch(
    promise: &JsPromise,
    on_rejected: PromiseHandler,
    queue: &MicrotaskQueue,
) -> JsPromise {
    promise_then(promise, None, Some(on_rejected), queue)
}

/// ECMAScript §27.2.5.3 `Promise.prototype.finally(onFinally)`.
///
/// Calls `onFinally()` regardless of the settlement outcome:
///
/// - `onFinally` returns `Ok(())` → the original value or rejection reason
///   passes through unchanged.
/// - `onFinally` returns `Err(r)` → the returned promise is rejected with `r`.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_resolve, promise_finally};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let ran = std::rc::Rc::new(std::cell::Cell::new(false));
/// let ran2 = std::rc::Rc::clone(&ran);
/// let p = promise_resolve(JsValue::Smi(1), &queue);
/// let p2 = promise_finally(&p, Box::new(move || { ran2.set(true); Ok(()) }), &queue);
/// queue.drain();
/// assert!(ran.get());
/// assert_eq!(p2.value(), Some(JsValue::Smi(1)));
/// ```
pub fn promise_finally(
    promise: &JsPromise,
    on_finally: Box<dyn Fn() -> Result<(), JsValue>>,
    queue: &MicrotaskQueue,
) -> JsPromise {
    let on_finally = Rc::new(on_finally);
    let on_finally_f = Rc::clone(&on_finally);
    let on_fulfilled = Some(Box::new(move |v: JsValue| match on_finally_f() {
        Ok(()) => Ok(v),
        Err(e) => Err(e),
    }) as PromiseHandler);

    let on_finally_r = Rc::clone(&on_finally);
    let on_rejected = Some(Box::new(move |r: JsValue| match on_finally_r() {
        Ok(()) => Err(r),
        Err(e) => Err(e),
    }) as PromiseHandler);

    promise_then(promise, on_fulfilled, on_rejected, queue)
}

// ── Static: Promise.all ───────────────────────────────────────────────────────

/// ECMAScript §27.2.4.1 `Promise.all(promises)`.
///
/// Returns a promise that:
/// - Resolves with a [`JsValue::Array`] of all fulfillment values (in input
///   order) when **all** input promises have fulfilled.
/// - Rejects with the first rejection reason as soon as **any** input promise
///   rejects.
///
/// An empty input resolves immediately with an empty array.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_resolve, promise_all};
/// use stator_core::objects::value::JsValue;
/// use std::rc::Rc;
///
/// let queue = MicrotaskQueue::new();
/// let p = promise_all(
///     vec![promise_resolve(JsValue::Smi(1), &queue), promise_resolve(JsValue::Smi(2), &queue)],
///     &queue,
/// );
/// queue.drain();
/// if let Some(JsValue::Array(arr)) = p.value() {
///     assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(2)]);
/// } else {
///     panic!("expected Array");
/// }
/// ```
pub fn promise_all(promises: Vec<JsPromise>, queue: &MicrotaskQueue) -> JsPromise {
    let count = promises.len();
    if count == 0 {
        return promise_resolve(JsValue::new_array(Vec::new()), queue);
    }

    let results: Rc<RefCell<Vec<Option<JsValue>>>> = Rc::new(RefCell::new(vec![None; count]));
    let remaining: Rc<RefCell<usize>> = Rc::new(RefCell::new(count));
    let p_result = JsPromise::new_pending();

    for (i, p) in promises.into_iter().enumerate() {
        let results_f = Rc::clone(&results);
        let remaining_f = Rc::clone(&remaining);
        let p_result_f = p_result.clone();
        let q_f = queue.clone();

        let p_result_r = p_result.clone();
        let q_r = queue.clone();

        let on_fulfilled = Some(Box::new(move |v: JsValue| {
            results_f.borrow_mut()[i] = Some(v);
            let mut rem = remaining_f.borrow_mut();
            *rem -= 1;
            if *rem == 0 {
                // SAFETY: all `count` slots have been written before `rem` reaches 0.
                let all: Vec<JsValue> = results_f
                    .borrow()
                    .iter()
                    .map(|r| r.clone().expect("slot filled by preceding reaction"))
                    .collect();
                p_result_f.resolve(JsValue::new_array(all), &q_f);
            }
            Ok(JsValue::Undefined)
        }) as PromiseHandler);

        let on_rejected = Some(Box::new(move |r: JsValue| {
            p_result_r.reject(r.clone(), &q_r);
            Err(r)
        }) as PromiseHandler);

        promise_then(&p, on_fulfilled, on_rejected, queue);
    }

    p_result
}

// ── Static: Promise.allSettled ────────────────────────────────────────────────

/// ECMAScript §27.2.4.2 `Promise.allSettled(promises)`.
///
/// Always resolves (never rejects) with a [`JsValue::Array`] of one result per
/// input promise.  Each result is itself a two-element [`JsValue::Array`]:
///
/// - `["fulfilled", value]` for a fulfilled promise.
/// - `["rejected",  reason]` for a rejected promise.
///
/// An empty input resolves immediately with an empty array.
pub fn promise_all_settled(promises: Vec<JsPromise>, queue: &MicrotaskQueue) -> JsPromise {
    let count = promises.len();
    if count == 0 {
        return promise_resolve(JsValue::new_array(Vec::new()), queue);
    }

    let results: Rc<RefCell<Vec<Option<JsValue>>>> = Rc::new(RefCell::new(vec![None; count]));
    let remaining: Rc<RefCell<usize>> = Rc::new(RefCell::new(count));
    let p_result = JsPromise::new_pending();

    for (i, p) in promises.into_iter().enumerate() {
        let results_f = Rc::clone(&results);
        let remaining_f = Rc::clone(&remaining);
        let p_result_f = p_result.clone();
        let q_f = queue.clone();

        let results_r = Rc::clone(&results);
        let remaining_r = Rc::clone(&remaining);
        let p_result_r = p_result.clone();
        let q_r = queue.clone();

        let on_fulfilled = Some(Box::new(move |v: JsValue| {
            results_f.borrow_mut()[i] = Some(JsValue::new_array(vec![
                JsValue::String("fulfilled".to_string()),
                v,
            ]));
            settle_all_settled(&results_f, &remaining_f, &p_result_f, &q_f);
            Ok(JsValue::Undefined)
        }) as PromiseHandler);

        let on_rejected = Some(Box::new(move |r: JsValue| {
            results_r.borrow_mut()[i] = Some(JsValue::new_array(vec![
                JsValue::String("rejected".to_string()),
                r.clone(),
            ]));
            settle_all_settled(&results_r, &remaining_r, &p_result_r, &q_r);
            Ok(JsValue::Undefined)
        }) as PromiseHandler);

        promise_then(&p, on_fulfilled, on_rejected, queue);
    }

    p_result
}

/// Decrement `remaining` and, if it reaches zero, resolve `p_result` with
/// all collected values.
fn settle_all_settled(
    results: &Rc<RefCell<Vec<Option<JsValue>>>>,
    remaining: &Rc<RefCell<usize>>,
    p_result: &JsPromise,
    queue: &MicrotaskQueue,
) {
    let mut rem = remaining.borrow_mut();
    *rem -= 1;
    if *rem == 0 {
        // SAFETY: all `count` slots have been written before `rem` reaches 0.
        let all: Vec<JsValue> = results
            .borrow()
            .iter()
            .map(|r| r.clone().expect("slot filled by preceding reaction"))
            .collect();
        p_result.resolve(JsValue::new_array(all), queue);
    }
}

// ── Static: Promise.any ───────────────────────────────────────────────────────

/// ECMAScript §27.2.4.3 `Promise.any(promises)`.
///
/// Returns a promise that:
/// - Resolves with the first fulfillment value.
/// - Rejects with a [`JsValue::Array`] of all rejection reasons (in input
///   order) when **all** input promises have rejected.
///
/// An empty input rejects immediately with an empty array.
pub fn promise_any(promises: Vec<JsPromise>, queue: &MicrotaskQueue) -> JsPromise {
    let count = promises.len();
    if count == 0 {
        return promise_reject(JsValue::new_array(Vec::new()), queue);
    }

    let errors: Rc<RefCell<Vec<Option<JsValue>>>> = Rc::new(RefCell::new(vec![None; count]));
    let remaining: Rc<RefCell<usize>> = Rc::new(RefCell::new(count));
    let p_result = JsPromise::new_pending();

    for (i, p) in promises.into_iter().enumerate() {
        let p_result_f = p_result.clone();
        let q_f = queue.clone();

        let errors_r = Rc::clone(&errors);
        let remaining_r = Rc::clone(&remaining);
        let p_result_r = p_result.clone();
        let q_r = queue.clone();

        let on_fulfilled = Some(Box::new(move |v: JsValue| {
            p_result_f.resolve(v.clone(), &q_f);
            Ok(v)
        }) as PromiseHandler);

        let on_rejected = Some(Box::new(move |r: JsValue| {
            errors_r.borrow_mut()[i] = Some(r.clone());
            let mut rem = remaining_r.borrow_mut();
            *rem -= 1;
            if *rem == 0 {
                // SAFETY: all `count` slots have been written before `rem` reaches 0.
                let all: Vec<JsValue> = errors_r
                    .borrow()
                    .iter()
                    .map(|e| e.clone().expect("slot filled by preceding reaction"))
                    .collect();
                p_result_r.reject(JsValue::new_array(all), &q_r);
            }
            Err(r)
        }) as PromiseHandler);

        promise_then(&p, on_fulfilled, on_rejected, queue);
    }

    p_result
}

// ── Static: Promise.race ──────────────────────────────────────────────────────

/// ECMAScript §27.2.4.6 `Promise.race(promises)`.
///
/// Returns a promise that settles with the same value/reason as the **first**
/// input promise to settle.
///
/// An empty input returns a forever-pending promise.
pub fn promise_race(promises: Vec<JsPromise>, queue: &MicrotaskQueue) -> JsPromise {
    let p_result = JsPromise::new_pending();

    for p in promises {
        let p_result_f = p_result.clone();
        let q_f = queue.clone();
        let p_result_r = p_result.clone();
        let q_r = queue.clone();

        let on_fulfilled = Some(Box::new(move |v: JsValue| {
            p_result_f.resolve(v.clone(), &q_f);
            Ok(v)
        }) as PromiseHandler);

        let on_rejected = Some(Box::new(move |r: JsValue| {
            p_result_r.reject(r.clone(), &q_r);
            Err(r)
        }) as PromiseHandler);

        promise_then(&p, on_fulfilled, on_rejected, queue);
    }

    p_result
}

// ── Static: Promise.withResolvers ─────────────────────────────────────────────

/// The result of [`promise_with_resolvers`].
pub struct PromiseWithResolvers {
    /// The promise itself.
    pub promise: JsPromise,
    /// Resolves the promise with the given value.
    pub resolve: Box<dyn FnOnce(JsValue)>,
    /// Rejects the promise with the given reason.
    pub reject: Box<dyn FnOnce(JsValue)>,
}

/// ECMAScript §27.2.4.9 `Promise.withResolvers()`.
///
/// Returns a [`PromiseWithResolvers`] containing a fresh promise together with
/// its `resolve` and `reject` callbacks.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{MicrotaskQueue, promise_with_resolvers};
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let wr = promise_with_resolvers(&queue);
/// (wr.resolve)(JsValue::Smi(7));
/// assert!(wr.promise.is_fulfilled());
/// assert_eq!(wr.promise.value(), Some(JsValue::Smi(7)));
/// ```
pub fn promise_with_resolvers(queue: &MicrotaskQueue) -> PromiseWithResolvers {
    let p = JsPromise::new_pending();
    let p1 = p.clone();
    let q1 = queue.clone();
    let resolve: Box<dyn FnOnce(JsValue)> = Box::new(move |v| p1.resolve(v, &q1));
    let p2 = p.clone();
    let q2 = queue.clone();
    let reject: Box<dyn FnOnce(JsValue)> = Box::new(move |r| p2.reject(r, &q2));
    PromiseWithResolvers {
        promise: p,
        resolve,
        reject,
    }
}

// ── Unhandled Rejection Tracking ──────────────────────────────────────────────

/// Tracks promises that were rejected without a handler attached.
///
/// Register promises via [`track`](Self::track) and then call
/// [`collect_unhandled`](Self::collect_unhandled) after draining the microtask
/// queue to discover rejections that were never caught.
///
/// # Examples
///
/// ```
/// use stator_core::builtins::promise::{
///     MicrotaskQueue, UnhandledRejectionTracker, promise_reject, promise_catch,
/// };
/// use stator_core::objects::value::JsValue;
///
/// let queue = MicrotaskQueue::new();
/// let mut tracker = UnhandledRejectionTracker::new();
///
/// let p1 = promise_reject(JsValue::String("oops".into()), &queue);
/// tracker.track(p1);
///
/// let p2 = promise_reject(JsValue::String("handled".into()), &queue);
/// promise_catch(&p2, Box::new(|_| Ok(JsValue::Undefined)), &queue);
/// tracker.track(p2);
///
/// queue.drain();
/// let unhandled = tracker.collect_unhandled();
/// assert_eq!(unhandled.len(), 1);
/// assert_eq!(
///     unhandled[0].reason(),
///     Some(JsValue::String("oops".into())),
/// );
/// ```
pub struct UnhandledRejectionTracker {
    promises: Vec<JsPromise>,
}

impl Default for UnhandledRejectionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UnhandledRejectionTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            promises: Vec::new(),
        }
    }

    /// Register a promise for tracking.
    pub fn track(&mut self, promise: JsPromise) {
        self.promises.push(promise);
    }

    /// Return all tracked promises that are rejected and have no handler attached.
    ///
    /// This should be called **after** draining the microtask queue so that
    /// late-attached handlers have had a chance to run.
    pub fn collect_unhandled(&mut self) -> Vec<JsPromise> {
        let unhandled: Vec<JsPromise> = self
            .promises
            .drain(..)
            .filter(|p| p.is_rejected() && !p.is_handled())
            .collect();
        unhandled
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MicrotaskQueue ────────────────────────────────────────────────────────

    #[test]
    fn test_microtask_queue_fifo_ordering() {
        let queue = MicrotaskQueue::new();
        let log: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(Vec::new()));
        for i in 0..3_i32 {
            let log = Rc::clone(&log);
            queue.enqueue(Box::new(move || log.borrow_mut().push(i)));
        }
        queue.drain();
        assert_eq!(*log.borrow(), vec![0, 1, 2]);
    }

    #[test]
    fn test_microtask_queue_drain_picks_up_new_tasks() {
        let queue = MicrotaskQueue::new();
        let log: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(Vec::new()));

        let log1 = Rc::clone(&log);
        let queue2 = queue.clone();
        queue.enqueue(Box::new(move || {
            log1.borrow_mut().push(1);
            let log2 = Rc::clone(&log1);
            queue2.enqueue(Box::new(move || log2.borrow_mut().push(2)));
        }));
        queue.drain();
        assert_eq!(*log.borrow(), vec![1, 2]);
    }

    #[test]
    fn test_microtask_queue_len_and_is_empty() {
        let queue = MicrotaskQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        queue.enqueue(Box::new(|| {}));
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);
        queue.drain();
        assert!(queue.is_empty());
    }

    // ── promise_new ───────────────────────────────────────────────────────────

    #[test]
    fn test_promise_new_resolves_synchronously() {
        let queue = MicrotaskQueue::new();
        let p = promise_new(|resolve, _| resolve(JsValue::Smi(42)), &queue);
        assert!(p.is_fulfilled());
        assert_eq!(p.value(), Some(JsValue::Smi(42)));
    }

    #[test]
    fn test_promise_new_rejects_synchronously() {
        let queue = MicrotaskQueue::new();
        let p = promise_new(
            |_, reject| reject(JsValue::String("err".to_string())),
            &queue,
        );
        assert!(p.is_rejected());
        assert_eq!(p.reason(), Some(JsValue::String("err".to_string())));
    }

    #[test]
    fn test_promise_new_stays_pending_when_executor_does_nothing() {
        let queue = MicrotaskQueue::new();
        let p = promise_new(|_, _| {}, &queue);
        assert!(p.is_pending());
    }

    // ── promise_resolve / promise_reject ──────────────────────────────────────

    #[test]
    fn test_promise_resolve_static() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Boolean(true), &queue);
        assert!(p.is_fulfilled());
        assert_eq!(p.value(), Some(JsValue::Boolean(true)));
    }

    #[test]
    fn test_promise_reject_static() {
        let queue = MicrotaskQueue::new();
        let p = promise_reject(JsValue::Smi(0), &queue);
        assert!(p.is_rejected());
        assert_eq!(p.reason(), Some(JsValue::Smi(0)));
    }

    // ── promise_then chains ───────────────────────────────────────────────────

    #[test]
    fn test_promise_then_fulfilled_chain() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(1), &queue);
        let result: Rc<RefCell<JsValue>> = Rc::new(RefCell::new(JsValue::Undefined));
        let r2 = Rc::clone(&result);
        promise_then(
            &p,
            Some(Box::new(move |v| {
                *r2.borrow_mut() = v.clone();
                Ok(v)
            })),
            None,
            &queue,
        );
        queue.drain();
        assert_eq!(*result.borrow(), JsValue::Smi(1));
    }

    #[test]
    fn test_promise_then_rejection_propagates_without_handler() {
        let queue = MicrotaskQueue::new();
        let p = promise_reject(JsValue::Smi(99), &queue);
        let result: Rc<RefCell<JsValue>> = Rc::new(RefCell::new(JsValue::Undefined));
        let r2 = Rc::clone(&result);
        // No reject handler on p2 → rejection propagates; add catch on p3
        let p2 = promise_then(&p, None, None, &queue);
        promise_then(
            &p2,
            None,
            Some(Box::new(move |r| {
                *r2.borrow_mut() = r.clone();
                Ok(JsValue::Undefined)
            })),
            &queue,
        );
        queue.drain();
        assert_eq!(*result.borrow(), JsValue::Smi(99));
    }

    #[test]
    fn test_promise_then_transforms_value() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(5), &queue);
        let p2 = promise_then(
            &p,
            Some(Box::new(|v| {
                if let JsValue::Smi(n) = v {
                    Ok(JsValue::Smi(n * 2))
                } else {
                    Ok(v)
                }
            })),
            None,
            &queue,
        );
        queue.drain();
        assert_eq!(p2.value(), Some(JsValue::Smi(10)));
    }

    #[test]
    fn test_promise_then_handler_returns_err_rejects_downstream() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(1), &queue);
        let p2 = promise_then(
            &p,
            Some(Box::new(|_v| Err(JsValue::String("boom".to_string())))),
            None,
            &queue,
        );
        queue.drain();
        assert!(p2.is_rejected());
        assert_eq!(p2.reason(), Some(JsValue::String("boom".to_string())));
    }

    #[test]
    fn test_promise_then_multi_hop_chain() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(1), &queue);
        let p2 = promise_then(
            &p,
            Some(Box::new(|v| {
                if let JsValue::Smi(n) = v {
                    Ok(JsValue::Smi(n + 1))
                } else {
                    Ok(v)
                }
            })),
            None,
            &queue,
        );
        let p3 = promise_then(
            &p2,
            Some(Box::new(|v| {
                if let JsValue::Smi(n) = v {
                    Ok(JsValue::Smi(n * 10))
                } else {
                    Ok(v)
                }
            })),
            None,
            &queue,
        );
        queue.drain();
        assert_eq!(p3.value(), Some(JsValue::Smi(20)));
    }

    #[test]
    fn test_promise_then_on_pending_fires_after_resolve() {
        let queue = MicrotaskQueue::new();
        let wr = promise_with_resolvers(&queue);
        let result: Rc<RefCell<JsValue>> = Rc::new(RefCell::new(JsValue::Undefined));
        let r2 = Rc::clone(&result);
        promise_then(
            &wr.promise,
            Some(Box::new(move |v| {
                *r2.borrow_mut() = v.clone();
                Ok(v)
            })),
            None,
            &queue,
        );
        // Handler should not have run yet
        assert_eq!(*result.borrow(), JsValue::Undefined);
        (wr.resolve)(JsValue::Smi(7));
        queue.drain();
        assert_eq!(*result.borrow(), JsValue::Smi(7));
    }

    // ── promise_catch ─────────────────────────────────────────────────────────

    #[test]
    fn test_promise_catch_handles_rejection() {
        let queue = MicrotaskQueue::new();
        let p = promise_reject(JsValue::String("fail".to_string()), &queue);
        let caught: Rc<RefCell<JsValue>> = Rc::new(RefCell::new(JsValue::Undefined));
        let c2 = Rc::clone(&caught);
        let p2 = promise_catch(
            &p,
            Box::new(move |r| {
                *c2.borrow_mut() = r.clone();
                Ok(JsValue::Undefined)
            }),
            &queue,
        );
        queue.drain();
        assert_eq!(*caught.borrow(), JsValue::String("fail".to_string()));
        assert!(p2.is_fulfilled());
    }

    // ── promise_finally ───────────────────────────────────────────────────────

    #[test]
    fn test_promise_finally_runs_on_fulfill() {
        let queue = MicrotaskQueue::new();
        let ran: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
        let ran2 = Rc::clone(&ran);
        let p = promise_resolve(JsValue::Smi(1), &queue);
        let p2 = promise_finally(
            &p,
            Box::new(move || {
                *ran2.borrow_mut() = true;
                Ok(())
            }),
            &queue,
        );
        queue.drain();
        assert!(*ran.borrow());
        assert_eq!(p2.value(), Some(JsValue::Smi(1)));
    }

    #[test]
    fn test_promise_finally_runs_on_reject() {
        let queue = MicrotaskQueue::new();
        let ran: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
        let ran2 = Rc::clone(&ran);
        let p = promise_reject(JsValue::Smi(0), &queue);
        let p2 = promise_finally(
            &p,
            Box::new(move || {
                *ran2.borrow_mut() = true;
                Ok(())
            }),
            &queue,
        );
        queue.drain();
        assert!(*ran.borrow());
        assert!(p2.is_rejected());
        assert_eq!(p2.reason(), Some(JsValue::Smi(0)));
    }

    #[test]
    fn test_promise_finally_can_replace_with_new_rejection() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(1), &queue);
        let p2 = promise_finally(
            &p,
            Box::new(|| Err(JsValue::String("new error".to_string()))),
            &queue,
        );
        queue.drain();
        assert!(p2.is_rejected());
        assert_eq!(p2.reason(), Some(JsValue::String("new error".to_string())));
    }

    // ── promise_all ───────────────────────────────────────────────────────────

    #[test]
    fn test_promise_all_resolves_when_all_fulfill() {
        let queue = MicrotaskQueue::new();
        let p = promise_all(
            vec![
                promise_resolve(JsValue::Smi(1), &queue),
                promise_resolve(JsValue::Smi(2), &queue),
                promise_resolve(JsValue::Smi(3), &queue),
            ],
            &queue,
        );
        queue.drain();
        if let Some(JsValue::Array(arr)) = p.value() {
            assert_eq!(
                *arr.borrow(),
                vec![JsValue::Smi(1), JsValue::Smi(2), JsValue::Smi(3)]
            );
        } else {
            panic!("expected Array");
        }
    }

    #[test]
    fn test_promise_all_rejects_on_first_rejection() {
        let queue = MicrotaskQueue::new();
        let p = promise_all(
            vec![
                promise_resolve(JsValue::Smi(1), &queue),
                promise_reject(JsValue::String("err".to_string()), &queue),
                promise_resolve(JsValue::Smi(3), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_rejected());
        assert_eq!(p.reason(), Some(JsValue::String("err".to_string())));
    }

    #[test]
    fn test_promise_all_empty_resolves_with_empty_array() {
        let queue = MicrotaskQueue::new();
        let p = promise_all(vec![], &queue);
        assert!(p.is_fulfilled());
        if let Some(JsValue::Array(arr)) = p.value() {
            assert!(arr.borrow().is_empty());
        } else {
            panic!("expected Array");
        }
    }

    // ── promise_all_settled ───────────────────────────────────────────────────

    #[test]
    fn test_promise_all_settled_never_rejects() {
        let queue = MicrotaskQueue::new();
        let p = promise_all_settled(
            vec![
                promise_resolve(JsValue::Smi(1), &queue),
                promise_reject(JsValue::String("boom".to_string()), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_fulfilled());
        if let Some(JsValue::Array(arr)) = p.value() {
            assert_eq!(arr.borrow().len(), 2);
            if let JsValue::Array(inner0) = &arr.borrow()[0] {
                assert_eq!(
                    *inner0.borrow(),
                    vec![JsValue::String("fulfilled".to_string()), JsValue::Smi(1)]
                );
            } else {
                panic!("expected inner Array at [0]");
            }
            if let JsValue::Array(inner1) = &arr.borrow()[1] {
                assert_eq!(
                    *inner1.borrow(),
                    vec![
                        JsValue::String("rejected".to_string()),
                        JsValue::String("boom".to_string()),
                    ]
                );
            } else {
                panic!("expected inner Array at [1]");
            }
        } else {
            panic!("expected JsValue::Array");
        }
    }

    // ── promise_any ───────────────────────────────────────────────────────────

    #[test]
    fn test_promise_any_resolves_with_first_fulfill() {
        let queue = MicrotaskQueue::new();
        let p = promise_any(
            vec![
                promise_reject(JsValue::Smi(0), &queue),
                promise_resolve(JsValue::Smi(2), &queue),
                promise_resolve(JsValue::Smi(3), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_fulfilled());
        assert_eq!(p.value(), Some(JsValue::Smi(2)));
    }

    #[test]
    fn test_promise_any_rejects_when_all_reject() {
        let queue = MicrotaskQueue::new();
        let p = promise_any(
            vec![
                promise_reject(JsValue::Smi(1), &queue),
                promise_reject(JsValue::Smi(2), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_rejected());
        if let Some(JsValue::Array(arr)) = p.reason() {
            assert_eq!(*arr.borrow(), vec![JsValue::Smi(1), JsValue::Smi(2)]);
        } else {
            panic!("expected Array reason");
        }
    }

    // ── promise_race ──────────────────────────────────────────────────────────

    #[test]
    fn test_promise_race_first_fulfill_wins() {
        let queue = MicrotaskQueue::new();
        let p = promise_race(
            vec![
                promise_resolve(JsValue::Smi(1), &queue),
                promise_resolve(JsValue::Smi(2), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_fulfilled());
        assert_eq!(p.value(), Some(JsValue::Smi(1)));
    }

    #[test]
    fn test_promise_race_first_rejection_wins() {
        let queue = MicrotaskQueue::new();
        let p = promise_race(
            vec![
                promise_reject(JsValue::String("no".to_string()), &queue),
                promise_resolve(JsValue::Smi(2), &queue),
            ],
            &queue,
        );
        queue.drain();
        assert!(p.is_rejected());
        assert_eq!(p.reason(), Some(JsValue::String("no".to_string())));
    }

    // ── promise_with_resolvers ────────────────────────────────────────────────

    #[test]
    fn test_promise_with_resolvers_resolve() {
        let queue = MicrotaskQueue::new();
        let wr = promise_with_resolvers(&queue);
        assert!(wr.promise.is_pending());
        (wr.resolve)(JsValue::Smi(42));
        assert!(wr.promise.is_fulfilled());
        assert_eq!(wr.promise.value(), Some(JsValue::Smi(42)));
    }

    #[test]
    fn test_promise_with_resolvers_reject() {
        let queue = MicrotaskQueue::new();
        let wr = promise_with_resolvers(&queue);
        (wr.reject)(JsValue::String("fail".to_string()));
        assert!(wr.promise.is_rejected());
        assert_eq!(
            wr.promise.reason(),
            Some(JsValue::String("fail".to_string()))
        );
    }

    // ── Microtask ordering ────────────────────────────────────────────────────

    #[test]
    fn test_microtask_ordering_in_chain() {
        let queue = MicrotaskQueue::new();
        let log: Rc<RefCell<Vec<i32>>> = Rc::new(RefCell::new(Vec::new()));

        let p = promise_resolve(JsValue::Smi(0), &queue);
        let log1 = Rc::clone(&log);
        let p2 = promise_then(
            &p,
            Some(Box::new(move |v| {
                log1.borrow_mut().push(1);
                Ok(v)
            })),
            None,
            &queue,
        );
        let log2 = Rc::clone(&log);
        promise_then(
            &p2,
            Some(Box::new(move |v| {
                log2.borrow_mut().push(2);
                Ok(v)
            })),
            None,
            &queue,
        );

        // Before drain, no handler has run
        assert!(log.borrow().is_empty());
        queue.drain();
        // After drain, handlers ran in order
        assert_eq!(*log.borrow(), vec![1, 2]);
    }

    #[test]
    fn test_resolve_then_fires_asynchronously() {
        let queue = MicrotaskQueue::new();
        let wr = promise_with_resolvers(&queue);
        let fired: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));
        let fired2 = Rc::clone(&fired);
        promise_then(
            &wr.promise,
            Some(Box::new(move |_| {
                *fired2.borrow_mut() = true;
                Ok(JsValue::Undefined)
            })),
            None,
            &queue,
        );
        (wr.resolve)(JsValue::Smi(1));
        // Handler should be queued but not yet run
        assert!(!*fired.borrow());
        queue.drain();
        assert!(*fired.borrow());
    }

    // ── promise_state ─────────────────────────────────────────────────────────

    #[test]
    fn test_promise_state_variants() {
        let queue = MicrotaskQueue::new();
        let p = promise_resolve(JsValue::Smi(5), &queue);
        assert_eq!(p.state(), PromiseState::Fulfilled(JsValue::Smi(5)));
        let p2 = promise_reject(JsValue::String("err".to_string()), &queue);
        assert_eq!(
            p2.state(),
            PromiseState::Rejected(JsValue::String("err".to_string()))
        );
        let p3 = promise_new(|_, _| {}, &queue);
        assert_eq!(p3.state(), PromiseState::Pending);
    }

    // ── Unhandled rejection tracking ──────────────────────────────────────────

    #[test]
    fn test_unhandled_rejection_detected() {
        let queue = MicrotaskQueue::new();
        let mut tracker = UnhandledRejectionTracker::new();
        let p = promise_reject(JsValue::String("oops".into()), &queue);
        tracker.track(p);
        queue.drain();
        let unhandled = tracker.collect_unhandled();
        assert_eq!(unhandled.len(), 1);
        assert_eq!(unhandled[0].reason(), Some(JsValue::String("oops".into())));
    }

    #[test]
    fn test_handled_rejection_not_reported() {
        let queue = MicrotaskQueue::new();
        let mut tracker = UnhandledRejectionTracker::new();
        let p = promise_reject(JsValue::String("caught".into()), &queue);
        promise_catch(&p, Box::new(|_| Ok(JsValue::Undefined)), &queue);
        tracker.track(p);
        queue.drain();
        let unhandled = tracker.collect_unhandled();
        assert!(unhandled.is_empty());
    }

    #[test]
    fn test_tracker_ignores_fulfilled_promises() {
        let queue = MicrotaskQueue::new();
        let mut tracker = UnhandledRejectionTracker::new();
        let p = promise_resolve(JsValue::Smi(1), &queue);
        tracker.track(p);
        queue.drain();
        let unhandled = tracker.collect_unhandled();
        assert!(unhandled.is_empty());
    }

    #[test]
    fn test_tracker_default() {
        let tracker = UnhandledRejectionTracker::default();
        assert_eq!(tracker.promises.len(), 0);
    }

    // ── JsPromise identity equality ──────────────────────────────────────────

    #[test]
    fn test_promise_identity_equality() {
        let queue = MicrotaskQueue::new();
        let p1 = promise_resolve(JsValue::Smi(1), &queue);
        let p1_clone = p1.clone();
        let p2 = promise_resolve(JsValue::Smi(1), &queue);
        assert_eq!(p1, p1_clone);
        assert_ne!(p1, p2);
    }
}
