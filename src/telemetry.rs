use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::core::collection::Collection;
use crate::core::metadata;
use crate::core::repo::pile::{Pile, ReadError};
use crate::prelude::blobencodings::LongString;
use crate::prelude::inlineencodings::{GenId, Handle, ShortString, U256BE};
use crate::prelude::*;
use ed25519_dalek::SigningKey;
use rand_core06::OsRng;
use thread_local::ThreadLocal;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::EnvFilter;

const ENV_TELEMETRY_PILE: &str = "TELEMETRY_PILE";
const ENV_PILE: &str = "PILE";
const ENV_TELEMETRY_COLLECTION_SCOPE: &str = "TELEMETRY_COLLECTION_SCOPE";
const ENV_TELEMETRY_FLUSH_MS: &str = "TELEMETRY_FLUSH_MS";

pub mod schema {
    use super::*;

    attributes! {
        "3E062AA7E3554C8F2DB94883CE639BFE" unsafe as pub session: GenId;
        "146E5AA2F7CB3D8B654BC7742A13CAB3" unsafe as pub parent: GenId;
        "CCB0147D20C4C6FCAC0E3D87FAFF71D1" unsafe as pub name: Handle<LongString>;
        "8A4BE2C4D0E90D2B9EE0E1A07ECA2CFA" unsafe as pub category: ShortString;
        "E11A84A30CC112650DC860B66B8BD8A9" unsafe as pub begin_ns: U256BE;
        "2786FA563372FB6EF469EC7710719A49" unsafe as pub end_ns: U256BE;
        "7593602383D0B0D21BBE382A67E5BD9F" unsafe as pub duration_ns: U256BE;
        "7E96DD9A0B5002796B645ED25F5E99AC" unsafe as pub source: Handle<LongString>;
    }

    #[allow(non_upper_case_globals)]
    pub const kind_session: Id = crate::macros::id_hex!("2701F7019B865D461F0169B1303026D6");
    #[allow(non_upper_case_globals)]
    pub const kind_span: Id = crate::macros::id_hex!("0AF9FEB9A2BFEB1BE8A8229829181085");

    #[allow(non_upper_case_globals)]
    pub const telemetry_metadata: Id = crate::macros::id_hex!("BCFDE38F7E452924C72803239392EA05");

    /// Build the telemetry-protocol metadata fragment.
    ///
    /// After the Fragment-self-contained refactor, `describe` is no longer
    /// fallible and doesn't take a blob store — each entity!{} auto-puts
    /// long-form bytes (descriptions, names) into the fragment's own
    /// MemoryBlobStore. The returned Fragment is self-contained.
    pub fn build_telemetry_metadata() -> Fragment {
        let attrs = describe();

        let mut protocol = entity! { ExclusiveId::force_ref(&telemetry_metadata) @
            metadata::name: "triblespace_telemetry",
            metadata::description:
                "Span-based profiling events emitted by TribleSpace telemetry.",
            metadata::tag: metadata::KIND_PROTOCOL,
            metadata::attribute*: attrs,
        };

        protocol += entity! { ExclusiveId::force_ref(&kind_session) @
            metadata::name: "telemetry_session",
            metadata::description:
                "A profiling session. Groups spans emitted during one telemetry run.",
            metadata::tag: metadata::KIND_TAG,
        };
        protocol += entity! { ExclusiveId::force_ref(&kind_span) @
            metadata::name: "telemetry_span",
            metadata::description:
                "A begin/end span with optional parent links.",
            metadata::tag: metadata::KIND_TAG,
        };

        protocol
    }
}

fn is_valid_short(value: &str) -> bool {
    value.as_bytes().len() <= 32 && !value.as_bytes().iter().any(|b| *b == 0)
}

struct ThreadTelemetry {
    batch: Fragment,
    last_flush: Instant,
}

struct TelemetryInner {
    collection: Mutex<Option<Collection<Pile>>>,
    batches: ThreadLocal<Arc<Mutex<ThreadTelemetry>>>,
    registry: Mutex<Vec<Arc<Mutex<ThreadTelemetry>>>>,
    session: Id,
    base: Instant,
    flush_interval: Duration,
    shutdown: AtomicBool,
}

fn self_describing(mut batch: Fragment) -> Fragment {
    batch.describe_with(schema::build_telemetry_metadata());
    batch
}

fn pile_refresh_diagnostic(path: &Path, error: &ReadError) -> String {
    let failure = format!(
        "telemetry pile {} failed to load ({error}); telemetry disabled.",
        path.display()
    );

    match error {
        ReadError::UnsupportedRecord { .. } => format!(
            "{failure} This is likely format/version skew; upgrade to a reader that recognizes \
             the record marker. The pile was left unchanged."
        ),
        ReadError::CorruptPile { .. } => format!(
            "{failure} A malformed or incomplete known record does not prove that the remaining \
             suffix is a disposable torn write. The pile was left unchanged; inspect it with \
             current pile diagnostics before choosing recovery."
        ),
        _ => format!("{failure} The pile was left unchanged."),
    }
}

/// Publish a clone of the pending batch and clear the retained in-memory copy
/// only after the caller's complete publication policy succeeds. A partial
/// backend failure can therefore be retried with the exact same
/// content-addressed commit.
fn publish_pending<E>(
    state: &mut ThreadTelemetry,
    publish: impl FnOnce(Fragment) -> Result<(), E>,
) -> Result<bool, E> {
    if state.batch.facts().is_empty() {
        return Ok(false);
    }

    publish(self_describing(state.batch.clone()))?;
    state.batch = Fragment::empty();
    Ok(true)
}

impl TelemetryInner {
    fn now_ns(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }

    fn get_or_init_thread(&self) -> &Arc<Mutex<ThreadTelemetry>> {
        self.batches.get_or(|| {
            let arc = Arc::new(Mutex::new(ThreadTelemetry {
                batch: Fragment::empty(),
                last_flush: Instant::now(),
            }));
            self.registry
                .lock()
                .expect("telemetry registry lock")
                .push(arc.clone());
            arc
        })
    }

    fn maybe_flush(&self, state: &mut ThreadTelemetry) {
        if state.last_flush.elapsed() < self.flush_interval {
            return;
        }

        if state.batch.facts().is_empty() {
            state.last_flush = Instant::now();
            return;
        }

        let mut collection_guard = self.collection.lock().expect("telemetry collection lock");
        if let Some(collection) = collection_guard.as_mut() {
            if let Err(e) = publish_pending(state, |batch| {
                collection
                    .commit(batch)
                    .map_err(|error| format!("{error:?}"))?;
                collection.flush().map_err(|error| format!("{error:?}"))
            }) {
                log::warn!("telemetry flush failed: {e:?}");
                return;
            }
        }
        state.last_flush = Instant::now();
    }
}

#[derive(Debug, Clone, Copy)]
struct TelemetrySpanData {
    span: Id,
    start_ns: u64,
}

#[derive(Default)]
struct FieldCapture {
    source: Option<String>,
}

impl tracing::field::Visit for FieldCapture {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        match field.name() {
            "source" if !value.is_empty() => self.source = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "source" => {
                let mut raw = format!("{value:?}");
                if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
                    raw = raw[1..raw.len() - 1].to_string();
                }
                if !raw.is_empty() {
                    self.source = Some(raw);
                }
            }
            _ => {}
        }
    }
}

/// Tracing layer that turns spans into TribleSpace telemetry.
///
/// Construct via [`Telemetry::layer_from_env`] and attach to your application's subscriber.
pub struct TelemetryLayer {
    inner: Arc<TelemetryInner>,
}

impl TelemetryLayer {
    fn parent_id<S>(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        ctx: &Context<'_, S>,
    ) -> Option<Id>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        if let Some(parent) = attrs.parent() {
            if let Some(span) = ctx.span(parent) {
                if let Some(data) = span.extensions().get::<TelemetrySpanData>() {
                    return Some(data.span);
                }
            }
        }

        if let Some(id) = ctx.current_span().id() {
            if let Some(span) = ctx.span(id) {
                if let Some(data) = span.extensions().get::<TelemetrySpanData>() {
                    return Some(data.span);
                }
            }
        }

        None
    }
}

impl<S> Layer<S> for TelemetryLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        if self.inner.shutdown.load(Ordering::Relaxed) {
            return;
        }

        let Some(span) = ctx.span(id) else {
            return;
        };

        let meta = attrs.metadata();
        let mut fields = FieldCapture::default();
        attrs.record(&mut fields);

        let start_ns = self.inner.now_ns();
        let span_id = *genid();
        let parent = self.parent_id(attrs, &ctx);

        span.extensions_mut().insert(TelemetrySpanData {
            span: span_id,
            start_ns,
        });

        let thread_state = self.inner.get_or_init_thread();
        let mut state = thread_state.lock().expect("telemetry thread state lock");

        let target = meta.target();
        let category = target.split("::").next().unwrap_or(target);
        let category = if !category.is_empty() && is_valid_short(category) {
            category
        } else {
            "span"
        };

        span_begin(
            &mut state.batch,
            self.inner.session,
            span_id,
            parent,
            start_ns,
            category,
            meta.name(),
            fields.source,
        );
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        if self.inner.shutdown.load(Ordering::Relaxed) {
            return;
        }

        let Some(span) = ctx.span(&id) else {
            return;
        };
        let Some(data) = span.extensions().get::<TelemetrySpanData>().copied() else {
            return;
        };

        let end_ns = self.inner.now_ns();

        let thread_state = self.inner.get_or_init_thread();
        let mut state = thread_state.lock().expect("telemetry thread state lock");

        span_end(
            &mut state.batch,
            data.span,
            end_ns,
            end_ns.saturating_sub(data.start_ns),
        );

        self.inner.maybe_flush(&mut state);
    }
}

pub struct Telemetry {
    inner: Arc<TelemetryInner>,
}

impl Telemetry {
    /// Start a telemetry sink and return a layer that writes spans into it.
    ///
    /// This does **not** install a tracing subscriber. Embed the returned layer into your
    /// application's subscriber, and keep the returned [`Telemetry`] guard alive to
    /// flush and close the sink on shutdown.
    pub fn layer_from_env(session_name: &str) -> Option<(TelemetryLayer, Self)> {
        let pile_path = std::env::var(ENV_TELEMETRY_PILE)
            .ok()
            .or_else(|| std::env::var(ENV_PILE).ok())?;
        let pile_path = pile_path.trim();
        if pile_path.is_empty() {
            return None;
        }
        let pile_path = PathBuf::from(pile_path);

        let scope_hex = std::env::var(ENV_TELEMETRY_COLLECTION_SCOPE).ok()?;
        let scope_hex = scope_hex.trim();
        if scope_hex.len() != 32 {
            log::warn!(
                "TELEMETRY_COLLECTION_SCOPE must be a 32-char hex ID, got {} chars",
                scope_hex.len()
            );
            return None;
        }
        let collection_scope = Id::from_hex(scope_hex)?;

        let flush_ms = std::env::var(ENV_TELEMETRY_FLUSH_MS)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(250);
        let flush_interval = Duration::from_millis(flush_ms.max(10));

        let base = Instant::now();
        let session_id = *genid();

        // Open the pile with the non-mutating refresh. Any read failure
        // disables telemetry without changing the pile; the error alone does
        // not establish that any suffix is safe to discard.
        if let Some(parent) = pile_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).ok()?;
        }
        let mut pile = Pile::open(&pile_path).ok()?;
        if let Err(err) = pile.refresh() {
            log::warn!("{}", pile_refresh_diagnostic(&pile_path, &err));
            let _ = pile.close();
            return None;
        }

        let signing_key = SigningKey::generate(&mut OsRng);
        let mut collection = Collection::new(pile, collection_scope, signing_key);

        // Commit session start entity.
        let session_entity = ExclusiveId::force_ref(&session_id);
        let mut init = Fragment::empty();
        let session_name = init.put(session_name.to_string());
        init += entity! { session_entity @
            metadata::tag: schema::kind_session,
            schema::category: "session",
            schema::name: session_name,
            schema::begin_ns: 0u64,
        };
        if collection.commit(self_describing(init)).is_err() || collection.flush().is_err() {
            let _ = collection.close();
            return None;
        }

        let inner = Arc::new(TelemetryInner {
            collection: Mutex::new(Some(collection)),
            batches: ThreadLocal::new(),
            registry: Mutex::new(Vec::new()),
            session: session_id,
            base,
            flush_interval,
            shutdown: AtomicBool::new(false),
        });

        let layer = TelemetryLayer {
            inner: inner.clone(),
        };

        Some((layer, Self { inner }))
    }

    /// Convenience for standalone processes: start telemetry and install a global subscriber
    /// (only if none exists).
    pub fn install_global_from_env(session_name: &str) -> Option<Self> {
        let (layer, guard) = Self::layer_from_env(session_name)?;

        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
        let subscriber = tracing_subscriber::registry().with(filter).with(layer);

        if tracing::subscriber::set_global_default(subscriber).is_err() {
            log::warn!("triblespace telemetry disabled: tracing subscriber already set");
            drop(guard);
            return None;
        }

        Some(guard)
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);

        // Flush all thread-local batches. Each flush takes the locks in the
        // same state-then-collection order as the tracing callbacks.
        let registry = self.inner.registry.lock().expect("telemetry registry lock");
        for state_arc in registry.iter() {
            let mut state = state_arc.lock().expect("telemetry thread state lock");
            let mut collection_guard = self
                .inner
                .collection
                .lock()
                .expect("telemetry collection lock");
            if let Some(collection) = collection_guard.as_mut() {
                if let Err(e) = publish_pending(&mut state, |batch| {
                    collection
                        .commit(batch)
                        .map_err(|error| format!("{error:?}"))?;
                    collection.flush().map_err(|error| format!("{error:?}"))
                }) {
                    log::warn!("telemetry shutdown flush failed: {e:?}");
                }
            }
        }
        drop(registry);

        // Commit the terminal session facts and close the pile.
        let end_ns = self.inner.now_ns();
        let session_entity = ExclusiveId::force_ref(&self.inner.session);
        let mut end = Fragment::empty();
        end += entity! { session_entity @
            schema::end_ns: end_ns,
            schema::duration_ns: end_ns,
        };

        let mut collection_guard = self
            .inner
            .collection
            .lock()
            .expect("telemetry collection lock");
        if let Some(mut collection) = collection_guard.take() {
            if let Err(e) = collection.commit(self_describing(end)) {
                log::warn!("telemetry session end commit failed: {e:?}");
            }
            if let Err(e) = collection.close() {
                log::warn!("telemetry pile close failed: {e:?}");
            }
        }
    }
}

fn span_begin(
    batch: &mut Fragment,
    session: Id,
    span_id: Id,
    parent: Option<Id>,
    at_ns: u64,
    category: &str,
    name: &str,
    source: Option<String>,
) {
    let span_entity = ExclusiveId::force_ref(&span_id);
    let name = batch.put(name.to_string());
    let source = source.map(|source| batch.put(source));
    *batch += entity! { span_entity @
        metadata::tag: schema::kind_span,
        schema::session: session,
        schema::category: category,
        schema::name: name,
        schema::begin_ns: at_ns,
    };
    if let Some(parent) = parent {
        *batch += entity! { span_entity @ schema::parent: parent };
    }
    if let Some(source) = source {
        *batch += entity! { span_entity @ schema::source: source };
    }
}

fn span_end(batch: &mut Fragment, span_id: Id, at_ns: u64, duration_ns: u64) {
    let span_entity = ExclusiveId::force_ref(&span_id);
    *batch += entity! { span_entity @
        schema::end_ns: at_ns,
        schema::duration_ns: duration_ns,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_record_diagnostic_prioritizes_version_skew_without_destructive_advice() {
        let diagnostic = pile_refresh_diagnostic(
            Path::new("telemetry.pile"),
            &ReadError::UnsupportedRecord {
                offset: 256,
                marker: [0xA5; 16],
            },
        );

        assert!(diagnostic.contains("likely format/version skew"));
        assert!(diagnostic.contains("upgrade to a reader"));
        assert!(diagnostic.contains("pile was left unchanged"));
        assert!(!diagnostic.contains("amputate"));
    }

    #[test]
    fn no_refresh_failure_diagnostic_recommends_amputation() {
        let failures = [
            ReadError::CorruptPile { valid_length: 7 },
            ReadError::IoError(std::io::Error::other("unavailable")),
            ReadError::FileTooLarge { length: usize::MAX },
        ];

        for failure in &failures {
            let diagnostic = pile_refresh_diagnostic(Path::new("telemetry.pile"), failure);
            assert!(diagnostic.contains("telemetry disabled"));
            assert!(diagnostic.contains("pile was left unchanged"));
            assert!(!diagnostic.contains("amputate"));
        }
    }

    fn pending_span() -> Fragment {
        let mut batch = Fragment::empty();
        span_begin(
            &mut batch,
            *genid(),
            *genid(),
            None,
            7,
            "telemetry",
            "pending_span",
            Some("telemetry::tests".to_owned()),
        );
        batch
    }

    #[test]
    fn failed_publish_keeps_the_exact_pending_batch_for_retry() {
        let batch = pending_span();
        let mut state = ThreadTelemetry {
            batch: batch.clone(),
            last_flush: Instant::now(),
        };

        let result = publish_pending(&mut state, |_| Err::<(), _>("backend unavailable"));

        assert_eq!(result, Err("backend unavailable"));
        assert_eq!(state.batch, batch);
    }

    #[test]
    fn successful_publish_is_self_describing_and_clears_the_batch() {
        let batch = pending_span();
        assert!(!batch.facts().is_empty());
        assert!(!batch.blobs().is_empty());

        let mut state = ThreadTelemetry {
            batch: batch.clone(),
            last_flush: Instant::now(),
        };
        let mut published = None;

        let result = publish_pending(&mut state, |candidate| {
            published = Some(candidate);
            Ok::<(), ()>(())
        });

        assert_eq!(result, Ok(true));
        assert_eq!(state.batch, Fragment::empty());

        let published = published.expect("publisher received the batch");
        assert_eq!(published.facts(), batch.facts());
        let description = schema::build_telemetry_metadata();
        for fact in description.facts() {
            assert!(published.metafacts().contains(fact));
        }
        for fact in description.metafacts() {
            assert!(published.metafacts().contains(fact));
        }
    }
}
