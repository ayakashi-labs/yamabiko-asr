use crate::{AudioSourceId, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DiarizationMode {
    #[default]
    Off,
    #[allow(dead_code)]
    On,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AudioSourceOptions {
    pub(crate) diarization: DiarizationMode,
}

impl AudioSourceOptions {
    pub(crate) const OFF: Self = Self {
        diarization: DiarizationMode::Off,
    };

    #[allow(dead_code)]
    pub(crate) const ON: Self = Self {
        diarization: DiarizationMode::On,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BackendSpeakerId(u64);

impl BackendSpeakerId {
    #[allow(dead_code)]
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// A finalized, non-overlapping region on one source-local PCM timeline.
///
/// One backend speaker means a single-speaker region. More than one means
/// overlapping speech and is deliberately transcribed once without a speaker
/// attribution. An empty list represents non-speech and does not create an ASR
/// job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiarizedRegion {
    pub(crate) start_sample: u64,
    pub(crate) end_sample: u64,
    pub(crate) speakers: Vec<BackendSpeakerId>,
}

/// Newly finalized regions and the PCM watermark that the worker may release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiarizationOutput {
    pub(crate) regions: Vec<DiarizedRegion>,
    pub(crate) finalized_until: u64,
}

pub(crate) trait Diarizer: Send {
    /// Maximum number of unfinalized PCM samples the session may retain across
    /// all enabled sources after processing one input command.
    ///
    /// The worker may temporarily hold one additional input command while the
    /// backend advances its finalized watermark.
    fn max_retained_samples(&self) -> usize;

    fn open_source(&mut self, source_id: AudioSourceId) -> Result<()>;

    fn push(
        &mut self,
        source_id: AudioSourceId,
        samples: &[f32],
        start_sample: u64,
    ) -> Result<DiarizationOutput>;

    fn finish(&mut self, source_id: AudioSourceId) -> Result<DiarizationOutput>;
}

pub(crate) trait DiarizerFactory: Send {
    fn create(&mut self) -> Result<Box<dyn Diarizer>>;
}

pub(crate) struct UnavailableDiarizerFactory;

impl DiarizerFactory for UnavailableDiarizerFactory {
    fn create(&mut self) -> Result<Box<dyn Diarizer>> {
        Err(crate::Error::Backend(
            "diarization backend is not configured".to_string(),
        ))
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use super::{BackendSpeakerId, DiarizationOutput, DiarizedRegion, Diarizer, DiarizerFactory};
    use crate::{AudioSourceId, Error, Result};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc as std_mpsc;
    use std::sync::{Arc, Mutex};

    pub(crate) enum Behavior {
        FinalizeEach {
            speakers: Vec<BackendSpeakerId>,
        },
        FinalizeAfter {
            pushes: usize,
            speakers: Vec<BackendSpeakerId>,
            processed: std_mpsc::Sender<()>,
        },
        Flush {
            speakers: Vec<BackendSpeakerId>,
        },
        FailPush(Error),
        BlockThenFinalize {
            speakers: Vec<BackendSpeakerId>,
            started: std_mpsc::Sender<()>,
            release: std_mpsc::Receiver<()>,
        },
    }

    pub(crate) type Pushes = Arc<Mutex<Vec<(AudioSourceId, Vec<f32>)>>>;
    pub(crate) type Parts = (
        Factory,
        Arc<AtomicUsize>,
        Arc<Mutex<Vec<AudioSourceId>>>,
        Pushes,
    );

    struct Backend {
        behavior: Behavior,
        opened: Arc<Mutex<Vec<AudioSourceId>>>,
        pushes: Pushes,
        source_ends: HashMap<AudioSourceId, u64>,
        source_finalized: HashMap<AudioSourceId, u64>,
        source_pushes: HashMap<AudioSourceId, usize>,
        max_retained_samples: usize,
    }

    impl Diarizer for Backend {
        fn max_retained_samples(&self) -> usize {
            self.max_retained_samples
        }

        fn open_source(&mut self, source_id: AudioSourceId) -> Result<()> {
            self.opened.lock().unwrap().push(source_id);
            self.source_ends.insert(source_id, 0);
            self.source_finalized.insert(source_id, 0);
            self.source_pushes.insert(source_id, 0);
            Ok(())
        }

        fn push(
            &mut self,
            source_id: AudioSourceId,
            samples: &[f32],
            start_sample: u64,
        ) -> Result<DiarizationOutput> {
            self.pushes
                .lock()
                .unwrap()
                .push((source_id, samples.to_vec()));
            let end_sample = start_sample.saturating_add(samples.len() as u64);
            self.source_ends.insert(source_id, end_sample);

            match &mut self.behavior {
                Behavior::FinalizeEach { speakers } => Ok(DiarizationOutput {
                    regions: (!samples.is_empty())
                        .then(|| DiarizedRegion {
                            start_sample,
                            end_sample,
                            speakers: speakers.clone(),
                        })
                        .into_iter()
                        .collect(),
                    finalized_until: end_sample,
                }),
                Behavior::FinalizeAfter {
                    pushes,
                    speakers,
                    processed,
                } => {
                    let push_count = self.source_pushes.entry(source_id).or_default();
                    *push_count = push_count.saturating_add(1);
                    let finalized_until = *self.source_finalized.get(&source_id).unwrap_or(&0);
                    processed.send(()).unwrap();
                    if *push_count < *pushes {
                        return Ok(DiarizationOutput {
                            regions: Vec::new(),
                            finalized_until,
                        });
                    }

                    *push_count = 0;
                    self.source_finalized.insert(source_id, end_sample);
                    Ok(DiarizationOutput {
                        regions: (finalized_until < end_sample)
                            .then(|| DiarizedRegion {
                                start_sample: finalized_until,
                                end_sample,
                                speakers: speakers.clone(),
                            })
                            .into_iter()
                            .collect(),
                        finalized_until: end_sample,
                    })
                }
                Behavior::Flush { .. } => Ok(DiarizationOutput {
                    regions: Vec::new(),
                    finalized_until: 0,
                }),
                Behavior::FailPush(err) => Err(err.clone()),
                Behavior::BlockThenFinalize {
                    speakers,
                    started,
                    release,
                } => {
                    started.send(()).unwrap();
                    release.recv().unwrap();
                    Ok(DiarizationOutput {
                        regions: vec![DiarizedRegion {
                            start_sample,
                            end_sample,
                            speakers: speakers.clone(),
                        }],
                        finalized_until: end_sample,
                    })
                }
            }
        }

        fn finish(&mut self, source_id: AudioSourceId) -> Result<DiarizationOutput> {
            let end_sample = *self.source_ends.get(&source_id).unwrap_or(&0);
            match &self.behavior {
                Behavior::Flush { speakers } if end_sample > 0 => Ok(DiarizationOutput {
                    regions: vec![DiarizedRegion {
                        start_sample: 0,
                        end_sample,
                        speakers: speakers.clone(),
                    }],
                    finalized_until: end_sample,
                }),
                Behavior::FinalizeAfter { speakers, .. } => {
                    let finalized_until = *self.source_finalized.get(&source_id).unwrap_or(&0);
                    Ok(DiarizationOutput {
                        regions: (finalized_until < end_sample)
                            .then(|| DiarizedRegion {
                                start_sample: finalized_until,
                                end_sample,
                                speakers: speakers.clone(),
                            })
                            .into_iter()
                            .collect(),
                        finalized_until: end_sample,
                    })
                }
                _ => Ok(DiarizationOutput {
                    regions: Vec::new(),
                    finalized_until: end_sample,
                }),
            }
        }
    }

    pub(crate) struct Factory {
        creates: Arc<AtomicUsize>,
        diarizer: Option<Backend>,
        create_error: Option<Error>,
    }

    impl DiarizerFactory for Factory {
        fn create(&mut self) -> Result<Box<dyn Diarizer>> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            if let Some(err) = self.create_error.take() {
                return Err(err);
            }
            self.diarizer
                .take()
                .map(|diarizer| Box::new(diarizer) as Box<dyn Diarizer>)
                .ok_or_else(|| Error::Backend("fake diarizer created more than once".to_string()))
        }
    }

    pub(crate) fn factory(behavior: Behavior) -> Parts {
        factory_with_capacity(behavior, 16_000)
    }

    pub(crate) fn factory_with_capacity(behavior: Behavior, max_retained_samples: usize) -> Parts {
        let creates = Arc::new(AtomicUsize::new(0));
        let opened = Arc::new(Mutex::new(Vec::new()));
        let pushes = Arc::new(Mutex::new(Vec::new()));
        (
            Factory {
                creates: Arc::clone(&creates),
                diarizer: Some(Backend {
                    behavior,
                    opened: Arc::clone(&opened),
                    pushes: Arc::clone(&pushes),
                    source_ends: HashMap::new(),
                    source_finalized: HashMap::new(),
                    source_pushes: HashMap::new(),
                    max_retained_samples,
                }),
                create_error: None,
            },
            creates,
            opened,
            pushes,
        )
    }

    pub(crate) fn failing_factory(error: Error) -> (Factory, Arc<AtomicUsize>) {
        let creates = Arc::new(AtomicUsize::new(0));
        (
            Factory {
                creates: Arc::clone(&creates),
                diarizer: None,
                create_error: Some(error),
            },
            creates,
        )
    }
}
