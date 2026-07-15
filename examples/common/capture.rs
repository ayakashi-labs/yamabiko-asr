use crate::audio::{
    ASR_CHUNK_SAMPLES, AudioResampler, TARGET_SAMPLE_RATE, downmix_to_mono, wasapi_capture_time,
};
use crate::common::{ExampleResult, print_transcript};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use yamabiko_asr::{AudioInput, AudioSourceId, TranscriptEvent};

type TimedPcm = (Duration, Vec<f32>);
const PCM_QUEUE_CAPACITY: usize = 64;

enum CaptureMessage {
    Pcm(TimedPcm),
    Wake,
}

#[derive(Default)]
struct CaptureFailure(OnceLock<String>);

impl CaptureFailure {
    fn record(&self, message: String) -> bool {
        self.0.set(message).is_ok()
    }

    fn message(&self) -> Option<&str> {
        self.0.get().map(String::as_str)
    }
}

pub struct CaptureDevice {
    label: &'static str,
    display_label: &'static str,
    device: cpal::Device,
    supported: cpal::SupportedStreamConfig,
}

impl CaptureDevice {
    pub fn microphone(host: &cpal::Host) -> ExampleResult<Self> {
        let device = host.default_input_device().ok_or("no input device")?;
        let supported = device.default_input_config()?;
        Self::new("microphone", "Microphone", device, supported)
    }

    pub fn system_audio(host: &cpal::Host) -> ExampleResult<Self> {
        let device = host.default_output_device().ok_or("no output device")?;
        let supported = device.default_output_config()?;
        Self::new("system", "System audio", device, supported)
    }

    fn new(
        label: &'static str,
        display_label: &'static str,
        device: cpal::Device,
        supported: cpal::SupportedStreamConfig,
    ) -> ExampleResult<Self> {
        require_f32(label, supported.sample_format())?;
        Ok(Self {
            label,
            display_label,
            device,
            supported,
        })
    }

    pub fn start(
        self,
        session_started: Instant,
        input: AudioInput,
        failure_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> ExampleResult<Capture> {
        let device_name = self
            .device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "Unknown device".to_string());
        let channels = self.supported.channels() as usize;
        let sample_rate = self.supported.sample_rate();
        let sample_format = self.supported.sample_format();
        let label = self.label;
        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel(PCM_QUEUE_CAPACITY);
        let failure = Arc::new(CaptureFailure::default());
        let data_pcm_tx = pcm_tx.clone();
        let data_failure = Arc::clone(&failure);
        let data_failure_tx = failure_tx.clone();
        let error_failure = Arc::clone(&failure);
        let error_failure_tx = failure_tx.clone();
        let stream = self.device.build_input_stream(
            self.supported.into(),
            move |data: &[f32], info| {
                if data_failure.message().is_some() {
                    return;
                }
                let captured_at = wasapi_capture_time(session_started, info);
                let samples = downmix_to_mono(data, channels);
                match data_pcm_tx.try_send(CaptureMessage::Pcm((captured_at, samples))) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => report_capture_failure(
                        label,
                        format!("PCM queue reached its {PCM_QUEUE_CAPACITY}-buffer limit"),
                        &data_failure,
                        &data_failure_tx,
                        &data_pcm_tx,
                    ),
                    Err(TrySendError::Disconnected(_)) => report_capture_failure(
                        label,
                        "audio forwarding thread stopped".to_string(),
                        &data_failure,
                        &data_failure_tx,
                        &data_pcm_tx,
                    ),
                }
            },
            move |err| {
                report_capture_failure(
                    label,
                    format!("stream error: {err}"),
                    &error_failure,
                    &error_failure_tx,
                    &pcm_tx,
                );
            },
            None,
        )?;

        print_audio_format(
            self.display_label,
            &device_name,
            channels,
            sample_rate,
            sample_format,
        );
        Ok(Capture {
            label: self.label,
            stream: Some(stream),
            forwarder: spawn_forwarder(self.label, input, sample_rate, pcm_rx, failure, failure_tx),
        })
    }
}

pub struct Capture {
    label: &'static str,
    stream: Option<cpal::Stream>,
    forwarder: JoinHandle<ExampleResult<()>>,
}

impl Capture {
    pub fn play(&self) -> ExampleResult<()> {
        self.stream
            .as_ref()
            .ok_or("capture already stopped")?
            .play()?;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.stream.take();
    }

    pub fn join(mut self) -> ExampleResult<()> {
        self.stop();
        match self.forwarder.join() {
            Ok(result) => result,
            Err(_) => Err(format!("{} forwarding thread panicked", self.label).into()),
        }
    }
}

pub fn print_event(
    event: TranscriptEvent,
    microphone_id: AudioSourceId,
    system_id: Option<AudioSourceId>,
) -> bool {
    match event {
        TranscriptEvent::Segment(segment) => {
            let source = if segment.source_id == microphone_id {
                "microphone"
            } else if Some(segment.source_id) == system_id {
                "system"
            } else {
                "unknown"
            };
            print_transcript(&segment, Some(source));
            true
        }
        TranscriptEvent::EndOfStream => false,
        _ => true,
    }
}

fn require_f32(source: &str, format: cpal::SampleFormat) -> ExampleResult<()> {
    if format != cpal::SampleFormat::F32 {
        return Err(format!(
            "this example expects f32 {source} audio, got {format:?}; convert the callback samples to f32 in production"
        )
        .into());
    }
    Ok(())
}

fn print_audio_format(
    label: &str,
    device_name: &str,
    channels: usize,
    sample_rate: u32,
    sample_format: cpal::SampleFormat,
) {
    println!("  {label} device {device_name}");
    if channels == 1 && sample_rate == TARGET_SAMPLE_RATE {
        println!("  {label} 1 ch / {TARGET_SAMPLE_RATE} Hz / {sample_format:?}");
    } else {
        println!(
            "  {label} {channels} ch / {sample_rate} Hz / {sample_format:?} -> ASR mono / {TARGET_SAMPLE_RATE} Hz / F32"
        );
    }
}

fn spawn_forwarder(
    label: &'static str,
    input: AudioInput,
    sample_rate: u32,
    pcm_rx: Receiver<CaptureMessage>,
    failure: Arc<CaptureFailure>,
    failure_tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> JoinHandle<ExampleResult<()>> {
    std::thread::spawn(move || {
        let forward_result = match forward_audio(&input, sample_rate, pcm_rx, &failure) {
            Ok(()) => Ok(()),
            Err(error) => {
                let (_, message) = report_failure(
                    label,
                    "forwarding",
                    &error.to_string(),
                    &failure,
                    &failure_tx,
                );
                Err(message.into())
            }
        };
        let close_result = input.blocking_close();
        let result: ExampleResult<()> = match (forward_result, close_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(format!("failed to close input: {error}").into()),
            (Err(forward_error), Err(close_error)) => Err(format!(
                "{forward_error}; additionally failed to close input: {close_error}"
            )
            .into()),
        };
        if let Err(error) = result {
            if failure.message().is_none() {
                let (_, message) = report_failure(
                    label,
                    "forwarding",
                    &error.to_string(),
                    &failure,
                    &failure_tx,
                );
                return Err(message.into());
            }
            return Err(error);
        }
        Ok(())
    })
}

fn forward_audio(
    input: &AudioInput,
    sample_rate: u32,
    pcm_rx: Receiver<CaptureMessage>,
    failure: &CaptureFailure,
) -> ExampleResult<()> {
    let mut resampler = AudioResampler::new(sample_rate)?;
    let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);
    let mut source_started_at = None;
    let mut timeline_anchored = false;

    loop {
        if let Some(message) = failure.message() {
            return Err(message.to_string().into());
        }
        let message = match pcm_rx.recv() {
            Ok(message) => message,
            Err(_) => break,
        };
        let CaptureMessage::Pcm((captured_at, samples)) = message else {
            continue;
        };
        if !timeline_anchored {
            source_started_at.get_or_insert(captured_at);
        }
        resampler.push(&samples, &mut asr_buffer)?;
        send_complete_chunks(
            input,
            &mut asr_buffer,
            &mut source_started_at,
            &mut timeline_anchored,
        )?;
    }

    if let Some(message) = failure.message() {
        return Err(message.to_string().into());
    }

    resampler.finish(&mut asr_buffer)?;
    if !asr_buffer.is_empty() {
        send_chunk(
            input,
            std::mem::take(&mut asr_buffer),
            &mut source_started_at,
            &mut timeline_anchored,
        )?;
    }
    Ok(())
}

fn report_capture_failure(
    label: &str,
    message: String,
    failure: &CaptureFailure,
    failure_tx: &tokio::sync::mpsc::UnboundedSender<String>,
    pcm_tx: &SyncSender<CaptureMessage>,
) {
    if report_failure(label, "capture", &message, failure, failure_tx).0 {
        let _ = pcm_tx.try_send(CaptureMessage::Wake);
    }
}

fn report_failure(
    label: &str,
    operation: &str,
    message: &str,
    failure: &CaptureFailure,
    failure_tx: &tokio::sync::mpsc::UnboundedSender<String>,
) -> (bool, String) {
    let message = format!("{label} {operation} failed: {message}");
    let recorded = failure.record(message.clone());
    let message = failure.message().unwrap_or(&message).to_string();
    if recorded {
        let _ = failure_tx.send(message.clone());
    }
    (recorded, message)
}

fn send_complete_chunks(
    input: &AudioInput,
    asr_buffer: &mut Vec<f32>,
    source_started_at: &mut Option<Duration>,
    timeline_anchored: &mut bool,
) -> yamabiko_asr::Result<()> {
    while asr_buffer.len() >= ASR_CHUNK_SAMPLES {
        let remainder = asr_buffer.split_off(ASR_CHUNK_SAMPLES);
        let chunk = std::mem::replace(asr_buffer, remainder);
        send_chunk(input, chunk, source_started_at, timeline_anchored)?;
    }
    Ok(())
}

fn send_chunk(
    input: &AudioInput,
    chunk: Vec<f32>,
    source_started_at: &mut Option<Duration>,
    timeline_anchored: &mut bool,
) -> yamabiko_asr::Result<()> {
    if *timeline_anchored {
        input.blocking_send(chunk)
    } else {
        let timestamp = source_started_at.take().unwrap_or(Duration::ZERO);
        input.blocking_send_at(timestamp, chunk)?;
        *timeline_anchored = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_failure_is_bounded_to_first_notification() {
        let failure = CaptureFailure::default();
        let (failure_tx, mut failure_rx) = tokio::sync::mpsc::unbounded_channel();
        let (pcm_tx, _pcm_rx) = std::sync::mpsc::sync_channel(1);
        assert!(pcm_tx.try_send(CaptureMessage::Wake).is_ok());

        report_capture_failure(
            "microphone",
            "first".to_string(),
            &failure,
            &failure_tx,
            &pcm_tx,
        );
        report_capture_failure(
            "microphone",
            "second".to_string(),
            &failure,
            &failure_tx,
            &pcm_tx,
        );

        assert_eq!(failure.message(), Some("microphone capture failed: first"));
        assert_eq!(
            failure_rx.try_recv().unwrap(),
            "microphone capture failed: first"
        );
        assert!(failure_rx.try_recv().is_err());
    }

    #[test]
    fn forwarding_failure_suppresses_disconnected_capture_notification() {
        let failure = CaptureFailure::default();
        let (failure_tx, mut failure_rx) = tokio::sync::mpsc::unbounded_channel();
        let (pcm_tx, _pcm_rx) = std::sync::mpsc::sync_channel(1);

        let (recorded, message) = report_failure(
            "microphone",
            "forwarding",
            "ASR input closed",
            &failure,
            &failure_tx,
        );
        assert!(recorded);
        assert_eq!(message, "microphone forwarding failed: ASR input closed");

        report_capture_failure(
            "microphone",
            "audio forwarding thread stopped".to_string(),
            &failure,
            &failure_tx,
            &pcm_tx,
        );

        assert_eq!(failure.message(), Some(message.as_str()));
        assert_eq!(failure_rx.try_recv().unwrap(), message);
        assert!(failure_rx.try_recv().is_err());
    }
}
