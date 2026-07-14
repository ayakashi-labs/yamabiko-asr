use super::audio::{
    ASR_CHUNK_SAMPLES, AudioResampler, TARGET_SAMPLE_RATE, downmix_to_mono, wasapi_capture_time,
};
use super::{ExampleResult, local_time};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use yamabiko_asr::{AudioInput, AudioSourceId, PcmChunk, TranscriptEvent};

type TimedPcm = (Duration, Vec<f32>);

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

    pub fn start(self, session_started: Instant, input: AudioInput) -> ExampleResult<Capture> {
        let device_name = self
            .device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "Unknown device".to_string());
        let channels = self.supported.channels() as usize;
        let sample_rate = self.supported.sample_rate();
        let sample_format = self.supported.sample_format();
        let (pcm_tx, pcm_rx) = std::sync::mpsc::channel::<TimedPcm>();
        let stream = self.device.build_input_stream(
            self.supported.into(),
            move |data: &[f32], info| {
                let captured_at = wasapi_capture_time(session_started, info);
                let samples = downmix_to_mono(data, channels);
                let _ = pcm_tx.send((captured_at, samples));
            },
            move |err| eprintln!("{} stream error: {err}", self.label),
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
            forwarder: spawn_forwarder(self.label, input, sample_rate, pcm_rx),
        })
    }
}

pub struct Capture {
    label: &'static str,
    stream: Option<cpal::Stream>,
    forwarder: JoinHandle<()>,
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
        self.forwarder
            .join()
            .map_err(|_| format!("{} forwarding thread panicked", self.label).into())
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
                "microphone".to_string()
            } else if Some(segment.source_id) == system_id {
                "system".to_string()
            } else {
                format!("source:{}", segment.source_id.get())
            };
            let inference_seconds = segment.inference_duration.as_secs_f64();
            let audio_seconds = segment.end.saturating_sub(segment.start).as_secs_f64();
            let rtf = if audio_seconds > 0.0 {
                inference_seconds / audio_seconds
            } else {
                0.0
            };

            println!("[{}] [{source}] {}", local_time(), segment.text);
            println!(
                "  Timeline {:.2}-{:.2}s / Inference {inference_seconds:.2}s / Audio {audio_seconds:.2}s / RTF {rtf:.2}",
                segment.start.as_secs_f64(),
                segment.end.as_secs_f64(),
            );
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
    pcm_rx: Receiver<TimedPcm>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(err) = forward_audio(&input, sample_rate, pcm_rx) {
            eprintln!("{label} forwarding failed: {err}");
        }
        if let Err(err) = input.blocking_close() {
            eprintln!("failed to close {label} input: {err}");
        }
    })
}

fn forward_audio(
    input: &AudioInput,
    sample_rate: u32,
    pcm_rx: Receiver<TimedPcm>,
) -> ExampleResult<()> {
    let mut resampler = AudioResampler::new(sample_rate)?;
    let mut asr_buffer = Vec::with_capacity(ASR_CHUNK_SAMPLES * 2);
    let mut source_started_at = None;
    let mut timeline_anchored = false;

    while let Ok((captured_at, samples)) = pcm_rx.recv() {
        if !timeline_anchored {
            source_started_at.get_or_insert(captured_at);
        }
        for chunk in resampler.push(&samples)? {
            asr_buffer.extend(chunk);
        }
        send_complete_chunks(
            input,
            &mut asr_buffer,
            &mut source_started_at,
            &mut timeline_anchored,
        )?;
    }

    for chunk in resampler.finish()? {
        asr_buffer.extend(chunk);
    }
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
        input.blocking_send(PcmChunk::new(chunk))
    } else {
        let timestamp = source_started_at.take().unwrap_or(Duration::ZERO);
        input.blocking_send_at(timestamp, PcmChunk::new(chunk))?;
        *timeline_anchored = true;
        Ok(())
    }
}
