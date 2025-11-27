import math
import wave
import numpy as np
import sounddevice as sd


# --------------------------------------
# Basic bandpass biquad implementation
# --------------------------------------

class BiquadBandpass:
    """
    Simple RBJ-style bandpass biquad (constant skirt gain).
    """

    def __init__(self, fs, f0, Q):
        self.fs = fs
        self.f0 = f0
        self.Q = Q

        self.b0 = self.b1 = self.b2 = 0.0
        self.a1 = self.a2 = 0.0

        self.z1 = 0.0
        self.z2 = 0.0

        self._design()

    def _design(self):
        w0 = 2.0 * math.pi * self.f0 / self.fs
        alpha = math.sin(w0) / (2.0 * self.Q)

        b0 = self.Q * alpha
        b1 = 0.0
        b2 = -self.Q * alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * math.cos(w0)
        a2 = 1.0 - alpha

        # normalize
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def process(self, x):
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


# --------------------------------------
# Duffing + resonant bank voice
# --------------------------------------

class DuffingVoice:
    """
    One Gutter-style 'resonant Duffing' voice.
    """

    def __init__(
        self,
        fs=44100,
        n_modes=16,
        base_freq=100.0,
        max_freq=6000.0,
        Q=20.0,
        alpha=-1.0,
        beta=1.0,
        base_k=0.2,
        B=0.4,
        drive_freq=40.0,
        gain=2.0,
        soften=0.05,
        pitch_shift=1.0,
        seed=None,
    ):
        self.fs = fs
        self.dt = 1.0 / fs

        # Duffing parameters
        self.alpha = alpha
        self.beta = beta
        self.base_k = base_k
        self.B = B
        self.drive_freq = drive_freq
        self.omega = 2.0 * math.pi * drive_freq

        # Control-ish parameters
        self.gain = gain
        self.soften = soften
        self.pitch_shift = pitch_shift

        # State
        self.x = 0.01
        self.y = 0.0
        self.phase = 0.0
        self.lp = 0.0

        # Filter bank
        rng = np.random.default_rng(seed)
        centres = np.logspace(
            math.log10(base_freq),
            math.log10(max_freq),
            n_modes
        ) * pitch_shift
        jitter = rng.uniform(0.95, 1.05, size=n_modes)
        centres *= jitter

        self.filters = [BiquadBandpass(fs, f, Q) for f in centres]

    def step(self, k_eff):
        # 1) Filter bank inside the loop: feed previous x
        fb_sum = 0.0
        for bp in self.filters:
            fb_sum += bp.process(self.x)

        fb_sum /= max(len(self.filters), 1)
        fb_sum *= self.gain

        # 2) Lowpass "soften"
        self.lp += self.soften * (fb_sum - self.lp)

        # 3) Arctan limiter to ±1-ish
        constrained = math.atan(self.lp) * (2.0 / math.pi)

        # 4) Duffing update using constrained as "position"
        drive = self.B * math.cos(self.phase)

        x_dot = self.y
        y_dot = -k_eff * self.y - self.alpha * constrained - self.beta * (constrained ** 3) + drive

        self.x += self.dt * x_dot
        self.y += self.dt * y_dot

        self.phase += self.omega * self.dt
        if self.phase > 2.0 * math.pi:
            self.phase -= 2.0 * math.pi

        return constrained


# --------------------------------------
# Network of interacting voices (offline)
# --------------------------------------

class GutterSynthNetwork:
    """
    Multi-voice Gutter-style network:
    - N resonant Duffing voices
    - 2000-sample delay from each voice to others' damping parameter k
    """

    def __init__(
        self,
        num_voices=4,
        fs=44100,
        seconds=10.0,
        interaction_amount=0.3,
        delay_samples=2000,
        base_k=0.2,
        **voice_kwargs,
    ):
        self.fs = fs
        self.seconds = seconds
        self.num_voices = num_voices
        self.interaction_amount = interaction_amount
        self.delay_samples = delay_samples
        self.base_k = base_k

        self.voices = [
            DuffingVoice(fs=fs, base_k=base_k, seed=i, **voice_kwargs)
            for i in range(num_voices)
        ]

        self.delay_buffers = np.zeros((num_voices, delay_samples), dtype=np.float64)
        self.delay_pos = 0

        # Initialise delay with tiny noise so it doesn't start dead
        self.delay_buffers[:] = np.random.uniform(-0.001, 0.001, self.delay_buffers.shape)

    def render(self):
        num_samples = int(self.seconds * self.fs)
        out = np.zeros(num_samples, dtype=np.float32)

        for n in range(num_samples):
            # 1) Read delayed outputs and compute k for each voice
            k_eff = np.zeros(self.num_voices, dtype=np.float64)
            delayed_mix = np.mean(self.delay_buffers[:, self.delay_pos])

            for i in range(self.num_voices):
                k_raw = self.base_k + self.interaction_amount * delayed_mix
                k_eff[i] = np.clip(k_raw, 1e-4, 1.0)

            # 2) Step each voice
            frame = np.zeros(self.num_voices, dtype=np.float64)
            for i, v in enumerate(self.voices):
                frame[i] = v.step(k_eff[i])

            # 3) Update delay lines
            self.delay_buffers[:, self.delay_pos] = frame
            self.delay_pos = (self.delay_pos + 1) % self.delay_samples

            # 4) Mixdown
            out[n] = np.mean(frame)

        # Light output scaling
        out *= 0.4
        return out


def save_wav_mono(path, audio, fs):
    """
    Save a mono float32 numpy array (-1..1) as 16-bit WAV using stdlib.
    """
    # Normalise to safe range
    max_val = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if max_val > 0:
        audio = audio / max_val * 0.9

    int_samples = (audio * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(int_samples.tobytes())


if __name__ == "__main__":
    fs = 44100

    synth = GutterSynthNetwork(
        num_voices=4,
        fs=fs,
        seconds=10.0,
        base_k=0.2,
        interaction_amount=0.3,
        delay_samples=2000,
        # Per-voice parameters
        n_modes=16,
        base_freq=100.0,
        max_freq=6000.0,
        Q=20.0,
        alpha=-1.0,
        beta=1.0,
        B=0.4,
        drive_freq=40.0,
        gain=2.0,
        soften=0.05,
        pitch_shift=1.0,
    )

    print("Rendering Gutter-style audio...")
    audio = synth.render()
    print("Render done. Playing...")

    # Normalise for playback
    max_val = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Play via sounddevice (realtime playback of pre-rendered buffer)
    sd.play(audio.astype(np.float32), fs)
    sd.wait()
    print("Playback finished.")

    # Optional: save the WAV in the same folder
    save_wav_mono("gutter_synth_offline.wav", audio, fs)
    print("Saved gutter_synth_offline.wav")
