# This script generates a dynamic, stereo "cloud" of supersaw waveforms
# that evolves over time and saves the result to a stereo WAV file.

import numpy as np
from scipy.io.wavfile import write
import os
from numpy.random import uniform

# --- Synthesis Parameters ---
# The sample rate of the audio in Hz. Standard CD quality is 44100.
SAMPLE_RATE = 44100  
# The total duration of the audio in seconds.
TOTAL_DURATION = 11.0 * 60  # 11 minutes
# The number of detuned sawtooth waves to create a single supersaw.
NUM_SAWS = 9
# The maximum detuning amount for the saws in cents.
DETUNE_CENTS = 60
# Number of supersaw instances to be layered. The actual number will change over time.
MAX_LAYERS = 15

# --- Harmonic Progression ---
# The frequencies (in Hz) for the initial, beautiful phase (C minor chord).
# C3, Eb3, G3, C4, Eb4, G4, C5...
BEAUTIFUL_FREQUENCIES = [130.81, 155.56, 196.00, 261.63, 311.13, 392.00, 523.25]
# Frequencies for the final, discordant phase. These are more dissonant.
DISCORDANT_FREQUENCIES = [130.81, 140.0, 165.0, 180.0, 205.0, 290.0, 350.0]

# --- Function Definitions ---
def cents_to_ratio(cents):
  """
  Converts a value in cents to a frequency ratio.
  A cent is a logarithmic unit of musical interval.
  """
  return 2**(cents / 1200)

def generate_sawtooth_wave(frequency, duration, sample_rate, amplitude=1.0):
  """
  Generates a single sawtooth waveform.
  """
  t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
  # The formula for a sawtooth wave: 2 * (x - floor(x + 0.5)) where x is the phase.
  phase = frequency * t
  waveform = 2 * (phase - np.floor(phase + 0.5))
  return waveform * amplitude

def generate_supersaw(frequency, num_saws, detune_cents, duration, sample_rate, random_detune=False):
  """
  Generates a single supersaw by layering multiple detuned sawtooth waves.
  Random detune can be used for more chaotic sounds.
  """
  supersaw_mix = np.zeros(int(sample_rate * duration))
  
  # Calculate the frequency ratios for each saw based on detuning.
  detune_step = detune_cents / (num_saws - 1)
  
  for i in range(num_saws):
    # Determine detune amount.
    if random_detune:
      current_detune = uniform(-detune_cents / 2, detune_cents / 2)
    else:
      current_detune = (i - (num_saws - 1) / 2) * detune_step
    
    # Apply the cent ratio to the base frequency.
    current_freq = frequency * cents_to_ratio(current_detune)
    
    # Generate the sawtooth wave for this frequency.
    saw = generate_sawtooth_wave(current_freq, duration, sample_rate)
    
    # Add to the mix.
    supersaw_mix += saw
    
  # Normalize the supersaw mix to prevent clipping.
  if np.max(np.abs(supersaw_mix)) > 0:
    supersaw_mix /= np.max(np.abs(supersaw_mix))
  
  return supersaw_mix

def main():
  """
  Main function to generate the stereo cloud and save the WAV file.
  """
  print("Generating exquisite and dynamic supersaw cloud...")
  
  total_samples = int(SAMPLE_RATE * TOTAL_DURATION)
  
  # Create a two-channel array for stereo audio.
  final_audio = np.zeros((total_samples, 2))
  
  # Define key timestamps in seconds.
  phase_1_end_time = 5.0 * 60  # 5 minutes
  phase_2_end_time = 11.0 * 60 # 11 minutes

  # Generate audio for the first phase (beautiful and growing).
  print("Phase 1: Growing beautiful harmony (0-5 minutes)...")
  phase_1_samples = int(SAMPLE_RATE * phase_1_end_time)
  
  # Create a time array for the first phase.
  t1 = np.linspace(0, phase_1_end_time, phase_1_samples, endpoint=False)
  
  # Generate a dynamic fade-in envelope for the number of layers.
  layer_count_envelope = np.interp(t1, [0, phase_1_end_time], [1, MAX_LAYERS]).astype(int)
  
  for i, current_layers in enumerate(layer_count_envelope):
    if current_layers > 0:
      # Generate supersaw instances for this time step.
      for j in range(current_layers):
        # Pick a base frequency from the beautiful scale.
        freq_idx = j % len(BEAUTIFUL_FREQUENCIES)
        base_freq = BEAUTIFUL_FREQUENCIES[freq_idx]
        
        # Add a small random offset for unique layering in each channel.
        offset_l = uniform(-2, 2)
        offset_r = uniform(-2, 2)
        
        # Generate supersaws for left and right channels.
        supersaw_l = generate_supersaw(base_freq + offset_l, NUM_SAWS, DETUNE_CENTS, 1.0/SAMPLE_RATE, SAMPLE_RATE)
        supersaw_r = generate_supersaw(base_freq + offset_r, NUM_SAWS, DETUNE_CENTS, 1.0/SAMPLE_RATE, SAMPLE_RATE)
        
        # Add to the final audio mix at the current time step.
        final_audio[i, 0] += supersaw_l[0] / current_layers
        final_audio[i, 1] += supersaw_r[0] / current_layers

  # Generate audio for the second phase (gentler and more discordant).
  print("Phase 2: Transitioning to gentle discord (5-11 minutes)...")
  phase_2_samples = int(SAMPLE_RATE * (phase_2_end_time - phase_1_end_time))
  
  # Create a time array for the second phase.
  t2 = np.linspace(phase_1_end_time, phase_2_end_time, phase_2_samples, endpoint=False)
  
  # Create a dynamic fade-out envelope for the number of layers.
  layer_count_envelope = np.interp(t2, [phase_1_end_time, phase_2_end_time], [MAX_LAYERS, 1]).astype(int)
  
  for i, current_layers in enumerate(layer_count_envelope):
    if current_layers > 0:
      # Generate supersaw instances for this time step.
      for j in range(current_layers):
        # Pick a base frequency from the discordant scale.
        freq_idx = j % len(DISCORDANT_FREQUENCIES)
        base_freq = DISCORDANT_FREQUENCIES[freq_idx]
        
        # Add a small random offset for unique layering in each channel.
        offset_l = uniform(-5, 5)
        offset_r = uniform(-5, 5)
        
        # Generate supersaws for left and right channels with increasing random detune.
        supersaw_l = generate_supersaw(base_freq + offset_l, NUM_SAWS, DETUNE_CENTS * (1 + i/phase_2_samples), 1.0/SAMPLE_RATE, SAMPLE_RATE, random_detune=True)
        supersaw_r = generate_supersaw(base_freq + offset_r, NUM_SAWS, DETUNE_CENTS * (1 + i/phase_2_samples), 1.0/SAMPLE_RATE, SAMPLE_RATE, random_detune=True)

        # Add to the final audio mix at the current time step.
        final_audio[phase_1_samples + i, 0] += supersaw_l[0] / current_layers
        final_audio[phase_1_samples + i, 1] += supersaw_r[0] / current_layers

  # Final amplitude envelope for the entire track to ensure a smooth finish.
  overall_envelope = np.linspace(1, 0, total_samples)**2
  final_audio[:, 0] *= overall_envelope
  final_audio[:, 1] *= overall_envelope
  
  # Normalize the entire cloud to prevent clipping and convert to 16-bit integer format.
  max_val = np.max(np.abs(final_audio))
  if max_val > 0:
      final_audio /= max_val
  final_audio = np.int16(final_audio * 32767)
  
  # Define the output file name.
  output_filename = "supersaw_cloud.wav"
  
  # Write the audio to a WAV file.
  write(output_filename, SAMPLE_RATE, final_audio)
  
  print(f"Successfully generated and saved '{output_filename}'.")

if __name__ == "__main__":
  main()
