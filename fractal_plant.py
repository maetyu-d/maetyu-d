from audio_sequence import AudioSequence  # or just run the class definition directly

seq = AudioSequence(sample_rate=44100, bit_depth=16, channels=1)

# Classic fractal plant L-system with branching
plant_rules = {'X': 'F+[[X]-X]-F[-FX]+X', 'F': 'FF'}
seq.add_l_system_pattern(n_events=800, axiom='X', rules=plant_rules, iterations=6,
                         apply_to='dur', base_dur=20, dur_scale=8, amp_scale=0.3)

# Add some random accents
seq.add_random_pattern(n_events=200, amp_min=0.4, amp_max=0.9, dur_min=5, dur_max=15)

seq.set_total_length(20, unit='seconds')
seq.generate_audio('fractal_plant_rhythm.wav')
seq.play(blocking=False)
