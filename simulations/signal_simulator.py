import numpy as np

def generate_composite_radio_signals(
    num_sources,
    num_samples,
    fs,
    freq_range=(50, 1000),
    components_per_source=(2, 6),
    amplitude_range=(0.5, 1.5),
    snr_db=None,
    random_seed=None
):
    if random_seed is not None:
        np.random.seed(random_seed)

    t = np.arange(num_samples) / fs
    # Use complex128 for analytic signal representation
    signals = np.zeros((num_sources, num_samples), dtype=np.complex128)
    metadata = []

    for src in range(num_sources):
        num_components = np.random.randint(
            components_per_source[0],
            components_per_source[1] + 1
        )
        freqs = np.random.uniform(freq_range[0], freq_range[1], num_components)
        amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_components)
        phases = np.random.uniform(0, 2 * np.pi, num_components)

        signal = np.zeros(num_samples, dtype=np.complex128)
        for f, a, p in zip(freqs, amplitudes, phases):
            # Complex exponential representation
            signal += a * np.exp(1j * (2 * np.pi * f * t + p))

        signals[src] = signal
        metadata.append({"frequencies": freqs, "amplitudes": amplitudes, "phases": phases})

    if snr_db is not None:
        signal_power = np.mean(np.abs(signals) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*signals.shape) + 1j * np.random.randn(*signals.shape)
        )
        signals = signals + noise

    return signals, t, metadata