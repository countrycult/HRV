import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

def generate_synthetic_data(n=10, start_time=0, interval_range=(0.8, 1.2)):
    timestamps = [start_time]
    for _ in range(n - 1):
        timestamps.append(timestamps[-1] + np.random.uniform(*interval_range))
    return np.array(timestamps)

def calculate_hrv(timestamps):
    if len(timestamps) < 2:
        return None, None, None, None, None
    
    rr_intervals = np.diff(timestamps) * 1000  # Convert seconds to ms
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    pnn50 = (np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)) * 100
    heart_rate = 60000 / mean_rr  # Convert RR interval to Heart Rate (BPM)
    
    return rr_intervals, mean_rr, sdnn, rmssd, pnn50, heart_rate

def generate_ecg_waveform(n_samples, heart_rate):
    t = np.linspace(0, n_samples / heart_rate * 60, n_samples * 10)
    ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + 0.2 * np.random.randn(len(t))
    return t, ecg_signal

st.title("HRV Calculator with Synthetic Data")

# User Input for Synthetic Data Generation
n_samples = st.number_input("Number of synthetic timestamps:", min_value=5, max_value=100, value=10)
interval_min = st.number_input("Minimum RR Interval (seconds):", min_value=0.5, max_value=2.0, value=0.8)
interval_max = st.number_input("Maximum RR Interval (seconds):", min_value=0.5, max_value=2.0, value=1.2)

timestamps = generate_synthetic_data(n_samples, start_time=time.time(), interval_range=(interval_min, interval_max))
st.write("### Synthetic Timestamps")
st.write(timestamps)

# Calculate HRV Metrics
rr_intervals, mean_rr, sdnn, rmssd, pnn50, heart_rate = calculate_hrv(timestamps)

if rr_intervals is not None:
    st.write("### RR Intervals (ms)", rr_intervals)
    st.write(f"**Mean RR Interval:** {mean_rr:.2f} ms")
    st.write(f"**SDNN:** {sdnn:.2f} ms")
    st.write(f"**RMSSD:** {rmssd:.2f} ms")
    st.write(f"**pNN50:** {pnn50:.2f} %")
    st.write(f"**Estimated Heart Rate:** {heart_rate:.2f} BPM")
    
    # RR Interval Visualization
    fig, ax = plt.subplots()
    ax.plot(rr_intervals, marker='o', linestyle='-', color='b')
    ax.set_title("RR Intervals Over Time")
    ax.set_xlabel("Heartbeat Count")
    ax.set_ylabel("RR Interval (ms)")
    ax.grid(True)
    st.pyplot(fig)
    
    # ECG Waveform Simulation
    t, ecg_signal = generate_ecg_waveform(n_samples, heart_rate)
    fig, ax = plt.subplots()
    ax.plot(t, ecg_signal, color='r')
    ax.set_title("Synthetic ECG Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error("Not enough timestamps to calculate HRV.")
