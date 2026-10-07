# import pyqtgraph as pg
from signal_reader import SignalReader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
reader = SignalReader("data/5. suction_HIGH_CONC_1ml_and_no_resuction.csv")
time, signal = reader.get_slice(0, 70, max_points=500, use_minmax=False)        

print(signal.shape)



# Detect peaks in real-time
window_size = 2000
threshold = 30
peaks = find_peaks(signal[:, 1], height=threshold, distance=window_size)[1]

print(peaks)

plt.plot(time, signal[:, 1], label='Signal')
plt.title('Signal with Detected Peaks')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()
