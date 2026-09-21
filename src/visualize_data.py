# import pyqtgraph as pg
from signal_reader import SignalReader

reader = SignalReader("data/IDP_signal_example_[sami_dmitry_september2023].csv")
time, signal = reader.get_slice(0, 5)                  
