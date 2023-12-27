#!/usr/bin/python
# -*- coding: utf-8 -*-

# Natural frequency analysis for digital scales @yaku
# https://github.com/ShigeoYakuno

import numpy as np
from scipy import fftpack
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
import tkinter.font as f
from tkinter import filedialog
import re

filename = "sanple.csv"


class Application(tk.Frame):
    # entry Widget val
    entry = None

    def invalidText(self):
        print("restriction fails !")

    def onValidate(self, S):
        # If the entered characters are half-width numbers
        if re.match(re.compile("[0-9]+"), S):
            return True
        elif re.match(re.compile("[.]+"), S):
            return True
        else:
            return False

    # Function that performs Fourier transform
    def calcFFT(self, data, samplerate):
        spectrum = fftpack.fft(data)  # Fourier transform
        amp = np.sqrt((spectrum.real**2) + (spectrum.imag**2))  # amplitude
        amp = amp / (len(data) / 2)  # Amplitude normalization
        phase = np.arctan2(spectrum.imag, spectrum.real)  # Phase calculation
        phase = np.degrees(phase)  # Convert phase from radians to degrees
        freq = np.linspace(0, samplerate, len(data))  # Create frequency axis
        return spectrum, amp, phase, freq

    # Perform Fourier transform sequentially from "csv" in the column direction
    def csvFFT(self, in_file, out_file):
        df = pd.read_csv(in_file, encoding="SHIFT-JIS")

        # Set data rate
        rate = float(self.rate_entry.get())  # Hz
        dt = float(1 / rate)  # sec
        # print(f"rate={dt}")

        # Initialize data frame
        df_amp = pd.DataFrame()
        df_phase = pd.DataFrame()
        df_fft = pd.DataFrame()

        dt_time = pd.DataFrame()

        # Perform sequential Fourier transform
        data = df.T.iloc[0]  # Place vibration data in the second column

        # Keep data within the range 0 to 1 (normalization)
        # normalized_data = (data - data.min()) / (data.max() - data.min())
        # spectrum, amp, phase, freq = self.calcFFT(normalized_data.values, 1 / dt)

        # Set the data mean to 0 and standard deviation to 1 (standardization)
        standardized_data = (data - data.mean()) / data.std()
        spectrum, amp, phase, freq = self.calcFFT(standardized_data.values, 1 / dt)
        # print(f"{standardized_data} {dt}")

        df_amp[df.columns[0] + "_amp"] = pd.Series(amp)
        df_phase[df.columns[0] + "_phase[deg]"] = pd.Series(phase)

        df_fft["freq[Hz]"] = pd.Series(freq)
        # Combine frequency, amplitude, and phase data frames
        df_fft = df_fft.join(df_amp).join(df_phase)
        # Truncate data at Nyquist frequency
        df_fft = df_fft.iloc[range(int(len(df) / 2) + 1), :]
        # Save the Fourier transform result to csv
        df_fft.to_csv(out_file)

        dt_time = df.copy()  # copy() is important. (equal method is inappropriate)
        for num in range(int(len(df))):
            dt_time.iat[num, 0] = float(num * dt)
        # print(f"{dt_time.T.iloc[0]}")

        return df, df_fft, dt_time

    def chooseFile(self):
        global filename

        filename = filedialog.askopenfilename(
            title="open csv file", filetypes=[("csv file", ".csv")], initialdir="./"
        )

    # Create graph of FFT results from csv file
    def createGraph(self):
        global filename

        df, df_fft, dt_time = self.csvFFT(in_file=filename, out_file="fft.csv")

        plt.rcParams["font.size"] = 14
        plt.rcParams["font.family"] = "Times New Roman"

        # scale inward
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        # Grid lines on the top, bottom, left and right of the graph
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(222)  # 224
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_ticks_position("both")

        ax2 = fig.add_subplot(121)
        ax2.yaxis.set_ticks_position("both")
        ax2.xaxis.set_ticks_position("both")

        # Set axis label
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("fresh data")

        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Amplitude")
        ax2.set_title("natural frequency analysis")

        # Setting the scale
        ax2.set_xticks(np.arange(0, 81, 10))
        ax2.set_xlim(0, 80)

        ax1.plot(dt_time.T.iloc[0], df.T.iloc[0], label=df.columns[0], lw=1)
        ax2.plot(df_fft.T.iloc[0], df_fft.T.iloc[1], label=df_fft.columns[1], lw=1)
        # ax1.legend()
        ax2.legend()

        fig.tight_layout()

        plt.show()
        plt.close()

    def __init__(self, master=None):
        # init to Window
        super().__init__(master)

        self.master.title(" Natural frequency analysis for digital scales Ver1.0 @yaku")
        self.master.geometry("320x240")

        frame = tk.Frame(self.master)
        frame.pack()

        vcmd = self.register(self.onValidate)

        font2 = f.Font(family="Lucida Console", weight="bold", size=22, slant="italic")
        font3 = f.Font(family="Lucida Console", weight="bold", size=10, slant="italic")

        rbl_pos_x = 40
        txt_pos_x = 80
        rbl_pos_y = 10
        txt_pos_y = 30

        # sampling rate box
        self.rate_rabel = tk.Label(
            root, text="STEP1 input sampling rate(Hz) ", font=font3
        )
        self.rate_rabel.place(x=rbl_pos_x, y=rbl_pos_y)

        # Create an entry widget using frame widget (Frame) as the parent element.
        self.rate_entry = tk.Entry(
            root,
            width=15,
            validate="key",
            validatecommand=(vcmd, "%S"),
            invalidcommand=self.invalidText,
        )
        self.rate_entry.place(x=txt_pos_x, y=txt_pos_y)

        self.rabel2 = tk.Label(root, text="STEP2", font=font3)
        self.rabel2.place(x=50, y=60)

        # file dialog
        self.calc_btn = tk.Button(
            root, text="choose csv", font=font2, command=self.chooseFile
        )
        self.calc_btn.place(x=50, y=80)

        self.rabel3 = tk.Label(root, text="STEP3", font=font3)
        self.rabel3.place(x=50, y=150)

        # calcurate button
        self.calc_btn = tk.Button(
            root, text="execution FFT", font=font2, command=self.createGraph
        )
        self.calc_btn.place(x=40, y=170)


if __name__ == "__main__":
    # maike a window
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
