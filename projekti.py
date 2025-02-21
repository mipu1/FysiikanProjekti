import streamlit as st 
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from math import radians, cos, sin, asin, sqrt


pathLoc = 'Location.csv'
pathLin = 'Linear Acceleration.csv'

dfLoc = pd.read_csv(pathLoc)
dfLin = pd.read_csv(pathLin)


st.title("Kävely")                                                      #

                    ### Suodatettu 
def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

n = len(dfLin['Time (s)'])
T = dfLin['Time (s)'][len(dfLin['Time (s)'])-1] - dfLin['Time (s)'][0]
fs = n/T
nyq = fs/2 
order = 3
cutoff = 1/(0.5)


f = dfLin["Linear Acceleration y (m/s^2)"]
#f_filter_y = butter_lowpass_filter(dfLin['Linear Acceleration y (m/s^2)'], cutoff, fs, nyq, order)
f_filter_z = butter_lowpass_filter(dfLin['Linear Acceleration z (m/s^2)'], cutoff, fs, nyq, order)
#f_filter_x = butter_lowpass_filter(dfLin['Linear Acceleration x (m/s^2)'], cutoff, fs, nyq, order)

jaksot = 0
for i in range(n-1):
    if f_filter_z[i]/f_filter_z[i+1] < 0:
        jaksot = jaksot + 1

askelmäärä_peaks = jaksot /2
st.write(f"Askelmäärä huippujen tunnistamisen avulla on: {int(askelmäärä_peaks)}")     #

                ## Fourier

t = dfLin["Time (s)"]
N = len(f)
dt = np.max(t)/N

fourier = np.fft.fft(f,N)
psd = fourier*np.conj(fourier)/N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N/2))


f_max = freq[L][psd[L] == np.max(psd[L])]
T = 1/f_max 
askelmäärä_f = np.max(t)*f_max

st.write(f"Askelmäärä Fourier analyysin avulla on: {int(askelmäärä_f)}")       #


            ## Kuljettu matka, nopeus ja kartta

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r



dfLoc['dist'] = np.zeros(len(dfLoc))
dfLoc['t_dist'] = np.zeros(len(dfLoc))

for i in range(len(dfLoc)-1):
    dfLoc.loc[i+1,'dist'] = haversine(dfLoc['Longitude (°)'][i], dfLoc['Latitude (°)'][i], dfLoc['Longitude (°)'][i+1], dfLoc['Latitude (°)'][i+1]) 
    dfLoc.loc[i+1,'t_diff'] = dfLoc['Time (s)'][i+1] - dfLoc['Time (s)'][i]

dfLoc['velocity'] = dfLoc['dist'] / dfLoc['t_diff'] * 1000
dfLoc['cum_dist'] = np.cumsum(dfLoc['dist'])

nopeuden_ka = dfLoc['velocity'].mean()
matka = dfLoc['cum_dist'].iloc[-1]

st.write(f"Keskinopeus {round(nopeuden_ka, 2)} m/s")                            #
st.write(f"Kuljettu matka {round(matka, 3)} kilometriä")                        #


            ## Askelpituus (askelmäärä ja matkanpituus)

askelpituus = (matka*1000)/askelmäärä_peaks

st.write(f"Askelpituus ~ {round(askelpituus, 2)} cm")    

            ## Kartta
start_lat = dfLoc['Latitude (°)'].mean()
start_long = dfLoc['Longitude (°)'].mean()
map = folium.Map(location = [start_lat,start_long], zoom_start = 14)


folium.PolyLine(dfLoc[['Latitude (°)','Longitude (°)']], color = 'blue', weight = 3.5, opacity = 1).add_to(map)

st_map = st_folium(map, width=900, height=650)


                          #

            ## Suodatettu kiihtyvyysdata, jota käytit askelmäärän määrittelemiseen
st.line_chart(f_filter_z, x_label="Kiihtyvyysdata", )

            ## Tehospektri

chart_data = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(chart_data, x = 'freq', y = 'psd' , y_label = 'Teho',x_label = 'Taajuus [Hz]')


