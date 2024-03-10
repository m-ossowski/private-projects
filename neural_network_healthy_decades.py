# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:01:44 2022

@author: macie
"""

import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
#%%
# Catalogs
os.chdir(r"C:\\Users\macie\\Desktop\\Semestr 5\\Sieci neuronowe")
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"heart_rhythm")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"data")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "visualisations")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
os.makedirs(KATALOG_DANYCH, exist_ok=True)

path = r"C:\\Users\macie\\Desktop\\Semestr 5\\Sieci neuronowe\\healthy_decades"
# insert your own path in order to work
#%%
# Grouping by gender and age
colnames = ['Interval','Number']

f20_files = glob.glob(os.path.join(path , "*f20*"))
l_f20 = []
for filename in f20_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f20.append(df)
    
frame_f20 = pd.concat(l_f20,ignore_index='True')
frame_f20.name = "f20"
    
f30_files = glob.glob(os.path.join(path , "*f30*"))
l_f30 = []
for filename in f30_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f30.append(df)
frame_f30 = pd.concat(l_f30,ignore_index='True')
frame_f30.name = "f30"
f40_files = glob.glob(os.path.join(path , "*f40*"))
l_f40 = []
for filename in f40_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f40.append(df)
frame_f40 = pd.concat(l_f40,ignore_index='True')
frame_f40.name = "f40"
f50_files = glob.glob(os.path.join(path , "*f50*"))
l_f50 = []
for filename in f50_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f50.append(df)
frame_f50 = pd.concat(l_f50,ignore_index='True')
frame_f50.name = "f50"
f60_files = glob.glob(os.path.join(path , "*f60*"))
l_f60 = []
for filename in f60_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f60.append(df)
frame_f60 = pd.concat(l_f60,ignore_index='True')
frame_f60.name = "f60"
f70_files = glob.glob(os.path.join(path , "*f70*"))
l_f70 = []
for filename in f70_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f70.append(df)
frame_f70 = pd.concat(l_f70,ignore_index='True')
frame_f70.name = "f70"
f80_files = glob.glob(os.path.join(path , "*f80*"))
l_f80 = []
for filename in f80_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_f80.append(df)
frame_f80 = pd.concat(l_f80,ignore_index='True')
frame_f80.name = "f80"

m20_files = glob.glob(os.path.join(path , "*m20*"))
l_m20 = []
for filename in m20_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m20.append(df)
frame_m20 = pd.concat(l_m20,ignore_index='True')
frame_m20.name = "m20"
m30_files = glob.glob(os.path.join(path , "*m30*"))
l_m30 = []
for filename in m30_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m30.append(df)
frame_m30 = pd.concat(l_m30,ignore_index='True')
frame_m30.name = "m30"    
m40_files = glob.glob(os.path.join(path , "*m40*"))
l_m40 = []
for filename in m40_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m40.append(df)
frame_m40 = pd.concat(l_m40,ignore_index='True')
frame_m40.name = "m40"
m50_files = glob.glob(os.path.join(path , "*m50*"))
l_m50 = []
for filename in m50_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m50.append(df)
frame_m50 = pd.concat(l_m50,ignore_index='True')
frame_m50.name = "m50"
m60_files = glob.glob(os.path.join(path , "*m60*"))
l_m60 = []
for filename in m60_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m60.append(df)
frame_m60 = pd.concat(l_m60,ignore_index='True')
frame_m60.name = "m60"    
m70_files = glob.glob(os.path.join(path , "*m70*"))
l_m70 = []
for filename in m70_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m70.append(df)
frame_m70 = pd.concat(l_m70,ignore_index='True')
frame_m70.name = "m70"
m80_files = glob.glob(os.path.join(path , "*m80*"))
l_m80 = []
for filename in m80_files:
    df = pd.read_csv(filename, delimiter = "\t",names=colnames, header=None)
    l_m80.append(df)
frame_m80 = pd.concat(l_m80,ignore_index='True')
frame_m80.name = "m80"

#%%
# Grouping all females and males together
frame_f = pd.concat([frame_f20,frame_f30,frame_f40,frame_f50,frame_f60,frame_f70,frame_f80],ignore_index='True')
frame_f.name = "females"
files_f = [frame_f20,frame_f30,frame_f40,frame_f50,frame_f60,frame_f70,frame_f80]
frame_m = pd.concat([frame_m20,frame_m30,frame_m40,frame_m50,frame_m60,frame_m70,frame_m80],ignore_index='True')
frame_m.name = "males"
files_m = [frame_m20,frame_m30,frame_m40,frame_m50,frame_m60,frame_m70,frame_m80]
#%%
# Grouping all together
frame_all = pd.concat([frame_f20,frame_f30,frame_f40,frame_f50,frame_f60,frame_f70,frame_f80,frame_m20,frame_m30,frame_m40,frame_m50,frame_m60,frame_m70,frame_m80],ignore_index='True')
frame_all.name = "all"
#%% Looking for NA values in all
print("Number of NA values in column 1: ", frame_all.isnull().sum()[0])
print("Number of missing values in column 2: ", frame_all.isnull().sum()[1])
print("Two zeros mean that all of our data is complete.")
#%%
# Searching for the most common value in all
print("The most frequent value in all is: ", frame_all['Interval'].value_counts().nlargest(1))
#%%
# Searching for the minimum and maximum values of all
interval = frame_all['Interval']
print(f'The minimum value for all is: {min(interval)}')
print(f'and the maximum value for all is: {max(interval)}')
#%% Calculating the mean value and the std for all
interval = frame_all['Interval']
print(f'The mean value for all is: {np.mean(interval)}')
print(f'The std of interval for all is: {np.std(interval)}')    
#%%
# Histogram of all values
frame_all['Interval'].hist(bins=50, figsize=(9,6))
plt.tight_layout()
plt.title("Histogram of values for all")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ frame_all.name+ '_.jpg'), dpi=300 ) 
plt.show() # sprawdzić jak się plik zapisal

#%%
# The visualisation of all
frame_all["Interval"].plot(title='Interval ' + frame_all.name)
plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+frame_all.name+ '_healthy.jpg'), dpi=300 ) 
plt.show()
#%%
# Searching for the most common value for females and males
print("The most frequent value for females is: ", frame_f['Interval'].value_counts().nlargest(1))
print("The most frequent value for males is: ", frame_m['Interval'].value_counts().nlargest(1))
#%%
# Searching for the minimum and maximum values of females and males
interval_f = frame_f['Interval']
interval_m = frame_m['Interval']
print(f'The minimum value for females is: {min(interval_f)}')
print(f'and the maximum value for females is: {max(interval_f)}')
print(f'The minimum value for males is: {min(interval_m)}')
print(f'and the maximum value for males is: {max(interval_m)}')
#%% Calculating the mean value and the std for each gender
interval_f = frame_f['Interval']
interval_m = frame_m['Interval']
print(f'The mean value for females is: {np.mean(interval_f)}')
print(f'The std of the interval for females is: {np.std(interval_f)}')  
print(f'The mean value for males is: {np.mean(interval_m)}')
print(f'The std of the interval for males is: {np.std(interval_m)}')
#%%
# Histogram of female intervals
frame_f['Interval'].hist(bins=50, figsize=(9,6))
plt.tight_layout()
plt.title("Histogram of values for females")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ frame_f.name+ '_.jpg'), dpi=300 ) 
plt.show() # sprawdzić jak się plik zapisal
#%%
# Histogram of male intervals
frame_m['Interval'].hist(bins=50, figsize=(9,6))
plt.tight_layout()
plt.title("Histogram of values for males")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ frame_m.name+ '_.jpg'), dpi=300 ) 
plt.show() # sprawdzić jak się plik zapisal
#%%
# The visualisation for females
frame_f["Interval"].plot(title='Interval ' + frame_f.name)
plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+frame_f.name+ '_healthy.jpg'), dpi=300 ) 
plt.show()
#%%
# The visualisation for males
frame_m["Interval"].plot(title='Interval ' + frame_m.name)
plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+frame_m.name+ '_healthy.jpg'), dpi=300 ) 
plt.show()
#%% # Searching for the most common value of interval for each file

files = [frame_f20,frame_f30,frame_f40,frame_f50,frame_f60,frame_f70,frame_f80,frame_m20,frame_m30,frame_m40,frame_m50,frame_m60,frame_m70,frame_m80]
mcv = []
for file in files:
    a = file['Interval'].value_counts().index[0]
    mcv.append(a)
    print(f"The most frequent value for {file.name} is: ", file['Interval'].value_counts().nlargest(1))
    print()
print(f'The vector of the most common values {mcv}')
#%% Searching for the minimum and maximum values for each file
minimums = []
maximums = []
for file in files:
    interval = file['Interval']
    mins = min(interval)
    maxs = max(interval)
    print(f'The minimum value for {file.name} is: {mins}')
    print(f'and the maximum value for {file.name} is: {maxs}')
    minimums.append(mins)
    maximums.append(maxs)
    print()

print(f'Vector of minimums {minimums} and of maximums {maximums}')
#%% Calculating the mean value and the std for each file
means = []
stds = []

for file in files:
    interval = file['Interval']
    m = np.mean(interval)
    s = np.std(interval)
    print(f'The mean value for {file.name} is: {m}')
    print(f'The std of the interval for {file.name} is: {s}')  
    means.append(m)
    stds.append(s)
    print()
    
print(f'The vector of mean values {means}')
print()
print(f'and of the std {stds}')
#%% Histograms for each file 
for file in files:
    file['Interval'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title(f"Histogram of values for {file.name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ file.name + '_.jpg'), dpi=300 ) 
    plt.show() # sprawdzić jak się plik zapisal
    
#%% Visualisations for each file

for file in files:
    file["Interval"].plot(title='Interval ' + file.name)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+ file.name+ '_healthy.jpg'), dpi=300 ) 
    plt.show()

#%%  Calculating the SDNN for each file
sdnn = []
for file in files: 
    NN=np.diff(file['Interval'])
    SDNN = np.std(NN)
    print(f'The SDNN for {file.name} is: {SDNN}')
    sdnn.append(SDNN)
    print()
    
print(f'The vector of SDNN {sdnn}')
#%% Creating a function which inplaces NA when the intervals are not in order
def check_if_na(diff):
    for i in range(1, len(diff)):
        if diff.index[i] - diff.index[i-1] > 1:
            diff.iloc[i] = np.nan
    return diff
#%% Calculating RMSSD for each file
rmssd = []

for file in files:
    diff = file['Interval'].diff()
    file['diff'] = check_if_na(diff)
    file['abs_diff'] = abs(file['diff'])
    RMSSD = (sum(file['diff'].dropna().pow(2))/len(file['Interval'].dropna()))**(1/2)
    '''NN=np.diff(file['Interval'])
    NN_square=[i**2 for i in NN]
    RMSSD=(sum(NN_square)/len(NN_square))**0.5
    '''
    print(f'The RMSSD for {file.name}: {RMSSD}')
    rmssd.append(RMSSD)
    print()

print(f'The vector of RMSSD {rmssd}')
#%% Calculating the pNN50 and pNN20
pnn50 = []
pnn20 = []
for file in files:
    NN=np.diff(file['Interval'])
    NN50 = sum(i > 50 for i in NN)
    PNN50=NN50/len(NN)
    NN20 = sum(i > 20 for i in NN)
    PNN20=NN20/len(NN)
    print(f'The PNN50 for {file.name}: {PNN50}')
    print(f'The PNN20 for {file.name}: {PNN20}')
    print()
    pnn50.append(PNN50)
    pnn20.append(PNN20)

print(f'The vector of PNN50: {pnn50}')
print()
print(f'The vector of PNN20: {pnn20}')
#%% 
# Stacjonarnosc sygnału, analiza sygnału
# Searching for the minimum average for each range in each file and calculating the minimum std for it

avg_list = []
std_list = []
for file in files:
    interval = file['Interval']
    chunk_size = 1000
    chunked_list = [interval[i:i+chunk_size] for i in range(0, len(interval), chunk_size)]
    # print(len(chunked_list))
    average = [sum(chunked_list[i])/len(chunked_list[i]) for i in range(len(chunked_list))]
    list_for_std = [chunked_list[average.index(min(average))]]
    std = [(sum(list_for_std[i] - min(average))**2/len(list_for_std))**0.5 for i in range(len(list_for_std))]
    print(f"For {file.name} the minimum average is {min(average)} and for that range the std is {std}")
    avg_list.append(min(average))
    std_list.append(std)
    print()
    
print(f"This is a list of minimum averages {avg_list}")
print()
print(f"This is a list of std's {std_list}")
#%%
# Asking the user for the chunk_size for male and female visualisations
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 10000 looks nice): '))
for file in files:
    interval = file['Interval']
    chunked_list_user_input = [interval[i:i+chunk_size_user_input] for i in range(0, len(interval), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {file.name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ file.name + '_.jpg'), dpi=300 ) 
    plt.show()
#%%
# Preparing a list of differences between i and i+1 in intervals
# The code might take a bit to compile
list_of_diff = []
for file in files:
    v = []
    for i in range(len(file)):
        if i == 0:
            a = np.nan
        else:
            a = file['Interval'][i] - file['Interval'][i-1]
        v.append(a)
    list_of_diff.append(v)
#%% 
# Declaring a new column for the difference between i and i+1 in intervals
frame_f20['Difference'] = list_of_diff[0]
frame_f30['Difference'] = list_of_diff[1]
frame_f40['Difference'] = list_of_diff[2]
frame_f50['Difference'] = list_of_diff[3]
frame_f60['Difference'] = list_of_diff[4]
frame_f70['Difference'] = list_of_diff[5]
frame_f80['Difference'] = list_of_diff[6]

frame_m20['Difference'] = list_of_diff[7]
frame_m30['Difference'] = list_of_diff[8]
frame_m40['Difference'] = list_of_diff[9]
frame_m50['Difference'] = list_of_diff[10]
frame_m60['Difference'] = list_of_diff[11]
frame_m70['Difference'] = list_of_diff[12]
frame_m80['Difference'] = list_of_diff[13]
#%% 
# Creating a column for the acceleration (a, d, 0, np.nan)
for file in files:
    conditions = [
        (file['Difference'] > 0),
        (file['Difference'] < 0), 
        (file['Difference'] == 0),
        (file['Difference'] == np.nan)
        ]
    
    values = ['d', 'a', 0, np.nan]
    
    file['Acceleration'] = np.select(conditions, values)
#%%
# Asking the user for the chunk_size for male and female visualisations of the signal difference
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 100 looks nice): '))
for file in files:
    diff = file['Difference']
    chunked_list_user_input = [diff[i:i+chunk_size_user_input] for i in range(0, len(diff), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {file.name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ file.name + '_.jpg'), dpi=300 ) 
    plt.show()
#%%
# Calculating p(a),p(d),p(0) for each file
# Estimated time of code compiling: 30s
a = []
d = []
l_0 = []

for file in files:
    ones = []
    for i in range(0, len(file['Acceleration'])-2):
        ones.append(str(file['Acceleration'].iloc[i]))
    
    count1 = Counter(ones)
    # print(count1)
    number1 = len(file['Interval'])
    a.append(round(count1['a']/number1*100,3))
    d.append(round(count1['d']/number1*100,3))
    l_0.append(round(count1['0']/number1*100,3))

#%%
# Calculating the probability for 2 elements
# estimated time of code compiling: 1 minute
ad = []
da = []
dd = []
aa = []
l_0a = []
d0 = []
a0 = []
l_0d = []
l_00 = []

for file in files:
    twos = []
    for i in range(0, len(file['Acceleration'])-2):
        twos.append(str(file['Acceleration'].iloc[i]) + str(file['Acceleration'].iloc[i+1]))
    
    count2 = Counter(twos)                                                                                                                             
    # print(count2)
    
    number2 = len(file['Interval'])
    
    ad.append(round(count2['ad']/number2*100,3))
    da.append(round(count2['da']/number2*100,3))
    dd.append(round(count2['dd']/number2*100,3))
    aa.append(round(count2['aa']/number2*100,3))
    l_0a.append(round(count2['0a']/number2*100,3))
    d0.append(round(count2['d0']/number2*100,3))
    a0.append(round(count2['a0']/number2*100,3))
    l_0d.append(round(count2['0d']/number2*100,3))
    l_00.append(round(count2['00']/number2*100,3))
#%%
# Calculating the probability for three elements
# estimated time of code compiling: 3 minutes
add = []
dda = []
aad = []
daa = []
dad = []
ada = []
ddd = []
aaa = []
d0a = []
ad0 = []
l_0ad = []
a0d = []
l_0aa = []
dd0 = []
da0 = []
l_0da = []
aa0 = []
l_0dd = []
a0a = []
d0d = []
l_00a = []
l_0a0 = []
d00 = []
a00 = []
l_0d0 = []
l_00d = []
l_000 = []

for file in files:
    threes = []
    for i in range(0, len(file['Acceleration'])-2):
        threes.append(str(file['Acceleration'].iloc[i]) + str(file['Acceleration'].iloc[i+1])+str(file['Acceleration'].iloc[i+2]))
    count3 = Counter(threes)                                                                                                                            
    # print(count3)
    
    number3 = len(file['Interval'])
    
    add.append(round(count3['add']/number3*100,3))
    dda.append(round(count3['dda']/number3*100,3))
    aad.append(round(count3['aad']/number3*100,3))
    daa.append(round(count3['daa']/number3*100,3))
    dad.append(round(count3['dad']/number3*100,3))
    ada.append(round(count3['ada']/number3*100,3))
    ddd.append(round(count3['ddd']/number3*100,3))
    aaa.append(round(count3['aaa']/number3*100,3))
    d0a.append(round(count3['d0d']/number3*100,3))
    ad0.append(round(count3['ad0']/number3*100,3))
    l_0ad.append(round(count3['0ad']/number3*100,3))
    a0d.append(round(count3['a0d']/number3*100,3))
    l_0aa.append(round(count3['0aa']/number3*100,3))
    dd0.append(round(count3['dd0']/number3*100,3))
    da0.append(round(count3['da0']/number3*100,3))
    l_0da.append(round(count3['0da']/number3*100,3))
    aa0.append(round(count3['aa0']/number3*100,3))
    l_0dd.append(round(count3['0dd']/number3*100,3))
    a0a.append(round(count3['a0a']/number3*100,3))
    d0d.append(round(count3['d0d']/number3*100,3))
    l_00a.append(round(count3['00a']/number3*100,3))
    l_0a0.append(round(count3['0a0']/number3*100,3))
    d00.append(round(count3['d00']/number3*100,3))
    a00.append(round(count3['a00']/number3*100,3))
    l_0d0.append(round(count3['0d0']/number3*100,3))
    l_00d.append(round(count3['00d']/number3*100,3))
    l_000.append(round(count3['000']/number3*100,3))
#%%
# Creating the final Dataframe with calculated values for each file
file_names = ["frame_f20","frame_f30","frame_f40","frame_f50","frame_f60","frame_f70","frame_f80","frame_m20","frame_m30","frame_m40","frame_m50","frame_m60","frame_m70","frame_m80"]
col_names = ["File name", "Gender", "Most common value", "Minimum", "Maximum", "Mean", "STD","SDNN","RMSSD","pNN50", "pNN20", "Minimum average interval", "The STD for the minimum average interval",
             "p(a)", "p(d)","p(0)",
             "p(ad)", "p(da)", "p(dd)","p(aa)", "p(0a)", "p(d0)","p(a0)", "p(0d)", "p(00)",
             "p(add)",
             "p(dda)",
             "p(aad)",
             "p(daa)",
             "p(dad)",
             "p(ada)",
             "p(ddd)",
             "p(aaa)",
             "p(d0a)",
             "p(ad0)",
             "p(0ad)",
             "p(a0d)",
             "p(0aa)",
             "p(dd0)",
             "p(da0)",
             "p(0da)",
             "p(aa0)",
             "p(0dd)",
             "p(a0a)",
             "p(d0d)",
             "p(00a)",
             "p(0a0)",
             "p(d00)",
             "p(a00)",
             "p(0d0)",
             "p(00d)",
             "p(000)"
             ]
gender = ["f","f","f","f","f","f","f","m","m","m","m","m","m","m"]
df_final = pd.DataFrame(list(zip(file_names,gender,mcv,minimums,maximums,means,stds,sdnn,rmssd,pnn50,pnn20,avg_list,std_list,
                                a, d, l_0,
                                ad, da, dd, aa, l_0a, d0, a0, l_0d, l_00,
                                add,
                                dda,
                                aad,
                                daa,
                                dad,
                                ada,
                                ddd,
                                aaa,
                                d0a,
                                ad0,
                                l_0ad,
                                a0d,
                                l_0aa,
                                dd0,
                                da0,
                                l_0da,
                                aa0,
                                l_0dd,
                                a0a,
                                d0d,
                                l_00a,
                                l_0a0,
                                d00,
                                a00,
                                l_0d0,
                                l_00d,
                                l_000)),columns = col_names)

#%%
# Creating female and male DataFrames for visualisation

df_final_females = df_final[df_final["Gender"] == "f"]
df_final_males = df_final[df_final["Gender"] == "m"]
df_final_females.plot(y="SDNN")
df_final_males.plot(y="SDNN")

#%% 
# Visualising the female and male SDNN
plt.plot(df_final_females["SDNN"], color = "g", label = "female")
plt.xticks([])
plt.legend(title = "Analysis of SDNN for females")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'female_sdnn_.jpg'), dpi=300 )
plt.show()
plt.plot(df_final_males["SDNN"], color = "b", label = "male")
plt.xticks([])
plt.legend(title = "Analysis of SDNN for males")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'male_sdnn_.jpg'), dpi=300 )
plt.show()
#%% 
# Visualising the female and male pNN50
plt.plot(df_final_females["pNN50"], color = "g", label = "female")
plt.xticks([])
plt.legend(title = "Analysis of pNN50 for females")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'female_pNN50_.jpg'), dpi=300 )
plt.show()
plt.plot(df_final_males["pNN50"], color = "b", label = "male")
plt.xticks([])
plt.legend(title = "Analysis of pNN50 for males")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'male_pNN50_.jpg'), dpi=300 )
plt.show()
#%% 
# Visualising the female and male pNN20
plt.plot(df_final_females["pNN20"], color = "g", label = "female")
plt.xticks([])
plt.legend(title = "Analysis of pNN20 for females")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'female_pNN20_.jpg'), dpi=300 )
plt.show()
plt.plot(df_final_males["pNN20"], color = "b", label = "male")
plt.xticks([])
plt.legend(title = "Analysis of pNN20 for males")
plt.savefig(os.path.join(KATALOG_WYKRESOW,'male_pNN20_.jpg'), dpi=300 )
plt.show()
#%%
# The Poincare plot for each file
for file in files:
    x = file['Interval'][:-1]
    y = file['Interval'][1:]
    y.reset_index(drop = True, inplace = True)
    plt.scatter(x, y, marker = '*')
    plt.plot(range(min(file['Interval']), max(file['Interval'])), range(min(file['Interval']), max(file['Interval'])), linestyle = '-', color = "red", label = "RR(i) = RR(i+1)")
    plt.legend(title = f"The Poincare plot for {file.name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'poincare_plot_'+ file.name + '_.jpg'), dpi=300)
    plt.show()

#%% Generating a list of each file
l_all = [l_f20,l_f30,l_f40,l_f50,l_f60,l_f70,l_f80, l_m20,l_m30,l_m40,l_m50,l_m60,l_m70,l_m80]

l_all_final = []
for i in range(len(l_all)):
    for j in range(len(l_all[i])):
        l_all_final.append(l_all[i][j])

#%% Creating a list of file names for the final df
file_list = os.listdir(path)
print(file_list)
#%% # Searching for the most common value of interval for each file
mcv2 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    a = file['Interval'].value_counts().index[0]
    mcv2.append(a)
    print(f"The most frequent value for {name} is: ", file['Interval'].value_counts().nlargest(1))
    print()
    i += 1
print(f'The vector of the most common values {mcv2}')
#%% Searching for the minimum and maximum values for each file
minimums2 = []
maximums2 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    interval = file['Interval']
    mins = min(interval)
    maxs = max(interval)
    print(f'The minimum value for {name} is: {mins}')
    print(f'and the maximum value for {name} is: {maxs}')
    minimums2.append(mins)
    maximums2.append(maxs)
    print()
    i += 1

print(f'Vector of minimums {minimums2} and of maximums {maximums2}')
#%% Calculating the mean value and the std for each file
means2 = []
stds2 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    interval = file['Interval']
    m = np.mean(interval)
    s = np.std(interval)
    print(f'The mean value for {name} is: {m}')
    print(f'The std of the interval for {name} is: {s}')  
    means2.append(m)
    stds2.append(s)
    print()
    
print(f'The vector of mean values {means2}')
print()
print(f'and of the std {stds2}')
#%% Histograms for each file 
i = 0
for file in l_all_final:
    name = file_list[i]
    file['Interval'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title(f"Histogram of values for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ name + '_.jpg'), dpi=300 ) 
    plt.show() # sprawdzić jak się plik zapisal
    i += 1
#%% Visualisations for each file
i = 0
for file in l_all_final:
    name = file_list[i]
    file["Interval"].plot(title='Interval ' + name)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+ name+ '_healthy.jpg'), dpi=300 ) 
    plt.show()
    i += 1
#%%  Calculating the SDNN for each file
sdnn2 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    NN = file['Interval'].diff()
    SDNN = np.std(NN)
    print(f'The SDNN for {name} is: {SDNN}')
    sdnn2.append(SDNN)
    print()
    i += 1
    
print(f'The vector of SDNN {sdnn2}')

#%% Calculating RMSSD for each file
rmssd2 = []

i = 0
for file in l_all_final:
    name = file_list[i]
    diff = file['Interval'].diff()
    file['diff'] = check_if_na(diff)
    file['abs_diff'] = abs(file['diff'])
    RMSSD = (sum(file['diff'].dropna().pow(2))/len(file['Interval'].dropna()))**(1/2)
    print(f'The RMSSD for {name}: {RMSSD}')
    rmssd2.append(RMSSD)
    print()
    i += 1
print(f'The vector of RMSSD {rmssd2}')
#%% Calculating the pNN50 and pNN20
pnn502 = []
pnn202 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    NN=np.diff(file['Interval'])
    NN50 = sum(i > 50 for i in NN)
    PNN50=NN50/len(NN)
    NN20 = sum(i > 20 for i in NN)
    PNN20=NN20/len(NN)
    print(f'The PNN50 for {name}: {PNN50}')
    print(f'The PNN20 for {name}: {PNN20}')
    print()
    pnn502.append(PNN50)
    pnn202.append(PNN20)
    i += 1
    
print(f'The vector of PNN50: {pnn502}')
print()
print(f'The vector of PNN20: {pnn202}')
#%% 
# Stacjonarnosc sygnału, analiza sygnału
# Searching for the minimum average for each range in each file and calculating the minimum std for it

avg_list2 = []
std_list2 = []
i = 0
for file in l_all_final:
    name = file_list[i]
    interval = file['Interval']
    chunk_size = 1000
    chunked_list = [interval[i:i+chunk_size] for i in range(0, len(interval), chunk_size)]
    # print(len(chunked_list))
    average = [sum(chunked_list[i])/len(chunked_list[i]) for i in range(len(chunked_list))]
    list_for_std = [chunked_list[average.index(min(average))]]
    std = [(sum(list_for_std[i] - min(average))**2/len(list_for_std))**0.5 for i in range(len(list_for_std))]
    print(f"For {name} the minimum average is {min(average)} and for that range the std is {std}")
    avg_list2.append(min(average))
    std_list2.append(std)
    print()
    i += 1
    
print(f"This is a list of minimum averages {avg_list2}")
print()
print(f"This is a list of std's {std_list2}")
#%%
# Asking the user for the chunk_size for visualisations
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 10 looks nice): '))
i = 0
for file in l_all_final:
    name = file_list[i]
    interval = file['Interval']
    chunked_list_user_input = [interval[i:i+chunk_size_user_input] for i in range(0, len(interval), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ name + '_.jpg'), dpi=300 ) 
    plt.show()
    i += 1

#%% 
# Creating a column for the acceleration (a, d, 0, np.nan)
i = 0
for file in l_all_final:
    name = file_list[i]
    conditions = [
        (file['diff'] > 0),
        (file['diff'] < 0), 
        (file['diff'] == 0),
        (file['diff'] == np.nan)
        ]
    
    values = ['d', 'a', 0, np.nan]
    
    file['Acceleration'] = np.select(conditions, values)
    i += 1
#%%
# Asking the user for the chunk_size for visualisations of the signal difference
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 10 looks nice): '))
i = 0
for file in l_all_final:
    name = file_list[i]
    diff = file['diff']
    chunked_list_user_input = [diff[i:i+chunk_size_user_input] for i in range(0, len(diff), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ name + '_.jpg'), dpi=300 ) 
    plt.show()
    i+=1
#%%
# Calculating p(a),p(d),p(0) for each file
# Estimated time of code compiling: 30s
a2 = []
d2 = []
l_02 = []

i = 0
for file in l_all_final:
    name = file_list[i]
    ones = []
    for j in range(0, len(file['Acceleration'])-2):
        ones.append(str(file['Acceleration'].iloc[j]))
    
    count1 = Counter(ones)
    # print(count1)
    number1 = len(file['Interval'])
    a2.append(round(count1['a']/number1*100,3))
    d2.append(round(count1['d']/number1*100,3))
    l_02.append(round(count1['0']/number1*100,3))
    i += 1

#%%
# Calculating the probability for 2 elements
# estimated time of code compiling: 1 minute
ad2 = []
da2 = []
dd2 = []
aa2 = []
l_0a2 = []
d02 = []
a02 = []
l_0d2 = []
l_002 = []

i = 0
for file in l_all_final:
    name = file_list[i]
    twos = []
    for j in range(0, len(file['Acceleration'])-2):
        twos.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1]))
    
    count2 = Counter(twos)                                                                                                                             
    # print(count2)
    
    number2 = len(file['Interval'])
    
    ad2.append(round(count2['ad']/number2*100,3))
    da2.append(round(count2['da']/number2*100,3))
    dd2.append(round(count2['dd']/number2*100,3))
    aa2.append(round(count2['aa']/number2*100,3))
    l_0a2.append(round(count2['0a']/number2*100,3))
    d02.append(round(count2['d0']/number2*100,3))
    a02.append(round(count2['a0']/number2*100,3))
    l_0d2.append(round(count2['0d']/number2*100,3))
    l_002.append(round(count2['00']/number2*100,3))
    i += 1
#%%
# Calculating the probability for three elements
# estimated time of code compiling: 3 minutes
add2 = []
dda2 = []
aad2 = []
daa2 = []
dad2 = []
ada2 = []
ddd2 = []
aaa2 = []
d0a2 = []
ad02 = []
l_0ad2 = []
a0d2 = []
l_0aa2 = []
dd02 = []
da02 = []
l_0da2 = []
aa02 = []
l_0dd2 = []
a0a2 = []
d0d2 = []
l_00a2 = []
l_0a02 = []
d002 = []
a002 = []
l_0d02 = []
l_00d2 = []
l_0002 = []

i = 0
for file in l_all_final:
    name = file_list[i]
    threes = []
    for j in range(0, len(file['Acceleration'])-2):
        threes.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1])+str(file['Acceleration'].iloc[j+2]))
    count3 = Counter(threes)                                                                                                                            
    # print(count3)
    
    number3 = len(file['Interval'])
    
    add2.append(round(count3['add']/number3*100,3))
    dda2.append(round(count3['dda']/number3*100,3))
    aad2.append(round(count3['aad']/number3*100,3))
    daa2.append(round(count3['daa']/number3*100,3))
    dad2.append(round(count3['dad']/number3*100,3))
    ada2.append(round(count3['ada']/number3*100,3))
    ddd2.append(round(count3['ddd']/number3*100,3))
    aaa2.append(round(count3['aaa']/number3*100,3))
    d0a2.append(round(count3['d0d']/number3*100,3))
    ad02.append(round(count3['ad0']/number3*100,3))
    l_0ad2.append(round(count3['0ad']/number3*100,3))
    a0d2.append(round(count3['a0d']/number3*100,3))
    l_0aa2.append(round(count3['0aa']/number3*100,3))
    dd02.append(round(count3['dd0']/number3*100,3))
    da02.append(round(count3['da0']/number3*100,3))
    l_0da2.append(round(count3['0da']/number3*100,3))
    aa02.append(round(count3['aa0']/number3*100,3))
    l_0dd2.append(round(count3['0dd']/number3*100,3))
    a0a2.append(round(count3['a0a']/number3*100,3))
    d0d2.append(round(count3['d0d']/number3*100,3))
    l_00a2.append(round(count3['00a']/number3*100,3))
    l_0a02.append(round(count3['0a0']/number3*100,3))
    d002.append(round(count3['d00']/number3*100,3))
    a002.append(round(count3['a00']/number3*100,3))
    l_0d02.append(round(count3['0d0']/number3*100,3))
    l_00d2.append(round(count3['00d']/number3*100,3))
    l_0002.append(round(count3['000']/number3*100,3))
    i += 1
#%%
# Creating the final Dataframe with calculated values for each file
col_names = ["File name", "Gender", "Most common value", "Minimum", "Maximum", "Mean", "STD","SDNN","RMSSD","pNN50", "pNN20", "Minimum average interval", "The STD for the minimum average interval",
             "p(a)", "p(d)","p(0)",
             "p(ad)", "p(da)", "p(dd)","p(aa)", "p(0a)", "p(d0)","p(a0)", "p(0d)", "p(00)",
             "p(add)",
             "p(dda)",
             "p(aad)",
             "p(daa)",
             "p(dad)",
             "p(ada)",
             "p(ddd)",
             "p(aaa)",
             "p(d0a)",
             "p(ad0)",
             "p(0ad)",
             "p(a0d)",
             "p(0aa)",
             "p(dd0)",
             "p(da0)",
             "p(0da)",
             "p(aa0)",
             "p(0dd)",
             "p(a0a)",
             "p(d0d)",
             "p(00a)",
             "p(0a0)",
             "p(d00)",
             "p(a00)",
             "p(0d0)",
             "p(00d)",
             "p(000)"
             ]
#%% Creating a list of genders
gender2 = []

for i in range(len(file_list)):
    g = file_list[i][0]
    gender2.append(g)
#%%    
df_file_final = pd.DataFrame(list(zip(file_list,gender2,mcv2,minimums2,maximums2,means2,stds2,sdnn2,rmssd2,pnn502,pnn202,avg_list2,std_list2,
                                a2, d2, l_02,
                                ad2, da2, dd2, aa2, l_0a2, d02, a02, l_0d2, l_002,
                                add2,
                                dda2,
                                aad2,
                                daa2,
                                dad2,
                                ada2,
                                ddd2,
                                aaa2,
                                d0a2,
                                ad02,
                                l_0ad2,
                                a0d2,
                                l_0aa2,
                                dd02,
                                da02,
                                l_0da2,
                                aa02,
                                l_0dd2,
                                a0a2,
                                d0d2,
                                l_00a2,
                                l_0a02,
                                d002,
                                a002,
                                l_0d02,
                                l_00d2,
                                l_0002)),columns = col_names)

#%% Correcting the data in one column to a float
split = pd.DataFrame(df_file_final['The STD for the minimum average interval'].to_list(), columns = ['The STD for the minimum average interval'])
df_file_final['The STD for the minimum average interval'] = split
#%%
# The Poincare plot for each file
i = 0
for file in l_all_final:
    name = file_list[i]
    x = file['Interval'][:-1]
    y = file['Interval'][1:]
    y.reset_index(drop = True, inplace = True)
    plt.scatter(x, y, marker = '*')
    plt.plot(range(min(file['Interval']), max(file['Interval'])), range(min(file['Interval']), max(file['Interval'])), linestyle = '-', color = "red", label = "RR(i) = RR(i+1)")
    plt.legend(title = f"The Poincare plot for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'poincare_plot_'+ name + '_.jpg'), dpi=300)
    plt.show()
    i += 1

#%%
l = l_all_final.copy()
chunk_size = 100
i = 0
l_of_max_mean_rr = []
l_of_max_std_rr = []
for file in l:
    # name = file_list[i]
    interval = file['Interval']
    chunked_list = [interval[i:i+chunk_size] for i in range(0, len(interval), chunk_size)]
    # print(len(chunked_list))
    list_of_std = []
    list_of_mean = []
    for k in chunked_list:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))
    # print(max(list_of_std))
    # print()
#%% Finding the range with the highest mean RR and std RR for each file

l_of_max_mean_rr = []  # lists of ranges from each file
l_of_max_std_rr = []

ind_of_max_std_range = [] # lists of indexes for each range
ind_of_max_mean_range = []
i = 0
chunk_size = 100
for file in l:
    interval = file['Interval']
    chunked_list = [interval[i:i+chunk_size] for i in range(0, len(interval), chunk_size)]
    # print(len(chunked_list))
    list_of_std = []
    list_of_mean = []
    for k in chunked_list:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))
    
    ind1 = list_of_std.index(max(list_of_std))
    ind_of_max_std_range.append(list_of_std.index(max(list_of_std)))
    l_of_max_std_rr.append(chunked_list[ind1])
    
    ind2 = list_of_mean.index(max(list_of_mean))
    ind_of_max_mean_range.append(list_of_mean.index(max(list_of_mean)))
    l_of_max_mean_rr.append(chunked_list[ind2])
#%% Changing the data to a dataframe
l_of_df_mean_rr = []
for file in l_of_max_mean_rr:
    df1 = pd.DataFrame(file, columns = ['Interval'])
    l_of_df_mean_rr.append(df1)

print(l_of_df_mean_rr)

l_of_df_std_rr = []
for file in l_of_max_std_rr:
    df1 = pd.DataFrame(file, columns = ['Interval'])
    l_of_df_std_rr.append(df1)

print(l_of_df_std_rr)
#%% # Searching for the most common value of interval for each file range
mcv2 = []
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    a = file['Interval'].value_counts().index[0]
    mcv2.append(a)
    print(f"The most frequent value for {name} is: ", file['Interval'].value_counts().nlargest(1))
    print()
    i += 1
print(f'The vector of the most common values {mcv2}')
#%% Searching for the minimum and maximum values for each file range
minimums2 = []
maximums2 = []
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    interval = file['Interval']
    mins = min(interval)
    maxs = max(interval)
    print(f'The minimum value for {name} is: {mins}')
    print(f'and the maximum value for {name} is: {maxs}')
    minimums2.append(mins)
    maximums2.append(maxs)
    print()
    i += 1

print(f'Vector of minimums {minimums2} and of maximums {maximums2}')
#%% Calculating the mean value and the std for each file
means2 = []
stds2 = []
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    interval = file['Interval']
    m = np.mean(interval)
    s = np.std(interval)
    print(f'The mean value for {name} is: {m}')
    print(f'The std of the interval for {name} is: {s}')  
    means2.append(m)
    stds2.append(s)
    print()
    
print(f'The vector of mean values {means2}')
print()
print(f'and of the std {stds2}')
#%% Histograms for each file 
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]

    file['Interval'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title(f"Histogram of values for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ name + '_.jpg'), dpi=300 ) 
    plt.show() # sprawdzić jak się plik zapisal
    i += 1
#%% Visualisations for each file
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    file["Interval"].plot(title='Interval ' + name)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+ name+ '_healthy.jpg'), dpi=300 ) 
    plt.show()
    i += 1
#%%  Calculating the SDNN for each file
sdnn2 = []
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    NN = file['Interval'].diff()
    SDNN = np.std(NN)
    print(f'The SDNN for {name} is: {SDNN}')
    sdnn2.append(SDNN)
    print()
    i += 1
    
print(f'The vector of SDNN {sdnn2}')
#%% Calculating RMSSD for each file
rmssd2 = []

i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    diff = file['Interval'].diff()
    file['diff'] = check_if_na(diff)
    file['abs_diff'] = abs(file['diff'])
    RMSSD = (sum(file['diff'].dropna().pow(2))/len(file['Interval'].dropna()))**(1/2)
    print(f'The RMSSD for {name}: {RMSSD}')
    rmssd2.append(RMSSD)
    print()
    i += 1
print(f'The vector of RMSSD {rmssd2}')
#%% Calculating the pNN50 and pNN20
pnn502 = []
pnn202 = []
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    NN=np.diff(file['Interval'])
    NN50 = sum(i > 50 for i in NN)
    PNN50=NN50/len(NN)
    NN20 = sum(i > 20 for i in NN)
    PNN20=NN20/len(NN)
    print(f'The PNN50 for {name}: {PNN50}')
    print(f'The PNN20 for {name}: {PNN20}')
    print()
    pnn502.append(PNN50)
    pnn202.append(PNN20)
    i += 1
    
print(f'The vector of PNN50: {pnn502}')
print()
print(f'The vector of PNN20: {pnn202}')

#%% 
# Creating a column for the acceleration (a, d, 0, np.nan)
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    conditions = [
        (file['diff'] > 0),
        (file['diff'] < 0), 
        (file['diff'] == 0),
        (file['diff'] == np.nan)
        ]
    
    values = ['d', 'a', 0, np.nan]
    
    file['Acceleration'] = np.select(conditions, values)
    i += 1
#%%
# Asking the user for the chunk_size for visualisations of the signal difference
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 10 looks nice): '))
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    diff = file['diff']
    chunked_list_user_input = [diff[i:i+chunk_size_user_input] for i in range(0, len(diff), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ name + '_.jpg'), dpi=300 ) 
    plt.show()
    i+=1
#%%
# Calculating p(a),p(d),p(0) for each file

a2 = []
d2 = []
l_02 = []

i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    ones = []
    for j in range(0, len(file['Acceleration'])-2):
        ones.append(str(file['Acceleration'].iloc[j]))
    
    count1 = Counter(ones)
    # print(count1)
    number1 = len(file['Interval'])
    a2.append(round(count1['a']/number1,3))
    d2.append(round(count1['d']/number1,3))
    l_02.append(round(count1['0']/number1,3))
    i += 1

#%%
# Calculating the probability for 2 elements

ad2 = []
da2 = []
dd2 = []
aa2 = []
l_0a2 = []
d02 = []
a02 = []
l_0d2 = []
l_002 = []

i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    twos = []
    for j in range(0, len(file['Acceleration'])-2):
        twos.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1]))
    
    count2 = Counter(twos)                                                                                                                             
    # print(count2)
    
    number2 = len(file['Interval'])
    
    ad2.append(round(count2['ad']/number2,3))
    da2.append(round(count2['da']/number2,3))
    dd2.append(round(count2['dd']/number2,3))
    aa2.append(round(count2['aa']/number2,3))
    l_0a2.append(round(count2['0a']/number2,3))
    d02.append(round(count2['d0']/number2,3))
    a02.append(round(count2['a0']/number2,3))
    l_0d2.append(round(count2['0d']/number2,3))
    l_002.append(round(count2['00']/number2,3))
    i += 1

#%%
# Calculating the probability for three elements

add2 = []
dda2 = []
aad2 = []
daa2 = []
dad2 = []
ada2 = []
ddd2 = []
aaa2 = []
d0a2 = []
ad02 = []
l_0ad2 = []
a0d2 = []
l_0aa2 = []
dd02 = []
da02 = []
l_0da2 = []
aa02 = []
l_0dd2 = []
a0a2 = []
d0d2 = []
l_00a2 = []
l_0a02 = []
d002 = []
a002 = []
l_0d02 = []
l_00d2 = []
l_0002 = []

i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    threes = []
    for j in range(0, len(file['Acceleration'])-2):
        threes.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1])+str(file['Acceleration'].iloc[j+2]))
    count3 = Counter(threes)                                                                                                                            
    # print(count3)
    
    number3 = len(file['Interval'])
    
    add2.append(round(count3['add']/number3,3))
    dda2.append(round(count3['dda']/number3,3))
    aad2.append(round(count3['aad']/number3,3))
    daa2.append(round(count3['daa']/number3,3))
    dad2.append(round(count3['dad']/number3,3))
    ada2.append(round(count3['ada']/number3,3))
    ddd2.append(round(count3['ddd']/number3,3))
    aaa2.append(round(count3['aaa']/number3,3))
    d0a2.append(round(count3['d0d']/number3,3))
    ad02.append(round(count3['ad0']/number3,3))
    l_0ad2.append(round(count3['0ad']/number3,3))
    a0d2.append(round(count3['a0d']/number3,3))
    l_0aa2.append(round(count3['0aa']/number3,3))
    dd02.append(round(count3['dd0']/number3,3))
    da02.append(round(count3['da0']/number3,3))
    l_0da2.append(round(count3['0da']/number3,3))
    aa02.append(round(count3['aa0']/number3,3))
    l_0dd2.append(round(count3['0dd']/number3,3))
    a0a2.append(round(count3['a0a']/number3,3))
    d0d2.append(round(count3['d0d']/number3,3))
    l_00a2.append(round(count3['00a']/number3,3))
    l_0a02.append(round(count3['0a0']/number3,3))
    d002.append(round(count3['d00']/number3,3))
    a002.append(round(count3['a00']/number3,3))
    l_0d02.append(round(count3['0d0']/number3,3))
    l_00d2.append(round(count3['00d']/number3,3))
    l_0002.append(round(count3['000']/number3,3))
    i += 1

#%% Preparing the columns for project 2

age = []
for i in range(len(file_list)):
    a = file_list[i][1:3]
    age.append(a)

#%%
# Creating the final Dataframe with calculated values for each file
col_names = ["File name", "Age", "Gender", "Most common value", "Minimum", "Maximum", "Mean", "STD","SDNN","RMSSD","pNN50", "pNN20",
             "p(a)", "p(d)","p(0)",
             "p(ad)", "p(da)", "p(dd)","p(aa)", "p(0a)", "p(d0)","p(a0)", "p(0d)", "p(00)",
             "p(add)",
             "p(dda)",
             "p(aad)",
             "p(daa)",
             "p(dad)",
             "p(ada)",
             "p(ddd)",
             "p(aaa)",
             "p(d0a)",
             "p(ad0)",
             "p(0ad)",
             "p(a0d)",
             "p(0aa)",
             "p(dd0)",
             "p(da0)",
             "p(0da)",
             "p(aa0)",
             "p(0dd)",
             "p(a0a)",
             "p(d0d)",
             "p(00a)",
             "p(0a0)",
             "p(d00)",
             "p(a00)",
             "p(0d0)",
             "p(00d)",
             "p(000)"
             ]

#%% Creating a list of genders
gender2 = []

for i in range(len(file_list)):
    g = file_list[i][0]
    gender2.append(g)

#%%    
df_mean_range_final = pd.DataFrame(list(zip(file_list,age,gender2,mcv2,minimums2,maximums2,means2,stds2,sdnn2,rmssd2,pnn502,pnn202,
                                a2, d2, l_02,
                                ad2, da2, dd2, aa2, l_0a2, d02, a02, l_0d2, l_002,
                                add2,
                                dda2,
                                aad2,
                                daa2,
                                dad2,
                                ada2,
                                ddd2,
                                aaa2,
                                d0a2,
                                ad02,
                                l_0ad2,
                                a0d2,
                                l_0aa2,
                                dd02,
                                da02,
                                l_0da2,
                                aa02,
                                l_0dd2,
                                a0a2,
                                d0d2,
                                l_00a2,
                                l_0a02,
                                d002,
                                a002,
                                l_0d02,
                                l_00d2,
                                l_0002)),columns = col_names)


#%%
# The Poincare plot for each file
i = 0
for file in l_of_df_mean_rr:
    name = file_list[i]
    x = file['Interval'][:-1]
    y = file['Interval'][1:]
    y.reset_index(drop = True, inplace = True)
    plt.scatter(x, y, marker = '*')
    plt.plot(range(min(file['Interval']), max(file['Interval'])), range(min(file['Interval']), max(file['Interval'])), linestyle = '-', color = "red", label = "RR(i) = RR(i+1)")
    plt.legend(title = f"The Poincare plot for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'poincare_plot_'+ name + '_.jpg'), dpi=300)
    plt.show()
    i += 1
#%% Doing the same things for l_of_df_std_rr
#%% # Searching for the most common value of interval for each file range
mcv2 = []
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    a = file['Interval'].value_counts().index[0]
    mcv2.append(a)
    print(f"The most frequent value for {name} is: ", file['Interval'].value_counts().nlargest(1))
    print()
    i += 1
print(f'The vector of the most common values {mcv2}')

#%% Searching for the minimum and maximum values for each file range
minimums2 = []
maximums2 = []
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    interval = file['Interval']
    mins = min(interval)
    maxs = max(interval)
    print(f'The minimum value for {name} is: {mins}')
    print(f'and the maximum value for {name} is: {maxs}')
    minimums2.append(mins)
    maximums2.append(maxs)
    print()
    i += 1

print(f'Vector of minimums {minimums2} and of maximums {maximums2}')
#%% Calculating the mean value and the std for each file
means2 = []
stds2 = []
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    interval = file['Interval']
    m = np.mean(interval)
    s = np.std(interval)
    print(f'The mean value for {name} is: {m}')
    print(f'The std of the interval for {name} is: {s}')  
    means2.append(m)
    stds2.append(s)
    print()
    
print(f'The vector of mean values {means2}')
print()
print(f'and of the std {stds2}')
#%% Histograms for each file 
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]

    file['Interval'].hist(bins=50, figsize=(9,6))
    plt.tight_layout()
    plt.title(f"Histogram of values for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogram_'+ name + '_.jpg'), dpi=300 ) 
    plt.show() # sprawdzić jak się plik zapisal
    i += 1
#%% Visualisations for each file
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    file["Interval"].plot(title='Interval ' + name)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval_'+ name+ '_healthy.jpg'), dpi=300 ) 
    plt.show()
    i += 1
#%%  Calculating the SDNN for each file
sdnn2 = []
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    NN = file['Interval'].diff()
    SDNN = np.std(NN)
    print(f'The SDNN for {name} is: {SDNN}')
    sdnn2.append(SDNN)
    print()
    i += 1
    
print(f'The vector of SDNN {sdnn2}')
#%% Calculating RMSSD for each file
rmssd2 = []

i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    diff = file['Interval'].diff()
    file['diff'] = check_if_na(diff)
    file['abs_diff'] = abs(file['diff'])
    RMSSD = (sum(file['diff'].dropna().pow(2))/len(file['Interval'].dropna()))**(1/2)
    print(f'The RMSSD for {name}: {RMSSD}')
    rmssd2.append(RMSSD)
    print()
    i += 1
print(f'The vector of RMSSD {rmssd2}')
#%% Calculating the pNN50 and pNN20
pnn502 = []
pnn202 = []
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    NN=np.diff(file['Interval'])
    NN50 = sum(i > 50 for i in NN)
    PNN50=NN50/len(NN)
    NN20 = sum(i > 20 for i in NN)
    PNN20=NN20/len(NN)
    print(f'The PNN50 for {name}: {PNN50}')
    print(f'The PNN20 for {name}: {PNN20}')
    print()
    pnn502.append(PNN50)
    pnn202.append(PNN20)
    i += 1
    
print(f'The vector of PNN50: {pnn502}')
print()
print(f'The vector of PNN20: {pnn202}')

#%% 
# Creating a column for the acceleration (a, d, 0, np.nan)
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    conditions = [
        (file['diff'] > 0),
        (file['diff'] < 0), 
        (file['diff'] == 0),
        (file['diff'] == np.nan)
        ]
    
    values = ['d', 'a', 0, np.nan]
    
    file['Acceleration'] = np.select(conditions, values)
    i += 1
#%%
# Asking the user for the chunk_size for visualisations of the signal difference
chunk_size_user_input = int(input('Choose a number for the chunk size(suggestion: 10 looks nice): '))
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    diff = file['diff']
    chunked_list_user_input = [diff[i:i+chunk_size_user_input] for i in range(0, len(diff), chunk_size_user_input)]
    list_of_std = []
    list_of_mean = []
    for k in chunked_list_user_input:
        list_of_std.append(np.std(k))
        list_of_mean.append(np.mean(k))

    plt.plot(list_of_std, color = "g", label = "STD")
    plt.plot(list_of_mean, color = "r", label = "MEAN")
    plt.legend(title = f"Analysis in windows of {chunk_size_user_input} for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'std_and_mean_'+ name + '_.jpg'), dpi=300 ) 
    plt.show()
    i+=1
#%%
# Calculating p(a),p(d),p(0) for each file

a2 = []
d2 = []
l_02 = []

i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    ones = []
    for j in range(0, len(file['Acceleration'])-2):
        ones.append(str(file['Acceleration'].iloc[j]))
    
    count1 = Counter(ones)
    # print(count1)
    number1 = len(file['Interval'])
    a2.append(round(count1['a']/number1,3))
    d2.append(round(count1['d']/number1,3))
    l_02.append(round(count1['0']/number1,3))
    i += 1

#%%
# Calculating the probability for 2 elements

ad2 = []
da2 = []
dd2 = []
aa2 = []
l_0a2 = []
d02 = []
a02 = []
l_0d2 = []
l_002 = []

i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    twos = []
    for j in range(0, len(file['Acceleration'])-2):
        twos.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1]))
    
    count2 = Counter(twos)                                                                                                                             
    # print(count2)
    
    number2 = len(file['Interval'])
    
    ad2.append(round(count2['ad']/number2,3))
    da2.append(round(count2['da']/number2,3))
    dd2.append(round(count2['dd']/number2,3))
    aa2.append(round(count2['aa']/number2,3))
    l_0a2.append(round(count2['0a']/number2,3))
    d02.append(round(count2['d0']/number2,3))
    a02.append(round(count2['a0']/number2,3))
    l_0d2.append(round(count2['0d']/number2,3))
    l_002.append(round(count2['00']/number2,3))
    i += 1
#%%
# Calculating the probability for three elements

add2 = []
dda2 = []
aad2 = []
daa2 = []
dad2 = []
ada2 = []
ddd2 = []
aaa2 = []
d0a2 = []
ad02 = []
l_0ad2 = []
a0d2 = []
l_0aa2 = []
dd02 = []
da02 = []
l_0da2 = []
aa02 = []
l_0dd2 = []
a0a2 = []
d0d2 = []
l_00a2 = []
l_0a02 = []
d002 = []
a002 = []
l_0d02 = []
l_00d2 = []
l_0002 = []

i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    threes = []
    for j in range(0, len(file['Acceleration'])-2):
        threes.append(str(file['Acceleration'].iloc[j]) + str(file['Acceleration'].iloc[j+1])+str(file['Acceleration'].iloc[j+2]))
    count3 = Counter(threes)                                                                                                                            
    # print(count3)
    
    number3 = len(file['Interval'])
    
    add2.append(round(count3['add']/number3,3))
    dda2.append(round(count3['dda']/number3,3))
    aad2.append(round(count3['aad']/number3,3))
    daa2.append(round(count3['daa']/number3,3))
    dad2.append(round(count3['dad']/number3,3))
    ada2.append(round(count3['ada']/number3,3))
    ddd2.append(round(count3['ddd']/number3,3))
    aaa2.append(round(count3['aaa']/number3,3))
    d0a2.append(round(count3['d0d']/number3,3))
    ad02.append(round(count3['ad0']/number3,3))
    l_0ad2.append(round(count3['0ad']/number3,3))
    a0d2.append(round(count3['a0d']/number3,3))
    l_0aa2.append(round(count3['0aa']/number3,3))
    dd02.append(round(count3['dd0']/number3,3))
    da02.append(round(count3['da0']/number3,3))
    l_0da2.append(round(count3['0da']/number3,3))
    aa02.append(round(count3['aa0']/number3,3))
    l_0dd2.append(round(count3['0dd']/number3,3))
    a0a2.append(round(count3['a0a']/number3,3))
    d0d2.append(round(count3['d0d']/number3,3))
    l_00a2.append(round(count3['00a']/number3,3))
    l_0a02.append(round(count3['0a0']/number3,3))
    d002.append(round(count3['d00']/number3,3))
    a002.append(round(count3['a00']/number3,3))
    l_0d02.append(round(count3['0d0']/number3,3))
    l_00d2.append(round(count3['00d']/number3,3))
    l_0002.append(round(count3['000']/number3,3))
    i += 1
#%%
# Creating the final Dataframe with calculated values for each file
col_names = ["File name", "Age", "Gender", "Most common value", "Minimum", "Maximum", "Mean", "STD","SDNN","RMSSD","pNN50", "pNN20",
             "p(a)", "p(d)","p(0)",
             "p(ad)", "p(da)", "p(dd)","p(aa)", "p(0a)", "p(d0)","p(a0)", "p(0d)", "p(00)",
             "p(add)",
             "p(dda)",
             "p(aad)",
             "p(daa)",
             "p(dad)",
             "p(ada)",
             "p(ddd)",
             "p(aaa)",
             "p(d0a)",
             "p(ad0)",
             "p(0ad)",
             "p(a0d)",
             "p(0aa)",
             "p(dd0)",
             "p(da0)",
             "p(0da)",
             "p(aa0)",
             "p(0dd)",
             "p(a0a)",
             "p(d0d)",
             "p(00a)",
             "p(0a0)",
             "p(d00)",
             "p(a00)",
             "p(0d0)",
             "p(00d)",
             "p(000)"
             ]
#%% Preparing the columns for project 2

age = []
for i in range(len(file_list)):
    a = file_list[i][1:3]
    age.append(a)

#%%    
df_std_range_final = pd.DataFrame(list(zip(file_list,age,gender2,mcv2,minimums2,maximums2,means2,stds2,sdnn2,rmssd2,pnn502,pnn202,
                                a2, d2, l_02,
                                ad2, da2, dd2, aa2, l_0a2, d02, a02, l_0d2, l_002,
                                add2,
                                dda2,
                                aad2,
                                daa2,
                                dad2,
                                ada2,
                                ddd2,
                                aaa2,
                                d0a2,
                                ad02,
                                l_0ad2,
                                a0d2,
                                l_0aa2,
                                dd02,
                                da02,
                                l_0da2,
                                aa02,
                                l_0dd2,
                                a0a2,
                                d0d2,
                                l_00a2,
                                l_0a02,
                                d002,
                                a002,
                                l_0d02,
                                l_00d2,
                                l_0002)),columns = col_names)

#%%
# The Poincare plot for each file
i = 0
for file in l_of_df_std_rr:
    name = file_list[i]
    x = file['Interval'][:-1]
    y = file['Interval'][1:]
    y.reset_index(drop = True, inplace = True)
    plt.scatter(x, y, marker = '*')
    plt.plot(range(min(file['Interval']), max(file['Interval'])), range(min(file['Interval']), max(file['Interval'])), linestyle = '-', color = "red", label = "RR(i) = RR(i+1)")
    plt.legend(title = f"The Poincare plot for {name}")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'poincare_plot_'+ name + '_.jpg'), dpi=300)
    plt.show()
    i += 1
    
#%% Final dataframes to use in project 2 are: df_std_range_final and df_mean_range_final

df_std_range_final.to_csv(os.path.join(KATALOG_DANYCH, "df_std_range_final"),index = None, header=True)

df_mean_range_final.to_csv(os.path.join(KATALOG_DANYCH, "df_mean_range_final"),index = None, header=True)
#%%
print("The end of project 1 :-)")
print("Author: Maciej Ossowski")