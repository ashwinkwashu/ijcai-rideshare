import pickle
import numpy as np
import os
vehs  = 1000
lamb = 7
nbhood = True
filename_frag = f'{vehs}_l{lamb}_2'
filename = f'{vehs}_l{lamb}_2_nbhood2021-10-20_074240.pkl'
# filename = f'{vehs}_l{lamb}_2_nbhood2021-10-18_202244.pkl'
files = os.listdir()
for file in files:
	if filename_frag in file:
		filename = file
		break
f = pickle.load(open(filename,'rb'))

dd = f['epoch_each_agent_profit']
dts = [0 for i in range(vehs)]

for dt in dd:
	seen = []
	for d, profit in dt:
		dts[d]+=1
		#Assuming each driver gets only one request at a time.
		#The way it is saved, it is actually one entry per request, so it adds up perfectly
		#0.659 service rate, seems correct
def gini_fn(x):
    #calculates the gini index of distribution x
    num = 0
    for i in x:
        for j in x:
            num+=abs(i-j)
    n = len(x)
    avg = sum(x)/n
    den=(2*n*n*avg)
    gini = num/den
    return gini, avg

d_data = {}
d_data['Trips per driver (TPD)'] = dts
d_data['Mean TPD'] = np.mean(dts)
d_data['StD TPD'] = np.std(dts)
d_data['Min TPD'] = np.min(dts)
d_data['Max TPD'] = np.max(dts)
d_data['TPD gini'] = gini_fn(dts)[0]
pickle.dump(d_data, open(f'DriverSummary{vehs}_l{lamb}_nbhood.pkl', 'wb'))
print(np.mean(dts),np.std(dts),np.min(dts),np.max(dts),gini_fn(dts)[0])

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,1)
# fig.set_size_inches(18.5, 2)
# axs.hist(dts, bins =100, range=(80,320))
axs.hist(dts, bins =100, range=(0,320))
# axs.hist(dts, bins =100)
axs.set_ylim([0,150])
plt.title(f'IJCAI(lambda=10^{lamb}), {vehs} vehicles')
plt.show()