import pickle
from copy import deepcopy
filename = '14Lambdas/1000_l11_2_2021-10-29_160732.pkl'
filename = '14/200_l10_2_nbhood2021-10-19_024439.pkl'
filename = '14Edit/50_l10_2_nbhood2021-10-21_032122.pkl'
filename = '14Lambdas/1000_l7_2_2021-10-30_175839.pkl'
filename = '10Lambdas/1000_l0.6666666666666666_3_nbhood2022-11-05_225120.pkl'
filename = '10Lambdas/1000_l0.6666666666666666_8_nbhood2022-11-11_041844.pkl'
filename = '14LambdasPair/1000_l7_2_nbhood2022-11-21_053759.pkl'
filename = '14LambdasPair/1000_l8_2_nbhood2022-11-21_195617.pkl'
epoch_data = pickle.load(open(filename,'rb'))

print(epoch_data['settings'])

ks = {i:key for i,key in enumerate(epoch_data.keys())}

labelfile = '../../data/ny/new_labels.pkl'
if 'nbhood' in filename:
    print("USING")
    labelfile = '../../data/ny/nbhood_labels.pkl'
labels = pickle.load(open(labelfile,'rb'))
label_set = set(labels)
print(label_set)
zsr = {z: [0,0,0] for z in label_set}
for e in epoch_data["epoch_locations_all"]:
    for l in e:
        z = labels[l]
        zsr[z][2] += 1
for e in epoch_data["epoch_locations_accepted"]:
    for l in e:
        z = labels[l]
        zsr[z][1] += 1
for k,v in zsr.items():
    v[0] = v[1]/v[2]

min_zsr = min([v[0] for v in zsr.values()])

total_num_reqs = 296238

total_accepted = epoch_data["total_requests_accepted"]
sr = total_accepted/total_num_reqs
lamb = len(str(epoch_data['settings']['lambda']))-1
print(f'Lambda: 10^{lamb}')
print(f'Overall SR: {sr}\nMinimum SR: {min_zsr}')

def gini(x):
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

gini_src = gini([v[0] for v in zsr.values()])
print(f'Gini source: {gini_src}')

#For interzone gini
def read_req_file():
    requests_file = open('test_flow_5000_11.txt', 'r').readlines()
    
    requests_file = [r.strip('\n') for r in requests_file]
    
    #requests for the entire day
    requests = {}
    resolution = 3600*24/int(requests_file[0])
    current_time = None

    for req in requests_file[1:]:
        if req.split(':')[0]=='Flows':
            current_time = int(req.split('-')[-1])*resolution
            requests[current_time] = []
        else:
            frm, to, _ = req.split(',')
            frm = int(frm)
            to = int(to)
            requests[current_time].append([frm,to])
    return requests

requests = read_req_file()
zsr2 = {z: {z2:[0,0,0] for z2 in label_set} for z in label_set}

for i,e in enumerate(epoch_data["epoch_locations_accepted"]):
    all_reqs = deepcopy(requests[i*60])
    for src in e:
        for start,end in all_reqs:
            if start==src:
                z1 = labels[src]
                z2 = labels[end]
                zsr2[z1][z2][1] += 1
                all_reqs.remove([start,end])
                break

for time_reqs in requests.values():
    for src, dest in time_reqs:
        z1 = labels[src]
        z2 = labels[dest]
        zsr2[z1][z2][2] += 1      

pairwise_zsrs = []
sums = [0,0]
for z1, v1 in zsr2.items():
    for z2, v2 in v1.items():
        sums[0]+=v2[1]
        sums[1]+=v2[2]
        v2[0] = v2[1]/v2[2]
        pairwise_zsrs.append(v2[0])

print(zsr2)
gini_pair = gini(pairwise_zsrs)
print('Gini zone-pairs :',gini_pair)
min_pair_zsr = min(pairwise_zsrs)
print('Min zone-pair service rate :',min_pair_zsr)
print(sums[0],sums[1])
print("Need to filter out the ignored zones from the request file. The numbers will be slightly off")
print(total_accepted,total_num_reqs)


print("Drivers data")
import numpy as np
vehs=1000
dd = epoch_data['epoch_each_agent_profit']
dts = [0 for i in range(vehs)]

for dt in dd:
	seen = []
	for d, profit in dt:
		dts[d]+=1
		#Assuming each driver gets only one request at a time.
		#The way it is saved, it is actually one entry per request, so it adds up perfectly
d_data = {}
d_data['Trips per driver (TPD)'] = dts
d_data['Mean TPD'] = np.mean(dts)
d_data['StD TPD'] = np.std(dts)
d_data['Min TPD'] = np.min(dts)
d_data['Max TPD'] = np.max(dts)
d_data['TPD gini'] = gini(dts)[0]
# pickle.dump(d_data, open(f'DriverSummary{vehs}_l{lamb}_nbhood.pkl', 'wb'))

# for i,d in enumerate(dts):
#     if dts[i]==0:
#         dts[i] = np.mean(dts)
#         print(i)
print(np.mean(dts),np.std(dts),np.min(dts),np.max(dts),gini(dts)[0])


summary = {}
summary['SR'] = sr
summary['Pair SR gini'] = gini_pair[0]
summary['Mean pair SR'] = gini_pair[1]
summary['Min pair SR'] = min_pair_zsr
summary['Pair SR'] = zsr2
summary['Pair SR Overall'] = zsr2
summary['Source SR gini'] = gini_src[0]
summary['Mean source SR'] = gini_src[1]
summary['Min source SR'] = min_zsr
summary['Source SR'] = zsr
summary['Source SR Overall'] = zsr
summary['Rewards'] = epoch_data['total_requests_accepted']
summary['Trips Per Driver'] = dts
summary['TPD gini'] = d_data['TPD gini']
summary['capacity'] = 4
# summary['lambda'] = 4/6
summary['lambda'] = 10**lamb
summary['numvehs'] = 1000

#Dump to summary.pickle
pickle.dump(summary, open(f'Summary2days14Pair{vehs}_l{lamb}_nbhood.pkl', 'wb'))

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,1)
# fig.set_size_inches(18.5, 2)
# axs.hist(dts, bins =100, range=(80,320))
axs.hist(dts, bins =100, range=(0,320))
# axs.hist(dts, bins =100)
axs.set_ylim([0,150])
plt.title(f'IJCAI(lambda=4/6), {1000} vehicles')
plt.show()