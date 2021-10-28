import pickle
from copy import deepcopy
filename = '14Edit/200_l8_2_nbhood2021-10-21_062047.pkl'
# filename = '14/1000_l10_2_nbhood2021-10-21_030841.pkl'
epoch_data = pickle.load(open(filename,'rb'))

print(epoch_data['settings'])

ks = {i:key for i,key in enumerate(epoch_data.keys())}

labelfile = '../../data/ny/new_labels.pkl'
if 'nbhood' in filename:
    print("USING")
    labelfile = '../../data/ny/nbhood_labels.pkl'
labels = pickle.load(open(labelfile,'rb'))
label_set = set(labels)

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

gini_pair = gini(pairwise_zsrs)
print('Gini zone-pairs :',gini_pair)
print(sums[0],sums[1])
print("Need to filter out the ignored zones from the request file. The numbers will be slightly off")
print(total_accepted,total_num_reqs)
