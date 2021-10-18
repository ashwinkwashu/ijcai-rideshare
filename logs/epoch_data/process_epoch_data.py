import pickle
filename = '2021-10-13_191934.pkl'
epoch_data = pickle.load(open(filename,'rb'))

print(epoch_data['settings'])

ks = {i:key for i,key in enumerate(epoch_data.keys())}

labelfile = '../../data/ny/new_labels.pkl'
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

print(f'Overall SR: {sr}\nMinimum SR: {min_zsr}')