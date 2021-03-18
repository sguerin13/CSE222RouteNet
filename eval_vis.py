import pickle
import numpy as np
import os

file_path = os.path.join(os.getcwd(),"evaluation_data/run_2_e_14.pkl")
eval_dict = pickle.load(open(file_path,'rb'))

d_perc_hist = np.histogram(eval_dict['delay_percent'],bins=10)
d_gt_hist = np.histogram(eval_dict['delay_gt'],bins=10)
d_0_abs_hist = np.histogram(eval_dict['delay_0_abs_error'],bins=10)
d_0_hist = np.histogram(eval_dict['delay_0_gt'])
j_perc_hist = np.histogram(eval_dict['jitter_percent'],bins=10)
j_gt_hist = np.histogram(eval_dict['jitter_gt'],bins=10)
j_0_abs_hist = np.histogram(eval_dict['jitter_0_abs_error'],bins=10)
j_0_hist = np.histogram(eval_dict['jitter_0_gt'])

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
# plt.hist(d_perc_hist[0],d_perc_hist[1])



################### Delay ######################

# delay distribution
plt.figure(figsize = (10,10))
plt.hist(eval_dict['delay_gt'],bins='sqrt')
plt.hist(eval_dict['delay_pred'],bins='sqrt',alpha = .5)
plt.title('Delay Distribution')
plt.ylabel('Number of Occurences')
plt.xlabel('Delay (s?)')
plt.xscale('log')
plt.legend(['ground truth','predicted'])
plt.show()

# delay percent error distribution
plt.figure(figsize = (10,10))
plt.hist([100*i for i in eval_dict['delay_percent']],density = True,bins = 'sqrt')
plt.ylabel('Fraction of Occurences')
plt.xlabel('Percent Error')
plt.title('Delay Prediction Error Distribution')
plt.xlim([0,200])
plt.show()

# median percent error:
print("median: ",np.median([100*i for i in eval_dict['delay_percent']]))
print("mean: ",np.mean([100*i for i in eval_dict['delay_percent']]))


delay_group = {1:[],2:[],3:[],4:[],5:[]}
for i,elem in enumerate(eval_dict['delay_percent']):
    if eval_dict['delay_gt'][i] < .01:
        delay_group[1].extend([elem])
        continue
    
    elif eval_dict['delay_gt'][i] < .05:
        delay_group[2].extend([elem])
        continue

    elif eval_dict['delay_gt'][i] < .1:
        delay_group[3].extend([elem])
        continue
    
    elif eval_dict['delay_gt'][i] < 1.0:
        delay_group[4].extend([elem])
        continue
    
    else:
        delay_group[5].extend([elem])
        continue


delay_avgs = [np.mean(delay_group[i]) for i in delay_group.keys()]
delay_std = [np.std(delay_group[i]) for i in delay_group.keys()]
plt.figure(figsize = (15,10))
x = [1,2,3,4,5]
y = [100*i for i in delay_avgs]
y_err = [100*i for i in delay_std]
bins = ['[0, .01]','[.01, .05]','[.05, .1]','[.1, 1.0]','>1.0']
plt.errorbar(x, y, y_err, fmt='s',ms = 10,mfc = 'r',elinewidth=5,capsize=15)
plt.title('Delay Percent Error Based on Ground Truth Delay')
plt.xlabel('Delay Range')
plt.ylabel('Percent Error')
plt.xticks(x,bins)
plt.show()


# jitter distribution
plt.figure(figsize = (10,10))
plt.hist(eval_dict['jitter_gt'],bins='sqrt')
plt.hist(eval_dict['jitter_pred'],bins='sqrt',alpha = .5)
plt.title('Jitter Distribution')
plt.ylabel('Number of Occurences')
plt.xlabel('Jitter (s?)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([.001,.01])
plt.legend(['ground truth','predicted'])
plt.show()

# jitter percent error distribution
plt.figure(figsize = (10,10))
plt.hist([100*i for i in eval_dict['jitter_percent']],density = True,bins = 'sqrt')
plt.ylabel('Number of Occurences')
plt.xlabel('Percent Error')
plt.xscale('log')

plt.show()

# median percent error:
print("median: ",np.median([100*i for i in eval_dict['jitter_percent']]))
print("mean: ",np.mean([100*i for i in eval_dict['jitter_percent']]))



