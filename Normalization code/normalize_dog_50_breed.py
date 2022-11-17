import os
from pathlib import Path
import numpy as np
import cv2
d = '.'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
print(subdirs)

dict_size={}
mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])
mean_lst=[]
stdTemp_lst=[]
std_lst=[]
for foldername in subdirs[1:]:
    temp_root=foldername
    sub_subdir= [os.path.join(temp_root, o) for o in os.listdir(temp_root) if os.path.isdir(os.path.join(temp_root,o))]
    #print(sub_subdir)
    add_files=0
    for one_folder in sub_subdir:
        mean = np.array([0.,0.,0.])
        stdTemp = np.array([0.,0.,0.])
        std = np.array([0.,0.,0.])

        imageFilesDir = Path(one_folder)
        files = list(imageFilesDir.rglob('*.jpg'))
        total_files=len(files)
        dict_size[one_folder]=total_files
        add_files+=total_files
        for i in range(total_files):
            im = cv2.imread(str(files[i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(float) / 255.
         
            for j in range(3):
                mean[j] += np.mean(im[:,:,j])
                mean = (mean/total_files)
                stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])
                
        mean_lst.append(mean)
        std_lst.append(stdTemp/total_files)

    print(add_files)

final_mean_lst=[sum(i) for i in zip(*mean_lst)]
final_mean_lst[:]=[x/100 for x in final_mean_lst]
print(final_mean_lst)

final_std_lst=[sum(i) for i in zip(*std_lst)]
final_std_lst[:]=[x/100 for x in final_std_lst]
print(final_std_lst)

