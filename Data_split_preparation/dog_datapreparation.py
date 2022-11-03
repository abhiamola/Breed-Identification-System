import os
import math

d = '.'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

#Run to make 120 folders in all 3 types: Train, Test, Val

f_names=[temp[2:] for temp in subdirs[1:]]
target_folder='./val'
for foldername in f_names:
    try:
        os.mkdir(os.path.join(target_folder, foldername))
    except WindowsError:
        # Handle the case where the target dir already exist.
        pass


#Sorting files into train test val {70,10,20} splits respectively.
s_n=["{0:03}".format(i) for i in range(1,400)]
total=0
for dirnames in subdirs[116:]:
    files=os.listdir(dirnames)
    total_files=len(files)
    train_files= math.floor(( len(files)*7 )/10)
    val_files= math.floor(( len(files)*1 )/10)
    test_files= total_files - train_files - val_files
    print("Directory : ",dirnames," Total : ",total_files," Train : ",train_files," Val : ",val_files," Test : ",test_files)
    
    
    for i in range(0,train_files):
        os.rename(dirnames+"/"+files[i],"./train/"+dirnames[2:]+"/"+s_n[i]+".jpg")
        
    for j in range(train_files,train_files+val_files):
        os.rename(dirnames+"/"+files[j],"./val/"+dirnames[2:]+"/"+s_n[j]+".jpg")
        
    for k in range(train_files+val_files,train_files+val_files+test_files):
        os.rename(dirnames+"/"+files[k],"./test/"+dirnames[2:]+"/"+s_n[k]+".jpg")
    
 