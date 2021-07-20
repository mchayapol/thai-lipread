import os
import sys
import subprocess
import time
dir_name = sys.argv[1]
viz_name = sys.argv[2]
out_dir_name = sys.argv[3]


print("{}: {} --> {}".format(viz_name,dir_name,out_dir_name))
file_list = os.listdir(dir_name)

s = time.time()
for filename in file_list:
  full_path_filename = os.path.join(dir_name,filename)
  # filename2 = filename.split('-')[1].lower().strip().replace('(1)','').replace(' ','-').replace('_','')
  filename2 = "{}-{}".format(viz_name,filename)
  print(filename,' => ',filename2)
  full_path_filename2 = os.path.join(dir_name,filename2)
  os.rename(full_path_filename,full_path_filename2)
  continue
