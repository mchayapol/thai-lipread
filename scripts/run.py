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
for f in file_list:
  # continue
  
  out_file_name = "{}-{}".format(viz_name,f)

  full_path_filename = os.path.join(dir_name,f)
  # print("Full path: {}".format(full_path_filename))
  
  # print("\t{}".format(f))
  filename, file_extension = os.path.splitext(f)
  file_extension = file_extension.lower()

  if file_extension.endswith("mp4") or file_extension.endswith("mov"):
    print("\t{}: {}".format(file_extension, full_path_filename))
    cmd = ["vid2vec","-vv",full_path_filename]
    print("Subprocess: {}".format(cmd))
    subprocess.run(cmd)

e = time.time()
et = e-s
print("Elapsed Time:",(et))