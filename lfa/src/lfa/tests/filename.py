import os

def prepare_filenames(csvFilename):
  csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.lfa.csv"
  filename, file_extension = os.path.splitext(csvFilename)
  # print(f"Name:\t{filename}\nExt:\t{file_extension}")

  csvFilename1kf = f"{filename}.1kf.csv"
  csvFilename3kf = f"{filename}.3kf.csv"
  return (csvFilename1kf,csvFilename3kf)


csvFilename = "D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi/v01.lfa.csv"
kf1,kf3 = prepare_filenames(csvFilename)
print(f"1KF\t{kf1}")
print(f"3KF\t{kf3}")
