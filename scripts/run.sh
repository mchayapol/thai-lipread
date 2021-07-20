# /bin/sh
# Sample
# ======
# $ sh run.sh D:/GoogleDrive-VMS/Research/lip-reading/datasets/angsawee/avi
# 
# References
# ==========
# https://tldp.org/LDP/abs/html/string-manipulation.html

echo "Base directory: $1";

for avi in $1/*;
do
  filename=$(basename -- "$avi")
  extension="${filename##*.}"
  filename="${filename%.*}"  
  # last /
  # filename="${fullfile##*/}" 
  # echo $filename $extension
  # continue

  if [ $extension != "avi" ];
  then
    echo "Skipping: $filename.$extension"
    continue
  fi


  echo "AVI: $avi"
  # echo "Length: ${#avi}"
  csv=${avi:0:${#avi}-4}.csv
  csv_pb=${avi:0:${#avi}-4}.pb.csv
  # echo "CSV: $csv"
  # echo "CSV PB: $csv_pb"


  echo "==== vid2vec ==== [$filename]"
  # echo "$filename $extension"
  vid2vec $avi
  lfa -q -s -m 4 $csv
  lfa -q -s -m 4 $csv_pb
done
