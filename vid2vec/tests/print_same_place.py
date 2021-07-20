import time

for i in range(0, 101, 5):
  print ('\r>> Processing %d%%' % i,end='',flush=True)
  time.sleep(.1)
print