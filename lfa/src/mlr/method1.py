def find_center(data):
  """
    Find center point of the clip
    RETURN
    ======
    a tuple of (c,d)
    c : center coordinate (x,y)
    d : data with frame's center

  """
  sumX = sumY = 0
  n = len(data)
  datac = []
  for f in data:
    f = f.copy()
    points = f["top_lip"] + f["bottom_lip"]
    x1 = x2 = points[0][0]
    y1 = y2 = points[0][1]
    for p in points:
      x,y = p
      x1 = min(x1,x)
      x2 = max(x2,x)
      y1 = min(y1,y)
      y2 = max(y2,y)
    f_center = ((x1+x2)/2, (y1+y2)/2)
    f["center"] = f_center
    datac.append(f) 
    sumX += f_center[0]
    sumY += f_center[1]

  c_center = (int(sumX/n),int(sumY/n))
  return (c_center,datac)

def shift_center(data,center):
  """
  Parameters
  ==========
  data is a list of frame with property "center"

  Return
  ======
  datas : shifted frames
  """
  cX,cY = center
  datas = []
  for f in data:
    f = f.copy()  # Important
    x,y = f["center"]
    dX = int(x - cX)
    dY = int(y - cY)
    print(dX,dY)
    # Shift every point with dX, dY
    top_lip = []
    for p in f["top_lip"]:
      x,y = p
      x += dX
      y += dY
      top_lip.append((x,y))  

    bottom_lip = []
    for p in f["bottom_lip"]:
      x,y = p
      x += dX
      y += dY
      bottom_lip.append((x,y))

    # print("Shifted----\n\t{}\n\t{}".format(f["top_lip"],top_lip))
    f["top_lip"] = top_lip
    f["bottom_lip"] = bottom_lip
    datas.append(f)

  return datas

def stabilize(data):  
  """
  Stabilize method 1
  use four corners for ROI to find center point and transform all frames using difference.
  """
  c_center, datac = find_center(data)
  print("Clip center: {}".format(c_center))
  # print(datac[0]['center'])

  datas = shift_center(datac,c_center)

  print("Origin\n",data[0])
  print("Origin with center\n",datac[0])
  print("Shifted\n",datas[0])
  return datas