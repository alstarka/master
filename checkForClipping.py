# check for clipping
if x1 < 0:
    x1 = 0
if y1 < 0:
    y1 = 0
if x2 > w:
    x2 = w
if y2 > h:
    y2 = h

    # re-calculate the size to avoid clipping
head_width = x2 - x1
head_height = y2 - y1

h, w = frame.shape[:2]
