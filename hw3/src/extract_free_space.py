import cv2
import numpy as np
import matplotlib.pyplot as plt

map = cv2.imread('./map/vivocity.png')
map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)

# light_free = (223, 213, 193)
# dark_free = (239, 227, 213)
light_free = (210, 200, 190)
dark_free = (250, 240, 230)
free_space = cv2.inRange(map, light_free, dark_free)

plt.subplot(1,2,1)
plt.imshow(map)
plt.subplot(1,2,2)
plt.imshow(free_space)
plt.show()

cv2.imwrite("./map/vivocity_freespace.png", free_space)