import cv2

"""
low_res_imgs = glob.glob('test/low_res/*.png')
low_res_imgs_npy = glob.glob('test/low_res/*.npy')
high_res_imgs_npy = glob.glob('test/high_res/*.npy')
topo_imgs_1_npy = glob.glob('test/topo/*_1.npy')
topo_imgs_2_npy = glob.glob('test/topo/*_2.npy')

high_res_imgs= glob.glob('test/high_res/*.png')

ind = 5
input_image = cv2.imread(low_res_imgs[ind])
gt_image = cv2.imread(high_res_imgs[ind])

temp = []
for i in range(64):
  for j in range(64):
    temp.append( (gt_image[i,j,0], gt_image[i,j,1], gt_image[i,j,2]) )
# print(np.unique(gt_image[:,:,0]))
r = set(temp)
print(r)

blk = (84, 1, 68)
white =  (36, 231, 253)
"""


def inter2high(input_image, method=cv2.INTER_CUBIC):
    dout = cv2.resize(input_image, (64, 64), interpolation=method)
    gray = cv2.cvtColor(dout, cv2.COLOR_BGR2GRAY)
    return gray
