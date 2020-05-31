#!/usr/bin/env python
# coding: utf-8

# In[22]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchsummary import summary
# from ipynb.fs.full.Preprocess_Image import preprocess_image_for_inference


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model0 = self.get_vgg()
        
        self.model1_1 = self.get_layer_1(38)
        self.model2_1 = self.get_layer_T(38)
        self.model3_1 = self.get_layer_T(38)
        self.model4_1 = self.get_layer_T(38)
        self.model5_1 = self.get_layer_T(38)
        self.model6_1 = self.get_layer_T(38)
        
        self.model1_2 = self.get_layer_1(19)
        self.model2_2 = self.get_layer_T(19)
        self.model3_2 = self.get_layer_T(19)
        self.model4_2 = self.get_layer_T(19)
        self.model5_2 = self.get_layer_T(19)
        self.model6_2 = self.get_layer_T(19)
        
    def forward(self, x):
        x_vgg = self.model0(x)
        
        layer1_paf = self.model1_1(x_vgg)
        layer1_confidence_map = self.model1_2(x_vgg)
        concat = torch.cat([layer1_paf, layer1_confidence_map, x_vgg], 1)
        
        layer2_paf = self.model2_1(concat)
        
        layer2_confidence_map = self.model2_2(concat)
        concat = torch.cat([layer2_paf, layer2_confidence_map, x_vgg], 1)
        
        layer3_paf = self.model3_1(concat)
        layer3_confidence_map = self.model3_2(concat)
        concat = torch.cat([layer3_paf, layer3_confidence_map, x_vgg], 1)
        
        layer4_paf = self.model4_1(concat)
        layer4_confidence_map = self.model4_2(concat)
        concat = torch.cat([layer4_paf, layer4_confidence_map, x_vgg], 1)
        
        layer5_paf = self.model5_1(concat)
        layer5_confidence_map = self.model5_2(concat)
        concat = torch.cat([layer5_paf, layer5_confidence_map, x_vgg], 1)
        
        layer6_paf = self.model6_1(concat)
        layer6_confidence_map = self.model6_2(concat)
        
        return layer6_paf, layer6_confidence_map
    def get_vgg(self):
        vgg_model = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, 1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2, 0),
                                nn.Conv2d(64, 128, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, 1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2, 0),
                                nn.Conv2d(128, 256, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, 1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2, 0),
                                nn.Conv2d(256, 512, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 3, 1, 1),
                                nn.ReLU(),
                                nn.Conv2d(256, 128, 3, 1, 1),
                                nn.ReLU())
        return vgg_model
    def get_layer_1(self, num_last_layer_channel):
        first_layer_model = nn.Sequential(nn.Conv2d(128, 128, 3 ,1 ,1),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 128, 3 ,1 ,1),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 128, 3 ,1 ,1),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 512, 1 ,1 ,0),
                                          nn.ReLU(),
                                          nn.Conv2d(512, num_last_layer_channel, 1 ,1 ,0),
                                          nn.ReLU())
        return first_layer_model

    def get_layer_T(self, num_last_layer_channel):
        layer_T = nn.Sequential(nn.Conv2d(185, 128, 7 ,1 ,3),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 7 ,1 ,3),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 7 ,1 ,3),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 7 ,1 ,3),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 7 ,1 ,3),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, 1 ,1 ,0),
                                nn.ReLU(),
                                nn.Conv2d(128, num_last_layer_channel, 1 ,1 ,0),
                                nn.ReLU())
        return layer_T


# In[3]:


model = Net()
model.cuda()
# print(model)
# summary(model, (3, 122, 122))
print("Model's state dictionary : ")
for model_param_name , value in model.state_dict().items():
    print(model_param_name , "\t", value.shape)


# In[4]:


model.load_state_dict(torch.load('pose_model.pth'))


# In[14]:


model.eval()
# test_tensor = torch.rand((1, 3, 674, 712))
# test_tensor = test_tensor.cuda()
# paf , confidence_map = model(test_tensor)
img = cv2.imread('images/self1.jpg')
plt.imshow(img[:, : ,[2,1,0]])
# img_processed = preprocess_image_for_inference(img)
# print(img_processed.shape)
# img = img.cpu()
# plt.imshow((np.transpose(np.squeeze(img), (1, 2 ,0))*255)[:, :, [2, 1, 0]])
# plt.show()
img_not_processed = cv2.resize(img ,None, fx = 0.25, fy = 0.25)
img_not_processed = img_not_processed.astype(np.float32)
img_not_processed = img_not_processed/255. - 0.5
img_not_processed = torch.from_numpy(np.transpose(img_not_processed, [2, 0, 1])[np.newaxis , ...].astype(np.float32))
print(img_not_processed.shape)
img_not_processed = img_not_processed.cuda().float()
with torch.no_grad():
    paf , confidence_map = model(img_not_processed)


# In[6]:


confidence_map.shape
con_np = confidence_map.cpu()
con_np = con_np.detach().numpy()
con_np.shape


# In[21]:


plt.imshow(cv2.resize(img, (con_np.shape[3], con_np.shape[2]))[:, :, [2,1,0]])
plt.imshow(np.squeeze(con_np)[13], alpha = 0.5)
layers_combined = np.squeeze(con_np)[0]
for i in range(1, 18):
    layers_combined += np.squeeze(con_np)[i]


# In[25]:


from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.imshow(layers_combined)
X = np.arange(0, layers_combined.shape[1])
Y = np.arange(0, layers_combined.shape[0])
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, layers_combined, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()


# In[ ]:




