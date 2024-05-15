import os
import torch
import matplotlib.pyplot as plt

from Dataset import ISBI
from Model.Vanila_UNet import VanilaUNet
from utils import Inference_processing


# Load inference dataset
ds_test = ISBI(mode='valid', transform=Inference_processing())

sample = ds_test.__getitem__(5)

img, lbl, ori = sample['image'], sample['label'], sample['origin']
view_img = ((img.permute(1,2,0) + .1662) * .491) * 255.0
test_img = img.to('cuda')
view_lbl = lbl.permute(1,2,0)
view_ori = ori.permute(1,2,0)

# Load best model
model = VanilaUNet().to('cuda')
best_model_state_path = './Model/Vanila_UNet/log/best_model/best_model.pth'
best_state = torch.load(best_model_state_path)
model.load_state_dict(best_state)


with torch.no_grad():
    model.eval()
    test_img = img.unsqueeze(0).to('cuda')
    pred = model(test_img)

view_pred = torch.argmax(pred, dim=1).detach().cpu().permute(1,2,0)

# Visualize the inference results
plt.subplot(1,3,1)
plt.imshow(view_ori)
plt.axis('off')
plt.title('Target tile')

plt.subplot(1,3,2)
plt.imshow(view_lbl)
plt.axis('off')
plt.title('Ground truth')

plt.subplot(1,3,3)
plt.imshow(view_pred)
plt.axis('off')
plt.title('Model prediction')

plt.show()

