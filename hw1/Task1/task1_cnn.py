import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ResNet-18 model
model = models.resnet18().to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image = Image.open("./Task1/001.jpg")
template = [173, 294, 121, 190]  # x,y,w,h
img_temp = image.crop([template[0], template[1], template[0] + template[2], template[1] + template[3]])

# plt.imshow(image)
# plt.show()
# plt.imshow(img_temp)
# plt.show()

img_temp_tensor = transform(img_temp).unsqueeze(0).to(device)
img_temp_feat = model(img_temp_tensor)

# Extract the features

max_score = 0
max_pos = [0, 0]
min_score = 1
min_pos = [0, 0]
with torch.no_grad():
    # slide the template over the image
    stride = 5
    win_pos = [0, 0]
    while win_pos[0] < image.size[0] - template[2]:
        win_pos[1] = 0
        while win_pos[1] < image.size[1] - template[3]:
            roi = image.crop([win_pos[0], win_pos[1], win_pos[0] + template[2], win_pos[1] + template[3]])
            roi_tensor = transform(roi).unsqueeze(0).to(device)
            roi_feat = model(roi_tensor)

            # score = torch.nn.functional.cosine_similarity(roi_feat, img_temp_feat, dim=1)
            # dot product
            score = torch.dot(roi_feat.view(-1), img_temp_feat.view(-1))
            # print(score)
            # if score > max_score:
            #     max_score = score
            #     max_pos = win_pos
            if score < min_score:
                min_score = score
                min_pos = win_pos
            # plt.cla()
            # plt.imshow(roi)
            # plt.text(0, 0, f"Score: {score.item()}", fontsize=12, color="red")
            # plt.pause(0.1)

            win_pos[1] += stride
        win_pos[0] += stride
    print(max_score, max_pos)
