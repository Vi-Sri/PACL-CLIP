import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from torchvision.datasets import CocoCaptions
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from core import PACLCLip, compute_patch_level_similarity, normalize_patch_level_similarity, compute_weighted_sum, pacl_compatibility_function, compute_infoNCE_loss
from tqdm import tqdm
import clip
import random

def collate_combine_description(batch):
    images, captions_list = zip(*batch)
    combined_captions = [' '.join(captions) for captions in captions_list]
    caption_tokens = clip.tokenize(combined_captions).to(device)
    images = torch.stack(images, 0)  
    return images, caption_tokens
    

def custom_collate_random(batch):
    images, captions_list = zip(*batch)
    selected_captions = [random.choice(captions) for captions in captions_list]
    caption_tokens = clip.tokenize(selected_captions).to(device)
    images = torch.stack(images, 0)
    return images, caption_tokens



clip_model, preprocess = clip.load("ViT-B/16", device="cuda") 

dataset = CocoCaptions(root="/lustre/fs1/groups/course.cap6411/Dataset/coco/val2017", 
                       annFile="/lustre/fs1/groups/course.cap6411/Dataset/coco/annotations/captions_val2017.json", 
                       transform=preprocess)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_random)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
lr = 1e-5  
vision_dim = 768
projection_dim = 512
step_size = 5  
gamma = 0.1  
clip_value = 1.0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PACLCLip(clip_model, vision_dim, projection_dim).to(device)
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
best_loss = float('inf')
save_dir = "./saved_models"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
    for images, caption_tokens in progress_bar:
        images = images.to(device)
        optimizer.zero_grad()
        compatibility = model(images, caption_tokens)
        loss = compute_infoNCE_loss(compatibility)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (epoch + 1)})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss}")
    
    scheduler.step()
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, 'best_model_checkpoint.pth'))
  







  
   
  
