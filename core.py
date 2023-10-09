import torch
import torch.nn as nn
import clip
from patchify import patchify


class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.size(0), -1, x.size(1), self.patch_size, self.patch_size)
        return patches

class VisionTransformerWrapper(nn.Module):
    def __init__(self, vision_transformer):
        super(VisionTransformerWrapper, self).__init__()
        self.vision_transformer = vision_transformer

    def forward(self, x: torch.Tensor):
        x = self.vision_transformer.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  
        x = x + self.vision_transformer.positional_embedding.to(x.dtype)[1:]
        x = self.vision_transformer.ln_pre(x)

        x = x.permute(1, 0, 2)  
        x = self.vision_transformer.transformer(x)
        x = x.permute(1, 0, 2)  

        x = self.vision_transformer.ln_post(x[:, 0, :])

        return x 

class PACLVisionEncoder(nn.Module):
    def __init__(self, clip_model, patch_size=16):
        super(PACLVisionEncoder, self).__init__()
        self.clip_model = VisionTransformerWrapper(clip_model.visual)
        self.patch_size = patch_size
        self.patch_extractor = PatchExtractor(patch_size)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, x):
        #patches = patchify(x, (1, 3, self.patch_size, self.patch_size), step=self.patch_size)
        #patches = patches.permute(0, 2, 3, 1, 4, 5, 6).reshape(x.size(0), -1, 3, self.patch_size, self.patch_size)
        #patches = patches.reshape(-1, 3, self.patch_size, self.patch_size)
        patches = self.patch_extractor(x)
        b, t, c, h, w = patches.size()
        patches_reshaped = patches.reshape(b * t, c, h, w)
        patches_upsampled = self.upsample(patches_reshaped)
        patch_embeddings = self.clip_model(patches_upsampled)
        patch_embeddings = patch_embeddings.reshape(x.size(0), t, -1)
        return patch_embeddings

class PACLVisionEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PACLVisionEmbedder, self).__init__()
        
        print("In dim, Out dim : ",in_dim, out_dim) 
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.residual = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.main_branch(x) + self.residual(x)
        

class PACLCLip(nn.Module):
    def __init__(self, clip_model, vision_dim, projection_dim):
        super(PACLCLip, self).__init__()
        
        self.clip_model = clip_model
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
  
        self.vision_encoder = PACLVisionEncoder(self.clip_model)
        self.vision_embedder = PACLVisionEmbedder(vision_dim, projection_dim)
        
 
    def forward(self, images, text_tokens):
        patch_embeddings = self.vision_encoder(images)
        patch_shared_embeddings = self.vision_embedder(patch_embeddings)

        text_embeddings = self.clip_model.encode_text(text_tokens)
        
        similarity = compute_patch_level_similarity(patch_shared_embeddings, text_embeddings)
        weights = normalize_patch_level_similarity(similarity)
        weighted_vision = compute_weighted_sum(patch_shared_embeddings, weights)
        compatibility = pacl_compatibility_function(weighted_vision, text_embeddings)
        
        return compatibility
        

def compute_patch_level_similarity(patch_vision_embeds, text_embed):
    return torch.matmul(patch_vision_embeds, text_embed.unsqueeze(-1)).squeeze(-1)

def normalize_patch_level_similarity(similarity):
    return torch.nn.functional.softmax(similarity, dim=1)
    
def compute_weighted_sum(patch_vision_embeds, weights):
    return torch.matmul(patch_vision_embeds.transpose(1, 2), weights.unsqueeze(-1)).squeeze(-1)
    
def pacl_compatibility_function(weighted_vision, text_embed):
    weighted_vision_expanded = weighted_vision.unsqueeze(1)
    text_embed_expanded = text_embed.unsqueeze(0)
    compatibility = torch.nn.functional.cosine_similarity(weighted_vision_expanded, text_embed_expanded, dim=-1)
    return compatibility
    
def compute_infoNCE_loss(compatibility):

    numerator = torch.exp(compatibility)
    
    denominator_x = numerator.sum(dim=1, keepdim=True)
    denominator_y = numerator.sum(dim=0, keepdim=True)
    
    epsilon = 1e-10
    Lx = torch.mean(torch.log(numerator / (denominator_x + epsilon) + epsilon))
    Ly = torch.mean(torch.log(numerator / (denominator_y + epsilon) + epsilon))
    
    loss = -0.5 * (Lx + Ly)
    
    return loss
