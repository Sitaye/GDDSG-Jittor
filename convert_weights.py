import torch
import timm

print("Loading ViT model from timm with pretrained weights from .npz...")
model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)

state_dict_to_save = model.state_dict()

if 'head.weight' in state_dict_to_save:
    print(f"Head weight shape: {state_dict_to_save['head.weight'].shape}")

output_path = "vit_base_patch16_224_in21k_from_timm.pth"
torch.save(state_dict_to_save, output_path)

print(f"\nSuccessfully converted and saved weights to {output_path}")
print(f"Total keys saved: {len(state_dict_to_save)}")