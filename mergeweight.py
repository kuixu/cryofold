import torch

# 加载三个模型的状态字典
model1_state = torch.load('data/weights/cf_v1.pth')
model2_state = torch.load('data/weights/cf_structure_encoder.pth')
model3_state = torch.load('data/weights/cf_structure_decoder.pth')

# 合并状态字典
merged_state = {}
merged_state.update(model1_state)
merged_state.update(model2_state)
merged_state.update(model3_state)

# 保存合并后的模型
torch.save(merged_state, 'data/weights/cf_v1.0.pth')
