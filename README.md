# SkipPress
## 使用skip 加速vlm推理，整体基于VLMEvalKit框架

### 在VLMEvalKit/vlmeval/vlm/qwen2_vl里面添加delta_press.py文件
### 将model.py替换为model.py
### delta_press.py实现了四种加速策略
- 1.根据attention变化量决定是否跳过token：可以在vit阶段实现，也可以在llm阶段实现，为class DeltaPressViT: 和 class DeltaPress
- 2.根据block的变化量决定是否跳过token：在llm阶段实现
- 3.随机跳过
