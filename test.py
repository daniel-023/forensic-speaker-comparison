from inference import ecapa_inference
import torch

KS1 = "/Users/daniel/Documents/VSCode/forensic-speaker-comparison/trump/KS1_trimmed.wav"
DS1 = "/Users/daniel/Documents/VSCode/forensic-speaker-comparison/trump/DS1_trimmed.wav"
DS3 = "/Users/daniel/Documents/VSCode/forensic-speaker-comparison/trump/DS3_trimmed.wav"

KS1_emb = ecapa_inference(KS1)
DS1_emb = ecapa_inference(DS1)      
DS3_emb = ecapa_inference(DS3)
print("KS1 vs DS1 (different speakers):", torch.cosine_similarity(KS1_emb.unsqueeze(0), DS1_emb.unsqueeze(0)).item())
print("KS1 vs DS3 (same speakers):", torch.cosine_similarity(KS1_emb.unsqueeze(0), DS3_emb.unsqueeze(0)).item())