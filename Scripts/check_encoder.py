from gliner import GLiNER
import torch

m = GLiNER.from_pretrained('urchade/gliner_small-v2.1')
m.eval()

input_ids = torch.randint(0, 1000, (1, 10))
attn_mask = torch.ones(1, 10, dtype=torch.long)

with torch.no_grad():
    enc = m.model.token_rep_layer
    print('Encoder type:', type(enc))
    print('Encoder children:', [name for name, _ in enc.named_children()])
    
    bert_out = enc.bert_layer(input_ids, attn_mask)[0]
    print('BERT output shape:', bert_out.shape)
    
    final_out = enc.encode_text(input_ids, attn_mask)
    print('Final encoder output shape:', final_out.shape)
