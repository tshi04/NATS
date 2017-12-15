import torch
from torch.autograd import Variable

def beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=7, 
    max_len=20
):
    beam_seq = Variable(torch.LongTensor(beam_size, max_len).fill_(vocab2id['<pad>']))
    beam_seq[:,0] = vocab2id['<s>']
    beam_prb = Variable(torch.FloatTensor(beam_size).fill_(0.0))
    src_text_rep = src_text.repeat(beam_size, 1)
    for j in range(max_len-1):
        logits = model(src_text_rep.cuda(), beam_seq.cuda())
        word_prob = model.decode(logits=logits)
        prob, wds = word_prob.topk(k=beam_size)
        if j == 0:
            beam_prb = prob[0][0]
            beam_seq[:,1] = wds[0][0]
            continue
            
        cand_seq = beam_seq.repeat(beam_size, 1)
        cand_seq = cand_seq.view(beam_size, beam_seq.size(0), beam_seq.size(1))
        cand_seq = cand_seq.transpose(0,1)
        cand_seq = torch.cat(cand_seq, 0)
        cand_seq[:,j+1] = torch.cat(wds[:,j], 0)
        
        cand_prob = torch.cat(beam_prb.repeat(beam_size, 1).transpose(0,1), 0)
        cand_prob += torch.cat(prob[:,j], 0)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size)
        tmp_idx = tmp_idx.data.cpu().numpy()
        for b in range(beam_size):
            beam_seq[b] = cand_seq[tmp_idx[b]]
            beam_prb[b] = tmp_prb[b]    
    
    return beam_seq, beam_prb
