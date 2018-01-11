'''
test
This codes together with the beam search codes are no longer in use.
They are extremely slow.
They help us a lot during building more efficient codes.
Use fastbeam instead.
'''
if opt.task == 'test':
    print 'Please use the fastbeam instead'
    model.load_state_dict(torch.load(
        os.path.join(opt.data_dir, opt.model_dir, opt.model_file+'.model')))

    start_time = time.time()
    fout = open(os.path.join(opt.data_dir, 'summaries.txt'), 'w')
    for batch_id in range(test_batch):
        src_var, trg_input_var, trg_output_var = process_minibatch(
            batch_id=batch_id, path_=opt.data_dir, fkey_='test', 
            batch_size=opt.batch_size, vocab2id=vocab2id, 
            max_lens=[opt.src_seq_lens, opt.trg_seq_lens]
        )
        beam_seq, beam_prb = batch_beam_search(
            model=model, 
            src_text=src_var, 
            vocab2id=vocab2id, 
            beam_size=opt.beam_size, 
            max_len=opt.trg_seq_lens
        )
        trg_seq = trg_output_var.data.numpy()
        for b in range(trg_seq.shape[0]):
            arr = []
            gen_text = beam_seq.data.cpu().numpy()[b,0]
            gen_text = [id2vocab[wd] for wd in gen_text]
            arr.append(' '.join(gen_text))
            trg_text = [id2vocab[wd] for wd in trg_seq[b]]
            arr.append(' '.join(trg_text))
            fout.write('<sec>'.join(arr)+'\n')

        end_time = time.time()
        print(batch_id, end_time-start_time)
    
    fout.close()
'''
can handle batches.
still very slow.
'''
def batch_beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=4, 
    max_len=20
):
    max_len += 1
    batch_size = src_text.size(0)
    beam_seq = Variable(torch.LongTensor(
        batch_size, beam_size, max_len).fill_(vocab2id['<pad>'])).cuda()
    beam_seq[:, :, 0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(batch_size, beam_size).fill_(0.0)
    src_text_rep = src_text.unsqueeze(1).clone().repeat(1, beam_size, 1).cuda()
    for j in range(max_len-1):
        logits, _ = model(
            src_text_rep.view(-1, src_text_rep.size(2)),
            beam_seq.view(-1, beam_seq.size(2))
        )
        prob, wds = model.decode(logits=logits).data.topk(k=beam_size)
        prob = prob.view(batch_size, beam_size, prob.size(1), prob.size(2))
        wds = wds.view(batch_size, beam_size, wds.size(1), wds.size(2))
        if j == 0:
            beam_prb = prob[:, 0, 0]
            beam_seq[:, :, 1] = wds[:, 0, 0]
            continue
        
        if j == 2:
            print beam_seq
            print beam_seq.view(-1, beam_seq.size(2))[:, j]
            print prob[:, :, j]
            break
        cand_seq = beam_seq.repeat(1, beam_size, 1)
        cand_seq = cand_seq.view(batch_size, beam_size, beam_size, beam_seq.size(2))
        cand_seq = cand_seq.transpose(1,2)
        cand_seq = cand_seq.contiguous().view(batch_size, beam_size*beam_size, cand_seq.size(3))
        cand_seq[:, :,j+1] = wds[:, :, j].contiguous().view(-1)
        
        cand_prob = beam_prb.unsqueeze(1).repeat(1, beam_size, 1).transpose(1,2)
        cand_prob += prob[:, :, j]
        cand_prob = cand_prob.contiguous().view(batch_size, beam_size*beam_size)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size, dim=1)
        for x in range(batch_size):
            for b in range(beam_size):
                beam_seq[x, b] = cand_seq[x, tmp_idx[x, b]]
                beam_prb[x, b] = tmp_prb[x, b]
    return      
    return beam_seq, beam_prb


'''
very old approach
very slow
'''
def beam_search(
    model, 
    src_text,
    vocab2id,
    beam_size=5, 
    max_len=20
):
    max_len += 1
    beam_seq = Variable(torch.LongTensor(beam_size, max_len).fill_(vocab2id['<pad>'])).cuda()
    beam_seq[:,0] = vocab2id['<s>']
    beam_prb = torch.FloatTensor(beam_size).fill_(0.0)
    src_text_rep = src_text.clone().repeat(beam_size, 1).cuda()
    for j in range(max_len-1):
        logits, _ = model(src_text_rep, beam_seq)
        word_prob = model.decode(logits=logits)
        prob, wds = word_prob.topk(k=beam_size)
        if j == 0:
            beam_prb = prob.data[0][0]
            beam_seq[:,1] = wds.data[0][0]
            continue

        cand_seq = beam_seq.repeat(beam_size, 1)
        cand_seq = cand_seq.view(beam_size, beam_size, beam_seq.size(1))
        cand_seq = cand_seq.transpose(0,1)
        cand_seq = cand_seq.contiguous().view(-1, cand_seq.size(2))
        cand_seq[:,j+1] = wds.data[:,j].contiguous().view(-1)
        
        cand_prob = beam_prb.repeat(beam_size, 1).transpose(0,1)
        cand_prob += prob[:,j].data
        cand_prob = cand_prob.contiguous().view(-1)
        
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size)
        for b in range(beam_size):
            beam_seq[b] = cand_seq[tmp_idx[b]]
            beam_prb[b] = tmp_prb[b]
    
    return beam_seq, beam_prb
