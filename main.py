'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re
import os
import argparse
import shutil
import glob
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model import *
from utils import *
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='train', help='train | validate | rouge | beam')
parser.add_argument('--data_dir', default='../sum_data/', help='directory that store the data.')
parser.add_argument('--file_vocab', default='vocab', help='file store training vocabulary.')
parser.add_argument('--file_corpus', default='train.txt', help='file store training documents.')
parser.add_argument('--n_epoch', type=int, default=35, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--src_seq_lens', type=int, default=400, help='length of source documents.')
parser.add_argument('--trg_seq_lens', type=int, default=100, help='length of trage documents.')
parser.add_argument('--src_emb_dim', type=int, default=128, help='source embedding dimension')
parser.add_argument('--trg_emb_dim', type=int, default=128, help='target embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=256, help='decoder hidden dimension')
parser.add_argument('--src_num_layers', type=int, default=1, help='encoder number layers')
parser.add_argument('--vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_mincount', type=int, default=5, help='min word frequency')
parser.add_argument('--src_vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--src_word_mincount', type=int, default=5, help='min word frequency')
parser.add_argument('--src_bidirection', type=str2bool, default=True, help='encoder bidirectional?')
parser.add_argument('--batch_first', type=str2bool, default=True, help='batch first?')
parser.add_argument('--shared_embedding', type=str2bool, default=True, help='source / target share embedding?')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

parser.add_argument('--attn_method', default='luong_concat', help='luong_dot | luong_concat | luong_general')
parser.add_argument('--repetition', default='vanilla', help='vanilla | temporal | asee (coverage). Repetition Handling')
parser.add_argument('--network_', default='lstm', help='gru | lstm')
parser.add_argument('--pointer_net', type=str2bool, default=True, help='Use pointer network?')
parser.add_argument('--attn_decoder', type=str2bool, default=True, help='attention decoder?')
parser.add_argument('--oov_explicit', type=str2bool, default=True, help='explicit OOV?')
parser.add_argument('--share_emb_weight', type=str2bool, default=True, help='share_emb_weight')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')
parser.add_argument('--checkpoint', type=int, default=100, help='How often you want to save model?')
parser.add_argument('--nbestmodel', type=int, default=50, help='How many models you want to keep?')
parser.add_argument('--val_num_batch', type=int, default=10, help='how many batches')
parser.add_argument('--use_move_avg', type=str2bool, default=True, help='move average')
parser.add_argument('--continue_training', type=str2bool, default=True, help='Do you want to continue?')
parser.add_argument('--debug', type=str2bool, default=False, help='if true will clean the output after training')
parser.add_argument('--file_val', default='val.txt', help='val data')
# beam search
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--beam_size', type=int, default=5, help='beam size.')
parser.add_argument('--beam_batch_size', type=int, default=1, help='batch size for beam search.')
parser.add_argument('--copy_words', type=str2bool, default=True, help='Do you want to copy words?')
parser.add_argument('--model_optimal', type=str2bool, default=True, help='Do you want to use the best model?')
parser.add_argument('--model_dir', default='seq2seq_results-0', help='directory that store the model.')
parser.add_argument('--model_file', default='seq2seq_0_0', help='file for model.')

args = parser.parse_args()

if args.repetition == 'asee' and args.task == 'train':
    args.repetition = 'asee_train'
if args.pointer_net:
    args.shared_embedding = True
else:
    args.repetition = 'vanilla'
    args.oov_explicit = False
if args.oov_explicit:
    args.shared_embedding = True
    
if not args.task == 'train':
    args.dropout = 0.0
    
if not args.task == 'rouge':
    vocab2id, id2vocab = construct_vocab(
        file_=args.data_dir+'/'+args.file_vocab,
        max_size=args.vocab_size,
        mincount=args.word_mincount)
    print('The vocabulary size: {}'.format(len(vocab2id)))
    src_vocab2id = vocab2id
    src_id2vocab = id2vocab
if not args.shared_embedding:
    src_vocab2id, src_id2vocab = construct_vocab(
        file_=args.data_dir+'/'+args.file_vocab,
        max_size=args.src_vocab_size,
        mincount=args.src_word_mincount)
    print('The vocabulary size: {}'.format(len(src_vocab2id)))

if args.task == 'train' or args.task == 'validate' or args.task == 'beam':
    model = Seq2Seq(
        src_emb_dim=args.src_emb_dim,
        trg_emb_dim=args.trg_emb_dim,
        src_hidden_dim=args.src_hidden_dim,
        trg_hidden_dim=args.trg_hidden_dim,
        src_vocab_size=len(src_vocab2id),
        trg_vocab_size=len(vocab2id),
        src_nlayer=args.src_num_layers,
        batch_first=args.batch_first,
        src_bidirect=args.src_bidirection,
        dropout=args.dropout,
        attn_method=args.attn_method,
        repetition=args.repetition,
        network_=args.network_,
        pointer_net=args.pointer_net,
        shared_emb=args.shared_embedding,
        attn_decoder=args.attn_decoder,
        share_emb_weight=args.share_emb_weight
    ).cuda()
    print(model)
'''
train
'''
if args.task == 'train':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # read the last check point and continue training
    uf_model = [0, -1]
    if args.continue_training:
        out_dir = os.path.join(args.data_dir, args.model_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        model_para_files = glob.glob(os.path.join(out_dir, '*.model'))
        if len(model_para_files) > 0:
            uf_model = []
            for fl_ in model_para_files:
                arr = re.split('\/', fl_)[-1]
                arr = re.split('\_|\.', arr)
                uf_model.append([int(arr[1]), int(arr[2])])
            uf_model = sorted(uf_model)[-1]
            fl_ = os.path.join(out_dir, 'seq2seq_'+str(uf_model[0])+'_'+str(uf_model[1])+'.model')
            model.load_state_dict(torch.load(fl_))
    else:
        lead_dir = args.data_dir+'/seq2seq_results-'
        for k in range(1000000):
            out_dir = lead_dir+str(k)
            if not os.path.exists(out_dir):
                break
        os.mkdir(out_dir)
    # train models
    losses = []
    start_time = time.time()
    cclb = 0
    for epoch in range(uf_model[0], args.n_epoch):
        n_batch = create_batch_file(
            path_=args.data_dir,
            fkey_='train',
            file_=args.file_corpus,
            batch_size=args.batch_size
        )
        print('The number of batches: {}'.format(n_batch))
        for batch_id in range(n_batch):
            if cclb == 0 and batch_id <= uf_model[1]:
                continue
            else:
                cclb += 1
            if args.oov_explicit:
                ext_id2oov, src_var, trg_input_var, \
                src_var_ex, trg_output_var_ex = process_minibatch_explicit(
                    batch_id=batch_id, path_=args.data_dir, fkey_='train', 
                    batch_size=args.batch_size, 
                    vocab2id=vocab2id, 
                    max_lens=[args.src_seq_lens, args.trg_seq_lens])
                src_var = src_var.cuda()
                trg_input_var = trg_input_var.cuda()
                src_var_ex = src_var_ex.cuda()
                trg_output_var_ex = trg_output_var_ex.cuda()
                
                weight_mask = torch.ones(len(vocab2id)+len(ext_id2oov)).cuda()
                weight_mask[vocab2id['<pad>']] = 0
                loss_criterion = torch.nn.NLLLoss(weight=weight_mask).cuda()
            else:
                src_var, trg_input_var, trg_output_var = process_minibatch(
                    batch_id=batch_id, path_=args.data_dir, fkey_='train', 
                    batch_size=args.batch_size, 
                    src_vocab2id=src_vocab2id, vocab2id=vocab2id, 
                    max_lens=[args.src_seq_lens, args.trg_seq_lens])
                
                weight_mask = torch.ones(len(vocab2id)).cuda()
                weight_mask[vocab2id['<pad>']] = 0
                loss_criterion = torch.nn.NLLLoss(weight=weight_mask).cuda()
                src_var = src_var.cuda()
                trg_input_var = trg_input_var.cuda()
                trg_output_var = trg_output_var.cuda()
            
            logits, attn_, p_gen, loss_cv = model(src_var, trg_input_var)
            logits = F.softmax(logits, dim=2)
            # use the pointer generator loss
            if args.pointer_net:
                if args.oov_explicit:
                    logits = model.cal_dist_explicit(src_var_ex, logits, attn_, p_gen, vocab2id, ext_id2oov)
                    logits = logits + 1e-20
                else:
                    logits = model.cal_dist(src_var, logits, attn_, p_gen, src_vocab2id)

            if batch_id%1 == 0:
                word_prob = logits.topk(1, dim=2)[1].squeeze(2).data.cpu().numpy()
                
            logits = torch.log(logits)
            if args.oov_explicit:
                loss = loss_criterion(
                    logits.contiguous().view(-1, len(vocab2id)+len(ext_id2oov)),
                    trg_output_var_ex.view(-1))
            else:
                loss = loss_criterion(
                    logits.contiguous().view(-1, len(vocab2id)),
                    trg_output_var.view(-1))

            if args.repetition == 'asee_train':
                loss = loss + loss_cv[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
            end_time = time.time()
            losses.append([
                epoch, batch_id, 
                loss.data.cpu().numpy(), 
                loss_cv.data.cpu().numpy()[0], 
                (end_time-start_time)/3600.0])
            if batch_id%args.checkpoint == 0:
                loss_np = np.array(losses)
                np.save(out_dir+'/loss', loss_np)
                fmodel = open(os.path.join(out_dir, 'seq2seq_'+str(epoch)+'_'+str(batch_id)+'.model'), 'wb')
                torch.save(model.state_dict(), fmodel)
                fmodel.close()
            if batch_id%1 == 0:
                end_time = time.time()
                if args.oov_explicit:
                    sen_pred = [id2vocab[x] if x in id2vocab else ext_id2oov[x] for x in word_prob[0]]
                else:
                    sen_pred = [id2vocab[x] for x in word_prob[0]]
                print('epoch={}, batch={}, loss={}, loss_cv={}, time_escape={}s={}h'.format(
                    epoch, batch_id, 
                    loss.data.cpu().numpy(), 
                    loss_cv.data.cpu().numpy()[0],
                    end_time-start_time, (end_time-start_time)/3600.0
                ))
                print(' '.join(sen_pred))
            if args.debug:
                break
            del logits, attn_, p_gen, loss_cv, loss
        if args.debug:
            break
        
        loss_np = np.array(losses)
        np.save(out_dir+'/loss', loss_np)
        fmodel = open(os.path.join(out_dir, 'seq2seq_'+str(epoch)+'_'+str(batch_id)+'.model'), 'wb')
        torch.save(model.state_dict(), fmodel)
        fmodel.close()
            
    if args.debug:
        shutil.rmtree(out_dir)
'''
validate
'''
if args.task == 'validate':
    best_arr = []
    val_file = os.path.join(args.data_dir, args.model_dir, 'model_validate.txt')
    if os.path.exists(val_file):
        fp = open(val_file, 'r')
        for line in fp:
            arr = re.split('\s', line[:-1])
            best_arr.append([arr[0], float(arr[1]), float(arr[2])])
        fp.close()

    model.eval()
    with torch.no_grad(): 
        while 1:
            model_para_files = []
            model_para_files = glob.glob(os.path.join(args.data_dir, args.model_dir, '*.model'))
            for j in range(len(model_para_files)):
                arr = re.split('\_|\.', model_para_files[j])
                arr = [int(arr[-3]), int(arr[-2]), model_para_files[j]]
                model_para_files[j] = arr
            model_para_files = sorted(model_para_files)
            for j in range(len(model_para_files)):
                model_para_files[j] = model_para_files[j][-1]

            for fl_ in model_para_files:           
                best_model = {itm[0]: itm[1] for itm in best_arr}
                if fl_ in best_model:
                    continue
                losses = []
                start_time = time.time()
                if os.path.exists(fl_):
                    time.sleep(3)
                    model.load_state_dict(torch.load(fl_))
                else:
                    continue
                val_batch = create_batch_file(
                    path_=args.data_dir,
                    fkey_='validate',
                    file_=args.file_val,
                    batch_size=args.batch_size
                )
                print('The number of batches (test): {}'.format(val_batch))
                if args.val_num_batch > val_batch:
                    args.val_num_batch = val_batch
                for batch_id in range(args.val_num_batch):

                    if args.oov_explicit:
                        ext_id2oov, src_var, trg_input_var, \
                        src_var_ex, trg_output_var_ex = process_minibatch_explicit(
                            batch_id=batch_id, path_=args.data_dir, fkey_='validate', 
                            batch_size=args.batch_size, 
                            vocab2id=vocab2id, 
                            max_lens=[args.src_seq_lens, args.trg_seq_lens])
                        src_var = src_var.cuda()
                        trg_input_var = trg_input_var.cuda()
                        src_var_ex = src_var_ex.cuda()
                        trg_output_var_ex = trg_output_var_ex.cuda()

                        weight_mask = torch.ones(len(vocab2id)+len(ext_id2oov)).cuda()
                        weight_mask[vocab2id['<pad>']] = 0
                        loss_criterion = torch.nn.NLLLoss(weight=weight_mask).cuda()
                    else:
                        src_var, trg_input_var, trg_output_var = process_minibatch(
                            batch_id=batch_id, path_=args.data_dir, fkey_='validate', 
                            batch_size=args.batch_size, 
                            src_vocab2id=src_vocab2id, vocab2id=vocab2id, 
                            max_lens=[args.src_seq_lens, args.trg_seq_lens])
                        weight_mask = torch.ones(len(vocab2id)).cuda()
                        weight_mask[vocab2id['<pad>']] = 0
                        loss_criterion = torch.nn.NLLLoss(weight=weight_mask).cuda()
                        src_var = src_var.cuda()
                        trg_input_var = trg_input_var.cuda()
                        trg_output_var = trg_output_var.cuda()

                    logits, attn_, p_gen, loss_cv = model(src_var.cuda(), trg_input_var.cuda())
                    logits = F.softmax(logits, dim=2)
                    # use the pointer generator loss
                    if args.pointer_net:
                        if args.oov_explicit:
                            logits = model.cal_dist_explicit(src_var_ex, logits, attn_, p_gen, vocab2id, ext_id2oov)
                            logits = logits + 1e-20
                        else:
                            logits = model.cal_dist(src_var, logits, attn_, p_gen, src_vocab2id)

                    logits = torch.log(logits)
                    if args.oov_explicit:
                        loss = loss_criterion(
                            logits.contiguous().view(-1, len(vocab2id)+len(ext_id2oov)),
                            trg_output_var_ex.view(-1))
                    else:
                        loss = loss_criterion(
                            logits.contiguous().view(-1, len(vocab2id)),
                            trg_output_var.view(-1))

                    losses.append(loss.data.cpu().numpy())
                    show_progress(batch_id+1, args.val_num_batch)
                    del logits, attn_, p_gen, loss_cv, loss
                print()
                losses = np.array(losses)
                end_time = time.time()
                if args.use_move_avg:
                    try:
                        losses_out = 0.9*losses_out + 0.1*np.average(losses)
                    except:
                        losses_out = np.average(losses)
                else:
                    losses_out = np.average(losses)
                best_arr.append([fl_, losses_out, end_time-start_time])
                for itm in best_arr[:args.nbestmodel]:
                    print('model={}, loss={}, time={}'.format(itm[0], itm[1], itm[2]))

                best_arr = sorted(best_arr, key=lambda bb: bb[1])
                for itm in best_arr[args.nbestmodel:]:
                    tarr = re.split('_|\.', itm[0])
                    if tarr[-2] == '0':
                        continue
                    if os.path.exists(itm[0]):
                        os.unlink(itm[0])
                fout = open(val_file, 'w')
                for itm in best_arr:
                    if len(itm) == 0:
                        continue
                    fout.write(' '.join([itm[0], str(itm[1]), str(itm[2])])+'\n')
                fout.close()
'''
beam
'''
if args.task == 'beam':
    args.batch_size = args.beam_batch_size
    
    test_batch = create_batch_file(
        path_=args.data_dir,
        fkey_='test',
        file_=args.file_test,
        batch_size=args.batch_size
    )
    print('The number of batches (test): {}'.format(test_batch))
    
    model.eval()
    with torch.no_grad(): 
        if args.model_optimal:
            model_valid_file = os.path.join(args.data_dir, args.model_dir, 'model_validate.txt')
            fp = open(model_valid_file, 'r')
            for line in fp:
                model_optimal_file = re.split('\s', line[:-1])[0]
                break
            fp.close()
        else:
            model_optimal_file = os.path.join(args.data_dir, args.model_dir, args.model_file+'.model')
        print("You choose to use {} for decoding.".format(model_optimal_file))
        print("Batch Size = {}, Beam Size = {}".format(args.batch_size, args.beam_size))
        model.load_state_dict(torch.load(model_optimal_file))

        start_time = time.time()
        if args.oov_explicit:
            fout = open(os.path.join(args.data_dir, 'summaries.txt'), 'w')
            for batch_id in range(test_batch):
                ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr \
                = process_minibatch_explicit_test(
                    batch_id=batch_id, path_=args.data_dir, 
                    batch_size=args.batch_size, vocab2id=vocab2id, 
                    src_lens=args.src_seq_lens
                )
                src_msk = src_msk.cuda()
                src_var = src_var.cuda()
                src_var_ex = src_var_ex.cuda()
                beam_seq, beam_prb, beam_attn_ = fast_beam_search(
                    model=model,
                    src_text=src_var,
                    src_text_ex=src_var_ex,
                    vocab2id=vocab2id,
                    ext_id2oov=ext_id2oov,
                    beam_size=args.beam_size,
                    max_len=args.trg_seq_lens,
                    network=args.network_,
                    pointer_net=args.pointer_net,
                    oov_explicit=args.oov_explicit,
                    attn_decoder=args.attn_decoder
                )
                src_msk = src_msk.repeat(1, args.beam_size).view(
                    src_msk.size(0), args.beam_size, args.src_seq_lens).unsqueeze(0)
                # copy unknown words
                if args.copy_words:
                    beam_attn_ = beam_attn_*src_msk
                    beam_copy = beam_attn_.topk(1, dim=3)[1].squeeze(-1)
                    beam_copy = beam_copy[:, :, 0].transpose(0, 1)
                    wdidx_copy = beam_copy.data.cpu().numpy()
                    for b in range(len(trg_arr)):
                        arr = []
                        gen_text = beam_seq.data.cpu().numpy()[b,0]
                        gen_text = [id2vocab[wd] if wd in id2vocab else ext_id2oov[wd] for wd in gen_text]
                        gen_text = gen_text[1:]
                        for j in range(len(gen_text)):
                            if gen_text[j] == '<unk>':
                                gen_text[j] = src_arr[b][wdidx_copy[b, j]]
                        arr.append(' '.join(gen_text))
                        arr.append(trg_arr[b])
                        fout.write('<sec>'.join(arr)+'\n')
                else:
                    for b in range(len(trg_arr)):
                        arr = []
                        gen_text = beam_seq.data.cpu().numpy()[b,0]
                        gen_text = [id2vocab[wd] if wd in id2vocab else ext_id2oov[wd] for wd in gen_text]
                        gen_text = gen_text[1:]
                        arr.append(' '.join(gen_text))
                        arr.append(trg_arr[b])
                        fout.write('<sec>'.join(arr)+'\n')

                end_time = time.time()
                show_progress(batch_id, test_batch, str((end_time-start_time)/3600)[:8]+"h")
            fout.close()
        else:
            fout = open(os.path.join(args.data_dir, 'summaries.txt'), 'w')
            for batch_id in range(test_batch):
                src_var, src_arr, src_msk, trg_arr = process_minibatch_test(
                    batch_id=batch_id, path_=args.data_dir, 
                    batch_size=args.batch_size, vocab2id=src_vocab2id, 
                    src_lens=args.src_seq_lens
                )
                src_msk = src_msk.cuda()
                src_var = src_var.cuda()
                beam_seq, beam_prb, beam_attn_ = fast_beam_search(
                    model=model,
                    src_text=src_var,
                    src_text_ex=src_var, 
                    vocab2id=src_vocab2id,
                    ext_id2oov=src_vocab2id,
                    beam_size=args.beam_size, 
                    max_len=args.trg_seq_lens,
                    network=args.network_,
                    pointer_net=args.pointer_net,
                    oov_explicit=args.oov_explicit,
                    attn_decoder=args.attn_decoder
                )
                src_msk = src_msk.repeat(1, args.beam_size).view(
                    src_msk.size(0), args.beam_size, args.src_seq_lens).unsqueeze(0)
                # copy unknown words
                if args.copy_words:
                    beam_attn_ = beam_attn_*src_msk
                    beam_copy = beam_attn_.topk(1, dim=3)[1].squeeze(-1)
                    beam_copy = beam_copy[:, :, 0].transpose(0, 1)
                    wdidx_copy = beam_copy.data.cpu().numpy()
                    for b in range(len(trg_arr)):
                        arr = []
                        gen_text = beam_seq.data.cpu().numpy()[b,0]
                        gen_text = [src_id2vocab[wd] for wd in gen_text]
                        gen_text = gen_text[1:]
                        for j in range(len(gen_text)):
                            if gen_text[j] == '<unk>':
                                gen_text[j] = src_arr[b][wdidx_copy[b, j]]
                        arr.append(' '.join(gen_text))
                        arr.append(trg_arr[b])
                        fout.write('<sec>'.join(arr)+'\n')
                else:
                    for b in range(len(trg_arr)):
                        arr = []
                        gen_text = beam_seq.data.cpu().numpy()[b,0]
                        gen_text = [src_id2vocab[wd] for wd in gen_text]
                        gen_text = gen_text[1:]
                        arr.append(' '.join(gen_text))
                        arr.append(trg_arr[b])
                        fout.write('<sec>'.join(arr)+'\n')

                end_time = time.time()
                show_progress(batch_id, test_batch, str((end_time-start_time)/3600)[:8]+"h")
            fout.close()
        print()
'''
rouge
'''
if args.task == 'rouge':
    rouge_path = os.path.join(args.data_dir, 'rouge')
    if os.path.exists(rouge_path):
        shutil.rmtree(rouge_path)
    os.makedirs(rouge_path)
    sys_smm_path = os.path.join(rouge_path, 'system_summaries')
    mod_smm_path = os.path.join(rouge_path, 'model_summaries')
    os.makedirs(sys_smm_path)
    os.makedirs(mod_smm_path)
    fp = open(os.path.join(args.data_dir, 'summaries.txt'), 'r')
    cnt = 1
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        rmm = re.split('<pad>|<s>|</s>', arr[1])
        rmm = list(filter(None, rmm))
        rmm = [' '.join(list(filter(None, re.split('\s', sen)))) for sen in rmm]
        rmm = list(filter(None, rmm))
        
        smm = re.split('<stop>', arr[0])
        smm = list(filter(None, smm))
        smm = re.split('<pad>|<s>|</s>', smm[0])
        smm = list(filter(None, smm))
        smm = [' '.join(list(filter(None, re.split('\s', sen)))) for sen in smm]
        smm = list(filter(None, smm))
        fout = open(os.path.join(sys_smm_path, 'sum.'+str(cnt).zfill(5)+'.txt'), 'w')
        for sen in rmm:
            arr = re.split('\s', sen)
            arr = list(filter(None, arr))
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        fout = open(os.path.join(mod_smm_path, 'sum.A.'+str(cnt).zfill(5)+'.txt'), 'w')
        for sen in smm:
            arr = re.split('\s', sen)
            arr = list(filter(None, arr))
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        cnt += 1
    fp.close()
    
    from pyrouge import Rouge155
    r = Rouge155()
    
    path_to_rouge = os.path.abspath(args.data_dir)
    r.system_dir = os.path.join(path_to_rouge, 'rouge/system_summaries')
    r.model_dir = os.path.join(path_to_rouge, 'rouge/model_summaries')
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'sum.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    fout = open(os.path.join(args.data_dir, 'rouge.txt'), 'w')
    fout.write(output)
    fout.close()
