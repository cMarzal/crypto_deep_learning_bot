from __future__ import print_function
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import model
from args import arg_parse
from extraFuncs import load_file, changeList, getWeights
from random import shuffle


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # load args
    args = arg_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)


    def single_batch_padding(train_X_batch, train_y_batch):
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(train_y_batch)
        length = [len(x) for x in train_X_batch]
        return padded_sequence, label, length


    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # params
    f_size = 6
    epochs = 100

    trainSet = load_file('trainBTC1')
    ytrainSet = load_file('ytrainBTC1').astype(int)
    testSet = load_file('testBTC1')
    ytestSet = load_file('ytestBTC1').astype(int)
    # Normalize training set
    trainSet = changeList(trainSet)
    testSet = changeList(testSet)
    weights = torch.tensor(getWeights(ytrainSet))

    # start model
    model = model.Model(f_size).cuda()
    # model_state = os.path.join('model_bck', 'modelLS8.pth.tar')
    # model.load_state_dict(torch.load(model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss(weights.cuda())

    model.train()
    vid_len = len(trainSet)
    eval_len = len(testSet)
    train_loss = []
    val_acc = []
    max_acc = 0

    print('starting training')
    for epoch in range(epochs):
        batch_size = args.train_batch
        c = list(zip(trainSet, ytrainSet))
        shuffle(c)
        trainSet, ytrainSet = zip(*c)
        avg_loss = 0
        for i in range(0, vid_len, batch_size):
            if batch_size != args.train_batch:
                break
            elif i + batch_size >= vid_len:
                # break
                batch_size = vid_len - i - 1

            model.zero_grad()
            trainBatch = trainSet[i:i + batch_size]
            acts = ytrainSet[i:i + batch_size]

            batch_patch, act_patch, length = single_batch_padding(trainBatch, acts)
            batch_patch = batch_patch.cuda()

            output = model(batch_patch, length)
            ls = loss(output, act_patch.cuda())
            ls.backward()
            optimizer.step()
            avg_loss += ls.cpu()

        print('Epoch: %d/%d\tAverage Loss: %.4f'
              % (epoch, args.epoch,
                 avg_loss / vid_len))
        # Test on the validation videos

        model.eval()
        acc = 0
        totWeight = [0, 0, 0, 0, 0]
        corrWeight = [0, 0, 0, 0, 0]
        accWeight = [0, 0, 0, 0, 0]
        fake4 = 0
        fake41 = 0
        test_batch2 = args.test_batch
        with torch.no_grad():
            for e in range(0, eval_len, test_batch2):
                if test_batch2 != args.test_batch:
                    break
                elif e + test_batch2 >= eval_len:
                    # break
                    test_batch2 = eval_len - e - 1

                data_e = testSet[e:e + args.test_batch]
                label_e = ytestSet[e:e + args.test_batch]
                data_patch, act_patch, length = single_batch_padding(data_e, label_e)
                data_patch = data_patch.cuda()
                out = model(data_patch, length)
                out_label = torch.argmax(out, 1).cpu().data
                acc += np.sum((out_label == act_patch).numpy())
                for pos, num in enumerate(act_patch):
                    totWeight[num] += 1
                    if out_label[pos] == num:
                        corrWeight[num] += 1
                    if (num != 4) and out_label[pos] == 4:
                        fake4 += 1
                    if (num == 0) and out_label[pos] == 4:
                        fake41 += 1

        acc = acc / eval_len
        for x in range(5):
            accWeight[x] = corrWeight[x] / totWeight[x]
        train_loss.append(avg_loss)
        val_acc.append(acc)
        weightacc = sum(accWeight) / len(accWeight)
        print('Average accuracy:', acc)
        print('Weighted accuracy:', weightacc)
        print('Average accuracy for classes:')
        print('0 :', accWeight[0], '(', corrWeight[0], ' / ', totWeight[0], ')')
        print('1: ', accWeight[1], '(', corrWeight[1], ' / ', totWeight[1], ')')
        print('2: ', accWeight[2], '(', corrWeight[2], ' / ', totWeight[2], ')')
        print('3: ', accWeight[3], '(', corrWeight[3], ' / ', totWeight[3], ')')
        print('4: ', accWeight[4], '(', corrWeight[4], ' / ', totWeight[4], ')')
        print(' Number of fake 4s: ', fake4/(eval_len - totWeight[4]), '(', fake4, '/', (eval_len - totWeight[4]), ') -  Fake 4s in class 0: ', fake41/totWeight[0], '(', fake41, '/', totWeight[0], ')')

        model.train()

        if weightacc > max_acc:
            print('New minimum diference: ', weightacc, ' ...Saving model')
            save_model(model, os.path.join('mods', 'RNN_model.pth.tar'))
            max_acc = weightacc
        print('__________________________________________')

    print('Finished training with', args.epoch, 'epochs. Maximum Accuracy on Validation:', max_acc)
