import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as  plt
import sys
import argparse


parser = argparse.ArgumentParser(description = "Add arguments here")
parser.add_argument('--logs', dest='logs', type=str, help="input log file here")


if __name__ == '__main__':    
    args = parser.parse_args()
    logs = args.logs
    print logs
    iter_ = []
    loss_ = []
    with open(logs, 'r') as f1:
        recs = [x.strip() for x in f1.readlines()]
        for rec in recs:
            keywd_1 = ', loss = '
            keywd_2 = ' Iteration '
            if ', loss = ' in rec:
                # Iteration 24630, loss = 2.44993
                items = rec.split(keywd_1)
                loss_.append(float(items[-1]))
                iter_.append(int(items[0].split(keywd_2)[-1]))
    plt.figure()
    plt.xlabel('#Iteration')
    plt.ylabel('Loss')
    plt.plot(iter_, loss_, linewidth=1)
    plt.savefig("loss_curve.jpg")






