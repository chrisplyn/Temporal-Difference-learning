import random
import numpy as np
import matplotlib.pyplot as plt


def GenerateRandomWalkSequence(numStates):
    pos = (numStates - 1) / 2
    r = [pos]
    while True:
        if random.random() > 0.5:
            next_pos = 1;
        else:
            next_pos = -1;

        pos += next_pos;
        if pos == numStates - 1:
            z = 1;
            break;
        elif pos == 0:
            z = 0;
            break;
        else:
            r.append(pos)

    return r,z


def TDLambdaUpdate(numStates, seq, z, w0, alpha, lambd):
    a = np.array(seq)
    x = np.zeros((a.size, numStates))
    x[np.arange(a.size),a]  = 1
    P = np.clip(np.dot(x,w0),0,1)
    deltaW = 0
    S = x[0]

    for t in range(1,len(seq)):
        deltaW += alpha*(P[t]-P[t-1])*S
        S = x[t] + (lambd * S)

    deltaW += alpha * (z - P[-1])*S;
    return deltaW

def doBatchUpdate(trainingSets,lambdas,numStates):
    result = np.zeros((len(lambdas),nTrainingSets))
    for l in range(len(lambdas)):
        for n in range(len(trainingSets)):
            w0 = np.ones(numStates)*0.5

            while True:
                deltaW = 0
                for k in range(nSequences):
                    curSeq,z = trainingSets[n][k];
                    deltaW += TDLambdaUpdate(numStates, curSeq, z, w0, 0.01,lambdas[l]);

                if np.max(np.abs(deltaW)) < 0.001:
                    break
                w0 += deltaW

            result[l][n] = np.linalg.norm(trueProp-w0[1:-1],2)/np.sqrt(5)
    return result

def doOnlineUpdate(trainingSets,lambdas,alphas,numStates):
    result = np.zeros((len(lambdas),len(alphas),nTrainingSets))

    for l in range(len(lambdas)):
        for a in range(len(alphas)):
            for n in range(len(trainingSets)):
                w0 = np.ones(numStates)*0.5
    
                for k in range(nSequences):
                    curSeq,z = trainingSets[n][k];
                    w0 += TDLambdaUpdate(numStates, curSeq, z, w0, alphas[a],lambdas[l])
    
                result[l][a][n] = np.sqrt(np.sum(np.power(trueProp-w0[1:-1],2))/5)
    return result


def generateTrainingData(nTrainingSets,nSequences,numStates):
    trainingSets = []

    for numSet in range(nTrainingSets):
        trainingSet = {};
    
        for s in range(nSequences):
            trainingSet[s] = GenerateRandomWalkSequence(numStates)
        trainingSets.append(trainingSet)
    return trainingSets


############################
## experiment setting and data
############################

nTrainingSets = 100
nSequences = 10
numStates = 7
trueProp = np.array([1./6,1./3,0.5,2./3,5./6])
trainingData = generateTrainingData(nTrainingSets,nSequences,numStates)

############################
## experiment 1
############################
lambdas1 = [0,0.1,0.3,0.5,0.7,0.9,1]
exp1Res = doBatchUpdate(trainingData,lambdas1,numStates)

avg_rms_exp1 = np.mean(exp1Res,1)
fig = plt.figure(figsize=(10,10))

plt.plot(lambdas1, avg_rms_exp1, 'r-',marker='o')
plt.xlabel('$\lambda$', fontsize=18)
plt.ylabel('RMSE', fontsize=16)
fig.savefig('figure1.jpg')

############################
## experiment 2
############################
lambdas2 = [0,0.3,0.8,1]
alphas = np.arange(0,0.65,0.05)
exp2Res = doOnlineUpdate(trainingData,lambdas2,alphas,numStates)
avg_rms_exp2 = np.mean(exp2Res,2)

fig = plt.figure(figsize=(10,10))
for m in range(len(lambdas2)):
    x = alphas
    y = avg_rms_exp2[m]
#    
    if m == 3:
        x = x[:8]
        y = y[:8]
    plt.plot(x,y,marker='o')
    plt.text(x[-1], y[-1], '$\lambda={i}$'.format(i=lambdas2[m]), fontsize=18)

plt.xlabel('$\\alpha$', fontsize=18)
plt.ylabel('RMSE', fontsize=16)
fig.savefig('figure2.jpg')



############################
## experiment 3
############################
lambdas3 = np.arange(0,1.1,0.1)
alphas = np.arange(0,0.65,0.05)
exp3Res = doOnlineUpdate(trainingData,lambdas3,alphas,numStates)
avg_rms_exp3 = np.mean(np.min(exp3Res,1),1)

fig = plt.figure(figsize=(10,10))
plt.plot(lambdas3, avg_rms_exp3,linestyle='-', marker='o', color='r')
plt.xlabel('$\lambda$', fontsize=18)
plt.ylabel('RMSE', fontsize=16)
fig.savefig('figure3.jpg')