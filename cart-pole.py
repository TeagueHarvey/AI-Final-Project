import gym
import random
#r(theta)=-.2,.2
#r(dTheta)=-2.5,2.5
def getRange():
    env = gym.make('CartPole-v0')
    minTheta=9999
    maxTheta=-9999
    minDTheta=9999
    maxDTheta=-9999
    for i_episode in range(100):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            theta=observation[2]
            dTheta=observation[3]
            if(theta > maxTheta):
                maxTheta = theta
            if(theta < minTheta):
                minTheta=theta
            if(dTheta > maxDTheta):
                maxDTheta = dTheta
            if(dTheta < minDTheta):
                minDTheta=dTheta
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print("minTheta="+str(minTheta)+" maxTheta="+str(maxTheta))
    print("minDTheta="+str(minDTheta)+" maxDTheta="+str(maxDTheta))

    env.close()
    



def discretize(observation):
    theta=observation[2] #[-.2,.2]->6 buckets
    dTheta=observation[3] #[-2.5,2.5] -> 12 buckets
    normT=(theta/0.4+0.5)*6.0 #[0,6]
    normDT=(dTheta/5.0+0.5)*12.0 #[0,12]
    if(normT > 5):
        normT=5
    if(normT < 0):
        normT=0
    if(normDT > 11):
        normDT=11
    if(normDT < 0):
        normDT=0
    return int(normT),int(normDT) #discretized

#returns Q-table with all left-prob set to 0.5
#Q(state)=Q(state,left) Q(state,right)=1-Q(state)
def initQ():
    Q={} #key:(theta,dTheta) val:(prob(go_left))
    for theta in range(6):
        for dTheta in range(12):
            Q[(theta,dTheta)]=0.5 #left-prob is 0.5
    return Q

#returns dict with key:state val:max_(all actions) reward(state)
def initV():
    V={}
    
    


def Reward(state):
    theta=state[0]
    if(theta==3 or theta==2):#most balanced
        return 1.5
    return 1

def getAction(Q, state):
    leftProb=Q[state]*100;
    r= random.randint(1, 100)
    if(r < leftProb):
        return 0 #we go left
    return 1 #we go right

#returns higher of the two probabilities (l or r)
def maxQ(state, Q):
    left=Q[state]
    if (left > 0.5):
        return left
    return 1-left
    
    

#Q-update: Q(state,action)=oldQ(state,action)+lr*(reward+discount*maxQ_action(newstate)-oldQ(state,action))

#lr=0 totalsteps=1034, 1090
#lr=0.1 totalsteps=1231, 3675
#lr=0.5 totalsteps=3844

def valueIteration(learningRate, discountRate, explorationRate, episodes):
    Q=initQ() #Q-table
    lr=learningRate #learning rate
    dr=discountRate #discount rate
    er=explorationRate
    tsTotal=0 #total number of timesteps
    completed=0 #number of times to 100 steps
    env = gym.make('CartPole-v0')
    for i_episode in range(episodes):
        observation = env.reset()
        er=explorationRate-explorationRate*(i_episode/episodes)
        for t in range(200):
            env.render()
            state=discretize(observation)
            explore=random.randint(0,100)
            if(explore < er*100 and i_episode<30): #get random action
                action=random.randint(0,1)
            else: #get action normally
                action = getAction(Q, state)
            observation, reward, done, info = env.step(action)
            newState=discretize(observation)
            update=lr*(reward+dr*maxQ(newState,Q)-Q[state]) #update Q-value
            if(action==0): #left
                Q[state]=Q[state]+update
            else: #right
                Q[state]=Q[state]-update
            if(i_episode > 180): #we only start counting once the agent has learned something
                tsTotal=tsTotal+1
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
    #print(Q)
    #print("timeSteps="+str(tsTotal)+" 100s="+str(completed))
    avgTime=tsTotal/20
    env.close()
    print(Q)
    return avgTime


def main1():
    learningRates=[0.1,0.2,0.3]
    explorationRates=[0.0,0.1,0.2]
    total=0 
    for lr in learningRates:
        for er in explorationRates:
            total=0
            for i in range(3):
                total=total+valueIteration(lr,1.0,er,50)
            avg=total/3
            print("Learning Rate:"+str(lr)+" Exploration Rate:"+str(er)+" Average Steps:"+str(avg))

def main():
    learningRate=0.1
    explorationRate=0.0
    time=valueIteration(learningRate,1.0, explorationRate,50)
    print(time)

    


if __name__ == "__main__":
    main()
    
