import gym
import random
import math
#r(theta)=-.2,.2
#r(dTheta)=-2.5,2.5
#keep min and max states in order to reward

pBuckets=20
vBuckets=50
minP=pBuckets
maxP=0
minV=vBuckets
maxV=0
lr=0.25
er=0.1
maxSteps=5000
def getRange():
    env = gym.make('MountainCar-v0')
    minTheta=9999
    maxTheta=-9999
    minDTheta=9999
    maxDTheta=-9999
    for i_episode in range(100):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            theta=observation[0]
            dTheta=observation[1]
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
    position=observation[0] #[-1.2,0.6]
    velocity=observation[1] #[-0.07,0.07] 
    normP=((position+0.3)/1.8+0.5)*pBuckets 
    normV=(velocity/0.14+0.5)*vBuckets
    if(normP >= pBuckets):
        normT=pBuckets-1
    if(normP < 0):
        normP=0
    if(normV >= vBuckets):
        normV=vBuckets-1
    if(normV < 0):
        normV=0
    return int(normP),int(normV) #discretized

#returns Q-table with all left-prob set to 0.5
#Q(state)=Q(state,left) Q(state,right)=1-Q(state)
def initQ():
    Q={} #key:(theta,dTheta) val:(prob(go_left))
    for position in range(pBuckets):
        for velocity in range(vBuckets):
            state=(position, velocity)
            for action in range(3):
                Q[(state,action)]=0.33 #equal weight for all actions
    return Q

#returns dict with key:state val:max_(all actions) reward(state)
def initV():
    V={}
    
    


def Reward(state): #we want max speed in position 4
    global minP
    global maxP
    global minV
    global maxV
    position=state[0]
    velocity=state[1]
    r=0
    if(position < minP):
        r=r+(minP-position)
        minP=position
    if(position > maxP):
        r=r+(position-maxP)
        maxP=position
    if(velocity < minV):
        r=r+(minV-velocity)
        minV=velocity
    if(velocity > maxV):
        r=r+(velocity-maxV)
        maxV=velocity
    return r
    
        
    

def getAction(Q, state):
    r=random.randint(0,100)
    leftProb=Q[(state,0)]*100;
    rightProb=leftProb+Q[(state,2)]*100
    rExplore=random.randint(0,100) #need separate random to not affect going left
    if(rExplore < 100*er): #take random action
        return random.randint(0,2)
    if(r < leftProb):
        return 0
    if(r > rightProb):
        return 2
    return 1 #we do not move
     
    

#returns higher of the two probabilities (l or r)
def maxQ(state, Q):
    left=Q[(state,0)]
    nothing=Q[(state,1)]
    right=Q[(state,2)]
    return max(left, nothing, right)
    
    
    

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
    stateCounts=state_count_init() 
    env = gym.make('MountainCar-v0')
    for i_episode in range(episodes):
        observation = env.reset()
        #er=get_exploration_rate(i_episode)
        for t in range(maxSteps):
            env.render()
            x=observation[0]
            if(x >= env.goal_position): #SUCCESS
                print("The car made it after "+str(i_episode)+" episodes in "+str(t)+" time steps") 
            state=discretize(observation)
            explore=random.randint(0,100)
            action = getAction(Q, state)
            observation, reward, done, info = env.step(action)
            newState=discretize(observation)
            count=stateCounts[(state,action)] #number of times we have seen this state action pair
            stateCounts[(state,action)]=stateCounts[(state,action)]+1
            lr=0.25/float(count+1) #learning rate is proportional to how many times we have seen this state action pair
            update=lr*(Reward(state)+dr*maxQ(newState,Q)-Q[(state,action)]) #update Q-value
            if(action==0): #left
                Q[(state,action)]=Q[(state,action)]+update
            else: #right
                Q[(state,action)]=Q[(state,action)]-update

        print("minP:"+str(minP)+" maxP:"+str(maxP)+" minV:"+str(minV)+" maxV:"+str(maxV))
            
    #print(Q)
    #print("timeSteps="+str(tsTotal)+" 100s="+str(completed))
    avgTime=tsTotal/20
    env.close()
    print(Q)
    return avgTime

def state_count_init():
    counts={} #key: (theta, dTheta) val: number of times state has been encountered
    for position in range(pBuckets):
        for velocity in range(vBuckets):
            for action in range(3):
                counts[((position,velocity),action)]=0
    return counts
            


def get_exploration_rate(t):
    print(t)
    return max(0.0, min(1.0,1.0-math.log10((float(t)+1.0)/10.0)))

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
    valueIteration(0.1,1.0,0.0,2000)
    


if __name__ == "__main__":
    main()
    
