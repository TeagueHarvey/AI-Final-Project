import gym
import random
import math
#r(theta)=-.2,.2
#r(dTheta)=-2.5,2.5
class solver:
    

    def __init__(self, pBuckets, vBuckets, learningRate, explorationRate):
        self.pBuckets=pBuckets #number of buckets to discretize into
        self.vBuckets=vBuckets
        self.minP=pBuckets
        self.maxP=0
        self.minV=vBuckets
        self.maxV=0
        self.lr=learningRate
        self.er=explorationRate
        self.maxSteps=5000
        
 
        



    def discretize(self, observation):
        position=observation[0] #[-1.2,0.6]
        velocity=observation[1] #[-0.07,0.07] 
        normP=((position+0.3)/1.8+0.5)*self.pBuckets 
        normV=(velocity/0.14+0.5)*self.vBuckets
        if(normP >= self.pBuckets):
            normP=self.pBuckets-1
        if(normP < 0):
            normP=0
        if(normV >= self.vBuckets):
            normV=self.vBuckets-1
        if(normV < 0):
            normV=0
        return int(normP),int(normV) #discretized


    def initQ(self): #initializes Q table with equal probability for actions
        Q={} 
        for position in range(self.pBuckets):
            for velocity in range(self.vBuckets):
                state=(position, velocity)
                for action in range(3):
                    Q[(state,action)]=0.33 #equal weight for all actions
        return Q


        
        


    def Reward(self, state): #rewards for reaching new location or velocity
        position=state[0]
        velocity=state[1]
        r=0
        if(position < self.minP):
            r=r+(self.minP-position)
            self.minP=position
        if(position > self.maxP):
            r=r+(position-self.maxP)
            self.maxP=position
        if(velocity < self.minV):
            r=r+(self.minV-velocity)
            self.minV=velocity
        if(velocity > self.maxV):
            r=r+(velocity-self.maxV)
            self.maxV=velocity
        return r
        
            
        

    def getAction(self, Q, state):
        r=random.randint(0,100)
        leftProb=Q[(state,0)]*100;
        rightProb=leftProb+Q[(state,2)]*100
        rExplore=random.randint(0,100) #need separate random to not affect going left
        if(rExplore < 100*self.er): #take random action
            return random.randint(0,2)
        if(r < leftProb):
            return 0 #left
        if(r > rightProb):
            return 2 #right
        return 1 #we do not move
         
        

    #returns highest of the probabilities
    def maxQ(self, state, Q):
        left=Q[(state,0)]
        nothing=Q[(state,1)]
        right=Q[(state,2)]
        return max(left, nothing, right)
        
        
        

  

    def valueIteration(self, episodes):
        originalER=self.er
        Q=self.initQ() #Q-table
        stateCounts=self.state_count_init() 
        env = gym.make('MountainCar-v0')
        for i_episode in range(episodes):
            observation = env.reset()
            self.er=originalER-originalER*(i_episode/100) #decay er
            for t in range(self.maxSteps):
                #env.render()
                x=observation[0]
                if(x >= env.goal_position): #SUCCESS
                    er=0.0 # we no longer need to explore
                    return i_episode, t
                state=self.discretize(observation)
                explore=random.randint(0,100)
                action = self.getAction(Q, state)
                observation, reward, done, info = env.step(action)
                newState=self.discretize(observation)
                count=stateCounts[(state,action)] #number of times we have seen this state action pair
                stateCounts[(state,action)]=stateCounts[(state,action)]+1
                LR=self.lr/float(count+1) #learning rate is proportional to how many times we have seen this state action pair
                update=LR*(self.Reward(state)+self.maxQ(newState,Q)-Q[(state,action)]) #update Q-value
                if(action==0): #left
                    Q[(state,action)]=Q[(state,action)]+update
                else: #right
                    Q[(state,action)]=Q[(state,action)]-update
        env.close()
        return 99999, 99999 #failure

    def state_count_init(self):
        counts={} #key: (state,action) val: number of times state has been encountered
        for position in range(self.pBuckets):
            for velocity in range(self.vBuckets):
                for action in range(3):
                    counts[((position,velocity),action)]=0
        return counts
                




def main():
    pBuckets=20
    vBuckets=50
    lrs=[0.1,0.2,0.3,0.4]
    ers=[0.1,0.2,0.3,0.4]
    for lr in lrs:
        for er in ers:
            s=solver(pBuckets, vBuckets, lr, er)
            results=s.valueIteration(100)
            print("Learning Rate:"+str(lr)+" Exploration Rate:"+str(er))
            print(results)
               
            
    
    


if __name__ == "__main__":
    main()
    
