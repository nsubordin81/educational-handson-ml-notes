# Neural Networks

## from reinforcement learning to Q learning

when learning reinforcement learning with markov decision processes, we were going over storing the values of taking each action for a state into a Q table. 

another definition of reinforcement learning: branch of machine learning where agents submit actions and environments return observations which is often though of as the state of the system but coudl be some other type of observation, as well as a reward. The agent's goal is to determine the best action to take. usually this is desccribed as the agent interacting with a previously unknown environment and trying to maximize the total or overall reward.

### what is deep reinforcement learning doing? 

agent is using non-linear function approximators to estimate the action value function directly from the observations it is receiving form the environment. These nonlinear function approximators are rerpresented with a deep Neural Network. Then deep learning is used to find the best values for the parameters for the approximators. difference from supervised learning, the inference engine isn't returning a best guess label. when done in reinforcement learning context without labeled data the output is the best guess action to take instead. When deep RL is used for the entire RL pipeline this is known as 'pixels to actions' I guess for robotics where the state space is pixels because you are showing the robot a picture of what it is doing. You can otherwise give other things as the states, for robots this could be raw sensor data as well. The agent then takes the action it things based on the approximation will maximize its reward. Deep RL Networks results in intuitive, human like behaviors to be learned. They are good at things like path planning and imitating behaviors becuase the networks incorporate things like exploration and knowledge gathering. robots benefit from rl agents becuase it provides them a way to make sense of their environment which is hard to model in advance. 

Key insight to help you connect to what you have already learned: Deep Q Learning is approximating the Action Value Function that in your earlier implementation you were maintaining in that table. Instead of approximating it with the table though you will now do it with a neural network based on that function over your input state vector and parameter vectors. So, maybe kind of think of it like you were trying to figure out which state maximized the action value by keeping a running average, now there will be a gradient descent step that determines how to nudge the parameters to create something that does a better job of completing the task or not. 

### Pitfalls of using reinforcement learning with neural nets

Rolling history of the past data with a replay pool so the behavioor distrubtions is averaged over many of the previous states smoothing out learning and avoiding oscillations. Each step of the experience is then used in many weight updates. There is a target network also that represents the old Q function that will be used to represent the target loss at every step of training. Having a target network to compare against instead of having the network compare against itself prevents from value updates doubling down and spiring out of control. 

### deep mind able to beat humans

learned to play a bunch of atari games from scratch, exposed only to pixels. used a deep q network. the overall structure is that it is a deep neural n4etwork powering the function approximator, the agent. It takes in screenshots of the state of a game and then on the other side it outputs a vector of possible action values 2with the highest value being the one that corresponds to the action the agent should take at that time. 

the network is provided the game score at the end of each time  step for reinforcement. it starts out really bad with random actions but over time it learns to play the games well. 210/160 pixels with 128 possible colors per pixel. This is discrete but very large to process. 

the deepmind team did some preprocessing like taking the color away and making a square shaped screen size so they could perform some optimizations on GPUs that required square shaped screens, and then batching the screens in 4s so they coudl deal twith the fact that it was sequential. 

differently from the reinforcement learning with markov decision processes that they have already covered which will only provide one q value for a time step, corresponding to the one action that the approximate q funciton thinks should be taken, the neural network deep q network is capable of returning all the action values in a single forward pass. If you were to only return one value per time step , then you would have to run the network once per every possible action in the action space to know which one to take. this way you can choose one stochastically or choose the greediest one to know what to have the agent do. 

### Architecure

first there are two convolutional neural networks that the image input goes through. it allows the agent to exploit spatial relationships. The convoutional neural nets also take advantage of the fact that the input is four stacked frames in sequence to extract some temporal properties from the frames. The original deepmind was constructed with three convolutional nets followed by a fully connected hidden layer and then another fully connected linear output layer that produced the vector of action values. The other layers used relu as their activation function. 

### the difficulty of training it

training the network reqquires a lot of data, and even then it is not guaranteed to converge bedcause there is a high correlation between teh states and actions. the issue is that this makes a policy that is unstable and also ineffective. so what do you do? 

### Experience Replay

key idea: instead of discarding your state action reward next state tuple after every time step and action, you store the prior experiences in a replay buffer and then sample from it. 

A big reason this helps in deep q learning is that there is a strong correlation in a sequence of experience tuples because the action of one tuple impacts the state of the next one. this correlation could throw off learning, in a naive agent, they would process these experiences in order and it is likely that their learning would be impacted because they would attach significance to the cause and effect going on between the states and action pairs. 

if you store the experiences ina buffer and sample them at random there is no more correlation that the network can find between the experiences and you don't have the action value function oscillating because it is trying to learn from the correlation instead of the values. 

the example provided was learning to play tennis against a wall. So if you hit relatiavely straight and so you always got your ball back to the same place, it would reinforce your idea that your shot from that location was the best shot you could take. because of the correlation of taking the action and getting a good result the network woudl assign a high value to that action and it would just get larger in a feedback loop, failing to explore other states and try other actions and therefore learning the value disproportionately high for that action across all possible states instead of just where you happen to be experiencing it at the time. 

the way to overcome this with experience replay is to record a bunch of experiences to a sor tof database or offline area and then learn them in a batch. As a side benefit, you can then apply supervised learning approaches to the problem or your acn take advantage of the fact that you are saving yoru tuples to prioiritize tuples that are rare or more important to the problem than others. 

### fixed q targets

the second kind of correlation that q learning is susceptible to is temporal in nature. the goal of q learning as qa temporal difference problem solving technique is to approximate the td target, by reducing the difference between the true target, q(s,a) and the one we are approximating. we don't have the true qpi s a function we just ahve the R + gamma times the max of the qhat of the next state, with action and parameter. 

TD value the reward achieved in the current state plus a gamma term times the action value for the next state. That is supposed to correspond to the actual q function value for that given state and action, which we don't know. you are taking that TD target stand in, finding the difference between that and the current value and that is your TD error or how far away you are from the true value of the q function at that point. 

for the q learning update function, you started with an expected value of the squared error loss between the true q value and the approximate q value with a parameter. then it was shown how you could find the derivative of that to be the -2 times that difference. Then if you select w to be -alpha times 1/2 the original J(w) then you end up with cancelled out termsand just alfpha times the difference times the derivative with respect to q. I'm not as sure what these terms mean, need some calculus review. 

The important point made for the update function is that it is not mathematically correct to replace the qpi funciton in the derivative directly with the one that depends on the r and gamma and s' a and w terms, but you can get away with it in practice. This is only because we are making only small adjustments to w on each step and getting gradually closer to the target. If you set alpha to 1 and tried to get there all at once you would probabaly overshoot. 

There is another reason this isn't as big a deal for the traditonal reinforcement learning method of maintaining a table, because the table has a separate entry for each state action pair. this is not so much the case when we have function approximation because all the q values are tied together through the parameters. 

experience replay doesn't help this it helps a similar problem. the difference is that with experience replay you play the experiences out of order to address the correlation between successive experiences having similar values. the problem you have to address with fixed q targets is that the parameters belong to your current state and they also belong to your target so the fact that they are related means you are kind of in a cycle trying to chase your own values. 

good analogy for how the parameters affecting the target value affects your model as well: the donkey. If you are riding a donkey, dangling a carrot in front of it, it will cause the carrot to move randomly on every step and it will have to go in directions other than straight trying to figure out where to get the carrot and get frustrated and likely quit. However, if you were to stand in front of the donkey with the carrot and then move backwards in a straight line, the donkey will do amuch better job walking towards the carrot because the carrots movements and the donkey's movements are no longer coupled together. 

all you really have to do for the fixed q targets approach is make a copy of your parameter vector w for the target network, have several steps of leraning go by where you are updating the original w but the target has the fixed w parameters, and then at some point you update w for the next round of learning. 

