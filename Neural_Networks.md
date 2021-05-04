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

### My Notes From The Deep Q Learning Paper

key motivation, even though reinforcement learning theory works for agents in simple environments, it doesn't work as well when the agent needs to try to get control over a more realistic or complex environment. 

The agent's job gets harder, they have to make an efficient representation of the environment out of high dimensional s3ensory inputs and use that to generalize past experience to new situations. 

It turns out that animals and human agents solve control problems by using both reinforcement learning and also hierarchical sensory processing systems. we have a lot of data about the nervous systems of animals that shows parallels to the phasic signals that dopaminergic neurons emit and reinforcement learning algorithms specifically temporal difference. 

a limitation of using reinforcement learning agents to solve control problems has been they need domains where you can feasibly handcraft your features or you have state spaces that are both fully observed or low dimensional vs. continuous or infinite. 

Deep Q Networks learn from high dimensional sensory inputs and use end to end reinforcement learning. Atari 2600 games are the subject, the inputs are the pixels and the game score, and with just these two things the agent was able to beat out the performance of all prior algorithms and get to a level comparable to a professional human games tester. The Deep Q Network used to achieve this was used across 49 different games and the algorithm, network architecture and hyperparameters were able to stay the same. So you could say the deep q network is the first bridge between high dimensional inputs and the resulting actions an agent should take in the environment. this makes an agent that is capable of learning to e cel at a diverse array of challenging tasks. 

deep nueral nets have been getting more layers of nodes within them so they have more ways to represent the data that is coming in with sets of abstractions. deep convolutional neural nets are one of the ones that are used in deep q networks because they are capable of taking hierarchical layers of tiled filters to mimic the receptive fields that we use for vision, the 'feed-forward' processing in the visual cortex. this takes advantage of the fact that there are local spatial correlations in images and conv nets are also able to hands changes with the point of view of the image or the scale of it. 

For deep q the situation is the same as with reinforcement learning. an agent interacts with an enviornment and that is shown through observations, actions and rewards. The agent is trying to maximize their cumulative future reward. The goal is to use the deep convolutional neural network to approximate the optimal action-value function

<place to get the mathematical notation for the action value function.> 

This action value function is the maximum sum over the rewards that have been discounted by gamma at each  time step t. So for every time step you earn a reward and then for each reward you apply a discount because that reward came in the future. So say there was some behavioral policy pi that you were following, it makes observations and then recommends actions and those actions lead to rewards and transitions between states. You really want to get the most reward you can get across all the actions you can take while acting in the environment, but the actions you take in the future will be of less value than the ones you'll take right away so they have some discount. 
 
 there have been issues with reinforcement learning because it becomes unstable or divergent when a nerual network is used to approximate the action value function. This is also true for other non linear approximators. There are many reasons for this: 
	1. there are correlations present in the sequence of observations
	2. small updates to the action value function itself can sifnificantly change the policy and therefore the data distribution itself. 
	3. there are correlations between the action-values and also the target values represented by the reward plus the gamma discounted value of the maximum action value for the following state. 

There is a novel varieat of q learning introduced in the paper which addresses the instabilites found with these other nonlinear attempts. 

	1. Experience replay - this randomizes over the data, the observations don't get processed in the order in which they temporally appear. This helps get rid of issues with sequantial correlations between observations and it smoothes over the changes in the data distribution because actions don't get amplified because they are similar and happen together. 
	2. then there is the fixed parameter update which keeps some target values fixed and only updates them every once in a while, and then has the update adjust the action values towards those slowly changing targets. this will reduce correlations between the action value and the target action value. 

there are other stable applications of neural network s to reinforcement learning, one is called neural fitted q iteration. abut this method requires you to repeatedly train the networks from scratch on hundrends of iterations. they are too inefficient to be used successfully with large neural networks. 

the approach taken by deep q learning is to parameterize the approximate value function q(s, a; thea). this is done with the deep convolutional neural network  and the thea are the parameters, the weights if you will of the q network at ech iteration. to perform the experience replay the experiences are stored in an experiance tuple called et and consist of the state, the action, the reward, and then the following state. one of these such tuples is stored for each time step in a dataset Dt which has e1 through et inside. while learning there are q learning updates applied on some minibatches of experience (so some group of timestep together) and those are drawn from the Dt dataset uniformly and at random to remove the chance for correlation. 

updating the q learning for each iteration of the network (each time it processes one of these minibatches of experiences) is done through a loss function given by 

<show the loss function>

so that e means it si the expected value in terms of the state, action, reward, next state tuples drawn with uniform random distribution from Dt
of the square of the quantity given by
the difference between
the sum of the reward and the approximate value function of the following timestep using fixed theta and 
the approximate value of the current state and action with a non fixed theta. 

it is a difference because the value of the reward combined with the action value of the next step is standing in for the optimal action value. it is reward the agent stands to gain, and the amount of value that is currently had is the other term. So we are essentially saying, "you haven't got all the reward yet, try to minimize how far off you are from getting that reward."

we square be cause this is the squared error and this is a convenient mathematical trick which helps the data be more normal and also not negative

the data transformations required to make the deep q network work were impressive in their own right. so they started with high dimensional data that was 210 by 160 resolution color video runnin at 60 frames per second and they converted it to 84 by 84 by 4 microbatches of square images. 

the neural network learning agent was provided with only the visual images experienced by the game and the range of actions available to the agent for each game, but not the ways in which the actions would affect the visuals, as in how you could use the actions to make the things you wanted to happen on the screen happen or even what that was. 

there were two indices of learning used to measure the progress of the agent. first there was an average score per episode maintained for the agent and the second part was an average predicted q value. 

the performance of the deep q network is undeniably good. it outperformed the best other neural network based reinforcement learning approaches that had been measured on the atari games and then it also performed consistently well as the human equivalent on many of the games. 

to understand why the DQN does so well the team that created it looked at the representations that it learned when performing on the space invaders game. There is a technique used called t-SNE. this algorithm maps the DQN representations of states that we can perceive as similar to nearby points on a visual ization. the algorithm also was aable to generate similar embeddings for a dqfn state that are similar in terms of the reward they can provide but don' tlook similar to human viewers of the game. this tsne visualization helped confirm that the network was taking in high dimensional data and then figuring out a way to make adaptive behavioural decisions from it. It also turns out that the dqn representations aren't limited to data that has come about through solely its own policy for behavior. other agent and human player game states were fed in and the representations from the last hidden layer were recorded and put into tSNE. 

deep q networks were shown to exhibit some amount of mid to longer term strategy such as in breakout where they figured out the optimal strategy was to get the ball hidden behind all the bricks. However, a really longer term temporal strategy like one that requires taking no action or a few vvery specific actions that don't yield their value until a long way off, those are still a big challenge for deep q networks. 

the deep q network presented in the paper draws on the biological evidence that as an animal is engaging in perceptual learning if it gets reward signals then that may influence the characteristics of representations within the visual cortext. In the same way, the rewards from the reinforcement learning cause the convolutional neural net to figure out additional representations of the enviornment tha tit finds useful to maximizing values. The replay algorithm is also evolutionalrily based and was critical to getting the results tha the deep q learning approach had. In humans and mammals the hippocampus might actually be responsible for what the replay algorithm is doing, storing and representing the recently experienced transitions. When we rest or our brains are idling after some experience, the hippocampus will siometimes trigger the reactivation of recently experienced learned behaviors. Then 'value functions' are updated as the basal ganglia interacts with this hippocampal phenomena and that cases updates to optimize the experience to a new reward or goal. 



