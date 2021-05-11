# My Notes From The Deep Q Learning Paper

## Notes To Self - This is Too Much A Verbatim From The Paper, You Won't Remember It, Try Again In Your Own Words Later

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

## Methods 

preprocessing - the input was frames from atari games, every frame could have 210 vertical pixels by 160 horizontal ones. Then there are 128 different colors that each of the pixels could be, so the input space for just one of these frames was 210x160x128. That means a lot of memory to store the values and lots of computation added for every transformation. first there were steps taken to deal with atari's pixel flickering issue. Atari games, being old and having limitations on the hardware for how many sprites it could paint per frame, sometimes accounted for this by not painting sprites every frame but instead every other frame. so there is a max() operation done between every two sequential frames to capture sprites that might have been erased between frames. 

There is a step also to remove the luminance channel from the RGB color frame, I guess because that doesn't provide very much value for the extra representation it requires, and then the width and height are rescaled to 84x84. there is a function that applies these prprocessing steps to m most recent frames from the gameplay and then stack them to create the input that the Q function uses. The m used for the deep q network that was published was 4 but it didn' thave to be, you could choose different values of m. 

you can view the cource code for the dqn on https://sites.google.com/a/deepmind.com/dqn

model architecture - There is a very important point made here and I didn't get it the first time I read it. They mention that there are different ways to parameterize a Q network, and that the way that was most common before they made theirs was to have both the state space and one selected action from the action space as input and then a scalar value representing the Q value for that action as the output. They then explain that you have to do a forward pass through the neural network for every action value in the action space on every state. that means you have a cost that scales linearly with the size of your action space. 

their workaround for this was to have a different neural net architecture that took in just the state representation and not the action and then instead of just outputting a single scalar value for the actions, it outputs a vector of values, one for each possible action, and then the agent can choose to use the one that has the highest value for exploitation or at randomly for exploration. Only one forward pass required. 

Some more architecure details described in plain english to the extent possible. you start with the 84x84x4 input tensor that comes from the preprocessor and represents 4 frames of the game. then there is a hidden layer next which convolves the 32 filters of 8x8 with a stride of 4, I need to review the convnet lectures from my other course to make sure I understand that, and you apply a recitfier nonlinearity for activation. then you use 64 filters in another hidden layer, the filters are smaller, only 4x4, and your stride is shorter too this time with only 2, and you also use ReLU? for activation. then there is one more convolution layer that has 64 filters 3x3 with stride 1 followed by a rectivfier. At the and of all that you have a fully connected hidden layer with 512 rectifier units. the final output layer after the last hidden layer is a fully connected layer that will give an output for each of the valid actions that can be taken. For info, on the 49 atari games this network was first applied to, there were a variety of numbers of actions ranging from 4 to 18.

training details - the deep q network was purposefully only used on the 49 games that already had other methods being practiced on them so there could be a basis for performance comparison. 

It is important to the team building the deep q network that they showed that even though they didn't use the same network to train for all the games, they did keep the architecture, learning algorithm, and the hyperparameters constant, so it was essentially a copy of the same network that they were training each time. 

While there was no true modification to the games when the trained networks were playing them, they did have to make a modification to the reward structure of the games while training them. this was to scale the scores of the games because in some games the scores were high values and others lower, so tehy were normalized and clipped to a range between -1 and 1. This helped to limit the scale of the error derivatives and not have to adjust alpha for different games. However, this means if the agent was used in an environment where the scores were not within this range it may not perform well. The network also took as input the number of lives remaining that the atari emulator emits, and this was used to determine when an episode was over. 

For the experiments they used RMSProp. this is a form of mini batch gradient descent. They provided some slides on it. So some of the challenges to stochastic gradient descent convergence, the gradient that is calculated has different values to adjust each of the weights by. Also, the amount of adjustment to each weight varies as the learning process continues. You want to be able to choose a single global learning rate over the whole network so you'd rather these weight adjustments were more incremental in nature and had constant rates of change across them. The only too you have when you learn the full batch of observations at once is to use the sign of the gradient. you can't change the magnitude of the weight updates you apply. There is a method called rprop which keeps the idea from full batch learning of changing the gradient sign, and combines that with changing the step size for a weight by multiplying it by a signed constant. instead of having the step size be a constant 1 for all the weights, they could be any size, but recommendation is to be less than 50 and more than 0.0001. when the learning rate is small you average the gradients over sucessive mini batches, that is the stochastic gradient descent. you want the weight not to be affected by encountering one stray gradient value if the rest are fairly consistent. rprop though doesn't do this right, because across the time steps if the same value was encountered multiple times it would have a stronger effect on the weight. example, say the gradient was +1 for nine successive mini batches and then there was one gradient that was -0.9. Instead of recognizing that this -0.9 means that the other adjustments are likely coincidental and the weight hasn't stabilized yet, the rprop method will add the gradients to the weight proportionate to their occurrence, which means that weight will be increased 9 times and decreased 1 time when it should not be increased so much. 

So rms prop seeks to give the advantage of adjusting timesteps that rprop provides for robustness, along with still having minibatches like stochastic gradient descent uses to efficiently converge to a local maximum (minimum? I forget), and then finally average the gradient values over successive mini batches effectively so that the weights don't grow or shrink too much in response to the gradient signal. 

rprop uses the gradient but the gradient that it uses is also scaled by the size of the gradient. the problem with this is that the gradient is a different size for each mini batch the model sees. so the insight of rms prop is to instead of scaling by a variable term like the gradient, scale by something that is a moving average of the squared gradient for every weight. that is accomplished by having 90 percent of the mean square gradient by the mean square from the last mini batch and then 10 percent be the new gradient with respect to the current minibatch . you then take the square root of that mean square term and you have something that is scaled consistently and will not allow the weights to grow or shrink too much like rprop does. 

there is some guidance about whether or not it is necessary to use rmsprop or rmsprop with momentum or lecun's model ro something else that has surfaced in the last 5 years or so since this paper was written. generally it says use the full batch method for small data sets or data sets that don't have data that is the same a lot. For larger, redundant data sets though the recommendation is to use rmsprop. 

the policy for behavior was epsilon greedy during training. and this epsilon value was annealed linearly from 1 to .1 over a million frames (I guess /4 so 250,000) and then it was fixed once it reached 0.1. I don't know what constant factor that would put the decay of epsilon at but you could figure it out with some algebra. is it 0.9999996? hmm, maybe, math is hard. Once epsilon stabilizes at .1 the agent continues for 49 million more frames, which they did the math for us, total time training is the equivalent of playing the atari game for 38 days straight (without pauses for water or bathroom breaks haha lucky machines). The experiential replay memory was 1 million frames, which we will probalby get to later. 

there is frame skipping trick used in the domain of machine learning for atari games. instead of selecting actions in every single frame, the agent allows the emulator to run forward k frames before selecting an action. going from frame to frame is much less expensive than the compute required to select an action and so when k is 4 that is a 4x improvement in performance. The agent can get through that many more games and the runtime overhead stays the same. 

hyperparameters and optimization params were arrived at through informal search using pong breakout, seaquest, spac3e invaders and beam rider. There could have been a grid search done to find the best performing combination of hyperparameters over a range of values but this was not opted for because it would be computationally very expensive. Once arrived at, the hyperparameters stayed the same for all other games. 

experimental setup was to have the minimal prior knowledge for the agent. the agent wasn't given access to the internal state of the emulator, only things that a human would be able to ascertain, such as an image representing the current frame of play, the score of the game (without modifying it for the test split as it was modified during training), the number of lives the agent had left in the game, and the number of available actions without describing to the algorithm in any way what effects those actions would have (like "this button makes your character move in an upward direction"). 

### Evaluation Procedure

agents played each game 30 times for 5 minutes at a time during evaluation. initial state of the game was determined from a random seed. and there was an epsilon greedy policy used where epsilon was fixed at a value of 0.05. that helps reduce the chance that an agent won't explore enough. The baseline to beat for evaluation was an agent acting randomly in the space. there was an action chosen at random by the baseline agent at a rate of 10 times per second or every 6th frame. that is becausae the fastest  human player would be able to press the 'fire' button about this many times, so for a handful of games it was necessary to limit the speed at which the agent could execute actions in case high performance could be achieved through extremely fast button mashing. If a random agent was allowed to select a random action every frame, then this had the effect of making the DQN normalized performance higher across six specific games by about 5% and it made it outperform the human tester in these 6 games by a lot (just because the ability to do random actions from a small subset of actions really fast is more valuable than thinking about and executing the right action).  

human testers used the same engine for emulating atari games that the agents did and they played in conditions that were similar for example they weren't permitted to ppause their game or reload from a saved state. the emulator was set to the frequency that an old atari game would ahve run at, the CRT tv standard of 60 frames per second. there was no sound feedback to either human or agent players. the human performance was measured as the average rewards achieved from 20 episodes of the game that were ach 5 minutes long after having practiced for 2 hours (I guess the equivalent of the agent having trained on the game) 

### Algorithm

the insight of deep q learning was that this atari emulator setup allowed for the framing of the tasks in a way that would fit into the formal definition of a markov decision process. you have actions and observations and those lead to rewards and changes in the state over a series of timesteps and you can set the objective of taking the sequence of actions that would maximize your expected discounted reward. actions are chosen from an action space and then the emulator modifies its internal state and the game score and then provides info back to the agent about the new game state and also the reward which is a reflection of the change in game score. This can be stochastic in nature or deterministic, the agent doesn't know how the state transitions are managed in the emulator only that there are new frames shown to it as a result of time ssteps elapsing and actions being taken. The game score could depend on a longer sequence of game steps than just the current screen, so the reward could be dependent on a series of states and not get updated until several actions are taken. it may take several thousands of time steps to elapse before the game score reflects a reward that is useful to the agent. 

emulator states are perceptually aliased, which means you can just look at the current screen and understand enough about the state. So the actions and observations are provided as sequences that are put into the algorithm and then the game strategies are learned depending on the sequences that are put into it. There is assumed to not e an infinite sequence scenario where the reward may never be realized always a finite one. So that means you can have a Markov Decision Process to try and find the optimal policy, action value function. Every state in the MDP would be a sequence of observation and action and next observation tuples leading to an incremental reward. you can have the sequence St as the representation of the state at a given time step t. 

in the MDP then the agent is tryign to interact with the atari emulator to select actions taht will amximize the future rewards represented by the game score. you ahve to gamma discount the rewards ( this is .99 in deep q paper). and you can get the discounted return by summing the returns from the current time step to the T timestep where the game erminates and discounting by gamma to the power of the timestep so future returns are appropriately falling off. 

you then try to get the optimal action value function as the maximum expected return following any policy, defined by the maxiumum expected value seeing a sequence s and then taking action a following policy pi the maximum expected reward. 

There is a Bellman equation that has already been proven which states that so long as you know the optmial value of the state action pair following the current one for every action that you could take from the current state, then the best strategy would just be to select the action a' which will maximize the current return of a summed with the discounted return of the following state. 
o for reinforcement learning, what is usually done ois to estimate the second part of the bellman equation, the action value function Q by doing an iterative update. As you go through time steps and those time steps approach infinity, then the estimated values function will actually converge to the optimal policy. 

The problem with doing this reinforcement learning approach is that it won't generalize to action value functions in other situations. For every sequence of actions, every task you might apply the approach to, you need to spend some time going through the sequence to learn the estimated action value function. Since you don't want to actually exhaustively go through all possible values of the sequence in your goal of figuring out the optimal path, you instead might want to use function approximation to estimate the action value function. you would use a linear function approximation for typical reinforcement learning problem but then you coudl use a nonlinear one for truly nonlinear problems. 

so the non linear function approximator used in the deep q is the neural network approach known as a Q network. The Q network works by adjusting the parameter vector thea at each iteration i to reduce the mean squared error in the bellman equation. the optimal target values are substituted with approximate target valus using a parameter that was part of a previous iteration. you end up with loss functions that chagne for each iteratoin i. Big difference for supervised learning is that the targets depend on the network weights instead of like in supervised learning where they are ffixed and independent from the network weights. Since the 'label' is really an approximate action value estimate from an earlier time step in the sequence, we are aiming for a moving target. The trick is to make the target move more slowly. apparently the variance term of the targets is also part of the loss function, but one that is not involving the parameter vector that needs optimization so it can just be ignored (that's kind of neat, just ignoring a term because it doesn't get affected by differing values of the parameter vector). 

you then take the derivative of the loss function with respect to the weights to get the gradient. this is now the expected value with respect to a tuple of state, action, reward, next state. of the current state/action pair's reward plus a gamma discounted maximum value of the approximate q function for the state we end up in and the action taht we would take there, given we are using a parameter vector that is fixed to some value from a few or several time steps back, minus the value of the q function for the current state action pair and the most recent set or parameter weights) all taken as the derivative with respect to the apprimate action value function with current weights. 

So generally you don't try to figure out the full expectation, I guess this would be the dynamic programming approach. instead you use stochastic gradient descent to incrementally update the weights at every time step, then update your expectations, then eventually updating the fixed parameters to use the latest parameter weights. There is no model involved the reinforcement task is solved using samples that are observed from the atari emulator. This means you don't have to explicitly estimate the rewards and the dynamics of transition for every state. It doesn't use the same  policy that it is approximating during the training. it will learn the greedy policy by using an epsilon greedy policy that learns from exploration. 

#### Deep Q Algorithm

the agent will use an epsilon greedy policy based on Q. a function phi is used to create a fixed length 'history' of images to feed into the forward pass. there are 2 ways in which the q learning algorithm has to be modified so that using a neural network will not lead to divergence. you have experience replay first where the agent's experiences at each time step are stored as state, action, reward, next state tuples. in a data set Dt. This dataset gets pooled over sets of episodes into replay memory. The algorithm's inner loop will set mini batch updates to the samples of experience that are drawn at random from the pool of samples. Advantages of experience replay are that you can run the same experience for multiple weight updates so that is more efficiency of data. also you can break the correlation between sequential updates where the subsequent state is the state that was reached  so you get less variance. A third advantage is that in sequential updates one decision reinforces the next wso you are more likely to get caught in feedback loops where you only explore one part of the space. Experience replay just because it is randomly sampling is more likely to create an even distribtuion of the agents behavior to learn from, learning is more likely to converge because it won't be oscillating because the agent is going very far in one direction and then very far in the other. Experience replay has to be off policy which is part of why Q Learning was chosen as the algorithm. You ahve to do it off policy because your agent has several experiences first and then they are saved, they aren't directly learning from each of those experiences right away, so they can't be updating that policy in place. the algorithm only stores n experience tuples in the replay and then takes from D at random when updating. the memory buffer doesn't know what is an important transition of state from an unimportant one and overwrites the old transitions with the more recent ones. also uniform sampli ng will give equal importance to all transitions in the replay memory as well. you'd probably want ot give more important to transitions that you could learn more from, keep them in the memory longer and select them more often if you wanted to improve this algorithm. 

The other modivication is the separate netowork used to generate the targets, the y vector in the q learning update. for every c updtates we perform we clone the whole network to obtain a target network Qhat adn the qhat network generates the q learning targets for the next set of C updates to Q that makes the algorithm more stable than in regular qf learning because in the standard one if you increase the q value of a state and actoin then you are implicitly increasing the q value of the next state for all actions since values depend on earliear arcionts and so the target, the overal q value fucntion will have changed so there could be divergence or oscillations in the policy. if you ahve an older set of parameters to generate the targets then you have a delay introduced between when Q is updated and when the update has an effect on the targets. So the correlation is still there but it is weaker. 

last thing they clipped the error term from the the update function to be between -1 and 1. this is because the loss function's derivative for x is -1 for all values that are negative of x and 1 for all positive values. so if the squared errror living between those value means that you are essentially using absolute value loss function which makes the model more stable. I am not as sure how. I guess when you clip extreme errors then the gradients don't move the weights around too much so they have an easier time converging. 

#### Analysis of the workspace code: 

So there is an Agent class in dqn_agent.py and the model.py code. I will look at those in a minute, but first the cell that is actually running the trained agent. 

##### The Code that Runs the Agent:

the code in the notebook that runs that trained agent initializes a new agent to the right n umber of states and actions for the lunar lander environment, 8 states and 4 actions, there is a seed parameter but we aren't setting one. 

Then, we reset the lunar lander environment - this resets the environment and returns an initial observation. This is used either when calling fo rthe first time or after the episode of that environment's task returns 'done'. typical patterns looks like we reset the environment in some outer loop and then loop through the steps in that environment until the environment step returns 'done'

we set img to the return value of `plt.imshow(env.render(mode='rgb_array'))` which is an AxesImage object. This will allow us to update, display and rescale the image that we are displaying the game state on. imshow just says to take the data you pass it and display it on a 2d raster surface. the open ai gym environment has an rgb_array mode for its display, which uses the gym.envs.classic_control.rendering.Viewer object. the viewer accepts width and height on initialization and it allso accepts bounds that are a scaled quantity with respect to the width and height to give the actual size of the screen in pixels. The viewer looks to be something of a canvas object, and when it is passed the rgb_mode flag, then it will use pyglet.image.get_bugffer_manager().get_color_buffer() and it will get the immage data from the buffer and use numpy's frombuffer command to transform that data into an array of integers and then it reshapes the array to an array with buffer.height x buffer.width x 4. Not sure if the 4 is r,g,b,a or why it is 4 of them. So I guess this is just an array that the imshow will know how to draw to the screen based on their contracts with each other. 

then we iterate through an array from 0 to 199, where we: 
have the agent act adn record the action into 'action'
we update the data in our image object with the current value of env.render, assuming it has changed because our agent has acted. 
we turn off the plot axis because we are displaying a raster image of game state not an actual plot of anything
we then display the data we have written to the image by passing plt.gcf (get current figure) to the display object that we created during imports. this is coming from pyvirtualdisplay we imported Display and then set it and started it at the top of the notebook, but also looks like since we are in an ipython notebook we might have just completely overriden that by just importing display directly from IPython afterward, that is a little confusing to me. 

in any case we then call clear_output on the display, but with a flag of `wait=True` which will not actually clear the output until you have new output avaialble. This is very much the typical pattern for display buffers, draw to the new one while leaving the old one displayed and then swap them. 

The final step in the running code is to actually take the step in the environment with the action that you got from the agent.act policy function and get the reward and state from that. I guess you ignore the reward here because you've already trained the agent? it does not continue updating its policy so the reward isn't very useful anymore. Seems like you could let it continue learning if it was your goal but I guess that is a whole other aspect of this that we haven't worked on yet you'd have to still be exploring more and also have the dual network approach running as you had the agent navigated the space. 

If you get done from the environment step function then you are done with this episode so you stop and quit. Looks like you only get 200 chances to land this lunar lander module. 


##### The Code That Trains the Agent

training starts with the dqn function, which takes in a number of episodes to train on, a maximum timesteps per episode, a start value and a min value of epsilon for exploration, and a decay rate for epsilon. epsilon is going to start at 1 for each episode of the task and go down to .1 in the default set of hyperparams, adn the decay rate is .995, so in other words it is linearly losing .005 every step, and that means it will take 200 timesteps to go from totally random exploration of actions to being greedy 90% of the time and exploring only 10% of the time. 

we start by initializing the scores array to an empty python list. we then have a scores window that is set up to be a deque object from the collections package which is set to take the last 100 scores from the scores list, but at the start it doesn't do anything. then initialize epsilon to the start value. 

ok, now that everything is initialized, let's loop. 

First there is the outer loop. That one we do as many times as we have num_episodes, starting with 1 and ending with one more than the number of episodes, because range doesn't include its last item and is zero indexed by default, so that way we are going from 1 to 200 instead of 0 to 199.

we then initialize the state for this episode to the env.reset(), makes sense becuase either this is our first episode or we just finished one from a prior iteration of the loop, this outer loop is the episode loop. we also reset the score becuase for each episode we should have zero score at the beginning. 


we use the max timesteps parameter to loop over that many steps unless the episode finishes. Then, 
just like in the running version of this above, we have the agent call its 'act' method, but in this case we pass not only the state representation but also the epsilon value. Then, also unlike the running example where we update the display first etc., we instead immediately use that action we got back from the agent's act method by running the next step on the environment, passing in the action teh agent selected and storing what the environment gives us 
back. in this case that is next_state, reward, and whether the episode is done or not. We then use what the environment gave us to have the agent step through its own iteration, given we now have the SARS tuple, and we pass done in as well. this has no output, but we do update the state var from whatever it was set to, eitehr the environment reset or the state from the last time step. then we also update the score with whatever reward we received from the prior state. Question here, I remember they normalized the score to be clipped between -1 and 1 for the deep q network before, do they do that again here? 

fter updating the state and the reward, you are obliged at this point to see if your episode is compolete and you are ready to go to the next iteration of the episode loop. if you are still in the current expisode, you don't break and you end up adding the score you received to the deque structure that has the lst 100 scores on it and you also applend it to the overall scores list. Then you also decrease epsilon for this series of timesteps by multiplying it by tyhe decay factor unless it is less than epsilon_end the minimu value of expxilong. 

You then will get the running average of scores over the last hundred episodes. this allows you to check if you have achieved the target score. You are going for an average score of over 200 over 100 episodes. There is a pytorch command that allows you to save your state dictionary into the checkpoint.pth variable when you have achieve d thegoal. Once you have achieved the goal you can break out of the episode loop because you have created an agent that is capable of maintaining an average score of the target score for the number of episodes you were running. 

You then return the scores array that you were able to acheve when you have successfully met the goal. That's the dqn method. 

call the dqn method to train the agent, and then plot the scores below by creating a new figure with pyplot and then adding a subplot to it. then into that plot you can just add a new numpy array of indices that is as long as your scores array and then the scores themselves as the y axis. then you have the scores as the y and the episodes as the x, since you only get one score per episode. 

##### Dqn agent code

import all the numpy and random libraries and the named tuple and dque data structures from the collections library. 

get the QNetwork object from the model
import pytorch object as well as the F functional neural network
get the optim izer from pytorch also. 

set some of the global constants buffer size which is 100000
the batch size of 64
the gamma, our discount factor to .99
the tau value which updates the target parametersto .001
the learning rate which is set to .0005
and our C value which is update the network every 4 iterations. 

set the device to cuda:0 if using a gpu instance but otherwise use a cpu. 

The Agent Class is defined. The constructor takes in a state_size, action_size and a seed value. 
state size is the dimension of the state
action size is the dimensions of the action space
the seed is a random seed, which if not provided is actually picked at random otherwise for testing and verification of results can be set to a pre-determined seed. 

After the three passed in parameters are defined, we initialize the qnetwork local with the same parameters and then use a .to function and set the device on it. remember this device var is referencing the pytorch object taht could either be a gpu or cpu. 

we also initialize a copy of the network which is to hold the fixed parameter set and act as our targets for squared error loss purposes. 
we also at this time define an Adam optimizer based on our q network parameter vector and the learning rate which was defined earlier. It takes in the local parameter vector of the q learning model as well as the learning rate. we also initialize the amount of memory the agent has available for its experience replay buffer and that takes the action_size, the buffer size, the batch size and the seed for the randomness. 
the time step is then initialized to 0 so that we can use it as our counter to check the UPDATE_EVERY var against. 

###### step method

this is a side-effecting method that sometimes learns and sometimes doesn't depending on what time step it is, and it loads experience.

1. the agent has a step function that takes a state, action, reward and next state argument set and then also a boolean indicating if the episode is done, and it will add the new experience tuple to its memory buffer

2. check to see if the number of the update every counter. 

3. when the modulus for above returns zero, in this case meaning we are at a time step divisible by 4, then we will check to see if we have put enough experience tuples in our memory to satisfy one minibatch of rmsprop for learning. If we do, put them into an experiences var and then pass this into the learn() function of this agent along with the GAMMA discount factor. 

###### act method

this takes the state representation and a value for epsilon for exploration, and it will return the actions for a given state following a policy. 
the states will be array_like objects (numpy?) and the epsilon will be the value of epsilon for whether an action should be uniform random selection of action or choosing the greedy action. 

1. Ause pytorch to load the current state in from the numpy array it is in to the pytorch tensor type, convert it to float, unsqueeze with param 0. What is unsqueeze? It is new to me. Docs say it is, without changing the data, returning another tensor that has a new dimension added at a given index. So if you had a one dimensionsal tensor with 4 values as in the example on their docs page, then the result would be you now have a 2d tensor with shape (1, 4) since you inserted at 0. it means your result will have 1 row with 4 columns. if you had made the index a 1 then you have 4 rows each with 1 column. So the data is the same but the structure is different. This is hard for me to conceptualize above the 2 provided dimensions, I guess you could just add it anywhere, but it would need to extend the dimensionality appropriately and shuffle the data into the correct position in the tensor. However, important thing is, this is essentually wrapping our data in this case with a []. After that there is the call to to which ensures that the tensor datatype and device of the tensor is correct, setting it to the correct device passing in the device as a positional parameter depending on whether you are using cuda or cpu. So then you have a state var with an object which is a pytorch tensor that is correctly formatted. 

2. the next step is to call the eval() method on the qnetwork_local instance of the deepqnet that we have. this doesn't return a value. then we use a context manager predicated on torch.no_grad(). I guess if it is non null then we are going to call qnetwork_local(state) with our new state tensor and store the result as the action values. So I guess when we ran eval() it was doing something that prepared it to receive the state tensor and output the actions from its softmax, but only if no_grad is true, so guessing if we don't have a gradient then we need to get one by running it through to the last layer and then we need

3. to then train it with the self.qnetwork_local.train() call on the deepqnetwork. so got that? 

eval()
if no_grad()
	qnetwork_local(state)
train()

4. after training then we need to select our next action. do this by choosing a random between 0 and 1 and if it is above the epsilon threshold then we will do the greedy version where we take the action values we got from before and pass them througfh a few filters, not sure exactly what they do although they ahve intuitive names, so looks like first there is a cpu function that gives back data in a format and then whatever object that returns has a data attribute and we are selecting the numpy array version of that data and given we have a numpy array we can take the argmax of it and get the greedy action. 

5. if you end up below the epsilon threshold, then you do a stochastic method wehere you select the action randomly from an array that is a sequence of integers the same size as your number of actions. 


###### learn method 

this method will accept a batch of experience tuples and a gamma value for discounting and it will update the value parameters. It is side effecting because it doesn't return the value parameters that it updates. 

the type of the experiences is a tuple made up of {s,a,r,s, done} experience tuples. 
the gamma discount is a float

I'm responsible for writing the code that minimizes the loss function according to the formula in the whitepaper. 

###### soft_update method

this model is reposnible for the soft parameter update. that is, it will update the parameters of the target network so that they are a tau weighted average of the current target parameters and the parameters that have been updating against the target network, the local parameters. if tau is bigger, there will be more value added from the local params and if tau is smaller there will be more value added from the target params.

this looks to be done with a .copy underscore method off of the target param's data, allows an update in place I supposed and that happens for each of the params in the target network. 

##### ReplayBuffer class

this is the data structure used to house the replay experiences before they are sampled for minibatches. 

we have a constructor that takes the action size the buffer size , the batch size and the seed for randomness. 

we initialize each of these parameters as the value for that object in the constructor. we also initialize a namedtuple called experience that will be used to hold the experience tuples in a format that we can easily dereference from later. 

###### add
this allows adding a new experience {s,a,r,s,done} tuple to the buffer. we just call the experience method and pass all of those arguments in. this returns the experience tuple in the format we want, that namedtuple. Then we add the named tuple experience to the memory deque object. now I know what a deque is, it is a double ended queue which allows efficient pushes and pops from either the front or the back of the data structure. so instead of first in first out, you get something that is any in any out essentially as long as you can't pop from something that isn't at the end of the structure already. The main advantage of a deque (pronounced 'deck') over a list is that efficient pushing and popping of objects off of the ends, which are both O(n) operations on a list. so we initialized memory to be a deque that can ony grow to be as big as the buffer size that was passed in. 

###### sample 

this will randomly sample the batch of experiences from memory  according to its comment in the example code of the course. 
we get the batch by performing a random.sample() call with the memory (the deque of experiences we've saved so far) and the number of items to sample is provided by the gbatch size. 

then, we have to get the states, actions, rewards, next_states, and dones for all these experiences and return them in a tuple of tuples. But we actually start with the experiences each being an independent tuple of {s,a,r,s,done}. So, we want ot combine these each into their own pytorch device specific variables. in order to do this, we have to, for each of the variables: 
1. use a list comprehension to get the indivivual field from the named tuple for each experience tuple in our batch.
2. use numpy to vstack that array, I guess turn it from a list to a vertical numpy array, much as you would have for a tensor. 
3. convert that numpy array into a tensor. 
4. cast it to a data type that it should have in pytorch
5. make sure the tensor has the correct device (e.g. cuda, cpu)

then we return the tuples. 
 

###### __len__

this duner method will return the size of the internal memory deque. since we need to be able to get the len and this class is mostly a wrapper around this internal state. 

#### model.py QNetwork 

this has largely not been implemented, just the skeleton is there for me to fill in. All they seem to think I need to do is  finishe initializing the QNetwork and then implement the forward propagation step. Well I'll go over what is here at least. 

##### Init method

this constructor accepts three params, state_size which is the dimension of the state, actoin_size which is the dimension of the action space, and seed which is a seed so that if you don't want a random start you can test with a specific set of intial values. 

##### forward 

all it takes is state and my mission if I choose to accept it is to have a network that will map state to action values, so in other words learns the approximate optimal policy for the environment. 


So, I need to implement the loss computation as well as the forward propagation step, they have taken care of such things as experience replay, epsilon decay, even fixed state updates, so all of the special considerations of the deepqlearning paper except for the propagation and the update. 

Since I don't know pytorch very well I'm a little nervous around pytorch so might need to do whatever project starts with that. So let's go to the extracurriculars section and see if we can't learn some things

### Neural Networks review

neural networks have their name because perceptrons kind of represent neurons in the brain. perceptrons calculate some equations on the inputs and decide to return 1 or 0. Neurons are similar because dendrites get some nervous impulses and does something with them and then decides whether or not to send out an impulse itself. in that way the neurons netowrk their impulse sending capability with respect to each other much in the same way that the perceptrons do. cool that makes sense. 

lets create a nonlinear model by combining two linear ones. you could easily do this by superimposing two linear models and then merging what you have done. so in math: 
linear models are a probability space, for every point we have the probability of some variable for that point. the linear model will decide what regions to subdivide points into. what if you add them together? just add the probabilities? well no, because then it isn't a probability anymore whenever you are going over 1. So you need a function like the sigmoid to reshape the result of the combination into something that fits into a probability function.

what if you don't want ot do a straight combination of the two linear models? you can have weights that represent each of the models differently, and you can have a bias. What you have done is created a linear combination of these different models instead. so you kind of made a meta model that is turning each of the two lines into another line based on that, but the resulting linear combination of the earlier linear combinations isn't going to represent a linear function it is going to represent a weighted sum of linear combinations which will actually be a non linear function. That is probably the clearest explanation I've ever seen for how neural networks are non linear. 

so they show how there is a way to go from a single layer perceptron to a multi layer perceptron, which shows how you arrive at the foundation for neural networks

linear model = value * constant + value * constant + bias

so you have linear model a and linear model b. if you draw them so you have the x values as the nodes in the first layer, then you can see how those nodes are really just getting stacked so your input is really being treated as a series of linear models stacked on top of each other. then the output s of each are going to be input into yet another set of stacked linear models where your inputs are really just the outputs of the results of all the linear models form the first layer. 

the demonstration was that you have a linear combination that is one set of coefficients for the X vector x1, x2 and another linear combination that applies different coefficients to those same x values, then there is another linear combination with yet a third set of coefficients that it will apply to the outputs of the first two models to a linear combination to arrive at the third linear combination result. you can then have an extra node in the first layer which is the bias and the weights of that are set as the actual bias terms. you then compute the linear combination of the x values with their coefficient weights that are contributed for each x and the bias, and then take the sigmoid to set the result to a value between zero and 1 and then that becomes an activated neuron from the next layer, then you do tha tfor the next one down and that becomes another neuron but based on the weights of anothe rlinear combination of the x values and bias with a sigmoid to fit it to the appropriate range. then you do the sigmoid of your final result and that is your output. 

#### nomenclature - 
inputs x1, x2, etc. 

hidden layers - the set of linear models created with the first input layer. 

output layer - linear models are combined to create one non linear model. 

things you can do to change the architecture: 
1. adding nodes to the hidden layer, that means you are essentially adding linear models to your layer, each new node is a new linear model in terms of the input. 
2. adding nodes to the input layer, that is tantamount to increasing the dimensionality of your problem as a whole, that means your linear models are going to be in terms of a higher dimensional space and the boundary you are drawing in the output is going to be a hyperplane in that same higher dimensional space. 
3. adding nodes to the output, that just means you have more output meaning you are doing a multi class classification problem for instance. 
4. adding more layers - this is where the 'deep' in deep neural networks comes from. when you have another hidden layer, that means you are taking the output of the first set of linear models, which will be non-linear, and then inputting them into additional linear combinations of which the outputs will be even more non-linear functions. architectures can be very complex and will split the higher dimensional space with a highly nonlinear boundary, and that is where the magic happens, it is able to learn a very definite boundary for the data in higher dimensional space. Sounds like it would have problems with overfitting though, I wonder if they are going to say anything about that. 


multi class classification - neural networks are really good at binary classification problems out of the box. if you want to do a multi class classification problem you might think you need to make a neural network for each of the possible classes you want and then use softmax across those multiple networks to tell you what the result should be, whichever network's output showed the highest probability of the image belonging to that class. However, that is overkill because you have enough information from the earlier layers of the network to deduce what is in the image and should be able to use just that to figure out which thing is represented there out of a set of possible things. It turns out that if you add nodes to the output layer to represent one of each type of thing in your set of things that you think it could be, the network will assign a value for each of those nodes and you can do softmax to figure out which it is. 

#### Feed Forward

training, what data should the network have on the edges in order to predict well. 
in the perceptron with only one layer and only 2 dimensions, and it is only predicting a binary classification whether or not a point is red or blue. 
you have an equation w1x1 + w2x2 + b 
perceptron plots point x1,x2 and gives the probability that the point is red
let's say that the weight w1 is bigger so its edge is thicker. 
in the example the model was bad and the point was blue but the network was heavily weighted and the blue point was on the red side of the boundary. 

the process of plotting and outputting a probability is known as feed forward. 
in a more complex network you might have 2 hidden layer nodes
the model will also have the nonlinear output which has a different probability. 

so in feedforward, you have input vector x1 x2 and bias, then you multiply that by the weight matrix w2 for the second layer. you do a linear transformation with the weight vector, then you sigmoid them to put them between 0 and 1, adn then that gets multipled by another set of weights in a new matrix and then the sigmoid of that is your y hat. 

even if there was another layer, way of looking at it is start with vector x with weights applied and then sigmoid and then dot product with weights and then sigmoid and then weights applied with dot product and then sigmoid and then you have your output. 




