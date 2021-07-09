# Policy Gradient and Actor Critic Methods

notes from the grokking deep reinforcement learning book by Miguel Morales

## Policy Methods

what you learned so far were monte carlo approaches, temporal difference methods, you learned about sarsa and q learning. These methods are all learning a value function. In these methods, you learn a value for each state you visit over time, and you then learn a policy that, when in any of these states, can suggest the right action to get the best value for all future states which will lead to the highest cumulative reward for visiting all the states. 

instead, you might want to just optimize the policy without estimating values for the states. Policy based and policy gradient methods do this quite well. 

you can combine policy based methods with value based methods to form actor critic methods. Actor-Critic nomenclature is called that because we are both directly learning a policy to select actions (actor) and then using a value function to evaluate how the policy did (critic)

## differences between value based and policy based methods in more detail

value based functions are putting parameters on the action value function and then learning to optimize that function relative to an ideal q function (which has to be approximated by using the actual return and discounted function for the following state).

Policy based functions are putting parameters on the policy direclty and then maximizing this policy. 

## advantage of policy gradient method

the learned policy isn't as constrained of a function from states to actions. value based methods like functions where the action space is discrete, for example. This seems to be the main advantage function wise, though worth thinking more about why this is the case and why it is useful

the relationship between the states and their action doesn't have to be deterministic. in value based methods you aren't truly stochastic, they rely on things like epsilon to add stochastisity to their learning. But there were good examples in the udacity nanodegree course where you couldn't rely on epsilon because you could end up in situations where the agent fails to explore a truly stochastic state transition and ultimately misses out on a more optimal policy through misestimation or failure to converge

so it is kind of a case of getting the good without sacrificing anything because a policy based method can learn a deterministic policy that is more optimal just fine. Starting with a stochastic, partially observable assumption about the environment means that you can always reduce to deterministic when optimizing

policy methods enjoy a huge advantage when calculating a state value function or action value function is overkill and you just want to know what action to take in each state. the example provided in the grokking book was a huge corridor grid space where it was essentially a line with goals on either end. The actions to take are obviously move towards the closer goal, but a value function estimation would entail visiting all of these intermediate states and estimating their value, when if you learn the policy directly from the policy space, will probably cut out a lot of this work. This was a kind of take the author's word for it one, so might be worth digging deeper on once I understand policy gradient methods better. It seems naively like this would be a 50 50 situation because the policy gradient method woudl still have a policy space that had to take into account the trajectories.

finally policy methods tend to converge better, the probabilities of the actions will change smoothly and incrementally as you follow the gradient up to a local optimum whereas value based methods tend to oscillate and diverge. Need to go back and review the reasons for divergence, one given was that if the value function changes then the actions the agent takes can significantly change. It would be great to see this visualized. 

## code analysis of grokking deep rl book

discrete action space stochastic policy8, using what looks like pytorch
init method takes in the input and output dimensions and the hidden layer dimensions, init_std which I guess is the standard deviation maybe? activation function is set to relu by default but coudl be something else, that is a nice way to do it. 

this sets up the layers with a linear layer for the inputs that transforms to the length of the first hedden layer's dimensions, then there is a list for the hidden layers, the cocde assumes they are all linear, and then sets them all up in a for loop. then the output one is last

then there is a forward pass which creates an x var that is equal to the input parameter, which would be the observation tensor I suppose, then if it is not a tensor it is conversted to one and then unsqueeze is applied to add a row dimension of 1 on the outside of it, there are some assumptions here too but they probably always hold and that is interesting. 

then we are ready to call the input layer on x and do the activation function on it and star that as the result, and then do it for each of the hidden layers and have an output layer at the end. This is a pretty standard network, looks like we just use the output layer's values directly without applying a softmax to make them probabilities. 

then a few more functions that aren't just the standard pytorch modules, we have full_pass that gets the logits from the output of that forward pass and then gets a categorical distribution over the logits and then samples from that distribution and gets the log probability of that sample and adds a new dimensioni of length 1 to the end of it and then similarly gets the entropy of the sample distribution and then figures out if the sample is exploratory or exploitative by determining if it is the argmax of the logits that were retrieved. it returns a tuple with the action that was sampled, whether it was exploratory (strange possible torch based language) athe log probabilities and the entropy

select_action is another function that will get the logits in the same way and then get a distributions over the logits and then instead of doing the log probability and the entropy etc. it just returns the sampled action. select_greedy_action is not sampling it is just returning the argmax of the logits from a forward pass.

The code is mostly shared between these options, so overall the full pass code is doing the other two and returning more information

### direct policy learning
Advantage, repeated, we can ignore the environment dynamics and also we can learn the policy that optimizes the value function without having to learn the value function itself. Grokking book breakdown of math

But how do you do it? the high level process is to have a parameterized policy (so if using a NN then the weights are the parameters) that gives some probability of taking any given action from an observed state. Then, you run a trajectory of some length through this network by doing several time steps, and then at the end of the trajectory you go back for each of the actions you took in turn, and if the trajectory as a whole led to reward, you adjust the weights to make the highest probable action even more probable in the future. Now we go over the math derivation of the REINFORCE method that is used to do this: 

high level: 
- collect episode
- chaange policy weights to make actions in episode more likely if you won and less likely if you lost

if it helps you can think of this as being similar to supervised learning for object recognition, where your weight adjustments are based on whether or not your guess for an object probability was what the label said. difference is that for policy gradient methods your dataset changes over time and as such you could get a similar state in the future but the action should be something different because you are at a different point in your trajectory, so you have to be aware of how you got to that state in some sense to know whether the same action is still a good idea or not. 

### Andrej Karapathy posting on RL that breaks down policy gradients and other topics
he proposes that the four things that are holding AI back are 
- compute
- data
- algorithms
- infrastructure

He points out that many of the groundbreaking things in the field of reinforcement learning and Ai in general since the 90's aren't new and interesting algorithms but tweaks on existing algorithms that leverage better compute/data/infrastructure. nobody seems to contest this point, Rich Sutton largely agrees, we are riding on compute, we should be working on improving and experimenting with lots of different algorithms, democratizing the hardware and knowledge as much as we can. 

He offers that it is astounding how simple some of the algorithms are that have managed to do some of these amazing things in RL and computer vision are really not that innovative. He goes as far as to say in a kind of tongue-in-cheek, joking but not really joking way that DQN specificaly is kind of dumb, and that is ok to say I guess if you are very smart. I find them to be not intuitive and take a while to grasp, but I get the sentiment that DQN and even policy gradients aren't reinventing the wheel, they are based on algorithms people came up with a long time ago, they just happen to be working now because of creative applications and available compute to do function approximation on non trivial problems. He did all this to introduce hsi topic which is Policy Gradients, which are a better alternative to DQN in his opinion. It is objectively verifiable that all current incarnations of DQN have bene outperformed by some version of policy gradients, though I think that the hyperparameters have to be well tuned, and actor critic models which use both are the best performers, so cooperation and collaboration of ideas seems to be the winningest approach in this field. 

His thoughts on Policy Gradient superiority: 
- it is end-to-end. By that he means instead of do some intermediate work to approximate a value function that will be able to maximize returns from some implicit policy, there is an explicit policy that you are working on at the one end, and then you are directly optimizing the expected reward for this policy. 

Back to finite state automatas, yay my favorite thing! so he just mentions that MDPs are represented with these where the state transitions are probabilistic. 

you have an image buffer as the byte array 0-255 grayscale pixel buffer. you get this as input adn then you decide an action in some fashion, either up or down. after the choice the simulator executes the action on behalf of the agent and then gives the agent a reward, this is either 0, 1, or -1, so you do get a reward after executing the action, I guess there are a bunch of frames though that don't matter, so one time step in pong is really an entire round trip of the ball from you to your opponent and back, unless it goes past you or your opponent. So the environment is in charge of handling all that and you just need to worry about what you get back from the environment interface you happen to set up. 

The neural network architecture used in this article is to use the atari screen pixels as inputs to a two layer neural network, so that is 210 pixels high by 160 pixels wide by 3 layers for the rgb colors and that is a total of 100,800 numbers. and that 2 layer network will do the usual linear combinations of the weights and biases and then nonlinear activation functions and the final network will output a probability logit that represents the likelihood that we should take the up action with the paddle. 

he does a 4 line neural network implementation, which is wild. then he says "we aren't using biases because meh" the probability is squashed to 0-1 by the sigmoid activation. 

there is an intuitive description of what is happening in the 'network's mind'. This is akin to the atari DQN example where the convolutional network was in front and its goal was just to interpret the different states that the game could take based on pixels, learn in other words what the game looked like and identify those scenarios, and then the later layers of the network were actually to decide the action. 

he explains that you do want to ahve multiple frames fed in at once for temporal learning, some things are more important when considered over time, so he uses a difference between frames as the input to make the example easier. 

slight aside to talk about how incredible it is that the neural network is able to figure out the difficult problem like this. you have 100,800 numbers as input to represent an image, you'd have more if you took in more frames than one, and then you have something like 1,000,000 parameters in the network. the network uses that to decide to go UP for example in PONG, you get some reward maybe 0 maybe 1 maybe -1, and then you get the next series of the same amount of numbers for input and you do another forward pass. There could be a long period for which you don't get anything for a positive reward. So maybe 100 or more timesteps into the future we get 1 as the reward but how do you use all of the data collected to figure out what in that series of actions got you to the 1? which action was more important to getting that reward? How do you adjust all of these knobs (parameters) to emphasize the reward gaining behavior and deemphasize the reward losing or constant reward behavior? Credit Assignment is the predicament you are experiencing. You can know, as a human mind, that this happened in pong because you had a good shot or because the opponent took an incorrect action or some collection of movements led to a ball position that was impossible for the opponent to get to. In pong especially the good action might have been something several frames before you actually get a point, and all the actions you took after that point don't really matter because the ball is already on a good course to make you win the point. 

Question I need to learn the answer to: different between log probabilities and probabilities and why we use log probabilities. Looks like log probabilities can be greater than one, I am sure it has to do with taking the logarithm of a probability, I know t hat there is a trick in which you can use equivalence of a product of probabilities in expectation to turn it into a sum but not sure what the intuition is there mathematically. This is important to grok. Reason Andrej gives is that log probabilities and probabilities can be used interchangably for optimization problems because they both grow monotonically. That makes sense, and then "makes math nicer" is shorthand for somethign I should try to understand, but I have a feeling it has to do with the derivation of the expected value function that doesn't involve needing to know the transition. 

Supervised learning comparison (again): 
- label in sup learning, so for Policy methods would be equivalent of each timestep having a prescribed action and the network tries to get a probability that matches that action. So for a supervised learning example of one forward pass of the policy network you could do gradient descent by having your target be a gradient of 1.0 on the log probability of UP action, then you would backpropogate through the network to compute the gradient with respect to the parameters of the log probability that action is UP given the input of that vector of numbers representing your game state. I'm a little fuzzy on the loss function but it seems to be some representation of the difference between a 100% probability of UP and what the network got for UP, so then the gradient at each weight knows the direction to move each weight so that the probability is more likely to indicate UP action next time. That's a great walkthrough at a low level of what is happening. 
- in RL setting, same problem, you have log probabilities calculated for each action UP and DOWN, then you sample the action from a distribution, so maybe it is UP, which we happen to know is the right action, but maybe it is DOWN. He points out that at this point if you wanted to you could treat this as a supervised learning problem and then turn the sampled action into a Label and do gradient descent by minimizing the loss between your (randomly sampled) label with 100% probability and your network's selected label. But that parenthetical is very important, we just chose DOWN this time but don't know if it is a good action, the reward that would confirm that or invalidate it won't come for some indeterminate number of timesteps. So the key is that instead of forming the gradient for this single timestep right when we encounter it, we allow all of the actions up to the reward to be taken up front. What we would do in that case is get the reward, and then use that as the gradient for each of the actions that led to it. So we'd go back to the action we took, knowing that ultimately we didn't do as well in this case (we got -1 reward and lost the round of pong), and we would set the desired log probability for that action we took to be -1 which would cause the gradient to nudge the parameters so that we were less likely to have that action occur in the future. 

So that is the whole idea, kind of like a series of supervised learning problems that wait to be evaluated until a reward has been collected, and then every 'observation' of a state-action transition is then adjusted so that the parameters lead to a more favorable outcome based on that reward. One cool feature of doing it this way si that the actions are probabilistic by default, not deterministic. The reward can be any quantity and it still works, it will create a proportional gradient and adjust the parameters accordingly. I really like this line he gives about how incredulous this is: 

"That's the beauty of neural nets, Using them can feel like cheating: You're allowed to have 1 million parametersembedded in 1 teraflop of compute and you can make it do arbitrary things with [stochastic gradient descent]. It shouldn't work, but amusingly we live in a universe where it does."

so for training you have policy "rollouts" which are just the episodes or trajectories that lead to reward that you are playing through. in Pong, it makes sense taht these correspond to rounds where the ball eventually goes to one or the other side, win lose or draw. so you can limit them to 200 frames and in some cases maybe by that point the game hasn't finished yet and the agent's reward is zero. He gives a good sense of the actual numbers flowing here, your agent will have made 20,000 decisions in 100 games of 200 frames per game and each time you had to pick UP or DOWN. every decision will have an associated parameter gradient that talks about how to adjust the parameters to achieve a better result, you just have to label what is good and what is bad. So he gives won 12 games, lost 88 games so that 12 game series of 2400 decisions will be updated as a  +1 in this case as the gradient for the sampled action and then backprop down the network for that, and then you correspondingly do a -1 gradient on the sampled action for the 17600






### math derivation of policy gradient approach for REINFORCE

- Objective function, the performance measure the true value function of the parameterized policy from every initial state
- you want the gradient, so put the gradient on both sides of teh objective function, it is the delta with respect to the parameters  of the objective function
- tau is the representation of a trajectory, which in this book's version contains rewards as well so it is a set of states, actions and rewards over timesteps 0 through T
- then G of tau is going ot represent the return of a given trajectory Tau, so it would be the gamma discounted reward over all steps in that trajectory
- then you can take the probability of the trajectory, which is the product of the probabilities of each of the states and transitions with ultimate reward given the parameterized policy
- 





