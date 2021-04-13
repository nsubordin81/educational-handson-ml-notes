# Markov Decision Process

Markov Decision Processes are about solving decision making problems where you don't always know what will happen when you make a decision. 

let's say you are in a situation where you need to make a sequence of decisions.
what are some examples of this? 
* Games are a good one, as in many games you are making a sequence of decisions. 
* Living in general involves making decisions because of your
environment. 

But breaking life down into small, finite chunks, many activities also involve making a sequence of decisions. 

* When I'm sitting at my desk, maybe my goal is to eventually be able to apply reinforcement learning techniques. 
* I have lots of actions I can take, like reading different sources, posing and solving problems that are reinforcement learning, framing the problems. 
* Also, I need to stay hydrated and fed, I need to take care of several tasks that aren't related to reinforcement learning and which require energy
* I don't know what the best strategy is for me to reach my goal, and there are things I haven't seen yet that could help or hurt my ability to make the right decisions at the right
time. 
* Additionally, decisions I make in the future might be better or worse for me based on the decisions I'm making now and what happens after I make them. 

another common name for markov decision processes is stochastic optimal control problems. 

The sutton barto book's contribution to the formal definition of markov decision processes
is the notation of p(s', r | s, a) to describe their dynamics. Traditionally these dynamics are described with state transition probabilities p(s'|s,a) and expected immediate reward 
r(s, a) as separate quantities. 

Combining these into one probability statement in which they have one time step index makes it more concrete to the subject that the new state 
and the reward for transitioning to that state are jointly determined by the action taken. there is a Martin Minsky paper to read about this that he wrote in 1967 I should add some
notes from there. 

## Anatomy of an MDP

states s, actions a, probabilities a, rewards r, discount factor gamma

you can visualize MDPs as a Finite State Machine, where the states are the states of the mdp, the transitions are the actions you take from one state to get to another. 

Because Markov Decision Processes can represent problems where the results of an action may be uncertain, and that is why there are probabilities. the probability is modeled as the 
likelihood of getting a state and reward in the next time step given the agent was in a given state and took a given action. 

So given that, let's also add to that finite state machine the fact that the transitions are probabilistic. The aren't just the action but a probability mapping that given the state and action you will get to the next state and get some reward. 

## The Markov Property

The markov property is a name given to a property of a stochastic process (process that has randomness) if the conditional probability distribution of future states of the process only depends on the present state and not the states that came before it. So in the example of Markov Decision Processes, the probability of the future reward and state only depends on the current state and action, not the series of states and actions leading up to the current state and action. 

Let's dwell on this a little longer. What is the alternative to MDP and this markov property? The alternative would be that your probability of getting to one state from another wouldn't just depend on your current state and action, it would also depend on how you got to that state. It is certainly possible that you could arrive at a state in a finite state machine through two alternate paths. there could me multiple transitions leading to your current state and multiple transitions leading to those states. Imagine if your probability of getting to the next state had to be aware of all those prior 'histories' of states. It would certainly be a more complicated probability. By not taking that into account, what do we lose? 

Well, suppose there are real environments you are trying to model where the graph of the finite state machine has probabilities that depend on a sequence of states your agent has visited rather than just one. what kind of problem would that be? What would it look like? Could you model it with a markov property? for a really simple one, maybe there are four states, s1, s2, s3, s4. S1 and s2 connect to s3, and the correct way to model a transition from s3 to s4 is that if you are coming from s1 you will go from s3 to s4 with 99% probability and otherwise loop back to s3 with 1% probability. Otherwise, if you are traveling from s3 to s4 with the same action but coming from s2, the probabilities are reversed. So here you have a situation where the state you came from matters a great deal when you are in s3. if you use the markov property and a method like monte carlo or temporal difference learning, your action value function might eventually just average your expected return based on how often you have visited s3 from s1 vs. s2, so it will kind of bake in probability of the prior state in the policy. You won't actually know this is why this his happening though, on the surface it would just look like taking that action from s3 has that consequence. 

## State Value and Action Value Functions

Markov Decision Processes goal is to find two mappings, functions. V(S) the value of every state, and Q(S, A) the value of an action in a given state. we are trying to estimate or predict the V(S), and Q(S, A) is giving us the action to take. Together they help us arrive at the optimal policy Pi star, which will maximize our total _expected_ reward. it is expected instead of actual because MDPs have uncertainty built into them about what amount of reward we could get for a given state action pairing and also how often we can get it. Because of the uncertainty we create a discount term gamma so that the values we add to the Q function emphasize the rewards we are given now over the rewards we expect to get in the future. In simple problems where we know all of the transitions are deterministic, we wouldn't have to treat the reward as expected and we wouldn't need a discount factor because older information about rewards is just as good as current information about rewards.   


## RL Algorithms based on MDP

### Model Based Apporaches

* policy based iteration
* value iteration

you have to have a transition and reward model. Dynamic Programming approaches are then used to figuer out the idea state value and action value functions based on the model they already have. 

### Model-Free Learning

* Monte Carlo Methods
* Temporal Difference Learning

These don't have a model up front of the transitions or rewards. They don't know how likely a reward is based on a given action from a given state (that is what I'll call the dynamics, I hope I'm using that term correctly). Their method to find the optimal policy, state value and action value function is to 'sample' the environment. A sample here is taking exploratory actions over some limited number of time steps. They then use the experience they get from taking these actions to estimate the proper value functions. 

## How Deep Reinforcement Learning Differs from Traditional Reinforcement Learning in the Context of MDP

In traditional reinforcement learning algorithms, there is the implicit assumption that the parameter space of the Markov Decision Process is finite and/or discrete. 

Dynamic programming requires full knowledge of the transitions and rewards, so it needs a model based on a finite number of states or actions. Additionally, model free learning techniques use finite data structures to keep track of their experience, which means to this point they need a place in that structure to hold every state and action combination. If there are infinite or continuous states or actions, these methods break down, there just isn't an easy way to store the representations of these MDPs. 

Also, with infinite or continuous state and action spaces, you will probably be seeing more methods involving integration or derivation to be able to determine values as the slices of discrete chunks approach a continuous space, again where you don't get values until you approximate their limit. 

## Continuous MDP

### Why more complex? 

With discrete state or action spaces, you can easily represent the mapping functions for state value or action value as a dictionary or as a matrix in the case of Q tables. You can do this because your index in these cases is either your states or actions or both and so you can represent them in the dictionary or matrix with the sequence of integers. 

Also, think about the estimation and update portions of both model based and model free methods. they each loop over the states or actions within a state so that can't be done with infinite numbers of states or actions. 

now you don't have individual states or actions, if you were to plot your state value function, it'd no longer be able to be plotted as a histogram of discrete values become a density function under some continuous plot of a state function

### Why do we need to have continuous MDPs? 

for MDP of simpler reinforcement learning or simple environments that are more like the playspace of a game like chess or checkers, you can have discrete spaces to model out the environment. If all you care about can be represented in a discrete way, then this should be all you use it will save you complexity. 

However, in the real world most things are continuous, think about a vacuum cleaner robot navigating a floor. This is more of a continuous plane than a grid, you can't get away with representing it as a grid. 

generally speaking, if your agent is going to interact in the physical world or a world that is based on the actual physical world, then it will need a continuous action space and probably a continuous state space as well. 

### Discretization

This is a technique for converting a continuous space into a discrete space. 

for example, vacuum cleaner bot could be represented as a continuous state space. to discretize this, you can overlay a grid on it just like the one you would have in the gridworld example that has centers for each of its grid squares, and as long as the robot's center is within the boundaries of a grid square, you round the location of the robot to that square's center. This will have approximation probelems, but depending on the problem you are trying to solve this kind of approximation can have little or no impact on your ability to learn a good policy. You could then apply existing algorithms and have them still work in this space. 

You can do the same type of things for actions. you can round angles for example to the nearest 45 or 90 degrees. 

also, say then you have obstacles in the environemnt that are real world obstacles. Then you may need to do something like binary space partitioning or quad trees, i.e. you create an  occupancy grid to tell you what grid squares the obstacles are occupying, and then depending on how precise you want to be you can also subdivide the grid but just for those overall grid squares to get a more precise picture of where the objects are for your navigation task without wasting computation on the squares that it doesn't matter for. 

in non grid like state spaces you can also subdivide the different parts of a continuous state space. The example given was the state space for what the optimal gear in terms of fuel efficiency was for a car depending on how fast it was traveling. 

#### Tile Coding

the methods for discretization covered so far, in the lab where you just break up the space into bins or in the example where you have a known relationship between fuel consumption and speed to help guide where states should be broken up according to gear. however, if you don't know about the environment in advance enough to know what shoudl be used for the way space is broken up and if that space is more irregular, then you need a more general method. One of these is Tile Coding. 

For tile coding, you start with a 2d state space which is continuous, so think of on axis for state feature 1 and one for state feature 2 and they make a continuous plane. Then, you overlay or superimpose on top of this state space some tile spaces that are discrete, and those tile spaces are at offsets from each other. So then if you want to know the position of a point on the continuous state space, you can look at what tiles are activated on the respective tile space overlays. 

you can assign a bit to each of the tiles in all of the tile spaces and then you can represent the discretized location as a bit vector where you have ones for the activated and 0s for all other tiles. 

then, instead of the value function having a separate value for every state V(s) in the state space in terms of the bit vector, it instead uses a combination of the bit vector and a weight that every tile in the vector gets. the weights of each tile are then updated iteratively. that way, nearby states that share tiles share a component of the learned value function, so it smooths the boundaries betweeen states in the learned value function. 

there are drawbacks, you have to manually set up the tile grids and sizes ahead of time. Adaptive tile coding starts with a fixed size tile grid and then splits the tiles after learning slows down. adaptive tile coding doesn't 
rely on a human to specify the tile grid ahead of time and instead gets as complex as neede to learn what it needs to learn.






### Function Approximation

