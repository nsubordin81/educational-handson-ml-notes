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
