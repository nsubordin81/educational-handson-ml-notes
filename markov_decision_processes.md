# Markov Decision Process

Markov Decision Processes came from an attempt to formalize a pretty general set of problems. let's say you are in a situation where you need to make a sequence of decisions.
what are some examples of this? Games are a good one, as in many games you are making a sequence of decisions. Living in general involves making decisions because of your
environment. But breaking life down into small, finite chunks, many activities also involve making a sequence of decisions. When I'm sitting at my desk, I want to learn more
about reinforcement learning, I also need to stay hydrated and fed, I need to take care of several tasks which require energy, I also want to learn quickly and in a way that
if I am later tested on my knowledge or need to apply it in some novel context, I can. That describes potentially several examples of decision sequences, and also uncertainty. 
I don't know what the best strategy is for me to reach my goal, and there are things I haven't seen yet that could help or hurt my ability to make the right decisions at the right
time. Additionally, decisions I make in the future might be better or worse for me based on the decisions I'm making now and what happens after I make them. 

another common name for markov decision processes is stochastic optimal control problems. the sutton barto book's contribution to the formal definition of markov decision processes
is the notation of p(s', r | s, a) to describe their dynamics. Traditionally these dynamics are described with state transition probabilities p(s'|s,a) and expected immediate reward 
r(s, a) as separate quantities. combining these into one probability statemetn in which they have one time step index makes it more concrete to the subject that the new state 
and the reward for transitioning to that state are jointly determined by the action taken. there is a Martin Minsky paper to read about this that he wrote in 1967 I should add some
notes from there. 

## Anatomy of an MDP

states s, actions a, probabilities a, rewards r, discount factor gamma

you can visualize MDPs as a Finite State Machine, where the states are the states of the mdp, the transitions are the actions you take from one state to get to another. Because Markov Decision Processes can represent problems where the results of an action may be uncertain, and that is why there are probabilities. the probability is modeled as the 
likelihood of getting a state and reward in the next time step given the agent was in a given state and took a given action.

The markov property is a name given to a property of a stochastic process (process that has randomness) if the conditional probability distribution of future states of the process only depends on the present state and not the states that came before it. So in the example of Markov Decision Processes, the probability of the future reward and state only depends on the current state and action, not the series of states and actions leading up to the current state and action.

Markov Decision Processes goal is to find two mappings, functions. V(S) the value of every state, and Q(S, A) the value of an action in a given state. we are trying to estimate or predict the V(S), and Q(S, A) is giving us the action to take. Together they help us arrive at the optimal policy Pi star, which will maximize our total _expected_ reward. it is expected instead of actual because MDPs have uncertainty built into them about what amount of reward we could get for a given state action pairing and also how often we can get it. Because of the uncertainty we create a discount term gamma so that the values we add to the Q function emphasize the rewards we are given now over the rewards we expect to get in the future. In simple problems where we know all of the transitions are deterministic, we wouldn't have to treat the reward as expected and we wouldn't need a discount factor because older information about rewards is just as good as current information about rewards.   
