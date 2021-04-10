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

