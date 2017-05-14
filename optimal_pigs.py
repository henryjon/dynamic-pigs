import numpy as np
import pickle
import os
#from matplotlib import pyplot as plt

class Pigs(object):

	# stick:0
	# twist:1
	
	# Each player must bank at a score of 200
	
	# state = (score,current_player,turns_left,p_0,...,p_{n-1})
	
	def __init__(self,n_players,max_score=100):
			
		self.max_score=max_score
		self.filename = 'V' + str(max_score) + '.pickle'
		self.n_comps = 0
		# there are 9 rolls
		
		self.rolls = np.array([
			float('nan'),
			5,
			10,
			15,
			1,
			20,
			40,
			60,
			25
		])
		
		self.probs = np.array([
			0.21117594565, #nan
			0.292173675956+0.114782515554, #5
			0.0391304030298+0.0394950591154, #10
			0.00795651528273+0.0134642246984+0.00528951684581, #15
			0.213388927224, #1
			0.0502664388742+0.00775795804053+0.00273772568868+0.00107553509198, #20
			0.000901622189627, #40
			3.72770685289e-05, #60
			0.000366659690448 #25
		])
		self.probs = self.probs/sum(self.probs)
		
		self.n = n_players
		if os.path.isfile(self.filename):
			print 'loading...'
			with open(self.filename,'rb') as f:
				self.V_cache = pickle.load(f)
			print 'done.'
		else:
			self.V_cache = {}
	
	
	def twist(self,state,points):
		"Returns state after twist"
		next_state = list(state)
		if np.isnan(points): 
			# pig out
			# next player
			next_state[1] += 1
			next_state[1] %= self.n
			if next_state[1] == 0: 
				# next round
				next_state[2] += -1
			# zero the score 
			next_state[0] = 0
				
		else:
			# add points to score
			player = state[1]
			next_state[0] += points
			
		return tuple(next_state)
	
	def stick(self,state):
		"Returns state after stick"
		next_state = list(state)
		player = state[1]
		# add score to total
		next_state[player+3] += state[0]
		# next player
		next_state[1] += 1
		next_state[1] %= self.n
		if next_state[1] == 0: 
			# next round
			next_state[2] += -1
		# zero the score
		next_state[0] = 0
		
		return tuple(next_state)
		
	def Q(self,state,action):
		player = state[1]
		states,probs = self.next_step(state,action)
		ER = 0
		n = len(states)
		for i in range(n):
			s = states[i]
			p = probs[i]
			R = self.returns(player,s)
			ER += p*R
		
		return ER
	
	def greedy(self,state):
		"Returns optimal move"
		
		if state[0] >= self.max_score:
			# must stick
			return 0
		
		Q_vals = [self.Q(state,a) for a in [0,1]]
		# assume unique maximal action
		a = np.argmax(Q_vals)
		
		return a
		
	def next_step(self,state,a=None):
		"Returns distribution for next step"
		if a == None:
			a = self.greedy(state)
		score = state[0]
		
		states = []
		probs = []
		if a == 0 or score >= self.max_score:
			s = self.stick(state)
			states.append(s)
			probs.append(1)
		else:
			# a = 1
			for i in range(9):
				roll = self.rolls[i]
				p = self.probs[i]
				s = self.twist(state,roll)
				states.append(s)
				probs.append(p)
		
		return states,probs
		
	def returns(self,player,state):
		if state[2] == 0:
			# its terminal
			totals = state[3:]
			max_total = max(totals)
			win_players = [i for i in range(self.n) if totals[i]==max_total]
			
			if player in win_players:
				return 1.0/len(win_players)
			else:
				return 0
		else:
			return self.V(player,state)
	
	def V(self,player,state):
		
		# use that the sum over players = 1
		if player == 0:
			ER = 0
			for i in range(1,self.n):
				ER += self.V(i,state)
			ER = 1 - ER
			return ER
		
		# subtract the minimum total
		totals = state[3:]
		min_t = min(totals)
		state = list(state)
		state[3:] = [t - min_t for t in state[3:]]
		state = tuple(state)
			
		ix = (player,) + state
		# value according to the player 
		if not ix in self.V_cache:
			self.n_comps += 1
			ER = 0
			if state[2] == 0:
				# terminal 
				ER = self.returns(player,state)
			else:
				states,probs = self.next_step(state)
				n = len(states)
				for i in range(n):
					s = states[i]
					p = probs[i]
					R = self.returns(player,s)
					ER += p*R
					
			self.V_cache[ix] = ER
			
		return self.V_cache[ix]
		
	
		
	def printy(self,state):
			
		print 'player:          ',state[1]
		print 'score:           ',state[0]	
		print 'remaining turns: ',state[2]
		print 'scores:          ',state[3:]
		
		
	def plotty(self,player,turns_left,totals,x_min=0,x_max=41):
		X = [x for x in range(x_min,x_max)]
		Y_stick = []
		Y_twist = []
		
		for x in X:
			state = (x,player,turns_left,) + totals
			Y_stick.append(self.Q(state,0))
			Y_twist.append(self.Q(state,1))
		
		plt.plot(X,Y_stick)
		plt.plot(X,Y_twist)
		plt.xlabel('score')
		plt.ylabel('probability')
		plt.legend(['stick','twist'])
		plt.savefig('figure.png')
		plt.show()
		
		self.save()
	
	def save(self):
		print 'saving...'
		with open(self.filename,'wb') as f:
			pickle.dump(self.V_cache,f,protocol=pickle.HIGHEST_PROTOCOL)
		print 'done.'
		
if __name__ == "__main__":
		
	pigs = Pigs(3)

	state = (0,0,1,10,10,10)
	
	# state = (score,current_player,turns_left,p_0,...,p_{n-1})
	
	print 'Stick: ', pigs.Q(state,0)
	print 'Twist: ', pigs.Q(state,1)
	print 'Computations: ', pigs.n_comps

	pigs.save()
