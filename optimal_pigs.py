import numpy as np
import pickle
import os
#from matplotlib import pyplot as plt

def droplist(listy,i):
	"Drops ith element from a list"
	return listy[:i]+listy[i+1:] 

def product(listy):
	p = 1
	for i in listy:
		p*= i
	return p		

class Pigs(object):

	# stick:0
	# twist:1
	
	
	# state = (score,current_player,turns_left,p_0,...,p_{n-1})
	
	def __init__(self,n_players=3,max_score=100,n_turns=3):
			
		self.max_score=max_score
		self.n_turns = n_turns
		self.n = n_players
		self.filename = str(n_players)+'-'+str(max_score)+'V'+str(n_turns)
		self.n_comps = 0
		
		# game information
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
		self.n_rolls = len(self.rolls)
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
		
		
		
		max_2 = self.n*self.max_score
		
		memory = 0
		
		for t in range(1,self.n_turns+1):
		
			print 'array ',str(self.n_turns),'-',str(t)
			
			a = (self.n_turns-t+1)*self.max_score
			b = a - self.max_score
			list_1 = [(self.n-i) * b**(self.n-i-1) * a**i + i * b**(self.n-i) * a**(i-1) for i in range(self.n)]
			max_1 = sum(list_1)
			shape = (int(max_1),max_2)
			print 'array shape ',shape
			filename = self.filename+'-'+str(t)+'.pickle'
			
			if os.path.isfile(filename):
				print 'loading...'
				with open(filename,'rb') as f:
					setattr(self, 'V'+str(t), pickle.load(f))
				print 'done.'
			else:
				print 'creating...'
				setattr(self, 'V'+str(t), np.full(shape,None,dtype=float))		
			memory += shape[0]*shape[1]
			print '\n'
		
		print 'memory ',memory
	
	def encode(self,player,state):
		# needs tidying and optimising
		# still can't find the argmax
		
		score = state[0]
		current_player = state[1]
		turns_left = state[2]-1
		totals = state[3:]
		
		ix_1 = 0
		b = (self.n_turns-turns_left-1)*self.max_score # maximum score at beginning
		a = b + self.max_score
		
		totals_range = self.n*[b]
		for i in range(current_player):
			totals_range[i] += self.max_score
		
			# count digits passed
			ix_1 += (self.n-i) * b**(self.n-i-1) * a**i + i * b**(self.n-i) * a**(i-1)

		# subtract minimum
		min_t = min(totals)
		min_p = np.argmin(totals)
		totals = [t - min_t for t in droplist(totals,min_p)]
		
		# count digits passed
		
		for i in range(min_p):
			ix_1 += product(droplist(totals_range,i))
		
		# calculcate weights
		droplist(totals_range,min_p)
		weights = (self.n-1)*[1]
		weights[:self.n-2] = [p*totals_range[self.n-2] for p in weights[:self.n-2]]
		totals_weighted = [totals[i]*weights[i] for i in range(self.n-1)]
		ix_1 += sum(totals_weighted)
		
		weights = [1, self.max_score]
		ix_2 = score + player*self.max_score
		
		return (int(ix_1),int(ix_2))
		
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
			# truncate at maximum score
			next_state[0] = min(next_state[0],self.max_score)
			
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
			R = self.V(player,s)
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
			for i in range(self.n_rolls):
				roll = self.rolls[i]
				p = self.probs[i]
				s = self.twist(state,roll)
				states.append(s)
				probs.append(p)
		
		return states,probs
	
	def V(self,player,state):
		
		if state[2] == 0:
			# its terminal
			totals = state[3:]
			max_total = max(totals)
			win_players = [i for i in range(self.n) if totals[i]==max_total]
			
			if player in win_players:
				return 1.0/len(win_players)
			else:
				return 0
		
		# use that the sum over players = 1
		elif player == self.n-1:
			ER = 0
			for i in range(0,self.n-1):
				ER += self.V(i,state)
			ER = 1 - ER
			return ER
		
		else: 
			# we compute fully
			ix = self.encode(player,state)		
			turns_left = state[2]
			V = getattr(self,'V'+str(turns_left))
			
			if np.isnan(V[ix]):
				ER = 0
				states,probs = self.next_step(state)
				n = len(states)
				for i in range(n):
					s = states[i]
					p = probs[i]
					R = self.V(player,s)
					ER += p*R
					
				
				V[ix] = ER
				self.n_comps += 1
			
			return V[ix]		
		
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
		for t in range(1,self.n_turns+1):
			print 'array ',str(self.n_turns),'-',str(t)
			print 'saving...'
			filename = self.filename+'-'+str(t)+'.pickle'
			with open(filename,'wb') as f:
				pickle.dump(getattr(self,'V'+str(t)),f,protocol=pickle.HIGHEST_PROTOCOL)
			print 'done.'
		
if __name__ == "__main__":

	pigs = Pigs(3,n_turns=2)
	totals = (0,0,0)
	score = 0
	player = 0
	turns_left = 1

	state = (score,player,turns_left,) + totals
	
	# state = (score,current_player,turns_left,p_0,...,p_{n-1})

	print 'stick ', pigs.Q(state,0)
	print 'twist ', pigs.Q(state,1)
	print 'computations ', pigs.n_comps
	
	pigs.save()
