import numpy as np
import matplotlib.pyplot as plt


# On remplie la matrice de transition
P = np.array([[0.7, 0.2, 0.1],[0.3, 0.4, 0.3],[0.2, 0.3, 0.5]])

states = ["Sunny", "Cloudy", "Rainy"]

#on prend des cas aleatoirement
def simulate_markov_chain(P, start_state, n_steps):
    current_state = start_state
    history = [current_state]

    for _ in range(n_steps):
        current_state = np.random.choice(len(P),p=P[current_state])
        
        history.append(current_state)

    return history

#np.random.seed(0)

history = simulate_markov_chain(P, start_state=0, n_steps=50)

# affecter les noms
state_names = [states[i] for i in history]

print(state_names)


#simulation des cas 
plt.plot(history, marker='o')
plt.yticks([0, 1, 2], states)
plt.xlabel("temps")
plt.ylabel("etat")
plt.title("simulation des etats au cours du temps avec Markov Chain State")
plt.grid(True)
plt.show()

pi = np.array([1, 0, 0])  # on commence par l etat sunny

n_steps = 30
distributions = [pi]

for k in range(n_steps):
    new_pi = np.zeros(len(pi))
    
    for j in range(len(pi)):
        for i in range(len(pi)):
            new_pi[j] += pi[i] * P[i][j]
    
    pi = new_pi
    distributions.append(pi)

distributions = np.array(distributions)


plt.plot(distributions[:, 0], label="Sunny")
plt.plot(distributions[:, 1], label="Cloudy")
plt.plot(distributions[:, 2], label="Rainy")

plt.xlabel("teps")
plt.ylabel("Probabilité")
plt.title("Probabilité de Distribution au cours du temps")
plt.legend()
plt.grid(True)
plt.show()
