import numpy as np
import matplotlib.pyplot as plt

# Pega a memória ram
def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

def getPosicaoX(ram):
    x = ram[0x95]*256 + ram[0x94] #Posição x do Mario
    return x

def getPosicaoY(ram):
    y = ram[0x97]*256 + ram[0x96] #Posição y do Mario
    return y

def getLives(ram):
    vidas = ram[0x0DBE] #Vidas do Mario
    return vidas

def getFimFase(x, tamanho_fase):
    return x > tamanho_fase


def plotarGrafico(recompensa_por_episodio, episodios):
    episodios = len(recompensa_por_episodio)
    soma_recompensas = np.zeros(episodios)
    for episodio in range(episodios):
        soma_recompensas[episodio] = np.sum(recompensa_por_episodio[max(0, episodio - 10):(episodio + 1)])
    plt.plot(soma_recompensas)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title('Recompensa ao longo do tempo')
    plt.show()
    plt.savefig('recompensas_mario.png')