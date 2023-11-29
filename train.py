import numpy as np
import retro
from rominfo import getInputs
import time
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from utils import *
from collections import deque

acoes_possiveis = {0:128, 1:131, 2:391}

#Ações do Mario
#wait  0
#right 128
#left  64
#down  32
#jump  131 este jump ele vai para frente ao mesmo tempo
#spin  391 o mesmo para o spin

raio_estado = 4
tamanho_estados = 81 # tamanho da array do estado do getInput
tamanho_acoes   = len(acoes_possiveis)

#################################################################################

# Função de recompensas
def getRecompensa(recompensa, x_atual, x_novo, vidas_atual):

    if vidas_atual < 4: #Morreu em algum momento
        recompensa = recompensa - 1000

    if x_novo > x_atual: #Ele avançou na fase
        recompensa = recompensa + 1
    
    return recompensa

##################################################################################

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
input_size = tamanho_estados
output_size = tamanho_acoes
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = DQN(input_size, output_size)
modelo.to(device)

learning_rate = 0.01
optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
memory = deque(maxlen=2000)

##################################################################################

def salvar_modelo(modelo):
    torch.save(modelo.state_dict(), 'modelo_mario.pth')

##################################################################################

def carregar_modelo(modelo):
    modelo.load_state_dict(torch.load('modelo_mario.pth'))
    modelo.eval()

##################################################################################

def selecionarAcao(estado, epsilon):
    if np.random.rand() <=epsilon:
        return np.random.choice(range(output_size))
    else:
        with torch.no_grad():
            q_valores = modelo(torch.Tensor(estado))
            return np.argmax(q_valores.numpy())
        
##################################################################################

def run(episodios, treinamento=True, render=False):
    estados_batch = []
    acoes_batch = []
    recompensas_batch = []
    novos_estados_batch = []
    terminados_batch = []

    lr = 0.9 # Taxa de aprendizado
    desconto = 0.9 # gamma
    if treinamento:
        epsilon = 1
    else:
        epsilon = 0
    epsilon_decay = 0.0025

    recompensa_por_episodio = []
    peso_passos = 0.1
    peso_terminar_fase = 1000
    peso_perder_vida = -1000

    if os.path.exists('modelo_mario.pth'):
        carregar_modelo(modelo)


    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1) # Inicia o env
    
    for episodio in range(episodios):
        env.reset()
        ram = getRam(env)
        estado = getInputs(ram, radius=raio_estado) #Pega o primeiro estado existente

        inicio = time.time()
        morto = False
        terminado = False

        passos = []

        recompensa = 0
        
        while (not morto and not terminado):
            x_atual = getPosicaoX(ram) # Pega a posição x do estado atual
            vidas_atual = getLives(ram)   # Pega as vidas do estado atual
            ram = getRam(env)
            
            acao = selecionarAcao(estado, epsilon)
            #print(acao)
            
            acao_ram = acoes_possiveis[acao]
            env.step([acao_ram])
            if render:
                env.render()
            ram = getRam(env)
            passos.append(1)
            novo_estado = getInputs(ram, radius=raio_estado)
            x_novo = getPosicaoX(ram) # pega a nova posição x

            recompensa = getRecompensa(recompensa, x_atual, x_novo, vidas_atual)

            if (vidas_atual < 4):
                morto = True

            tempo_percorrido = time.time()
            tempo_total = tempo_percorrido - inicio
            if tempo_total > 500:
                recompensa += peso_perder_vida
                morto = True
            
            terminado = getFimFase(x_atual, 4800)
            if terminado:
                penalizacao_passo = len(passos) * peso_passos
                recompensa = recompensa + peso_terminar_fase - penalizacao_passo # Recompensa por terminar a fase

            estados_batch.append(estado)
            acoes_batch.append(acao)
            recompensas_batch.append(recompensa)
            novos_estados_batch.append(novo_estado)
            terminados_batch.append(terminado)

            estado = novo_estado

            # Treinamento
            if treinamento:
                if len(estados_batch) > batch_size:
                    estados_batch_np = np.array(estados_batch)
                    acoes_batch_np = np.array(acoes_batch)
                    recompensas_batch_np = np.array(recompensas_batch)
                    novos_estados_batch_np = np.array(novos_estados_batch)
                    terminados_batch_np = np.array(terminados_batch)

                    estados_batch_tensor = torch.Tensor(estados_batch_np)
                    novos_estados_batch_tensor = torch.Tensor(novos_estados_batch_np)
                    recompensas_batch_tensor = torch.Tensor(recompensas_batch_np)
                    acoes_batch_tensor = torch.Tensor(acoes_batch_np)
                    terminados_batch_tensor = torch.Tensor(terminados_batch_np)

                    q_valores = modelo(estados_batch_tensor)
                    novos_q_valores = modelo(novos_estados_batch_tensor)
                    targets = q_valores.clone()

                    
                    for i in range(batch_size):
                        indices_acoes = int(acoes_batch_tensor[i].item())
                        if terminados_batch_tensor[i]:
                            targets[i][indices_acoes] = recompensas_batch_tensor[i]
                        else:
                            targets[i][indices_acoes] = recompensas_batch_tensor[i] + desconto * torch.max(novos_q_valores[i])

                    optimizer.zero_grad()
                    loss = criterion(q_valores, targets)
                    loss.backward()
                    optimizer.step()

            estados_batch.clear()
            acoes_batch.clear()
            recompensas_batch.clear()
            novos_estados_batch.clear()
            terminados_batch.clear()

        if treinamento:
            print(f'Recompensa do episódio {episodio}: {recompensa:.2f}')
            recompensa_por_episodio.append(recompensa)

            epsilon = max(epsilon - epsilon_decay, 0)

            if(epsilon == 0):
                epsilon = epsilon_decay
            print(f'Epsilon: {epsilon}')

        salvar_modelo(modelo)

    env.close()

    if treinamento:
        plotarGrafico(recompensa_por_episodio, episodios)

if __name__ == '__main__':
    run(400, treinamento=False, render=True)