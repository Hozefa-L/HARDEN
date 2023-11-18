# HARDEN
Detecting Reentrancy Vulnerability in Smart Contracts using Graph Convolution Networks

This project's aim was to detect reentrancy vulnerability in Ethereum Smart Contracts.
Contains 342 smart contracts .dot dataset used for training and testing HARDEN Algorithm.

Below are important libraries required to implement HARDEN Algorithm in python:
DGL (Deep Graph Library) - Used to Load, train and test Graph Neural Network Model
Pytorch - backend environment for computation
Networkx - Can be used to manipulate graphs
Ethersolve - Takes bytecode as input and returns sound Control Flow graphs

#Reentrancy_dataset contains smart contract dot graphs dataset 
used for reentrancy detection

#HARDEN_Bytecode_Graph_Generation can be used to convert smart contracts to evm bytecodes 
to dot Control Flow graphs

#HARDEN_Load_Test_Train_GCN loads graph dataset and GCN model using DGL
to train and test GCN for reentrancy detection in Ethereum Smart Contracts.
