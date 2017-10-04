# Imitation-Learning

There are 2 task - 1) Reacher Evironment 2) Obstacle_reacher.
Implementation takes ideas from One-Shot Imitation Learning by Yan Duan, and Seq2Seq Models for language translation.

The Main Idea is 
Expertdemo --> LSTM --> Embedding, [Embedding,Curstate]-->Network-->Action.

Embedding acts a summary/context of the provided demo. It is concatenate with the current states to produce actions. 
