D:\Anaconda3\envs\py310\python.exe D:\code\others\multi_jsp_abc\t.py
[[12 15  8 13  2  6 10  4 11 14  9  7  1  5  3]
 [ 7  8  4  6  2 11  5 15 10  1 13 14 12  9  3]
 [ 9  7 13 14  2 10 12 15  6  3  8 11  1  4  5]
 [ 8 11  7  1  3 14 13  6 12  2  4  9 10  5 15]
 [15  4 10 11  7  1 12  5  3  6 13 14  8  9  2]
 [ 8  3  9 11  1 12  2  6 14  7 15  5  4 13 10]
 [ 8 15  1 13  9 11  2 14  7  5  4  3 12  6 10]
 [ 7 13  5  4 14  2  6  1  9 11 12  8 15  3 10]
 [12  5  6  1  9  7  3  4 11 13  8  2 14 10 15]
 [14  2  7  8  4  1  9  3 11 10  6 12 15 13  5]
 [13  1  9 14 11 15  4  3  8  7  2  6 10 12  5]
 [14  9  5  6  4  2  3 13 11 10  7 12  1  8 15]
 [ 8  1  6 10 12  9 11  4 14 13  5  2 15  3  7]
 [ 9 15  6 10  2 13 11 12  8 14  4  5  1  3  7]
 [ 1  8  7 14  3  6 10  9  5 11  2  4 12 15 13]]
初始ms  5257.0
initialize  0.13700604438781738
empploy 之后 5257.0
onlooker 之后 4844.0
trials  1
trials  0
trials  1
trials  0
trials  0
scout 之后 4844.0
empploy 之后 4844.0
onlooker 之后 4741.0
trials  0
trials  1
trials  5
trials  1
trials  0
scout 之后 4741.0
empploy 之后 4741.0
onlooker 之后 4741.0
trials  1
trials  2
trials  6
ch, [77  7]
trials  2
trials  3
scout 之后 4741.0
empploy 之后 4741.0
onlooker 之后 4741.0
trials  4
trials  0
trials  9
ch, [134   6]
trials  4
trials  4
scout 之后 4741.0
empploy 之后 4741.0
onlooker 之后 4741.0
trials  5
trials  3
trials  12
ch, [239  12]
Traceback (most recent call last):
  File "D:\code\others\multi_jsp_abc\t.py", line 55, in <module>
    best_solution = abc.optimize()
  File "D:\code\others\multi_jsp_abc\abc_\swarm.py", line 417, in optimize
    self.scout_bees_stage()
  File "D:\code\others\multi_jsp_abc\abc_\swarm.py", line 381, in scout_bees_stage
    g_list_new, opids_new = self.generate_agent_block_solution(i)
  File "D:\code\others\multi_jsp_abc\abc_\swarm.py", line 190, in generate_agent_block_solution
    rand_agent_block[rand_key] = agent_blocks[rand_key]
TypeError: unhashable type: 'list'

Process finished with exit code 1
