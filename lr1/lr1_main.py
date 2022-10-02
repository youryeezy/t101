from random import choice, shuffle, randint
from time import time


def generate_simple_rules(code_max, n_max, n_generate, log_oper_choice=["and","or","not"]):
    rules = []
    for j in range(0, n_generate):
        log_oper = choice(log_oper_choice)  #not means and-not (neither)
        if n_max < 2:
            n_max = 2
        n_items = randint(2, n_max)
        items = []
        for i in range(0, n_items):
            items.append(randint(1, code_max))
        rule = {
              'if': {
                  log_oper:	 items
               },
               'then': code_max+j
            }
        rules.append(rule)
    shuffle(rules)
    return(rules)


def generate_seq_facts(M):
    facts = list(range(0, M))
    shuffle(facts)
    return facts

def generate_rand_facts(code_max, M):
    facts = []
    for i in range(0, M):
        facts.append(randint(0, code_max))
    return facts
















#samples:
print(generate_simple_rules(100, 4, 10))


#generate rules and facts and check time
time_start = time()
N = 100000
M = 1000
rules = generate_simple_rules(100, 4, N)
facts = generate_rand_facts(100, M)
print("%d rules generated in %f seconds" % (N, time()-time_start))

time_start = time()

#time_result = time() - time_start
#print(res)
# YOUR CODE HERE

#print("%d facts validated vs %d rules in %f seconds" % (M, N, time_result))
