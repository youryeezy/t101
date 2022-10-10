from random import choice, shuffle, randint
from time import time


def generate_simple_rules(code_max, n_max, n_generate,):
    log_operation_choice = ["and", "or", "not"]
    seq_rules = []
    for j in range(0, n_generate):
        log_operation = choice(log_operation_choice)
        if n_max < 2:
            n_max = 2
        n_items = randint(2, n_max)
        items = []
        for i in range(0, n_items):
            items.append(randint(1, code_max))
        rule = {'if': {log_operation: items}, 'then': code_max + j}
        seq_rules.append(rule)
    shuffle(seq_rules)
    return seq_rules


def generate_seq_facts(m):
    seq_facts = list(range(0, m))
    shuffle(seq_facts)
    return seq_facts


def generate_rand_facts(code_max, m):
    seq_facts = []
    for i in range(0, m):
        seq_facts.append(randint(0, code_max))
    return seq_facts


def results(facts, rules):
    fact = set(facts)
    transit_results = []
    for i in rules:
        for j in i['if']:
            if j == 'or':
                for s in i['if'][j]:
                    if s in fact:
                        if len(transit_results) == 0:
                            fac_tmp = facts.copy()
                            transit_results.append({'if': fac_tmp, 'or': i['if'][j],
                                                    'then': i['then']})
                            facts.append(i['then'])
                            fact.add(i['then'])
                            break
                        else:
                            put = True
                            for ress in transit_results:
                                if 'or' in ress:
                                    if (ress['or'] == i['if'][j] and ress['then'] != i['then']) or (
                                            ress['or'] != i['if'][j] and ress['then'] == i['then']):
                                        put = False
                                        break
                                if 'and' in ress:
                                    if (ress['and'] == i['if'][j]) and (ress['then'] != i['then']):
                                        put = False
                                        break
                            if put is True:
                                fac = facts.copy()
                                transit_results.append({'if': fac, 'or': i['if'][j],
                                                        'then': i['then']})
                                facts.append(i['then'])
                                fact.add(i['then'])
            if j == 'not':
                count = len(i['if'][j])
                counter = 0
                for s in i['if'][j]:
                    if s not in fact:
                        counter = counter + 1
                    else:
                        break
                if counter == count:
                    if len(transit_results) == 0:
                        fac = facts.copy()
                        transit_results.append({'if': fac, 'not': i['if'][j], 'then': i['then']})
                        facts.append(i['then'])
                        fact.add(i['then'])
                    else:
                        put = True
                        for ress in transit_results:
                            if 'not' in ress:
                                if (ress['not'] == i['if'][j] and ress['then'] != i['then']) or (
                                        ress['not'] != i['if'][j] and ress['then'] == i['then']):
                                    put = False
                                    break
                        if put is True:
                            fac = facts.copy()
                            transit_results.append({'if': fac, 'not': i['if'][j], 'then': i['then']})
                            facts.append(i['then'])
                            fact.add(i['then'])
            if j == 'and':
                count = len(i['if'][j])
                counter = 0
                for s in i['if'][j]:
                    if s in fact:
                        counter = counter + 1
                    else:
                        break
                if counter == count:
                    if len(transit_results) == 0:
                        fac = facts.copy()
                        transit_results.append({'if': fac, 'and': i['if'][j], 'then': i['then']})
                        facts.append(i['then'])
                        fact.add(i['then'])
                    else:
                        put = True
                        for ress in transit_results:
                            if 'and' in ress:
                                if (ress['and'] == i['if'][j] and ress['then'] != i['then']) or (
                                        ress['and'] != i['if'][j] and ress['then'] == i['then']):
                                    put = False
                                    break
                        if put is True:
                            fac = facts.copy()
                            transit_results.append({'if': fac, 'and': i['if'][j],
                                                    'then': i['then']})
                            facts.append(i['then'])
                            fact.add(i['then'])
            if j == 'not':
                count = len(i['if'][j])
                counter = 0
                for s in i['if'][j]:
                    if s not in fact:
                        counter = counter + 1
                    else:
                        break
                if counter == count:
                    if len(transit_results) == 0:
                        fac = facts.copy()
                        transit_results.append({'if': fac, 'not': i['if'][j], 'then': i['then']})
                        facts.append(i['then'])
                        fact.add(i['then'])
                    else:
                        put = True
                        for ress in transit_results:
                            if 'not' in ress:
                                if (ress['not'] == i['if'][j] and ress['then'] != i['then']) or (
                                        ress['not'] != i['if'][j] and ress['then'] == i['then']):
                                    put = False
                                    break
                        if put is True:
                            fac = facts.copy()
                            transit_results.append({'if': fac, 'not': i['if'][j], 'then': i['then']})
                            facts.append(i['then'])
                            fact.add(i['then'])
    return transit_results


if '__main__' == __name__:

    time_start = time()

    total_rules = generate_simple_rules(100, 4, 10000)
    total_facts = generate_rand_facts(100, 1000)
    print("%d rules generated in %f seconds" % (1000, time()-time_start))

    time_start = time()
    result = results(total_facts, total_rules)
    time_result = time() - time_start
    print(time_result)

    print("%d facts validated for %d rules in %f seconds" % (1000, 10000, time_result))

