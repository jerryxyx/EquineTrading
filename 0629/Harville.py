import numpy as np
from operator import itemgetter, attrgetter, methodcaller

def power(list,n):
    return [x**n for x in list]

def Q2(p1, p2, prob, alpha):
    '''
    to calculate prob of runner 2 run 2nd when know runner 1 run first
    :param p1: prob of runner 1 win
    :param p2: prob of runner 2 win
    :param prob: prob list of every runner win
    :param alpha: para in Harville function
    :return: prob of runner 2 run 2nd when know runner 1 run first
    '''
    q2 = p2**alpha/(sum(power(prob, alpha)) - p1 ** alpha)
    return q2

def Q3(p1, p2, p3, prob, beta):
    '''
    to calculate prob of runner 3 run 3rd when know runner 1, 2 run first, second
    :param p1: prob of runner 1 win
    :param p2: prob of runner 2 win
    :param p3: prob of runner 3 win
    :param prob: prob list of every runner win
    :param beta: para in Harville function
    :return: prob of runner 3 run 3rd when know runner 1, 2 run first, second
    '''
    q3 = p3**beta/(sum(power(prob,beta))-p1**beta-p2**beta)
    return q3

def Q4(p1,p2,p3,p4,prob,sigma):
    '''
    to calculate prob of runner 4 run 4th when know runner 1, 2, 3 run first, second, third
    :param p1: prob of runner 1 win
    :param p2: prob of runner 2 win
    :param p3: prob of runner 3 win
    :param p4: prob of runner 4 win
    :param prob: prob list of every runner win
    :param sigma:  para in Harville function
    :return:  prob of runner 4 run 4th when know runner 1, 2, 3 run first, second, third
    '''
    q4 = p4 ** sigma / (sum(power(prob, sigma)) - p1 ** sigma - p2 ** sigma - p3 ** sigma)
    return q4

def harville_multiprob(id, win_probs, bet_type, alpha=1.0, beta=1.0, sigma=1.0):
    '''
    :param id: horse/runner place in a race, like 1, 2, 3...
    :param win_probs: the prob of win of the corresponding id
    :param bet_type: bet type, can be 'Exacta', 'Trifacta', 'Superfecta'
    :param alpha: parameter in discounted Harvlle formula
    :param beta: parameter in discounted Harvlle formula
    :param sigma: parameter in discounted Harvlle formula
    :return:2-d list, for each row, last element is the prob that combination win, first elements are id combination. It is ranked decending according to prob
    '''
    n = len(id)
    sump = sum(win_probs)
    prob = [x/sump for x in win_probs]# in case probs are not normalized
    M = []
    if bet_type == 'Win':
        for i in range(n):
            p1 = prob[i]
            row = [id[i], p1]
            M.append(row)
    elif bet_type == 'Exacta':#exacta
        for i in range(n):
            for j in range(n):
                if i!=j:
                    p1 = prob[i]
                    p2 = prob[j]
                    q2 = Q2(p1, p2, prob, alpha)
                    row = [id[i], id[j], p1 * q2]
                    M.append(row)
    elif bet_type == "Quinella":  # Exacta box, which is also equivalent to harville_box function with bet_type 'Exacta'
        for i in range(n):
            for j in range((i + 1), n):
                p1 = prob[i]
                p2 = prob[j]
                row = [id[i], id[j], p1 * Q2(p1, p2, prob, alpha) + p2 * Q2(p2, p1, prob, alpha)]
                M.append(row)
    elif bet_type == 'Trifecta':
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i-j)*(k-j)*(i-k)!=0:
                        p1 = prob[i]
                        p2 = prob[j]
                        p3 = prob[k]
                        q2 = Q2(p1, p2, prob, alpha)
                        q3 = Q3(p1, p2, p3, prob, beta)
                        row = [id[i],id[j],id[k],p1*q2*q3]
                        M.append(row)
    elif bet_type == 'Superfecta':  #Superfecta
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        nunique = len(np.unique([i,j,k,l]))
                        if nunique  == 4:
                            p1 = prob[i]
                            p2 = prob[j]
                            p3 = prob[k]
                            p4 = prob[l]
                            q2 = Q2(p1, p2, prob, alpha)
                            q3 = Q3(p1, p2, p3, prob, beta)
                            q4 = Q4(p1,p2,p3,p4,prob,sigma)
                            row = [id[i],id[j],id[k],p1*q2*q3*q4]
                            M.append(row)
    else:
        print('bet type unknown')
        return None
    M.sort(key=itemgetter(-1),reverse=True)
    return M

def harville_box(id, win_probs, bet_type, alpha=1.0, beta=1.0, sigma=1.0):
    '''
    to calculate box combination probability
    :param id: horse/runner place in a race, like 1, 2, 3...
    :param win_probs: the prob of win of the corresponding id
    :param bet_type: bet type, can be 'Exacta', 'Trifacta', 'Superfecta'
    :param alpha: parameter in discounted Harvlle formula
    :param beta: parameter in discounted Harvlle formula
    :param sigma: parameter in discounted Harvlle formula
    :return:2-d list, for each row, last element is the prob that combination win, first elements are id combination. It is ranked decending according to prob
    '''
    n = len(id);
    sump = sum(win_probs)
    prob = [x/sump for x in win_probs]# in case probs are not normalized
    M = [];
    if bet_type == 'Exacta':#exacta
        for i in range(n):
            for j in range((i+1),n):
                p1 = prob[i]
                p2 = prob[j]
                row = [id[i], id[j], p1 * Q2(p1, p2, prob, alpha) + p2 * Q2(p2, p1, prob, alpha)]
                M.append(row)

    else:
        print('bet type unknown')
        return None
    M.sort(key=itemgetter(-1),reverse=True)
    return M