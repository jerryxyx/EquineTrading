import scipy.spatial.distance as ssd
from scipy.stats import norm
import math
import itertools
import numpy as np
import pandas as pd

from math import ceil

def generate_combinations_factors(balls, num_balls):
    '''Generate combinations from N codes and k selections '''
    return [list(p) for p in itertools.permutations(balls, num_balls)]


def generate_system_space(balls, numballs2buckets):
    '''Generates all possible systems meeting criteria'''
    balls_in = {}
    for bucket, nballs in numballs2buckets.items():
        balls_in[bucket] = generate_combinations_factors(balls, nballs)
    return balls_in


def compute_probs_from_odds(odds):
    #TODO: handle NaNs and zeros
    dirty_probs = 1.0 / (odds + 1)
    clean_probs = dirty_probs / dirty_probs.sum()
    return clean_probs


def dmetric_L1_weighted(a_vector,b_vector, weight, funcdist):
    return ssd.minkowski(a_vector, b_vector, 1)


def log_safe(x,b):
    if x is None or x <= 0:
        # print("Log of {}".format(x))
        return 0
    else:
        return math.log(x,b)


def kl(p, q):
    """
    Specifically, the Kullback–Leibler divergence from Q to P, denoted DKL(P‖Q), is
    a measure of the information gained when one revises one's beliefs from the
    prior probability distribution Q to the posterior probability distribution P. In
    other words, it is the amount of information lost when Q is used to approximate P.
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def week_of_month(dt):
    """
    Returns int of the week of the month for the specified date. Will always be 1-5
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


def get_freq_wom(target_date):
    # Here we determine what dates to use for simulation

    target_datetime = pd.to_datetime(target_date)  # use pandas datetime for datetime functions
    target_weekday_name = pd.to_datetime(target_datetime).weekday_name  # 7:Sunday, 6:Saturday
    target_weekday_prefix = target_weekday_name[0:3].upper()
    target_wom = week_of_month(target_date)

    # 'WOM-1SUN is first sunday of month
    freq_wom = 'WOM-' + str(target_wom) + target_weekday_prefix

    return freq_wom


def mean_best_N_of_K(row, n, k):
    # e.g.
    # df[['HDWSpeedRating_0', 'HDWSpeedRating_1', 'HDWSpeedRating_2']].apply(lambda row: mean_best_N_of_K(row, n=2, k=3), axis=1)
    return row[0:k].nlargest(n).mean()


class ScoreToProbViaIntegral(object):
    def __init__(self, func, scoreLabel):
        self.func = func
        self.scoreLabel = scoreLabel

    def __call__(self, df, addIndex=False):
        scores = self.func(df)

        try:
            scores = pd.Series(scores)
            clean_scores = scores[scores > 0]
            clean_median = np.median(clean_scores)
            mean_score = scores[scores > 0].mean()
        except:
            print("no scores")
            return None

        try:
            scores = (scores - scores.mean()) / scores.std()
        except:
            print("could not compute normalized score")
            return None

        pdf, cdf = self.probDists(scores)
        pdfSeries = pd.Series(pdf).transpose()
        cdfSeries = pd.Series(cdf).transpose()
        probw = {}

        for winner in pdfSeries.index:
            probw[winner] = self.marginrunner(cdfSeries, pdfSeries, winner)
        probs = pd.Series(probw)
        probs = probs / probs.sum()

        if addIndex:
            probs_order = probs.order(ascending=False)
            idxABC = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
            idxRunners = probs_order.index.values
            idxZipABC = pd.MultiIndex.from_tuples(zip(idxABC, idxRunners))
            probs_order.index = idxZipABC
        return(probs)

    def marginrunner(self, cdf, pdf, runner):
        '''Computes the win probs from for each horse from cdf and pdf'''
        '''std : standard deviation of score'''
        '''incr: discretization for solving integral'''
        cdfdrop = cdf.drop(runner)
        pdfmult = pdf.ix[runner,]
        # print(("Starting {}:\n{}".format(runner, sum(pdfmult)))
        for w in cdfdrop.index:
            pdfmult = pdfmult * cdfdrop.ix[w,]
            # print(("After {}:\n{}".format(runner, sum(pdfmult)))
        sumtest = sum(pdfmult)
        # print(("{} {}".format(runner, sumtest))
        return sumtest

    def probDists(self, scores, incr=.25, width=8.0):
        '''computes probabilities by assuming normal distribution of outcomes relative to score'''
        range = np.arange(-width, width, incr)
        probintegral = {}
        pdfslice = {}
        for s in scores.index:
            cdfdict = {}
            pdfdict = {}
            dist = norm(scores[s], 1)
            for r in range:
                cdfdict[r] = dist.cdf(r)
                pdfdict[r] = dist.pdf(r)
            cdfseries = pd.Series(cdfdict)
            probintegral[s] = cdfseries
            pdfseries = pd.Series(pdfdict)
            pdfslice[s] = pdfseries
        return (pdfslice, probintegral)

    def __str__(self):
        return "ScoreToProbViaIntegral({!r})".format(self.scoreLabel)


# using load_benchmark functions to get advantage
def compute_payout(df, attr_model, bet_amount_equal, bet_amount_inequal, bet_on='final_tote_odds'):
    """
    Add columns for quick calculation of Win bets % payout
    :param df: Dataframe from dataset for multiple races
    :param attr_model: (string)an attribute / prob or score in the dataframe that can be ranked
    :param bet_amount_equal: bet_amount when ranking 1 runner according to attr_model is the same with favourate runner according to final tote odds
    :param bet_amount_inequal: bet_amount when ranking 1 runner according to attr_model is different with favourate runner according to final tote odds
    :param bet_on: if bet on ranking 1 runner according to attr_model or favourate runner according to final tote odds
    :return: Dataframe with columns added

    """
    df['is_win'] = df['official_finish_position'].map(lambda x: int(x == 1))
    df['rank_' + attr_model] = df.groupby('race_id')[attr_model].transform(lambda x: x.rank(ascending=False))
    if bet_amount_equal == 'strat_double':
        df['bet_amount'] = df['rank_' + bet_on].map(lambda x: int(x < 1.5) * 2.0)
    elif bet_amount_equal == 'strat_pass':
        df['bet_amount'] = df['rank_' + bet_on].map(lambda x: int(x < 1.5) * 0.0)
    elif bet_amount_equal == 'strat_unchanged':
        df['bet_amount'] = df['rank_' + bet_on].map(lambda x: int(x < 1.5) * 1.0)
    elif bet_amount_equal == 'strat_inverse_scaled':
        df['bet_amount'] = df['rank_' + bet_on].map(lambda x: int(x < 1.5) * 1.0) / df[attr_model]
    else:
        print('bet_amount_equal error!')
        return (None)

    symbol = ['final_tote_odds', attr_model]
    symbol.remove(bet_on)
    symbol_left = symbol[0]

    if bet_amount_inequal == 'strat_double':
        df.loc[(df['rank_' + bet_on] < 1.5) & (df['rank_' + symbol_left] > 1.5), 'bet_amount'] = 2.0
    elif bet_amount_inequal == 'strat_pass':
        df.loc[(df['rank_' + bet_on] < 1.5) & (df['rank_' + symbol_left] > 1.5), 'bet_amount'] = 0.0
    elif bet_amount_inequal == 'strat_unchanged':
        df.loc[(df['rank_' + bet_on] < 1.5) & (df['rank_' + symbol_left] > 1.5), 'bet_amount'] = 1.0
    elif bet_amount_inequal == 'strat_inverse_scaled':
        df.loc[(df['rank_' + bet_on] < 1.5) & (df['rank_' + symbol_left] > 1.5), 'bet_amount'] = 1 / df.loc[
            (df['rank_' + bet_on] < 1.5) & (df['rank_' + symbol_left] > 1.5), attr_model]
    else:
        print('bet_amount_inequal error!')
        return (None)

    df['is_wager'] = df['bet_amount'].map(lambda x: int(x > 0))
    df['is_paid'] = df['is_wager'] * df['is_win']
    df['payout'] = df['is_win'] * df['bet_amount'] * df['payout_win'].fillna(0.0)

    return df


def compute_advantage(df):
    if sum(df['bet_amount']) == 0:
        advantage = 0
    else:
        pct_win = df.groupby('race_id')['is_paid'].sum().value_counts(normalize=True)[1]

        pct_loss = 1.0 - pct_win
        mean_odds = df[df.is_paid > 0]['final_tote_odds'].mean()
        advantage = pct_win - pct_loss / mean_odds
        # print(advantage)
    return advantage


def compute_Return(df):
    if sum(df['bet_amount']) == 0:
        Return = 0
    else:
        Return = (sum(df[df.is_paid > 0]['payout']) - sum(df['bet_amount'])) / sum(df['bet_amount'])
    return Return