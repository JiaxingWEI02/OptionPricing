import pandas as pd
from IPython.display import display


annRT = lambda s: (s + 1).cumprod().iat[-1] ** (250 / len(s)) - 1
annVol = lambda s: s.std() * ((250) ** 0.5)
SR = lambda s: annRT(s) / annVol(s)
DD = lambda s: 1 - (s + 1).cumprod()/(s + 1).cumprod().expanding().max()
maxDD = lambda s: DD(s).max()
avgDD = lambda s: DD(s).mean()
maxDDStart = lambda s:DD(s).loc[:DD(s).idxmax()].iloc[::-1].idxmin().strftime("%Y-%m-%d")
maxDDEnd = lambda s:DD(s).idxmax().strftime("%Y-%m-%d")
dd_start = lambda s:s.sort_index().index[0].strftime("%Y-%m-%d")
dd_end = lambda s:s.sort_index().index[-1].strftime("%Y-%m-%d")

def pnl_stats(pnl):
    """
    pnl: pd.DataFrame(index= DateTime,columns=Factor, value = Daily Return) 
    """
    output=pnl.agg([dd_start,dd_end,SR,annRT,annVol,avgDD,maxDD,maxDDStart,maxDDEnd],axis=0).T
    if isinstance(pnl,pd.DataFrame):
        output.columns = ['From','To','SR','AnnRT','AnnVol',"AvgDD",'MaxDD',"MaxDDStart","MaxDDEnd"]

    elif isinstance(pnl,pd.Series):
        output.index = ['From','To','SR','AnnRT','AnnVol',"AvgDD",'MaxDD',"MaxDDStart","MaxDDEnd"]
    return output

def pnl_stats_byYear(pnl):
    output=pnl.groupby(pnl.index.strftime("%Y"),group_keys=True).apply(pnl_stats).unstack()
    return output

def stats(pnl):
    res=pnl.copy()
    res.index = pd.to_datetime(res.index)
    stats = pnl_stats(res)
    stats.name = "ALL"
    stats = pd.DataFrame(stats).T
    stats_year = pnl_stats_byYear(res)
    return pd.concat([stats_year,stats],axis=0)

pnl1 = ...
pnl2 = ...

# modified the strategy
# need to input the bar return of each strategy
res = pd.concat([pnl1,pnl2],axis=1).dropna(how='any')
res.columns = ['gd3w','mo3w','idx']
res.loc[:,'alpha_1'] = res.gd3w - res.idx
res.loc[:,'alpha_2'] = res.mo3w - res.idx
res.index=pd.to_datetime(res.index)
display(stats(res.alpha_1))
(res + 1).cumprod().loc[:,['gd3w','alpha_1','idx']].plot(figsize=(25,5),
                                                         grid=True,
                                                         title="gd3w cumprod pnl",
                                                         color = {'gd3w': 'red',"idx":"blue",'alpha_1':'yellow'})


# comparison of bar excess return
alpha_stats = pd.concat([pnl_stats(res[res.index.strftime("%Y") != '2017'].alpha_1),pnl_stats(res[res.index.strftime("%Y") != '2017'].alpha_2)],axis=1).T


# trading execution analysis
def execution_feedback(pos, cash, amt):
    '''
    pos, amt: pivot_table
    cash: initial aum
    The ratio of trading position's amt over whole market amt,
    represent the impact that each strategy has on the overall trading position
    '''
    threshold = ...
    feedback=((pos * cash).fillna(0).diff().abs()/amt).dropna(how='all',axis=1).dropna(how='all',axis=0).mean(axis=1)
    return feedback