import numpy as np
import pandas as pd
import json
# lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
plt.style.use('tableau-colorblind10')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

# homemade
from libs.mtd.features import trnd_scan, tautil
from libs.mtd.labeling import labeling
from libs.mtd.triple_barrier import get_barrier, make_rt
from libs.mtd.backtest import print_round_trip_stats, round_trip

class Config(dict): 
    '''
    MTD Strategy 매매규칙
    configuration json을 읽어들이는 class
    '''
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

class MTDTradingStrategy:
    """
    MTD 매매 시뮬레이션 및 결과 도출 Class
    Arguments
    ---------
    Momentum 확률 : DataFrame
    """

    def __init__(self, mtd_prob=None):
        self.mtd_prob = mtd_prob
    
    def simulate(self, feature, signals):
        #signals = pd.read_csv('./data/mtd/momentum_signals.csv')
        #signals=signals['mean_'].rename('signals').loc['2010':]
        # signals.columns = ['signals']
        signals= signals.loc['2010':]
        
        scaler = normalize
        scaler2 = MinMaxScaler()
        signals = pd.Series(scaler2.fit_transform(normalize(signals.values.reshape(-1,1),axis=0)).reshape((-1,)), 
                                index=signals.index).rename('signals')
        thresholds = [0, 0.5]
        
        enter_ml_list=[]
        for h in thresholds:
            enter_ml_list.append(signals.loc[signals>h].index)
                
        open = feature.open
        close = feature.close
        rsi = tautil.RSIIndicator(open,14).rsi().dropna()
        long = ~(~((rsi>=50) & (rsi<70)) & ~((rsi<=30) & (rsi.pct_change()>0)))
        enter_ta = rsi.loc[long].index
        
        enter_list = []
        for i in range(len(enter_ml_list)):
            enter_list.append((enter_ml_list[i] & enter_ta).sort_values())
            
        # no Rule (benchmark)
        pt_sl_bm = [1000,1000]
        max_holding_bm = [60, 0]
        no_exit_rule = [pt_sl_bm,max_holding_bm]
        
        #fixed target rules
        max_holding = [60, 0]
        
        #dynamic target rule
        close_ = feature.close
        changes = close_.pct_change(1).to_frame()
        for i in range(2,121):
            changes = changes.join(close_.pct_change(i).rename('close {}'.format(i)))
        dynamic_target = changes.abs().dropna().mean(axis=1)['2010':]
        
        # trading simulation

        barrier_exit_list=[]
        #for i in range(len(exit_rule_list)):
        #    barrier_exit_list.append(get_barrier(close, enter_list[0], exit_rule_list[i][0],exit_rule_list[i][1]))
            
        barrier_exit_list.append(get_barrier(close, enter_list[1], [1,1], max_holding, target = dynamic_target))  #dynamic  

        rts_exit_list=[]
        for i in range(len(barrier_exit_list)):
            rts_exit_list.append(make_rt(close,barrier_exit_list[i].dropna()))
        
        barrier_bm1 = get_barrier(close, close.index, no_exit_rule[0], no_exit_rule[1])
        barrier_bm2 = get_barrier(close, close.index, [1,1], max_holding, target = dynamic_target)
        barrier_bm3 = get_barrier(close, enter_list[1], no_exit_rule[0], no_exit_rule[1])
        
        rts_bm1 = make_rt(close,barrier_bm1.dropna())
        rts_bm2 = make_rt(close,barrier_bm2.dropna())
        rts_bm3 = make_rt(close,barrier_bm3.dropna())
        round_trip.get_df_ann_sr_tb(rts_bm1,'No Rule (BM)',years=11)
        
        # trading simulation
        barrier_enter_list=[]
        for i in range(len(enter_list)):
            barrier_enter_list.append(get_barrier(close, enter_list[i], [1,1], max_holding, target = dynamic_target))
            
        rts_enter_list=[]
        for i in range(len(barrier_enter_list)):
            rts_enter_list.append(make_rt(close,barrier_enter_list[i].dropna()))
            
        enter_result_df = pd.concat([round_trip.get_df_ann_sr_tb(rts_bm2,'No Enter Rule',years=11)], axis=1)
        for i in range(len(rts_enter_list)):
            enter_result_df = enter_result_df.join(round_trip.get_df_ann_sr_tb(rts_enter_list[i],'Enter Rule {}'.format(i+1)))
        
        # Exit Rule
        exit_result_df = pd.concat([round_trip.get_df_ann_sr_tb(rts_bm3,'No Exit Rule')], axis=1)
        for i in range(len(rts_exit_list)):
            exit_result_df = exit_result_df.join(round_trip.get_df_ann_sr_tb(rts_exit_list[i],'Exit Rule'))
            
        # For Next Research ( 서강대 연구 필요 )
        barrier = get_barrier(close, enter_list[1], [1,1], max_holding, target = dynamic_target)
        rts = make_rt(close,barrier.dropna())
        round_trip.get_df_ann_sr_tb(rts,'Chose for 2nd model')
        
        #barrier.to_csv('./data/mtd/barrier.csv')
        
        return barrier