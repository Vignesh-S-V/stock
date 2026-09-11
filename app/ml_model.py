from __future__ import annotations
import threading, time
from dataclasses import dataclass
from typing import Any
import numpy as np, pandas as pd, requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from app.trading import add_indicators

FEATURES=["RSI_14","MACD","MACD_HIST","ATR_PCT","BB_WIDTH","VWAP_DIST","ROC_12","RET_1","RET_5","RET_20","ADX_14","STOCH_K","STOCH_D","CCI_20","EMA9_21","EMA21_50","VOLUME_RATIO"]
MODEL_TTL_SECONDS=600; MODEL_THRESHOLD_DEFAULT=95.0; MODEL_MIN_SAMPLES=700; TRAIN_PERIOD_SECONDS=60*86400
_MODEL_CACHE:dict[str,tuple[float,Any,dict[str,Any]]]={}; _MODEL_LOCK=threading.Lock(); _TRAINING_SYMBOLS:set[str]=set()

@dataclass
class MLSignal:
    action:str; confidence:float; trained:bool; samples:int; validation_accuracy:float|None; threshold:float; reason:str

def _prepare_features(df):
    x=df.copy(); close=pd.to_numeric(x.get("Close"),errors="coerce"); volume=pd.to_numeric(x.get("Volume"),errors="coerce").fillna(0.0); x["Volume"]=volume
    x["ATR_PCT"]=pd.to_numeric(x.get("ATR_14"),errors="coerce")/close.replace(0,np.nan)*100
    x["VWAP_DIST"]=(close-pd.to_numeric(x.get("VWAP"),errors="coerce"))/close.replace(0,np.nan)*100
    x["EMA9_21"]=(pd.to_numeric(x.get("EMA_9"),errors="coerce")/pd.to_numeric(x.get("EMA_21"),errors="coerce")-1)*100
    x["EMA21_50"]=(pd.to_numeric(x.get("EMA_21"),errors="coerce")/pd.to_numeric(x.get("EMA_50"),errors="coerce")-1)*100
    volume_sma=pd.to_numeric(x.get("VOLUME_SMA_20"),errors="coerce"); x["VOLUME_RATIO"]=volume/volume_sma.replace(0,np.nan)
    return x.replace([np.inf,-np.inf],np.nan)

def _training_set(df):
    x=_prepare_features(df); close=pd.to_numeric(x["Close"],errors="coerce"); future_return=close.shift(-3)/close-1.0; volatility=close.pct_change().rolling(30).std()
    adaptive=(volatility*3.0*0.75).clip(lower=0.0006,upper=0.0035)
    y=pd.Series(np.nan,index=x.index,dtype=float); y.loc[future_return>adaptive]=1.0; y.loc[future_return<-adaptive]=0.0
    data=x[FEATURES].copy(); mask=data.notna().all(axis=1)&y.notna(); X,y=X_y=data.loc[mask],y.loc[mask].astype(int)
    # Keep the volatility-aware target when it has enough examples, but avoid
    # calibration failure when a long one-sided market period removes a class.
    if len(y) and y.nunique()<2 or (len(y) and y.value_counts(normalize=True).min()<0.05):
        fallback=pd.Series(np.nan,index=x.index,dtype=float)
        fallback.loc[future_return>0.0005]=1.0; fallback.loc[future_return<-0.0005]=0.0
        mask=data.notna().all(axis=1)&fallback.notna(); X,y=data.loc[mask],fallback.loc[mask].astype(int)
    if len(y) and y.nunique()<2:
        fallback=(future_return>0).astype(float).where(future_return.notna())
        mask=data.notna().all(axis=1)&fallback.notna(); X,y=data.loc[mask],fallback.loc[mask].astype(int)
    return X,y

def _fetch_training_frame(symbol):
    now=int(time.time()); url=f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"; params={"period1":now-TRAIN_PERIOD_SECONDS,"period2":now+5,"interval":"5m","includePrePost":"true","events":"div,splits","_":str(now)}
    try:
        r=requests.get(url,params=params,headers={"Cache-Control":"no-cache","Pragma":"no-cache","User-Agent":"Mozilla/5.0"},timeout=12); r.raise_for_status(); result=(r.json().get("chart",{}).get("result") or [None])[0]
        if not result:return pd.DataFrame()
        ts=result.get("timestamp",[]); q=(result.get("indicators",{}).get("quote") or [{}])[0]
        if not ts:return pd.DataFrame()
        f=pd.DataFrame({"Open":q.get("open",[]),"High":q.get("high",[]),"Low":q.get("low",[]),"Close":q.get("close",[]),"Volume":q.get("volume",[])},index=pd.to_datetime(ts,unit="s",utc=True).tz_convert(None)); f=f.dropna(subset=["Open","High","Low","Close"]); f["Volume"]=pd.to_numeric(f["Volume"],errors="coerce").fillna(0.0); return f.loc[~f.index.duplicated(keep="last")].sort_index()
    except Exception:return pd.DataFrame()

def _calibrated_model(base,X,y):
    # Time-ordered validation remains the quality gate in _fit. Calibration
    # itself uses stratified folds so every calibration fold contains both
    # classes even during strongly one-sided market periods.
    if y.nunique()<2 or y.value_counts().min()<3:return None
    folds=min(3,int(y.value_counts().min()))
    if folds<2:return None
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    model=CalibratedClassifierCV(estimator=base,method="sigmoid",cv=splitter,ensemble=True); model.fit(X,y); return model

def _fit(df,symbol):
    training=_fetch_training_frame(symbol)
    if len(training)<1000:training=df.copy()
    if training.empty:return None,{"samples":0,"validation_accuracy":None,"reason":"No training data was returned."}
    X,y=_training_set(add_indicators(training))
    if len(X)<MODEL_MIN_SAMPLES or y.nunique()<2:return None,{"samples":int(len(X)),"validation_accuracy":None,"reason":f"Training set has {len(X)} rows and {y.nunique()} classes."}
    split=int(len(X)*0.80); X_train,X_test=X.iloc[:split],X.iloc[split:]; y_train,y_test=y.iloc[:split],y.iloc[split:]
    if y_train.nunique()<2 or y_test.nunique()<2:return None,{"samples":int(len(X)),"validation_accuracy":None,"reason":"Chronological validation window contains only one class."}
    rf=RandomForestClassifier(n_estimators=140,max_depth=10,min_samples_leaf=6,class_weight="balanced_subsample",random_state=42,n_jobs=1); hgb=HistGradientBoostingClassifier(max_iter=140,learning_rate=0.035,max_leaf_nodes=15,l2_regularization=1.0,random_state=42); base=VotingClassifier(estimators=[("rf",rf),("hgb",hgb)],voting="soft",weights=[1.0,1.3],n_jobs=1)
    try:
        vm=_calibrated_model(base,X_train,y_train)
        if vm is None:return None,{"samples":int(len(X)),"validation_accuracy":None,"reason":"Calibration training window has insufficient class diversity."}
        acc=float(balanced_accuracy_score(y_test,vm.predict(X_test))); fm=_calibrated_model(base,X,y)
        if fm is None:return None,{"samples":int(len(X)),"validation_accuracy":acc,"reason":"Final calibration window has insufficient class diversity."}
    except (ValueError,RuntimeError,TypeError) as exc:return None,{"samples":int(len(X)),"validation_accuracy":None,"reason":f"Calibration failed: {type(exc).__name__}."}
    return fm,{"samples":int(len(X)),"validation_accuracy":acc,"reason":"Calibrated model ready."}

def _prediction_frame(df):
    x=df[["Open","High","Low","Close","Volume"]].copy()
    for c in ["Open","High","Low","Close","Volume"]:x[c]=pd.to_numeric(x[c],errors="coerce")
    x["Volume"]=x["Volume"].fillna(0.0); x=x.dropna(subset=["Open","High","Low","Close"])
    if len(x)>=30:x=x.resample("5min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}); x=x.dropna(subset=["Open","High","Low","Close"]); x=add_indicators(x)
    return x

def model_is_ready(symbol):
    now=time.time()
    with _MODEL_LOCK:
        cached=_MODEL_CACHE.get(symbol); return bool(cached and cached[1] is not None and now-cached[0]<MODEL_TTL_SECONDS)

def prewarm_ml_model(df,symbol):
    now=time.time()
    with _MODEL_LOCK:
        cached=_MODEL_CACHE.get(symbol)
        if cached and cached[1] is not None and now-cached[0]<MODEL_TTL_SECONDS:return
        if symbol in _TRAINING_SYMBOLS:return
        _TRAINING_SYMBOLS.add(symbol)
    try:
        model,meta=_fit(df,symbol)
        with _MODEL_LOCK:_MODEL_CACHE[symbol]=(time.time(),model,meta)
    finally:
        with _MODEL_LOCK:_TRAINING_SYMBOLS.discard(symbol)

def predict_ml_signal(df,symbol,threshold=MODEL_THRESHOLD_DEFAULT,train_if_missing=True):
    threshold=min(100.0,max(50.0,float(threshold))); now=time.time()
    with _MODEL_LOCK:
        cached=_MODEL_CACHE.get(symbol)
        if cached and now-cached[0]<MODEL_TTL_SECONDS: model,meta=cached[1],cached[2]
        elif not train_if_missing:return MLSignal("HOLD",0.0,False,0,None,threshold,"Model is training in the background.")
        else:model=None; meta=None
    if model is None:
        model,meta=_fit(df,symbol)
        with _MODEL_LOCK:_MODEL_CACHE[symbol]=(time.time(),model,meta)
    if model is None:return MLSignal("HOLD",0.0,False,int(meta.get("samples",0)),meta.get("validation_accuracy"),threshold,meta.get("reason","Model did not meet minimum data/calibration requirements."))
    prediction_df=_prediction_frame(df)
    if prediction_df.empty:return MLSignal("HOLD",0.0,True,int(meta["samples"]),meta.get("validation_accuracy"),threshold,"Live prediction timeframe is incomplete.")
    features=_prepare_features(prediction_df)[FEATURES]; complete=features.dropna(how="any")
    if complete.empty:
        missing=[c for c in FEATURES if features[c].isna().all()]; return MLSignal("HOLD",0.0,True,int(meta["samples"]),meta.get("validation_accuracy"),threshold,f"Live feature history is incomplete ({', '.join(missing) if missing else 'no complete row'}).")
    latest_index=features.index[-1]; latest_complete_index=complete.index[-1]; latest=complete.iloc[[-1]]; probabilities=model.predict_proba(latest)[0]; classes=list(model.classes_); p_down=float(probabilities[classes.index(0)]) if 0 in classes else 0.0; p_up=float(probabilities[classes.index(1)]) if 1 in classes else 0.0; confidence=round(min(100.0,max(0.0,max(p_up,p_down)*100.0)),1)
    if p_up>=threshold/100.0 and p_up>p_down:action="BUY"
    elif p_down>=threshold/100.0 and p_down>p_up:action="SELL"
    else:action="HOLD"
    reason=f"Calibrated ML probability {confidence:.1f}% {'reached' if action!='HOLD' else 'is below'} the {threshold:.1f}% trade threshold."
    if latest_complete_index!=latest_index:reason+=" Latest 5-minute bar was incomplete, so the last complete bar was used."
    return MLSignal(action,confidence,True,int(meta["samples"]),meta.get("validation_accuracy"),threshold,reason)
