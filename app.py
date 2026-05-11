


"""
GNR-Val v5.0 — Streamlit Industrial CAD Compliance Dashboard
Varroc Eureka 3.0 | Problem Statement 9
Run via: streamlit run app.py
"""

from __future__ import annotations
import io, json, logging, os, random, re, sys, time, warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

warnings.filterwarnings("ignore")

st.set_page_config(page_title="GNR-Val v5.0 | CAD Compliance",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gnrval")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    SEED: int = 42
    N_COMPLIANT: int = 400; N_REVIEW_NEEDED: int = 300; N_NONCOMPLIANT: int = 300
    MIN_COMPONENTS: int = 4; MAX_COMPONENTS: int = 14
    TRAIN_RATIO: float = 0.70; VAL_RATIO: float = 0.15
    IN_CHANNELS: int = 8; HIDDEN_DIM: int = 128; LATENT_DIM: int = 64
    DROPOUT: float = 0.25; EPOCHS: int = 120; BATCH_SIZE: int = 32
    LR: float = 1e-3; WEIGHT_DECAY: float = 1e-4; RECON_WEIGHT: float = 0.10
    GRAD_CLIP: float = 1.0; EARLY_STOP_PATIENCE: int = 20; LOG_EVERY: int = 10
    MODEL_PATH: str = "gnrval_final_combined.pt"
    REAL_MODEL_PATH: str = "gnrval_real_trained.pth"
    NONCOMPLIANT_THRESH: float = 70.0; REVIEW_THRESH: float = 40.0

cfg = Config()

def set_seed(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MATERIAL_CODES: Dict[str,int] = {"steel":0,"aluminum":1,"plastic":2,"titanium":3,"composite":4}
COMPONENT_TYPES: Dict[str,int] = {"shaft":0,"bearing":1,"housing":2,"bracket":3,"fastener":4,"gear":5}
INV_MAT  = {v:k for k,v in MATERIAL_CODES.items()}
INV_COMP = {v:k for k,v in COMPONENT_TYPES.items()}
RULE_CATEGORIES = {"TOL":"Tolerance","COMP":"Component-Specific","SURF":"Surface Finish",
    "DIM":"Dimensional","DFM":"Design for Manufacture","MAT":"Material Compatibility","JNTI":"Joint/Clearance"}
STANDARDS = {"ISO2768_FINE":0.10,"ISO2768_MEDIUM":0.30,"ISO2768_COARSE":0.80,
    "RA_MATING":1.60,"RA_GENERAL":3.20,"RA_ROUGH":6.30,"DIM_MIN":5.0,"DIM_MAX":500.0,
    "CL_CLOSE":0.05,"CL_MEDIUM":0.20,"CL_LOOSE":0.50,"CL_CRITICAL":1.00,"WEIGHT_LIMIT":5000.0}
COMPONENT_RULES = {
    "shaft":   {"max_tol":0.050,"max_ra":1.6,"min_dim":5.0, "max_dim":300.0,"max_ar":10,"std":"IT6-IT8"},
    "bearing": {"max_tol":0.025,"max_ra":0.8,"min_dim":10.0,"max_dim":200.0,"max_ar":2, "std":"IT5-IT6"},
    "gear":    {"max_tol":0.050,"max_ra":1.6,"min_dim":20.0,"max_dim":400.0,"max_ar":5, "std":"ISO 1328"},
    "housing": {"max_tol":0.300,"max_ra":3.2,"min_dim":20.0,"max_dim":500.0,"max_ar":5, "std":"IT8-IT11"},
    "bracket": {"max_tol":0.500,"max_ra":6.3,"min_dim":10.0,"max_dim":500.0,"max_ar":8, "std":"IT11-IT14"},
    "fastener":{"max_tol":0.150,"max_ra":3.2,"min_dim":3.0, "max_dim":100.0,"max_ar":8, "std":"ISO 4759"},
}
MATERIAL_COMPAT = {
    ("steel","steel"):True,("steel","aluminum"):True,("steel","composite"):True,
    ("aluminum","aluminum"):True,("aluminum","composite"):True,("titanium","steel"):True,
    ("titanium","composite"):True,("plastic","plastic"):True,("plastic","aluminum"):True,
    ("steel","plastic"):False,("titanium","aluminum"):False,("composite","plastic"):False,
}
JOINT_CL_LIMITS = {"fixed":0.10,"revolute":0.30,"prismatic":0.20,"contact":0.40}

# Floating-point epsilon: ensures that values exactly equal to a limit always PASS
_EPS = 1e-9

def material_compatible(m1_code,m2_code):
    m1=INV_MAT.get(int(m1_code),"steel"); m2=INV_MAT.get(int(m2_code),"steel")
    return MATERIAL_COMPAT.get((m1,m2),MATERIAL_COMPAT.get((m2,m1),True))

# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Violation:
    node:Any; rule_id:str; category:str; rule:str; value:float; limit:str
    severity:str; suggestion:str=""; passed:bool=False
    def to_dict(self): return asdict(self)

# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _check_tol(nominal,tolerance,actual):
    lo=nominal-abs(tolerance); hi=nominal+abs(tolerance); return (lo<=actual<=hi),lo,hi

def run_rule_engine(node_features,nominal_dims=None,nx_graph=None):
    violations,passed_checks=[],[]
    def _record(v): (passed_checks if v.passed else violations).append(v)
    for i,feat in enumerate(node_features):
        dim_x,dim_y,dim_z,tol,mat_code,type_code,weight,roughness=(float(f) for f in feat)
        nom=nominal_dims[i] if nominal_dims and i<len(nominal_dims) else (dim_x,dim_y,dim_z)
        nom_x,nom_y,nom_z=nom
        comp=INV_COMP.get(int(type_code),"housing"); cr=COMPONENT_RULES.get(comp,COMPONENT_RULES["housing"])
        for d,ax in ((dim_x,"X"),(dim_y,"Y"),(dim_z,"Z")):
            ok=d>0
            _record(Violation(node=i,rule_id=f"DIM-POSITIVE-{ax}",category="DIM",
                rule=f"Dim {ax}>0",value=round(d,3),limit=">0mm",
                severity="CRITICAL" if not ok else "INFO",
                suggestion="Invalid dimension" if not ok else "OK",passed=ok))
        for actual,nominal,ax in [(dim_x,nom_x,"X"),(dim_y,nom_y,"Y"),(dim_z,nom_z,"Z")]:
            ok,lo,hi=_check_tol(nominal,tol,actual)
            deviation=abs(actual-nominal)
            sev="INFO" if ok else ("WARNING" if deviation<=tol*1.20 else "CRITICAL")
            _record(Violation(node=i,rule_id=f"TOL-DIM-{ax}",category="TOL",
                rule=f"ISO 2768 Dim {ax}",value=round(actual,4),limit=f"[{lo:.4f},{hi:.4f}]",
                severity=sev,suggestion="Tolerance exceeded" if not ok else "Within tolerance",passed=ok))
        ok=tol<=cr["max_tol"]+_EPS
        sev="INFO" if ok else ("WARNING" if tol<=cr["max_tol"]*1.20+_EPS else "CRITICAL")
        _record(Violation(node=i,rule_id="TOL-COMP",category="COMP",rule=f"{comp} tolerance",
            value=round(tol,5),limit=f"<={cr['max_tol']}",severity=sev,
            suggestion="Tolerance too large" if not ok else "OK",passed=ok))
        ok=roughness<=cr["max_ra"]+_EPS
        _record(Violation(node=i,rule_id="SURF-RA",category="SURF",rule="Surface roughness",
            value=round(roughness,3),limit=f"<={cr['max_ra']}",
            severity="WARNING" if not ok else "INFO",
            suggestion="Surface too rough" if not ok else "Surface OK",passed=ok))
        for d,ax in [(dim_x,"X"),(dim_y,"Y"),(dim_z,"Z")]:
            ok=cr["min_dim"]-_EPS<=d<=cr["max_dim"]+_EPS
            sev="INFO" if ok else ("CRITICAL" if d<cr["min_dim"]*0.5 or d>cr["max_dim"]*1.5 else "WARNING")
            _record(Violation(node=i,rule_id=f"DIM-BOUND-{ax}",category="DIM",rule="Dimension bounds",
                value=round(d,3),limit=f"{cr['min_dim']}-{cr['max_dim']}",severity=sev,
                suggestion="Out of bounds" if not ok else "OK",passed=ok))
        ds=sorted([dim_x,dim_y,dim_z]); ratio=ds[-1]/max(ds[0],1e-6)
        ok=ratio<=cr["max_ar"]+_EPS
        _record(Violation(node=i,rule_id="DFM-AR",category="DFM",rule="Aspect ratio",
            value=round(ratio,2),limit=f"<={cr['max_ar']}",severity="WARNING" if not ok else "INFO",
            suggestion="Aspect ratio high" if not ok else "OK",passed=ok))
        ok=weight<=STANDARDS["WEIGHT_LIMIT"]+_EPS
        sev="INFO" if ok else ("WARNING" if weight<=STANDARDS["WEIGHT_LIMIT"]*1.25+_EPS else "CRITICAL")
        _record(Violation(node=i,rule_id="DFM-WT",category="DFM",rule="Weight limit",
            value=round(weight,1),limit=f"<={STANDARDS['WEIGHT_LIMIT']}",severity=sev,
            suggestion="Weight too high" if not ok else "OK",passed=ok))
    if nx_graph is not None:
        n=len(node_features)
        for u,v_node,data in nx_graph.edges(data=True):
            if not(0<=u<n and 0<=v_node<n): continue
            m1,m2=int(node_features[u][4]),int(node_features[v_node][4])
            ok=material_compatible(m1,m2)
            _record(Violation(node=f"e({u}->{v_node})",rule_id="MAT-COMPAT",category="MAT",
                rule="Material compatibility",value=0,limit="compatible",
                severity="CRITICAL" if not ok else "INFO",
                suggestion="Galvanic risk" if not ok else "OK",passed=ok))
            cl=float(data.get("clearance",0.0)); jt=str(data.get("joint_type","contact"))
            lim=JOINT_CL_LIMITS.get(jt,0.40); ok=cl<=lim+_EPS
            sev="CRITICAL" if cl>STANDARDS["CL_CRITICAL"]+_EPS else ("WARNING" if not ok else "INFO")
            _record(Violation(node=f"e({u}->{v_node})",rule_id="JNTI-CL",category="JNTI",
                rule="Joint clearance",value=round(cl,4),limit=f"<={lim}",severity=sev,
                suggestion="Clearance too large" if not ok else "OK",passed=ok))
    return violations,passed_checks

# ─────────────────────────────────────────────────────────────────────────────
# SCORE FUSION
# ─────────────────────────────────────────────────────────────────────────────
def fuse_scores(ml_prob_noncompliant,violations):
    n_crit=sum(1 for v in violations if v.severity=="CRITICAL")
    n_warn=sum(1 for v in violations if v.severity=="WARNING")
    n_info=sum(1 for v in violations if v.severity=="INFO")
    rule_score=min(100.0,n_crit*40.0+n_warn*15.0+n_info*5.0)
    ml_score=float(ml_prob_noncompliant)*100.0
    if n_crit>0:
        fused=max(cfg.NONCOMPLIANT_THRESH,0.30*ml_score+0.70*rule_score)
        verdict="NON-COMPLIANT"; override_reason=f"{n_crit} CRITICAL → auto NON-COMPLIANT override"
    else:
        fused=0.35*ml_score+0.65*rule_score; override_reason=None
        if fused>=cfg.NONCOMPLIANT_THRESH: verdict="NON-COMPLIANT"
        elif fused>=cfg.REVIEW_THRESH:     verdict="REVIEW NEEDED"
        else:                              verdict="COMPLIANT"
    explanation=(f"GNN: {ml_score:.1f}/100 non-compliance prob. "
        f"Rule engine: {n_crit} CRITICAL, {n_warn} WARNING, {n_info} INFO → {rule_score:.1f}/100. "
        f"{'Override applied.' if override_reason else f'Balanced fusion = {fused:.1f}/100.'}")
    return {"ml_score":round(ml_score,2),"rule_score":round(rule_score,2),
        "fused_score":round(fused,2),"verdict":verdict,"critical":n_crit,"warnings":n_warn,"info":n_info,
        "override_reason":override_reason,"explanation":explanation}

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _add_edges_to_graph(G,n_components,target_label):
    jt_opts=list(JOINT_CL_LIMITS.keys())
    for i in range(n_components-1):
        jt=np.random.choice(jt_opts); lim=JOINT_CL_LIMITS[jt]
        if target_label==0:   cl=np.random.uniform(1e-3,lim*0.7)
        elif target_label==1: cl=np.random.uniform(lim*0.9,lim*1.15)
        else:                 cl=np.random.uniform(lim*1.2,lim*3.0)
        G.add_edge(i,i+1,clearance=cl,joint_type=jt)

def graph_to_pyg(G,feats,label):
    raw_x=torch.tensor(feats,dtype=torch.float); x=raw_x.clone()
    jt_map={"fixed":0,"revolute":1,"prismatic":2,"contact":3}
    edges=list(G.edges(data=True))
    if edges:
        src=[u for u,v,d in edges]+[v for u,v,d in edges]
        dst=[v for u,v,d in edges]+[u for u,v,d in edges]
        ei=torch.tensor([src,dst],dtype=torch.long)
        e_feats=[[float(d.get("clearance",0.2)),float(jt_map.get(d.get("joint_type","contact"),3))]
                 for u,v,d in edges]
        edge_attr=torch.tensor(e_feats+e_feats,dtype=torch.float)
    else:
        ei=torch.zeros((2,0),dtype=torch.long); edge_attr=torch.zeros((0,2),dtype=torch.float)
    return Data(x=x,edge_index=ei,edge_attr=edge_attr,
                y=torch.tensor([label],dtype=torch.long),raw_x=raw_x)

def generate_cad_graph(n_components,is_compliant,seed=42):
    rng=np.random.default_rng(seed)
    comp_keys=list(COMPONENT_TYPES.keys()); mat_keys=list(MATERIAL_CODES.keys())
    label=0 if is_compliant else 2; node_features=[]
    for _ in range(n_components):
        comp=rng.choice(comp_keys); mat=rng.choice(mat_keys); cr=COMPONENT_RULES[comp]
        if label==0:
            tol=rng.uniform(cr["max_tol"]*0.1,cr["max_tol"]*0.7)
            roughness=rng.uniform(0.2,cr["max_ra"]*0.7)
            dim_x=rng.uniform(cr["min_dim"]*1.1,cr["max_dim"]*0.9)
            dim_y=rng.uniform(cr["min_dim"]*1.1,cr["max_dim"]*0.9)
            dim_z=rng.uniform(cr["min_dim"]*1.1,cr["max_dim"]*0.9)
        else:
            tol=rng.uniform(cr["max_tol"]*1.5,cr["max_tol"]*4.0)
            roughness=rng.uniform(cr["max_ra"]*1.5,cr["max_ra"]*4.0)
            dim_x=rng.uniform(cr["min_dim"]*0.1,cr["min_dim"]*0.8)
            dim_y=rng.uniform(cr["min_dim"]*0.1,cr["min_dim"]*0.8)
            dim_z=rng.uniform(cr["min_dim"]*0.1,cr["min_dim"]*0.8)
        weight=dim_x*dim_y*dim_z*1e-4
        node_features.append([dim_x,dim_y,dim_z,tol,
            float(MATERIAL_CODES[mat]),float(COMPONENT_TYPES[comp]),weight,roughness])
    feats=np.array(node_features,dtype=np.float32); G=nx.Graph(); G.add_nodes_from(range(n_components))
    _add_edges_to_graph(G,n_components,label)
    return G,feats,label

def generate_dataset():
    dataset=[]; rng=np.random.default_rng(cfg.SEED)
    for i in range(cfg.N_COMPLIANT):
        n_comp=int(rng.integers(cfg.MIN_COMPONENTS,cfg.MAX_COMPONENTS+1))
        G,feats,_=generate_cad_graph(n_comp,is_compliant=True,seed=cfg.SEED+i)
        dataset.append(graph_to_pyg(G,feats,0))
    for i in range(cfg.N_REVIEW_NEEDED):
        n_comp=int(rng.integers(cfg.MIN_COMPONENTS,cfg.MAX_COMPONENTS+1))
        rng2=np.random.default_rng(cfg.SEED+2000+i)
        comp_keys=list(COMPONENT_TYPES.keys()); mat_keys=list(MATERIAL_CODES.keys()); nf=[]
        for _ in range(n_comp):
            comp=rng2.choice(comp_keys); mat=rng2.choice(mat_keys); cr=COMPONENT_RULES[comp]
            tol=rng2.uniform(cr["max_tol"]*0.8,cr["max_tol"]*1.6)
            roughness=rng2.uniform(cr["max_ra"]*0.8,cr["max_ra"]*1.5)
            dim_x=rng2.uniform(cr["min_dim"]*0.7,cr["max_dim"]*1.1)
            dim_y=rng2.uniform(cr["min_dim"]*0.7,cr["max_dim"]*1.1)
            dim_z=rng2.uniform(cr["min_dim"]*0.7,cr["max_dim"]*1.1)
            weight=dim_x*dim_y*dim_z*1e-4
            nf.append([dim_x,dim_y,dim_z,tol,
                float(MATERIAL_CODES[mat]),float(COMPONENT_TYPES[comp]),weight,roughness])
        feats=np.array(nf,dtype=np.float32); G_nx=nx.Graph(); G_nx.add_nodes_from(range(n_comp))
        _add_edges_to_graph(G_nx,n_comp,1); dataset.append(graph_to_pyg(G_nx,feats,1))
    for i in range(cfg.N_NONCOMPLIANT):
        n_comp=int(rng.integers(cfg.MIN_COMPONENTS,cfg.MAX_COMPONENTS+1))
        G,feats,_=generate_cad_graph(n_comp,is_compliant=False,seed=cfg.SEED+5000+i)
        dataset.append(graph_to_pyg(G,feats,2))
    return dataset

# ─────────────────────────────────────────────────────────────────────────────
# GNN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class GNNEncoder(nn.Module):
    def __init__(self,in_ch=cfg.IN_CHANNELS,h=cfg.HIDDEN_DIM,lat=cfg.LATENT_DIM,drop=cfg.DROPOUT):
        super().__init__(); self.drop=drop
        self.c1=TransformerConv(in_ch,h,edge_dim=2); self.c2=TransformerConv(h,h,edge_dim=2)
        self.c3=TransformerConv(h,lat,edge_dim=2)
        self.b1,self.b2,self.b3=nn.BatchNorm1d(h),nn.BatchNorm1d(h),nn.BatchNorm1d(lat)
    def forward(self,x,ei,batch,edge_attr=None):
        x=F.relu(self.b1(self.c1(x,ei,edge_attr))); x=F.dropout(x,self.drop,self.training)
        x=F.relu(self.b2(self.c2(x,ei,edge_attr))); x=F.dropout(x,self.drop,self.training)
        x=F.relu(self.b3(self.c3(x,ei,edge_attr))); return global_mean_pool(x,batch)

class GNRValModel(nn.Module):
    def __init__(self,in_ch=cfg.IN_CHANNELS,h=cfg.HIDDEN_DIM,lat=cfg.LATENT_DIM,drop=cfg.DROPOUT):
        super().__init__()
        self.encoder=GNNEncoder(in_ch,h,lat,drop)
        self.decoder=nn.Sequential(nn.Linear(lat,h),nn.ReLU(),nn.Linear(h,in_ch))
        self.classifier=nn.Sequential(nn.Linear(lat,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,3))
    def forward(self,x,ei,batch,edge_attr=None):
        z=self.encoder(x,ei,batch,edge_attr); return self.classifier(z),z,self.decoder(z)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _eval_loader(model,loader):
    model.eval(); all_preds,all_labels,all_probs=[],[],[]
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(DEVICE)
            logits,_,_=model(batch.x,batch.edge_index,batch.batch,batch.edge_attr)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
            all_probs.extend(F.softmax(logits,dim=1)[:,2].cpu().tolist())
    return accuracy_score(all_labels,all_preds)*100,all_preds,all_labels,all_probs

def train_model_fn(train_data,val_data,pretrained_model_path=None):
    train_loader=DataLoader(train_data,batch_size=cfg.BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(val_data,batch_size=cfg.BATCH_SIZE,shuffle=False)
    train_labels=[int(d.y.item()) for d in train_data]
    counts=[max(train_labels.count(i),1) for i in range(3)]
    weights=torch.tensor([1.0/c for c in counts],dtype=torch.float).to(DEVICE)
    weights=weights/weights.sum()*3.0
    model=GNRValModel().to(DEVICE)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            state_dict=torch.load(pretrained_model_path,map_location=DEVICE)
            model.load_state_dict(state_dict,strict=True)
            log.info(f"Loaded pretrained real-world model from: {pretrained_model_path}")
        except Exception as e:
            log.warning(f"Could not load pretrained model from {pretrained_model_path}: {e}")

    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY)
    criterion=nn.CrossEntropyLoss(weight=weights)
    best_acc,best_state=0.0,None
    pbar=st.progress(0,text=f"Training GNN... Epoch 0/{cfg.EPOCHS}")
    for ep in range(1,cfg.EPOCHS+1):
        model.train()
        for batch in train_loader:
            batch=batch.to(DEVICE); optimizer.zero_grad()
            logits,_,recon=model(batch.x,batch.edge_index,batch.batch,batch.edge_attr)
            loss=criterion(logits,batch.y)+cfg.RECON_WEIGHT*F.mse_loss(
                recon,global_mean_pool(batch.x,batch.batch))
            loss.backward(); optimizer.step()
        if ep%cfg.LOG_EVERY==0:
            val_acc,_,_,_=_eval_loader(model,val_loader)
            if val_acc>best_acc: best_acc,best_state=val_acc,deepcopy(model.state_dict())
        pbar.progress(ep/cfg.EPOCHS,text=f"Training GNN... Epoch {ep}/{cfg.EPOCHS} | Best Val Acc: {best_acc:.1f}%")
    if best_state: model.load_state_dict(best_state)
    pbar.empty()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_input_data(input_data):
    validated_data,input_warnings=[],[]
    allowed_materials=set(MATERIAL_CODES.keys()); allowed_types=set(COMPONENT_TYPES.keys())
    for idx,comp in enumerate(input_data):
        item=dict(comp)
        for key in ["actual_x","actual_y","actual_z"]:
            if key not in item: raise ValueError(f"Component {idx}: missing field '{key}'")
            item[key]=float(item[key])
        item["tolerance"]=abs(float(item.get("tolerance",0.1)))
        item["roughness"]=abs(float(item.get("roughness",1.6)))
        item["material"]=str(item.get("material","steel")).lower().strip()
        item["type"]=str(item.get("type","housing")).lower().strip()
        if item["material"] not in allowed_materials:
            input_warnings.append(f"Component {idx}: unknown material → replaced with 'steel'.")
            item["material"]="steel"
        if item["type"] not in allowed_types:
            input_warnings.append(f"Component {idx}: unknown type → replaced with 'housing'.")
            item["type"]="housing"
        if "nominal_dims" in item and item["nominal_dims"] is not None:
            item["nominal_dims"]=[float(x) for x in item["nominal_dims"]]
        else:
            item["nominal_dims"]=[item["actual_x"],item["actual_y"],item["actual_z"]]
            input_warnings.append(f"Component {idx}: nominal_dims missing, using actual dims.")
        validated_data.append(item)
    return validated_data,input_warnings

def get_top_violation_reasons(violations,top_k=3):
    sev_rank={"CRITICAL":3,"WARNING":2,"INFO":1}
    ranked=sorted(violations,key=lambda v:sev_rank.get(v.severity,0),reverse=True)
    reasons,seen=[],set()
    for v in ranked:
        if v.rule_id in seen: continue
        reasons.append({"node":v.node,"rule_id":v.rule_id,"severity":v.severity,
                        "category":v.category,"message":v.suggestion})
        seen.add(v.rule_id)
        if len(reasons)>=top_k: break
    return reasons

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC INFERENCE GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_inference_graph_from_components(input_data):
    """
    Build a deterministic nx.Graph from real validated component data.
    Same input_data always produces the same graph — zero randomness.

    Edge topology:
      1. Sequential backbone: every adjacent pair (i, i+1) is connected.
      2. Hierarchy edges: non-adjacent pairs connected when their component
         types have a known assembly relationship (shaft→bearing, etc.).

    Joint type: derived deterministically from the component-type pair.
    Clearance:  derived from the average of the two components' tolerance
                values, clamped to 70 % of the joint's clearance limit —
                a conservative, repeatable estimate from real CAD intent.
    """
    n = len(input_data)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Deterministic joint-type lookup by component-type pair
    _JT_MAP = {
        ("shaft",    "bearing"):  "revolute",
        ("shaft",    "gear"):     "revolute",
        ("gear",     "bearing"):  "revolute",
        ("housing",  "bearing"):  "fixed",
        ("housing",  "shaft"):    "fixed",
        ("housing",  "fastener"): "fixed",
        ("bracket",  "fastener"): "fixed",
        ("gear",     "fastener"): "fixed",
        ("shaft",    "fastener"): "fixed",
        ("bearing",  "fastener"): "fixed",
        ("housing",  "bracket"):  "contact",
        ("bracket",  "gear"):     "contact",
    }

    # Assembly hierarchy: types that u naturally connects to
    _HIER = {
        "shaft":    {"bearing", "fastener", "gear"},
        "housing":  {"bearing", "fastener", "shaft"},
        "gear":     {"fastener", "bearing", "shaft"},
        "bracket":  {"fastener", "housing"},
        "bearing":  {"fastener", "shaft"},
        "fastener": set(),
    }

    def _joint_type(t1, t2):
        return _JT_MAP.get((t1, t2), _JT_MAP.get((t2, t1), "contact"))

    def _clearance(comp_i, comp_j, jt):
        # Real tolerances from user input drive the clearance estimate
        tol_i = abs(float(comp_i.get("tolerance", 0.1)))
        tol_j = abs(float(comp_j.get("tolerance", 0.1)))
        limit = JOINT_CL_LIMITS.get(jt, 0.40)
        cl = min((tol_i + tol_j) / 2.0, limit * 0.70)
        return round(max(1e-4, cl), 6)

    # 1. Sequential backbone — always present, preserves graph connectivity
    for i in range(n - 1):
        t1 = str(input_data[i].get("type", "housing")).lower().strip()
        t2 = str(input_data[i + 1].get("type", "housing")).lower().strip()
        jt = _joint_type(t1, t2)
        cl = _clearance(input_data[i], input_data[i + 1], jt)
        G.add_edge(i, i + 1, clearance=cl, joint_type=jt)

    # 2. Hierarchy-based non-adjacent edges
    for i, ci in enumerate(input_data):
        t1 = str(ci.get("type", "housing")).lower().strip()
        for j, cj in enumerate(input_data):
            if j <= i or G.has_edge(i, j):
                continue
            t2 = str(cj.get("type", "housing")).lower().strip()
            if t2 in _HIER.get(t1, set()) or t1 in _HIER.get(t2, set()):
                jt = _joint_type(t1, t2)
                cl = _clearance(ci, cj, jt)
                G.add_edge(i, j, clearance=cl, joint_type=jt)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(input_data,nominal_dims=None,trained_model=None,node_scaler=None):
    if not input_data: return {"error":"No component data provided."}
    input_data,input_warnings=validate_input_data(input_data)
    if nominal_dims is None: nominal_dims=[tuple(item["nominal_dims"]) for item in input_data]
    feats_list=[]
    for item in input_data:
        ax,ay,az=float(item["actual_x"]),float(item["actual_y"]),float(item["actual_z"])
        tol=abs(float(item.get("tolerance",0.1))); ra=abs(float(item.get("roughness",1.6)))
        feats_list.append([ax,ay,az,tol,
            float(MATERIAL_CODES.get(item.get("material","steel"),0)),
            float(COMPONENT_TYPES.get(item.get("type","housing"),2)),ax*ay*az*1e-4,ra])
    node_features=np.array(feats_list,dtype=np.float32)
    gnn_nf=node_scaler.transform(node_features) if node_scaler else node_features.copy()
    G_nx=build_inference_graph_from_components(input_data)
    if trained_model is None: return {"error":"Model not trained yet."}
    trained_model.eval()
    pyg=graph_to_pyg(G_nx,gnn_nf,0)
    x=pyg.x.to(DEVICE); ei=pyg.edge_index.to(DEVICE); ea=pyg.edge_attr.to(DEVICE)
    bv=torch.zeros(x.size(0),dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits,_,recon=trained_model(x,ei,bv,ea)
        probs=F.softmax(logits/1.5,dim=1).cpu().numpy()[0]
    pred_class=int(logits.argmax(1).item())
    lmap={0:"compliant",1:"review-needed",2:"non-compliant"}
    lfull={0:"Compliant",1:"Review-Needed",2:"Non-Compliant"}
    violations,passed=run_rule_engine(node_features,nominal_dims,G_nx)
    top_reasons=get_top_violation_reasons(violations,top_k=3)
    scores=fuse_scores(float(probs[2]),violations)
    return {"ml_prediction":lmap.get(pred_class,"unknown"),"ml_pred_class":pred_class,
        "ml_confidence":round(float(max(probs))*100,2),
        "ml_probs":{lfull[i]:round(float(probs[i])*100,2) for i in range(3)},
        "reconstruction_error":round(float(F.mse_loss(recon[0],x.mean(0)).item()),4),
        "violations":violations,"passed_checks":passed,"scores":scores,
        "n_components":len(node_features),"n_violations":len(violations),"n_passed":len(passed),
        "input_warnings":input_warnings,"top_reasons":top_reasons}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _compute_decision_basis(res):
    sc=res.get("scores",{}); n_crit=sc.get("critical",0); recon=float(res.get("reconstruction_error",0.0))
    basis="Rule-Dominant Safety Override" if n_crit>0 else "Balanced Rule + GNN Assessment"
    if recon>=15.0: basis+=" | Elevated Model Uncertainty"
    return basis

def _get_release_recommendation(verdict):
    m={"COMPLIANT":{"label":"Release Approved","icon":"✅","color":"green"},
       "REVIEW NEEDED":{"label":"Engineering Review Required","icon":"⚠️","color":"orange"},
       "NON-COMPLIANT":{"label":"Production Hold","icon":"🚫","color":"red"}}
    return m.get(verdict,{"label":"Under Review","icon":"🔍","color":"gray"})

def _get_top3_remediations(res):
    v_list = res.get("violations", [])
    actions = []
    cat_seen = set()

    remap = {
        "TOL": "Recalibrate CNC tooling or tighten process control to stay within tolerance budget.",
        "SURF": "Switch to fine-grinding/superfinishing; verify grinding wheel grit and feed rate.",
        "MAT": "Apply isolating coating or substitute galvanic-neutral alloy per ASTM B117.",
        "DIM": "Revise CAD nominal or adjust process per ASME Y14.5 GD&T drawing callouts.",
        "DFM": "Re-evaluate geometry for manufacturability; reduce aspect ratio via topology optimisation.",
        "JNTI": "Adjust fit specification (H7/g6 etc.) and verify with go/no-go gauges.",
    }

    for v in v_list:
        cat = str(v.category).upper()
        for key, msg in remap.items():
            if key in cat and key not in cat_seen:
                actions.append(msg)
                cat_seen.add(key)
                break
        if len(actions) >= 3:
            break

    if float(res.get("reconstruction_error", 0.0)) >= 15.0 and len(actions) < 3:
        actions.append("Simplify CAD features to improve GNN prediction confidence.")

    if not actions:
        actions.append("No corrective actions required. Maintain current process settings.")

    return actions[:3]

def render_cad_3d_model(components, violations=None):
    """
    3D Digital Twin Preview — clean wireframe cuboids.

    Per component, three traces are added:
      1. Scatter3d lines  — 12 explicit box edges, bold and color-coded.
         No triangle geometry → zero stretched-triangle artifacts.
      2. Mesh3d           — correct 12-triangle closed cuboid at opacity=0.08.
         Opacity is intentionally very low so triangle seam lines are
         invisible; the trace exists only to give subtle depth/fill cues.
      3. Scatter3d text   — label just above the top face.

    Violated components → red edges/fill.
    Compliant components → type-based color.
    Dark theme preserved throughout.
    """
    if not components:
        return None

    violated_nodes = set()
    if violations:
        violated_nodes = {v.node for v in violations if isinstance(v.node, int)}

    type_colors = {
        "housing":  "#3b82f6",
        "shaft":    "#10b981",
        "bearing":  "#f59e0b",
        "gear":     "#8b5cf6",
        "bracket":  "#94a3b8",
        "fastener": "#06b6d4",
    }

    # Correct closed-cuboid triangle indices (12 triangles, outward normals)
    # Vertex layout:  0=x0y0z0  1=x1y0z0  2=x1y1z0  3=x0y1z0
    #                 4=x0y0z1  5=x1y0z1  6=x1y1z1  7=x0y1z1
    _FI = [0,0, 4,4, 0,0, 2,2, 0,0, 1,1]
    _FJ = [1,2, 5,6, 1,5, 3,7, 3,7, 2,6]
    _FK = [2,3, 6,7, 5,4, 7,6, 7,4, 6,5]

    fig = go.Figure()
    spacing = 0.0

    for idx, comp in enumerate(components):
        sx = max(float(comp.get("actual_x", 10.0)), 1.0)
        sy = max(float(comp.get("actual_y", 10.0)), 1.0)
        sz = max(float(comp.get("actual_z", 10.0)), 1.0)

        comp_type = str(comp.get("type", "housing")).lower().strip()
        base_color = type_colors.get(comp_type, "#94a3b8")
        color = "#ef4444" if idx in violated_nodes else base_color

        # Position: lay components along X axis, all starting at Y=0, Z=0
        x0 = spacing
        x1 = spacing + sx
        y0, y1 = -sy / 2.0,  sy / 2.0
        z0, z1 =  0.0,        sz
        cx = (x0 + x1) / 2.0
        cy = 0.0
        spacing = x1 + max(10.0, sx * 0.15)

        hover = (f"<b>N{idx} · {comp_type}</b><br>"
                 f"X: {sx:.2f} mm<br>Y: {sy:.2f} mm<br>Z: {sz:.2f} mm"
                 f"<extra></extra>")

        # ── 1. Wireframe edges ────────────────────────────────────────────
        # 12 edges of the cuboid, each represented as [start, end, None]
        # so Plotly draws separate line segments with no connecting diagonal.
        _edges = [
            # Bottom face (z0)
            (x0,y0,z0, x1,y0,z0), (x1,y0,z0, x1,y1,z0),
            (x1,y1,z0, x0,y1,z0), (x0,y1,z0, x0,y0,z0),
            # Top face (z1)
            (x0,y0,z1, x1,y0,z1), (x1,y0,z1, x1,y1,z1),
            (x1,y1,z1, x0,y1,z1), (x0,y1,z1, x0,y0,z1),
            # Vertical pillars
            (x0,y0,z0, x0,y0,z1), (x1,y0,z0, x1,y0,z1),
            (x1,y1,z0, x1,y1,z1), (x0,y1,z0, x0,y1,z1),
        ]
        lx, ly, lz = [], [], []
        for ax, ay, az, bx, by, bz in _edges:
            lx += [ax, bx, None]
            ly += [ay, by, None]
            lz += [az, bz, None]

        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            line=dict(color=color, width=3),
            name=f"N{idx} {comp_type}",
            hovertemplate=hover,
            legendgroup=f"c{idx}",
        ))

        # ── 2. Transparent face fill (depth cue only) ─────────────────────
        verts = np.array([
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
        ])
        fig.add_trace(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=_FI, j=_FJ, k=_FK,
            color=color,
            opacity=0.08,          # low enough that triangle seams vanish
            flatshading=False,
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"c{idx}",
        ))

        # ── 3. Label just above top face ──────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[z1 + max(2.0, sz * 0.06)],
            mode="text",
            text=[f"<b>N{idx}</b><br>{comp_type}"],
            textfont=dict(size=9, color="white"),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=f"c{idx}",
        ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(color="white"),
        scene=dict(
            bgcolor="#0a0e1a",
            xaxis=dict(title="X (mm)", backgroundcolor="#0a0e1a",
                       gridcolor="#1e293b", showbackground=True,
                       zerolinecolor="#334155", color="#64748b"),
            yaxis=dict(title="Y (mm)", backgroundcolor="#0a0e1a",
                       gridcolor="#1e293b", showbackground=True,
                       zerolinecolor="#334155", color="#64748b"),
            zaxis=dict(title="Z (mm)", backgroundcolor="#0a0e1a",
                       gridcolor="#1e293b", showbackground=True,
                       zerolinecolor="#334155", color="#64748b"),
            camera=dict(eye=dict(x=1.7, y=1.5, z=1.2)),
            aspectmode="auto",
        ),
        legend=dict(bgcolor="#0f172a", bordercolor="#334155",
                    borderwidth=1, font=dict(color="white")),
        margin=dict(l=0, r=0, t=20, b=0),
        height=650,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STEP PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_step_text(content: str) -> List[Dict[str, Any]]:
    """
    Single canonical STEP parser for the Streamlit app.
    Parses ISO 10303-21 ASCII STEP files (AP203/AP214).
    Extracts CARTESIAN_POINT coordinates → per-solid bounding boxes.
    Falls back to LENGTH_MEASURE values or stable index-based defaults.
    No pythonocc, OCC, or native CAD library required.
    """
    lengths = [float(x) for x in re.findall(r"LENGTH_MEASURE\s*\(\s*([\d.]+)\s*\)", content)]
    cart_pts = re.findall(
        r"CARTESIAN_POINT\s*\([^,]*,\s*\(\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\)\s*\)",
        content)
    face_count = len(re.findall(r"ADVANCED_FACE", content))
    n_comp = max(1, min(12, face_count // 40 + len(cart_pts) // 5 + 1))

    # Stable ordered lists — deterministic cycling by component index
    mat_keys  = list(MATERIAL_CODES.keys())    # steel, aluminum, plastic, titanium, composite
    type_keys = list(COMPONENT_TYPES.keys())   # shaft, bearing, housing, bracket, fastener, gear

    extracted = []
    for i in range(n_comp):
        if i < len(cart_pts):
            try:
                dx = abs(float(cart_pts[i][0])) or 50.0
                dy = abs(float(cart_pts[i][1])) or 50.0
                dz = abs(float(cart_pts[i][2])) or 20.0
            except Exception:
                dx, dy, dz = 50.0, 50.0, 20.0
        else:
            if lengths:
                dx = lengths[i % len(lengths)]
                dy = lengths[(i + 1) % len(lengths)]
                dz = lengths[(i + 2) % len(lengths)]
            else:
                dx = 50.0 + (i % 5) * 10.0
                dy = 50.0 + ((i + 1) % 5) * 10.0
                dz = 20.0 + (i % 3) * 5.0

        comp_type = type_keys[i % len(type_keys)]
        comp_mat  = mat_keys[i % len(mat_keys)]

        extracted.append({
            "actual_x":     round(max(1.0, dx), 3),
            "actual_y":     round(max(1.0, dy), 3),
            "actual_z":     round(max(1.0, dz), 3),
            "tolerance":    0.10,    # ISO 2768 medium — safe conservative default
            "material":     comp_mat,
            "type":         comp_type,
            "roughness":    1.6,     # Ra 1.6 µm — mating surface standard default
            "nominal_dims": [round(max(1.0, dx), 3),
                             round(max(1.0, dy), 3),
                             round(max(1.0, dz), 3)],
        })
    return extracted


def parse_step_file(content: str) -> List[Dict[str, Any]]:
    """Alias for parse_step_text — retained for call-site compatibility."""
    return parse_step_text(content)

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY GRAPH
# ─────────────────────────────────────────────────────────────────────────────
_ASSEMBLY_HIER={"shaft":["bearing","fastener","gear"],"housing":["bearing","seal","gasket"],
    "bearing":["fastener","seal"],"gear":["fastener"],"bracket":["fastener","seal"]}

def _build_assembly_graph(components):
    n=len(components); G=nx.DiGraph(); G.add_nodes_from(range(n))
    labels={i:f"N{i}\n{components[i].get('type','part')}" for i in range(n)}
    for i,comp in enumerate(components):
        for j,other in enumerate(components):
            if i!=j and other.get("type","").lower() in _ASSEMBLY_HIER.get(comp.get("type",""),[]): G.add_edge(i,j)
    coords=np.array([[c.get("actual_x",0),c.get("actual_y",0),c.get("actual_z",0)] for c in components],dtype=float)
    isolated=[v for v in G.nodes() if G.degree(v)==0]
    for v in isolated:
        dists=np.linalg.norm(coords-coords[v],axis=1); dists[v]=np.inf; G.add_edge(v,int(np.argmin(dists)))
    if nx.number_weakly_connected_components(G)>1:
        for i in range(n-1):
            if not(G.has_edge(i,i+1) or G.has_edge(i+1,i)): G.add_edge(i,i+1)
    return G,labels

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    set_seed()

    dataset=generate_dataset()
    all_labels=[int(d.y.item()) for d in dataset]
    train_idx,temp_idx=train_test_split(list(range(len(dataset))),
        test_size=(1.0-cfg.TRAIN_RATIO),stratify=all_labels,random_state=cfg.SEED)
    temp_labels=[all_labels[i] for i in temp_idx]
    val_idx,test_idx=train_test_split(temp_idx,test_size=0.5,stratify=temp_labels,random_state=cfg.SEED)

    train_data=[dataset[i] for i in train_idx]
    val_data=[dataset[i] for i in val_idx]
    test_data=[dataset[i] for i in test_idx]

    all_raw=np.vstack([d.raw_x.numpy() for d in train_data])
    scaler=StandardScaler()
    scaler.fit(all_raw)

    for d in train_data+val_data+test_data:
        d.x=torch.tensor(scaler.transform(d.raw_x.numpy()),dtype=torch.float)

    pretrained_path=cfg.REAL_MODEL_PATH if os.path.exists(cfg.REAL_MODEL_PATH) else None
    if pretrained_path:
        log.info(f"Using real-world pretrained weights: {pretrained_path}")
    else:
        log.info("No real-world pretrained weights found. Training from synthetic dataset only.")

    model=train_model_fn(train_data,val_data,pretrained_model_path=pretrained_path)
    torch.save(model.state_dict(),cfg.MODEL_PATH)

    loader=DataLoader(test_data,batch_size=cfg.BATCH_SIZE,shuffle=False)
    _,preds,labels,_=_eval_loader(model,loader)
    metrics={"Accuracy %":round(accuracy_score(labels,preds)*100,2),
        "Precision % (Macro)":round(precision_score(labels,preds,average="macro",zero_division=0)*100,2),
        "Recall % (Macro)":round(recall_score(labels,preds,average="macro",zero_division=0)*100,2),
        "F1 % (Macro)":round(f1_score(labels,preds,average="macro",zero_division=0)*100,2)}
    report=classification_report(labels,preds,
        target_names=["Compliant","Review-Needed","Non-Compliant"],zero_division=0)
    cm=confusion_matrix(labels,preds)
    return model,scaler,metrics,report,cm


    # ─────────────────────────────────────────────────────────────────────────────
# FUSION 360 CLI BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
def _json_safe_result(res):
    safe = dict(res)

    # Violation dataclasses -> plain dicts
    safe["violations"] = [v.to_dict() if hasattr(v, "to_dict") else v for v in safe.get("violations", [])]
    safe["passed_checks"] = [v.to_dict() if hasattr(v, "to_dict") else v for v in safe.get("passed_checks", [])]

    # Make sure everything is JSON serializable
    if "scores" in safe and isinstance(safe["scores"], dict):
        safe["scores"] = {
            str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in safe["scores"].items()
        }

    if "ml_probs" in safe and isinstance(safe["ml_probs"], dict):
        safe["ml_probs"] = {
            str(k): float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in safe["ml_probs"].items()
        }

    if "top_reasons" in safe:
        cleaned = []
        for item in safe["top_reasons"]:
            if isinstance(item, dict):
                cleaned.append({
                    str(k): (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in item.items()
                })
            else:
                cleaned.append(item)
        safe["top_reasons"] = cleaned

    for key in ["ml_confidence", "reconstruction_error"]:
        if key in safe and isinstance(safe[key], (np.floating, np.integer)):
            safe[key] = float(safe[key])

    for key in ["n_components", "n_violations", "n_passed", "ml_pred_class"]:
        if key in safe and isinstance(safe[key], (np.integer,)):
            safe[key] = int(safe[key])

    return safe


def validate_components_for_fusion(components):
    model, scaler, _, _, _ = load_pipeline()
    result = run_inference(components, trained_model=model, node_scaler=scaler)
    return _json_safe_result(result)


def fusion_cli_entry():
    """
    Usage:
        python app.py --fusion-cli input.json output.json
    """
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python app.py --fusion-cli input.json output.json")

    input_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    components = payload.get("components", [])
    if not isinstance(components, list):
        raise ValueError("Input JSON must contain a 'components' list.")

    result = validate_components_for_fusion(components)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#0a0e1a;}
.gnr-header{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);border:1px solid #1e40af;border-radius:16px;padding:28px 36px;margin-bottom:24px;text-align:center;box-shadow:0 0 40px rgba(59,130,246,0.15);}
.gnr-header h1{font-family:'Syne',sans-serif;font-size:2.2em;font-weight:800;color:#f1f5f9;margin:0 0 6px 0;letter-spacing:2px;}
.gnr-header p{color:#64748b;margin:0;font-size:0.88em;letter-spacing:1px;}
.metric-card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:16px 18px;text-align:center;}
.metric-card .label{color:#64748b;font-size:0.72em;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:5px;}
.metric-card .value{font-family:'JetBrains Mono',monospace;font-size:2em;font-weight:700;}
.verdict-compliant{background:#064e3b;border:2px solid #10b981;color:#a7f3d0;}
.verdict-review{background:#78350f;border:2px solid #f59e0b;color:#fde68a;}
.verdict-noncompliant{background:#7f1d1d;border:2px solid #ef4444;color:#fecaca;}
.verdict-box{border-radius:14px;padding:22px 28px;text-align:center;margin:16px 0;font-size:1.6em;font-weight:800;letter-spacing:2px;}
.section-title{color:#94a3b8;font-size:0.78em;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid #1e293b;padding-bottom:8px;margin:20px 0 14px 0;}
.viol-crit{border-left:4px solid #ef4444;background:#1c0a0a;padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;}
.viol-warn{border-left:4px solid #f59e0b;background:#1c1100;padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;}
.viol-info{border-left:4px solid #3b82f6;background:#0a0f1c;padding:8px 12px;border-radius:0 6px 6px 0;margin:4px 0;}
.viol-txt{font-family:'JetBrains Mono',monospace;font-size:0.82em;color:#cbd5e1;}
.stButton>button{background:#1e40af!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;letter-spacing:1px!important;}
.stButton>button:hover{background:#2563eb!important;box-shadow:0 0 20px rgba(59,130,246,0.4)!important;}
div[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid #1e293b;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""<div class="gnr-header"><h1>🏭 GNR-VAL v5.0</h1>
        <p>INDUSTRIAL CAD COMPLIANCE VALIDATOR · VARROC EUREKA 3.0 · PS-9</p>
        <p style="color:#475569;font-size:0.8em;margin-top:6px;">
        TransformerConv GNN + ISO 2768 / ASME Y14.5 / GD&T / DFM Rule Engine</p>
        </div>""", unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        org_name  = st.text_input("Organisation", value="Varroc Engineering")
        prod_name = st.text_input("Product", value="Eureka 3.0 Demo")
        st.markdown("---")
        st.markdown("### 🧠 Model")
        if st.button("🚀 Train / Load Model", use_container_width=True):
            with st.spinner("Building & training GNN pipeline (~2-3 min first run)…"):
                st.session_state["pipeline"] = load_pipeline()
            st.success("✅ Model ready!")
        if "pipeline" in st.session_state:
            _,_,metrics,_,_=st.session_state["pipeline"]
            st.markdown("**Test Metrics**")
            for k,v in metrics.items(): st.markdown(f"`{k}`: **{v}**")
        else:
            st.info("Click **Train / Load Model** to initialise.")
        st.markdown("---")
        st.markdown("### 📋 Input Mode")
        input_mode=st.radio("",["Manual Form","STEP File Upload","Random Demo"])

    tab_v, tab_m, tab_r = st.tabs(["🔍 Validate Design","📊 Model Analytics","📄 Export Report"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — VALIDATE
    # ═════════════════════════════════════════════════════════════════════════
    with tab_v:
        components_list=[]

        if input_mode=="Manual Form":
            st.markdown('<div class="section-title">Component Geometry Parameters</div>',unsafe_allow_html=True)
            if "form_components" not in st.session_state: st.session_state["form_components"]=[]
            with st.expander("➕ Add Component",expanded=True):
                c1,c2,c3=st.columns(3)
                with c1: nom_x=st.number_input("Nominal X",value=100.0,step=0.1); act_x=st.number_input("Actual X",value=100.1,step=0.1)
                with c2: nom_y=st.number_input("Nominal Y",value=100.0,step=0.1); act_y=st.number_input("Actual Y",value=100.1,step=0.1)
                with c3: nom_z=st.number_input("Nominal Z",value=50.0,step=0.1); act_z=st.number_input("Actual Z",value=50.1,step=0.1)
                c4,c5,c6,c7=st.columns(4)
                with c4: tol=st.number_input("Tolerance (mm)",value=0.05,format="%.4f")
                with c5: ra=st.number_input("Roughness Ra (µm)",value=0.8,step=0.1)
                with c6: mat=st.selectbox("Material",list(MATERIAL_CODES.keys()))
                with c7: ctyp=st.selectbox("Type",list(COMPONENT_TYPES.keys()))
                ba,bb=st.columns(2)
                with ba:
                    if st.button("➕ Add",use_container_width=True):
                        st.session_state["form_components"].append({
                            "actual_x":act_x,"actual_y":act_y,"actual_z":act_z,
                            "tolerance":tol,"material":mat,"type":ctyp,"roughness":ra,
                            "nominal_dims":[nom_x,nom_y,nom_z]})
                        st.success(f"Added {ctyp} ({len(st.session_state['form_components'])} total)")
                with bb:
                    if st.button("🗑️ Clear All",use_container_width=True):
                        st.session_state["form_components"]=[]
                        st.info("Cleared.")
            if st.session_state["form_components"]:
                st.dataframe(pd.DataFrame(st.session_state["form_components"]).drop(
                    columns=["nominal_dims"],errors="ignore"),use_container_width=True)
                components_list=st.session_state["form_components"]

        elif input_mode=="STEP File Upload":
            st.markdown('<div class="section-title">STEP File Import</div>',unsafe_allow_html=True)
            uploaded=st.file_uploader("Upload STEP/STP file",type=["step","stp"],
                help="ASCII STEP files. Real CARTESIAN_POINT & LENGTH_MEASURE tokens extracted.")
            if uploaded:
                with st.spinner("Parsing STEP file…"):
                    content=uploaded.read().decode("utf-8",errors="ignore")
                    components_list=parse_step_file(content)
                st.success(f"✅ Extracted {len(components_list)} components from `{uploaded.name}`")
                st.dataframe(pd.DataFrame(components_list).drop(
                    columns=["nominal_dims"],errors="ignore"),use_container_width=True)

        else:
            st.markdown('<div class="section-title">Random Demo — 20 Synthetic Components</div>',unsafe_allow_html=True)
            if st.button("🎲 Generate",use_container_width=False):
                comps=[]
                for _ in range(20):
                    comps.append({"actual_x":random.uniform(1,200),"actual_y":random.uniform(1,200),
                        "actual_z":random.uniform(1,200),
                        "nominal_dims":[random.uniform(1,200),random.uniform(1,200),random.uniform(1,200)],
                        "tolerance":random.uniform(0.0001,0.5),
                        "material":random.choice(list(MATERIAL_CODES.keys())),
                        "type":random.choice(list(COMPONENT_TYPES.keys())),
                        "roughness":random.uniform(0.1,10)})
                st.session_state["demo_components"]=comps
                st.success("20 random components generated.")
            if "demo_components" in st.session_state:
                components_list=st.session_state["demo_components"]
                st.dataframe(pd.DataFrame(components_list).drop(columns=["nominal_dims"],errors="ignore"),
                    use_container_width=True)

        st.markdown("---")
        run_col,_=st.columns([1,3])
        with run_col: run_btn=st.button("▶ RUN VALIDATION",use_container_width=True)

        if run_btn:
            if not components_list: st.error("No components loaded.")
            elif "pipeline" not in st.session_state: st.error("Train model first (sidebar).")
            else:
                model,scaler,_,_,_=st.session_state["pipeline"]
                with st.spinner("Running GNN + Rule Engine…"):
                    result=run_inference(components_list,trained_model=model,node_scaler=scaler)
                st.session_state["last_result"]=result
                st.session_state["last_components"]=components_list
                st.session_state["org_name"]=org_name; st.session_state["prod_name"]=prod_name

        if "last_result" in st.session_state:
            res=st.session_state["last_result"]; sc=res["scores"]; verdict=sc["verdict"]
            v_cls={"COMPLIANT":"verdict-compliant","REVIEW NEEDED":"verdict-review",
                   "NON-COMPLIANT":"verdict-noncompliant"}.get(verdict,"verdict-review")
            v_icon={"COMPLIANT":"✅","REVIEW NEEDED":"⚠️","NON-COMPLIANT":"🚫"}.get(verdict,"🔍")
            st.markdown(f'<div class="verdict-box {v_cls}">{v_icon} {verdict}<br>'
                f'<span style="font-size:0.55em;font-weight:400;">Fused Risk Score: {sc["fused_score"]} / 100</span>'
                f'</div>',unsafe_allow_html=True)

            basis=_compute_decision_basis(res); rec=_get_release_recommendation(verdict)
            st.markdown(f"""<div style="background:#0f172a;border-left:4px solid #3b82f6;
                border-radius:6px;padding:10px 16px;margin:8px 0;">
                <span style="color:#64748b;font-size:0.78em;">DECISION BASIS: </span>
                <span style="color:#93c5fd;font-size:0.88em;font-weight:600;">{basis}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <span style="color:#64748b;font-size:0.78em;">RELEASE: </span>
                <span style="font-size:0.88em;font-weight:700;">{rec['icon']} {rec['label']}</span>
                </div>""",unsafe_allow_html=True)

            st.markdown('<div class="section-title">Compliance Scores</div>',unsafe_allow_html=True)
            c1,c2,c3,c4,c5=st.columns(5)
            fused_color="#10b981" if sc["fused_score"]<40 else ("#f59e0b" if sc["fused_score"]<70 else "#ef4444")
            for col,(val,label_,color_) in zip([c1,c2,c3,c4,c5],[
                (sc["ml_score"],"ML Non-Comp Prob","#3b82f6"),
                (sc["rule_score"],"Rule Severity","#8b5cf6"),
                (sc["fused_score"],"Fused Risk",fused_color),
                (f"{res['ml_confidence']}%","ML Confidence","#06b6d4"),
                (res["reconstruction_error"],"Recon Error","#a78bfa")]):
                col.markdown(f'<div class="metric-card"><div class="label">{label_}</div>'
                    f'<div class="value" style="color:{color_};">{val}</div></div>',unsafe_allow_html=True)

            st.markdown('<div class="section-title">ML Class Probabilities</div>',unsafe_allow_html=True)
            probs=res.get("ml_probs",{})
            p1,p2,p3=st.columns(3)
            for col,cls_,color_ in zip([p1,p2,p3],
                ["Compliant","Review-Needed","Non-Compliant"],["#10b981","#f59e0b","#ef4444"]):
                pct=probs.get(cls_,0.0)
                col.markdown(f'<div class="metric-card"><div class="label">{cls_}</div>'
                    f'<div class="value" style="color:{color_};">{pct}%</div></div>',unsafe_allow_html=True)
                col.progress(int(pct))

            st.markdown('<div class="section-title">Violation Summary</div>',unsafe_allow_html=True)
            vc1,vc2,vc3,vc4=st.columns(4)
            vc1.metric("🔴 Critical",sc["critical"]); vc2.metric("🟡 Warning",sc["warnings"])
            vc3.metric("🔵 Info",sc["info"]); vc4.metric("✅ Passed",res["n_passed"])

            # Charts
            st.markdown('<div class="section-title">Analytics Dashboard</div>',unsafe_allow_html=True)
            fig,axes=plt.subplots(1,3,figsize=(18,5),facecolor="#0a0e1a")
            g_col=fused_color; score=sc["fused_score"]
            axes[0].pie([score,100-score],colors=[g_col,"#1e293b"],startangle=90,
                counterclock=False,wedgeprops={"width":0.32})
            axes[0].text(0,0,f"{score}%",ha="center",va="center",color="white",fontsize=26,fontweight="bold")
            axes[0].set_title("Compliance Risk Score",color="white",fontweight="bold",pad=15)
            axes[0].set_facecolor("#0a0e1a")
            axes[1].set_facecolor("#0a0e1a")
            cats=["Critical","Warning","Info"]; cnts=[sc["critical"],sc["warnings"],sc["info"]]
            if sum(cnts)==0:
                axes[1].bar(["No Violations"],[1],color=["#10b981"])
                axes[1].text(0,1.1,"✅ All Clear",ha="center",color="#10b981",fontweight="bold")
            else:
                bars=axes[1].bar(cats,cnts,color=["#ef4444","#f59e0b","#3b82f6"])
                for bar in bars: axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,
                    f"{int(bar.get_height())}",ha="center",color="white",fontweight="bold")
            axes[1].tick_params(colors="white"); axes[1].set_title("Severity Distribution",color="white",fontweight="bold")
            axes[1].grid(axis="y",color="#1e293b",linestyle="--")
            axes[2].set_facecolor("#0a0e1a")
            fl=["ML Score","Rule Score","Fused"]; fv=[sc["ml_score"],sc["rule_score"],sc["fused_score"]]
            bars3=axes[2].bar(fl,fv,color=["#3b82f6","#8b5cf6",g_col])
            for bar in bars3: axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                f"{bar.get_height():.1f}",ha="center",color="white",fontweight="bold")
            axes[2].set_ylim(0,115); axes[2].tick_params(colors="white")
            axes[2].set_title("Model Fusion Breakdown",color="white",fontweight="bold")
            axes[2].grid(axis="y",color="#1e293b",linestyle="--")
            plt.tight_layout(pad=3.0); st.pyplot(fig); plt.close()

            # Assembly graph
            comps=st.session_state.get("last_components",[])
            if comps:
                st.markdown('<div class="section-title">Assembly Topology Graph</div>',unsafe_allow_html=True)
                G_vis,labels_vis=_build_assembly_graph(comps)
                n_vis=len(comps); violated_nodes={v.node for v in res["violations"] if isinstance(v.node,int)}
                node_colors=["#ef4444" if i in violated_nodes else "#10b981" for i in range(n_vis)]
                fig2,ax_g=plt.subplots(figsize=(14,6),facecolor="#0a0e1a"); ax_g.set_facecolor("#0d1117")
                try:
                    pos=nx.spring_layout(G_vis,k=2.5/max(1,np.sqrt(n_vis)),seed=42)
                    nx.draw_networkx_edges(G_vis,pos,ax=ax_g,edge_color="#374151",width=1.5,arrowsize=15)
                    nx.draw_networkx_nodes(G_vis,pos,ax=ax_g,node_color=node_colors,node_size=900,
                        edgecolors="white",linewidths=1.2)
                    nx.draw_networkx_labels(G_vis,pos,labels=labels_vis,ax=ax_g,
                        font_color="white",font_weight="bold",font_size=8)
                    ax_g.legend(handles=[mpatches.Patch(color="#ef4444",label="Violated"),
                        mpatches.Patch(color="#10b981",label="Compliant")],
                        facecolor="#1e293b",edgecolor="#374151",labelcolor="white",fontsize=9)
                except Exception: ax_g.text(0.5,0.5,"Graph unavailable",transform=ax_g.transAxes,ha="center",color="white")
                ax_g.axis("off"); ax_g.set_title("Assembly Relationship View",color="white",fontweight="bold",pad=14)
                st.pyplot(fig2); plt.close()

                st.markdown('<div class="section-title">3D Digital Twin Preview</div>', unsafe_allow_html=True)
                fig_3d = render_cad_3d_model(comps, res.get("violations", []))
                if fig_3d is not None:
                    st.plotly_chart(fig_3d, use_container_width=True, config={"displaylogo": False})


            # Uncertainty
            recon_err=float(res.get("reconstruction_error",0.0))
            unc_level=("Low 🟢" if recon_err<1.0 else ("Moderate 🟡" if recon_err<15.0 else "High 🔴"))
            st.markdown('<div class="section-title">Model Uncertainty</div>',unsafe_allow_html=True)
            uc1,uc2=st.columns([3,1])
            with uc1: st.progress(min(100,int(recon_err*5)),text=f"Reconstruction Error: {recon_err:.4f}")
            with uc2: st.markdown(f"**{unc_level}**")

            # Top reasons
            top3=res.get("top_reasons",[])
            if top3:
                st.markdown('<div class="section-title">Top Violation Reasons</div>',unsafe_allow_html=True)
                for r in top3:
                    sev=r.get("severity","INFO")
                    tag_cls={"CRITICAL":"#ef4444","WARNING":"#f59e0b"}.get(sev,"#3b82f6")
                    st.markdown(f"""<div style="background:#0f172a;border:1px solid #1e293b;
                        border-radius:8px;padding:10px 14px;margin:5px 0;">
                        <span style="background:{tag_cls}20;color:{tag_cls};padding:2px 8px;
                        border-radius:4px;font-size:0.75em;font-weight:700;">{sev}</span>
                        <span style="color:#94a3b8;font-size:0.85em;margin-left:8px;">
                        Node {r.get('node','?')} | {r.get('rule_id','')}</span>
                        <span style="color:#e2e8f0;font-size:0.88em;margin-left:8px;">
                        → {r.get('message','')}</span></div>""",unsafe_allow_html=True)

            # Full violation log
            violations=res.get("violations",[])
            if violations:
                st.markdown('<div class="section-title">Full Violation Log</div>',unsafe_allow_html=True)
                with st.expander(f"Show all {len(violations)} violations"):
                    for v in violations:
                        sev_cls={"CRITICAL":"viol-crit","WARNING":"viol-warn"}.get(v.severity,"viol-info")
                        st.markdown(f'<div class="{sev_cls}"><span class="viol-txt">'
                            f'[{v.severity}] Node {v.node} | {v.rule_id} | '
                            f'Value: {v.value} | Limit: {v.limit} | {v.suggestion}'
                            f'</span></div>',unsafe_allow_html=True)

            # Passed checks
            passed=res.get("passed_checks",[])
            with st.expander(f"✅ {len(passed)} passed checks"):
                pdf=pd.DataFrame([{"Node":p.node,"Rule":p.rule_id,"Value":p.value,"Limit":p.limit}
                    for p in passed[:50]])
                if not pdf.empty: st.dataframe(pdf,use_container_width=True)

            # Remediation advisor
            st.markdown('<div class="section-title">🔧 Remediation Advisor</div>',unsafe_allow_html=True)
            for i,action in enumerate(_get_top3_remediations(res) or [],1):
                st.markdown(f"""<div style="background:#0f172a;border-left:4px solid #3b82f6;
                    border-radius:0 8px 8px 0;padding:10px 16px;margin:6px 0;">
                    <span style="color:#3b82f6;font-weight:700;margin-right:8px;">{i}.</span>
                    <span style="color:#cbd5e1;font-size:0.9em;">{action}</span></div>""",
                    unsafe_allow_html=True)
            st.markdown('<div class="section-title">AI Explainability</div>', unsafe_allow_html=True)

            explain_text = sc.get("explanation", "N/A")
            override_text = sc.get("override_reason", "")

            explain_html = (
                '<div style="background:#0f172a;'
                'border:1px solid #1e293b;'
                'border-radius:10px;'
                'padding:14px 18px;'
                'color:#94a3b8;'
                'font-size:0.88em;'
                'line-height:1.8;">'
                f'<div>{explain_text}</div>'
            )

            if override_text:
                explain_html += (
                    '<div style="margin-top:8px;">'
                    '<span style="color:#ef4444;font-weight:700;">Override:</span>'
                    f'<span style="color:#cbd5e1;"> {override_text}</span>'
                    '</div>'
                )

            explain_html += '</div>'

            st.markdown(explain_html, unsafe_allow_html=True)
            # Input warnings
            for w in res.get("input_warnings",[]): st.warning(w)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — MODEL ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_m:
        if "pipeline" not in st.session_state:
            st.info("Train model first.")
        else:
            model,scaler,metrics,report,cm=st.session_state["pipeline"]
            st.markdown('<div class="section-title">GNN Test Set Performance</div>',unsafe_allow_html=True)
            m1,m2,m3,m4=st.columns(4)
            for col,(k,v),color in zip([m1,m2,m3,m4],metrics.items(),
                ["#10b981","#3b82f6","#f59e0b","#8b5cf6"]):
                col.markdown(f'<div class="metric-card"><div class="label">{k}</div>'
                    f'<div class="value" style="color:{color};">{v}%</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="section-title">Classification Report</div>',unsafe_allow_html=True)
            st.code(report,language="")
            st.markdown('<div class="section-title">Confusion Matrix</div>',unsafe_allow_html=True)
            fig_cm,ax_cm=plt.subplots(figsize=(7,5),facecolor="#0a0e1a"); ax_cm.set_facecolor("#0a0e1a")
            im=ax_cm.imshow(cm,cmap="Blues")
            for axis,lbl in [(ax_cm.xaxis,"X"),(ax_cm.yaxis,"Y")]:
                pass
            ax_cm.set_xticks([0,1,2]); ax_cm.set_yticks([0,1,2])
            ax_cm.set_xticklabels(["Compliant","Review","Non-Compliant"],color="white",fontsize=9)
            ax_cm.set_yticklabels(["Compliant","Review","Non-Compliant"],color="white",fontsize=9)
            ax_cm.set_xlabel("Predicted",color="white"); ax_cm.set_ylabel("Actual",color="white")
            ax_cm.set_title("Confusion Matrix",color="white",fontweight="bold")
            for i in range(3):
                for j in range(3):
                    ax_cm.text(j,i,str(cm[i][j]),ha="center",va="center",
                        color="white",fontweight="bold",fontsize=12)
            plt.colorbar(im,ax=ax_cm); st.pyplot(fig_cm); plt.close()
            st.markdown('<div class="section-title">Model Architecture</div>',unsafe_allow_html=True)
            total=sum(p.numel() for p in model.parameters())
            st.markdown(f"""| Component | Details |
|---|---|
| **Encoder** | 3× TransformerConv (edge-aware, edge_dim=2) |
| **Dims** | {cfg.HIDDEN_DIM}→{cfg.HIDDEN_DIM}→{cfg.LATENT_DIM} |
| **Decoder** | Autoencoder (latent→{cfg.IN_CHANNELS}) |
| **Classifier** | {cfg.LATENT_DIM}→32→16→3 |
| **Total Params** | `{total:,}` |
| **Device** | `{DEVICE}` |
| **Standards** | ISO 2768, ASME Y14.5, GD&T, DFM |""")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — EXPORT
    # ═════════════════════════════════════════════════════════════════════════
    with tab_r:
        if "last_result" not in st.session_state:
            st.info("Run a validation first.")
        else:
            import datetime
            res=st.session_state["last_result"]; sc=res["scores"]
            org=st.session_state.get("org_name","N/A"); prod=st.session_state.get("prod_name","N/A")
            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            basis=_compute_decision_basis(res); rec=_get_release_recommendation(sc["verdict"])
            top3_rem=_get_top3_remediations(res)
            lines=["="*70,"  GNR-VAL v5.0 — INDUSTRIAL CAD COMPLIANCE REPORT",
                "  Varroc Eureka 3.0 · Problem Statement 9","="*70,
                f"  Generated   : {ts}",f"  Organisation: {org}",f"  Product     : {prod}","",
                "FINAL VERDICT","-"*70,
                f"  Verdict        : {rec['icon']} {sc['verdict']}",
                f"  Release Status : {rec['label']}",f"  Decision Basis : {basis}","",
                "COMPLIANCE SCORES","-"*70,
                f"  ML Non-Compliance Prob : {sc['ml_score']}",
                f"  Rule Severity Score    : {sc['rule_score']}",
                f"  Fused Risk Score       : {sc['fused_score']}",
                f"  ML Prediction          : {res['ml_prediction'].upper()}",
                f"  ML Confidence          : {res['ml_confidence']}%",
                f"  Reconstruction Error   : {res['reconstruction_error']}","",
                "VIOLATION SUMMARY","-"*70,
                f"  Critical: {sc['critical']} | Warning: {sc['warnings']} | Info: {sc['info']} | Passed: {res['n_passed']}","",
                "AI EXPLAINABILITY","-"*70,f"  {sc.get('explanation','N/A')}"]
            if sc.get("override_reason"): lines.append(f"  Override: {sc['override_reason']}")
            lines+=["","TOP REMEDIATION ACTIONS","-"*70]
            for i,a in enumerate(top3_rem,1): lines.append(f"  {i}. {a}")
            lines+=["","VIOLATIONS LOG","-"*70]
            for v in res.get("violations",[]):
                lines.append(f"  [{v.severity}] Node {v.node} | {v.rule_id} | Value:{v.value} | Limit:{v.limit} | {v.suggestion}")
            lines+=["","="*70,"  END OF REPORT — GNR-VAL v5.0","="*70]
            report_text="\n".join(lines)
            fname=f"GNRVal_Report_{ts.replace(' ','_').replace(':','-')}.txt"
            st.markdown('<div class="section-title">Report Preview</div>',unsafe_allow_html=True)
            st.code(report_text,language="")
            json_data={"timestamp":ts,"org":org,"product":prod,"verdict":sc["verdict"],
                "scores":sc,"ml_prediction":res["ml_prediction"],"ml_confidence":res["ml_confidence"],
                "violations":[v.to_dict() for v in res["violations"]],
                "top_reasons":res["top_reasons"],"remediations":top3_rem}
            col_t,col_j=st.columns(2)
            with col_t: st.download_button("⬇️ Download TXT",data=report_text,
                file_name=fname,mime="text/plain",use_container_width=True)
            with col_j: st.download_button("⬇️ Download JSON",
                data=json.dumps(json_data,indent=2),
                file_name=fname.replace(".txt",".json"),mime="application/json",use_container_width=True)

    st.markdown("""<div style="text-align:center;color:#1e293b;font-size:0.75em;margin-top:40px;
        border-top:1px solid #1e293b;padding-top:16px;">
        GNR-Val v5.0 · TransformerConv GNN + ISO 2768 / ASME Y14.5 Rule Engine · Varroc Eureka 3.0
        </div>""",unsafe_allow_html=True)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--fusion-cli":
        fusion_cli_entry()
    else:
        main()
