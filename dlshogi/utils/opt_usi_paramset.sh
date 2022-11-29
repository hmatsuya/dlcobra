#!/bin/bash

study_name=highlow

command=/home/hmatsuya/workspace/Shogi/DeepLearningShogiOriginal/usi/bin/usi

games=300
byoyomi=800

opening=/home/hmatsuya/workspace/Shogi/20221128_matsuyamasan/even/kgr.even28.sfen
opening_moves=28
opening_seed=42

threads=2

options=UCT_Threads:$threads,PV_Mate_Search_Threads:1,Resign_Threshold:200,Time_Margin:0,Byoyomi_Margin:0
options1=DNN_Model:/home/hmatsuya/workspace/Shogi/20221128_matsuyamasan/model/model_20220808_Ryfc15_610.onnx,$options
options2=DNN_Model:/home/hmatsuya/workspace/Shogi/dlcobra/model/model-dr2_exhi/model-dr2_exhi.onnx,$options

python usi_params_optimizer_set.py --trials 2 --byoyomi $byoyomi --games $games --opening $opening --opening_moves 28 --opening_seed $opening_seed --options1 $options1 --options2 $options2 --name testing --study_name $study_name $command $command
