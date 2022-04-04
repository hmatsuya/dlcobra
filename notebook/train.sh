#!/bin/bash

# 学習するエポックの閾値の上限値。72エポック（24分割した教師データ*3サイクル）学習する。
# last=72
last = 240

# 引数がある場合は上書きする
if [ $# -ge 1 ]; then
    last=$(($1))
fi

# 変数設定
name="resnet10_swish"
log_dir="./log"
model_dir="./model"
data_dir="./data"

# 最新のチェックポイント+1から学習を再開する。
for i in $(ls -v ${model_dir}/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$i; done
if [ -v chkp ]; then
    start=$(expr ${chkp: -7:3} + 1)
else
    start=1
fi
    
echo start: $1

for ((i=$start; i<=$last; i++)); do
    iii=$(printf "%03d" $i)
    jjj=$(printf "%03d" $((i-1)))
    kkk=$(printf "%03d" $(((i-1) % 24 + 1)))

    # floodgate、水匠3改、dlshogi_with_gctの教師データ（24分割）を順番に学習する。
    src="${data_dir}/floodgate_2019-2021_r3500-${kkk}.hcpe ${data_dir}/suisho3kai-${kkk}.hcpe ${data_dir}/dlshogi_with_gct-${kkk}.hcpe"

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続する。
    if [ $i -eq 1 ]; then
        resume=""
    else
        resume="-r ${model_dir}/checkpoint_${name}-${jjj}.pth"
    fi

    # 最終エポックの学習のみモデルファイルを保存する。
    if [ $i -eq $last ]; then
        model="--model ${model_dir}/model_${name}-{epoch:03}"
    else
#         model=""
        model="--model ${model_dir}/model_${name}-{epoch:03}"
    fi

    # チェックポイントファイル名
    checkpoint="${model_dir}/checkpoint_${name}-{epoch:03}.pth"

    # ログファイル名
    log="${log_dir}/${name}-${iii}.txt"

    # 前回中断時のログファイルがある場合、削除する。
    if [ -e ${log} ]; then
        rm -f ${log}
    fi

    echo epoch ${i} start.

    # 学習
#     python -m dlshogi.train                                            \
#               ${src}                                                   \
#               ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe \
#               --network ${name}                                        \
#               ${resume}                                                \
#               --checkpoint ${checkpoint}                               \
#               ${model}                                                 \
#               --lr_scheduler "StepLR(step_size=24,gamma=0.1)"          \
#               --use_swa                                                \
#               --swa_start_epoch 10                                     \
#               --use_average                                            \
#               --use_evalfix                                            \
#               $2 $3 $4 $5 $6 $7 $8 $9 | tee ${log}

    python -m dlshogi.train                                            \
              ${src}                                                   \
              ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe \
              --network ${name}                                        \
              ${resume}                                                \
              --checkpoint ${checkpoint}                               \
              ${model}                                                 \
              --lr_scheduler "StepLR(step_size=24,gamma=0.1)"          \
              --use_average                                            \
              --use_evalfix                                            \
              $2 $3 $4 $5 $6 $7 $8 $9 | tee ${log}
    
    if [ $? -ne 0 ]; then
        break
    fi
done
