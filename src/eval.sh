clear

echo "====   EVAL for BRATS-GLI 226    ===="

NUMBER=226
DATASET_NAME="Dataset${NUMBER}_BraTS2024-BraTS-GLI"

BASELINE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_150ep__ptPlans__CNN_BaseMAEBS8_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
LP_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_150ep__ptPlans__CNN_KVMAE_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_150ep__ptPlans__CNN_ConMAEFT_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
ALIGNED_CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_150ep__ptPlans__CNN_AlignedMAE_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
VAR_ALIGNED_CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_150ep__ptPlans__CNN_VarAligned2MAEFT_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/

EVA_BASELINE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_Primus_150ep__ptPlans__EVA_BaseMAETrainerBS8_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
EVA_LP_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_Primus_150ep__ptPlans__EVA_KVMAE_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
EVA_CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_Primus_150ep__ptPlans__EVA_ConMAEFTEva_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
EVA_ALIGNED_CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_Primus_150ep__ptPlans__EVA_AlignedMAEFTLR3Eva_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/
EVA_VAR_ALIGNED_CON_MAE_DIR=/deepstore/datasets/mia/HealthyAI/nnUNet_models/${DATASET_NAME}/PretrainedTrainer_Primus_150ep__ptPlans__EVA_VarAligned2MAEFTEva_0${NUMBER}_Dataset_745____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z__3d_fullres/fold_50-50/

# get the last log file in terms of data time in each directory
BASELINE_LOG=$(ls "${BASELINE_DIR}"/training_log* | sort | tail -1)
LP_MAE_LOG=$(ls "${LP_MAE_DIR}"/training_log* | sort | tail -1)
CON_MAE_LOG=$(ls "${CON_MAE_DIR}"/training_log* | sort | tail -1)
ALIGNED_CON_MAE_LOG=$(ls "${ALIGNED_CON_MAE_DIR}"/training_log* | sort | tail -1)
VAR_ALIGNED_CON_MAE_LOG=$(ls "${VAR_ALIGNED_CON_MAE_DIR}"/training_log* | sort | tail -1)

EVA_BASELINE_LOG=$(ls "${EVA_BASELINE_DIR}"/training_log* | sort | tail -1)
EVA_LP_MAE_LOG=$(ls "${EVA_LP_MAE_DIR}"/training_log* | sort | tail -1)
EVA_CON_MAE_LOG=$(ls "${EVA_CON_MAE_DIR}"/training_log* | sort | tail -1)
EVA_ALIGNED_CON_MAE_LOG=$(ls "${EVA_ALIGNED_CON_MAE_DIR}"/training_log* | sort | tail -1)
EVA_VAR_ALIGNED_CON_MAE_LOG=$(ls "${EVA_VAR_ALIGNED_CON_MAE_DIR}"/training_log* | sort | tail -1)

# === FINAL METRICS ===
echo ""
echo "==== Final Metrics (Pseudo dice) ===="

echo "RESNEC - L"
echo ""

echo "Baseline"
grep -oP 'Epoch [0-9]+' "$BASELINE_LOG" | tail -1
grep "Pseudo dice" "$BASELINE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "LP MAE"
grep -oP 'Epoch [0-9]+' "$LP_MAE_LOG" | tail -1
grep "Pseudo dice" "$LP_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Contrastive MAE"
grep -oP 'Epoch [0-9]+' "$CON_MAE_LOG" | tail -1
grep "Pseudo dice" "$CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Aligned Contrastive MAE"
grep -oP 'Epoch [0-9]+' "$ALIGNED_CON_MAE_LOG" | tail -1
grep "Pseudo dice" "$ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Var Aligned Contrastive MAE"
grep -oP 'Epoch [0-9]+' "$VAR_ALIGNED_CON_MAE_LOG" | tail -1
grep "Pseudo dice" "$VAR_ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo ""
echo "PRIMUS - M"
echo ""

echo "Eva Baseline"
grep -oP 'Epoch [0-9]+' "$EVA_BASELINE_LOG" | tail -1
grep "Pseudo dice" "$EVA_BASELINE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Eva LP MAE"
grep -oP 'Epoch [0-9]+' "$EVA_LP_MAE_LOG" | tail -1
grep "Pseudo dice" "$EVA_LP_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Eva Contrastive MAE"
grep -oP 'Epoch [0-9]+' "$EVA_CON_MAE_LOG" | tail -1
grep "Pseudo dice" "$EVA_CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Eva Aligned Contrastive MAE"
grep -oP 'Epoch [0-9]+' "$EVA_ALIGNED_CON_MAE_LOG" | tail -1
grep "Pseudo dice" "$EVA_ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'

echo "Eva Var Aligned Contrastive MAE"
if [ -n "$EVA_VAR_ALIGNED_CON_MAE_LOG" ]; then
    grep -oP 'Epoch [0-9]+' "$EVA_VAR_ALIGNED_CON_MAE_LOG" | tail -1
    grep "Pseudo dice" "$EVA_VAR_ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'np\.float32\(\K[0-9.]+'
else
    echo "No Var Aligned Contrastive MAE log available."
fi

echo ""

# === BEST METRICS ===
echo ""
echo "==== Best Metrics (Pseudo dice) ===="

echo "RESNEC - L"
echo ""

echo -n "Baseline: "
grep "Pseudo dice" "$BASELINE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "LP MAE: "
grep "Pseudo dice" "$LP_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Contrastive MAE: "
grep "Pseudo dice" "$CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Aligned Contrastive MAE: "
grep "Pseudo dice" "$ALIGNED_CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Var Aligned Contrastive MAE: "
grep "Pseudo dice" "$VAR_ALIGNED_CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo ""
echo "PRIMUS - M"
echo ""

echo -n "Eva Baseline: "
grep "Pseudo dice" "$EVA_BASELINE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Eva LP MAE: "
grep "Pseudo dice" "$EVA_LP_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Eva Contrastive MAE: "
grep "Pseudo dice" "$EVA_CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Eva Aligned Contrastive MAE: "
grep "Pseudo dice" "$EVA_ALIGNED_CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1

echo -n "Eva Var Aligned Contrastive MAE: "
if [ -n "$EVA_VAR_ALIGNED_CON_MAE_LOG" ]; then
    grep "Pseudo dice" "$EVA_VAR_ALIGNED_CON_MAE_LOG" | grep -oP 'np\.float32\(\K[0-9.]+' | sort -nr | head -1
else
    echo "No Var Aligned Contrastive MAE log available."
fi
echo ""

echo "==== Best Logged Metrics (Yayy!) ===="

echo "RESENC - L"
echo ""

# example string to get the number from: "2025-07-29 10:47:22.212630: Yayy! New best EMA pseudo Dice: 0.8055999875068665"
# we only want the number after "Yayy! New best EMA pseudo Dice: "
echo -n "Baseline: "
grep "Yayy!" "$BASELINE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "LP MAE: "
grep "Yayy!" "$LP_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Contrastive MAE: "
grep "Yayy!" "$CON_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Aligned Contrastive MAE: "
grep "Yayy!" "$ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Var Aligned Contrastive MAE: "
grep "Yayy!" "$VAR_ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo ""
echo "PRIMUS - M"
echo ""

echo -n "Eva Baseline: "
grep "Yayy!" "$EVA_BASELINE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Eva LP MAE: "
grep "Yayy!" "$EVA_LP_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Eva Contrastive MAE: "
grep "Yayy!" "$EVA_CON_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Eva Aligned Contrastive MAE: "
grep "Yayy!" "$EVA_ALIGNED_CON_MAE_LOG" | tail -1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'

echo -n "Eva Var Aligned Contrastive MAE: "
if [ -n "$EVA_VAR_ALIGNED_CON_MAE_LOG" ]; then
    grep "Yayy!" "$EVA_VAR_ALIGNED_CON_MAE_LOG" | tail - 1 | grep -oP 'Yayy! New best EMA pseudo Dice: \K\d+\.\d+'
else
    echo "No Var Aligned Contrastive MAE log available."
fi

echo ""

echo "====   EVAL for BRATS-GLI 226 Completed!   ===="
