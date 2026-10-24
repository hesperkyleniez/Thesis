"""Train the final AUC-selected F0 MLP used by Akıncı CQCC prediction fusion.
Do not overwrite the historical standard-F0 checkpoint. Output is F0-v25-AUC.
"""
from improvedCQCC_f0_train import train_auc_f0
if __name__=='__main__':train_auc_f0(stage='final')
